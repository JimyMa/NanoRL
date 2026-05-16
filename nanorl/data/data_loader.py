"""SlimeRPC client used by the train side to pull trajectory batches.

A thin wrapper around ``dlslime.rpc.proxy`` that hides the pickle/unpickle
step and offers a small in-process prefetch queue so train steps and RPC
fetches overlap. Failures back off exponentially and stop after a cap so a
broken fabric doesn't spam the log.
"""

from __future__ import annotations

import logging
import pickle
import queue
import threading
import time
from dataclasses import dataclass

from dlslime import PeerAgent

from .sample import Trajectory, TrajectoryBatch
from .trajectory_buffer import open_consumer, TrajectoryService

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryClientCfg:
    producer_alias: str
    pull_size: int = 16
    prefetch_depth: int = 2
    pull_timeout_s: float = 30.0
    initial_settle_s: float = 0.2
    backoff_min_s: float = 0.1
    backoff_max_s: float = 5.0
    max_consecutive_failures: int = 50


class TrajectoryClient:
    """Pull-side of the trajectory data plane.

    Spawns a background thread that keeps `prefetch_depth` batches inflight,
    draining the producer in pull-of-N chunks. ``next_batch`` blocks the
    train loop only when no prefetched batch is ready.
    """

    def __init__(self, agent: PeerAgent, cfg: TrajectoryClientCfg):
        self._agent = agent
        self._cfg = cfg
        self._proxy = open_consumer(agent, cfg.producer_alias)
        self._q: queue.Queue[list[Trajectory]] = queue.Queue(maxsize=cfg.prefetch_depth)
        self._stop = threading.Event()
        self._fatal: BaseException | None = None
        # The dlslime example sleeps ~100ms after serve()/proxy() before
        # the first RPC; without it the first WR can race recv-WR posting
        # on the remote side and hit IBV_WC_RETRY_EXC_ERR.
        if cfg.initial_settle_s > 0:
            time.sleep(cfg.initial_settle_s)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="traj-prefetch"
        )
        self._thread.start()

    def _loop(self):
        consecutive_failures = 0
        backoff = self._cfg.backoff_min_s
        while not self._stop.is_set():
            try:
                fut = self._proxy.pull_batch(
                    self._cfg.pull_size, self._cfg.pull_timeout_s
                )
                payload: bytes = fut.wait(timeout=self._cfg.pull_timeout_s + 5.0)
                samples: list[Trajectory] = pickle.loads(payload)
            except Exception as exc:
                consecutive_failures += 1
                logger.warning(
                    "pull_batch failed (#%d/%d): %s",
                    consecutive_failures,
                    self._cfg.max_consecutive_failures,
                    exc,
                )
                if consecutive_failures >= self._cfg.max_consecutive_failures:
                    self._fatal = exc
                    self._stop.set()
                    return
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, self._cfg.backoff_max_s)
                continue

            consecutive_failures = 0
            backoff = self._cfg.backoff_min_s
            if not samples:
                continue
            try:
                self._q.put(samples, timeout=1.0)
            except queue.Full:
                self._q.get_nowait()
                self._q.put_nowait(samples)

    def next_batch(
        self, batch_size: int, pad_id: int = 0, timeout: float | None = None
    ) -> TrajectoryBatch:
        """Block until at least `batch_size` trajectories are available.

        Raises if the prefetch thread has hit the failure cap.
        """
        collected: list[Trajectory] = []
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(collected) < batch_size:
            if self._fatal is not None:
                raise RuntimeError(f"trajectory prefetch thread died: {self._fatal!r}")
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            chunk = self._q.get(timeout=remaining)
            collected.extend(chunk)
        keep, leftover = collected[:batch_size], collected[batch_size:]
        if leftover:
            try:
                self._q.put_nowait(leftover)
            except queue.Full:
                pass
        return TrajectoryBatch.from_trajectories(keep, pad_id=pad_id)

    def close(self) -> None:
        self._stop.set()
