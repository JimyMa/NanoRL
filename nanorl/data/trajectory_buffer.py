"""Trajectory transport over SlimeRPC, plus the M3 weight-update RPC.

Defines the ``TrajectoryService`` contract shared between producer (rollout
actor) and consumer (train actor). The same class is imported on both
sides — DLSlime's RPC layer uses the class definition for method dispatch
on the server and method binding on the proxy.

Producer side: instantiate ``TrajectoryService(...)``, push samples with
``publish``, run ``run_rpc_server`` in a thread.

Consumer side: build a proxy with ``open_consumer`` and call ``pull_batch``.

M3 additions: the rollout side may also pass a ``llm`` and a
``weight_transport`` (a ``WeightTransportRollout``) into the service. When
present, the train side can RPC-call ``apply_weight_update`` with a
pickled ``WeightManifest``; we pull weights via RDMA, fan them out to
NanoInfra workers via ``llm.update_weights``, release the receive
buffers, and return per-worker counts.
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from collections import deque
from typing import Any, Iterable

from dlslime import PeerAgent
from dlslime.rpc import method, proxy, serve

from .sample import Trajectory

logger = logging.getLogger(__name__)


class TrajectoryService:
    """The wire contract. Methods on this class are the RPC surface.

    Both server and client must import this exact class. Server calls return
    pickled-bytes payloads (`@method` default); client unpickles before
    handing to ``TrajectoryClient``. We keep the wire shape minimal so we can
    swap to a flatbuffer / raw mode later without changing call sites.

    Optional M3 fields on the producer side:
        ``llm``               — a ``LLMComponent`` (the patched NanoInfra
                                engine with ``update_weights``)
        ``weight_transport``  — a ``WeightTransportRollout`` already paired
                                to the train alias

    Both default to None, in which case the weight-sync RPC raises a
    clean error rather than silently no-op'ing.
    """

    def __init__(
        self,
        capacity: int = 4096,
        *,
        llm: Any = None,
        weight_transport: Any = None,
    ):
        self._buf: deque[Trajectory] = deque()
        self._capacity = capacity
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._llm = llm
        self._weight_transport = weight_transport

    # --- producer-side helpers (NOT @method; called locally) ---

    def publish(self, samples: Iterable[Trajectory]) -> int:
        """Append `samples` to the buffer; returns new length."""
        added = 0
        with self._not_empty:
            for s in samples:
                if len(self._buf) >= self._capacity:
                    self._buf.popleft()  # drop-oldest backpressure
                self._buf.append(s)
                added += 1
            self._not_empty.notify_all()
            return len(self._buf)

    def buffered(self) -> int:
        with self._lock:
            return len(self._buf)

    def attach_weight_path(self, llm: Any, weight_transport: Any) -> None:
        """Late-bind the LLM + weight transport. Useful when the service is
        constructed before the engine has finished booting."""
        self._llm = llm
        self._weight_transport = weight_transport

    # --- RPC surface ---

    @method
    def pull_batch(self, n: int, timeout_s: float = 30.0) -> bytes:
        """Block until at least one trajectory available, then return up to `n`.

        Returns pickled ``list[Trajectory]``. Empty list on timeout.
        """
        deadline = time.monotonic() + timeout_s
        with self._not_empty:
            while not self._buf:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return pickle.dumps([])
                self._not_empty.wait(timeout=remaining)
            take = min(n, len(self._buf))
            out = [self._buf.popleft() for _ in range(take)]
        return pickle.dumps(out)

    @method
    def stats(self) -> bytes:
        with self._lock:
            return pickle.dumps(
                {"buffered": len(self._buf), "capacity": self._capacity}
            )

    @method
    def apply_weight_update(self, manifest_blob: bytes) -> bytes:
        """Pull a versioned weight set from the train side and apply to LLM.

        Fast path: the manifest is forwarded to each NanoInfra worker via
        ``LLMComponent.pull_and_apply_weights``; each worker uses its own
        ``PeerAgent`` to RDMA-read in parallel from the train side. This
        bypasses the rollout-driver→Ray cross-host serialization that was
        the bottleneck in the original design.

        Falls back to the slow ``llm.update_weights`` path (rollout
        driver pulls, then Ray-fans-out the dict) if no worker-side
        ``PeerAgent`` is available — useful for single-process tests.

        Returns pickled ``{version, n_tensors, pull_s, apply_s, counts}``.

        Raises ``RuntimeError`` if the service was constructed without an
        ``llm`` (i.e. the rollout-only CLI was started without the M3
        wiring).
        """
        if self._llm is None:
            raise RuntimeError(
                "TrajectoryService was started without llm; the rollout-only "
                "CLI needs to be invoked with M3 enabled."
            )
        t0 = time.monotonic()
        manifest = pickle.loads(manifest_blob)
        # Fast path: workers pull directly. The driver does NO RDMA, just
        # fan-outs the (small) manifest blob via Ray.
        per_worker_stats = self._llm.pull_and_apply_weights(
            manifest_blob,
            manifest.train_alias,
        )
        elapsed = time.monotonic() - t0
        # Synthesize a stats dict in the same shape the slow path returned
        # so callers don't need to branch.
        result = {
            "version": manifest.version,
            "n_tensors": len(manifest.entries),
            "pull_s": max(
                (s.get("pull_s", 0.0) for s in per_worker_stats), default=0.0
            ),
            "apply_s": max(
                (s.get("apply_s", 0.0) for s in per_worker_stats), default=0.0
            ),
            "wall_s": elapsed,
            "counts": [
                {
                    k: v
                    for k, v in s.items()
                    if k
                    in (
                        "loaded",
                        "skipped_unknown",
                        "used_loader_cb",
                        "used_direct_copy",
                    )
                }
                for s in per_worker_stats
            ],
        }
        logger.info("apply_weight_update (direct-pull) %s", result)
        return pickle.dumps(result)


def run_rpc_server(
    agent: PeerAgent,
    service: TrajectoryService,
    consumer_alias: str,
    daemon: bool = True,
    connection=None,
    connect_timeout_s: float = 600.0,
    serve_settle_s: float = 0.2,
) -> threading.Thread:
    """Start the SlimeRPC ``serve`` loop in a background thread.

    If `connection` is provided (the object returned by
    ``PeerAgent.connect_to(...)``), the serve thread waits for it to come up
    *before* calling ``serve``. Without that, dlslime's ``serve`` raises
    ``ValueError("requires a connected endpoint")`` on producers that come
    up before any consumer registers — see ``dlslime/rpc/proxy.py:76``.

    `serve_settle_s` is a brief sleep after `serve()` returns control of the
    main path to its background workers — gives dlslime a beat to post
    recv WRs before any inbound RPC arrives, mirroring the
    ``time.sleep(0.1)`` in dlslime's own ``rpc_example.py``. Without it the
    first remote send can hit IBV_WC_RETRY_EXC_ERR.

    The thread is daemonized by default so it exits with the process.
    """

    def _target():
        if connection is not None:
            try:
                connection.wait(timeout=connect_timeout_s)
            except Exception as exc:
                logger.error(
                    "trajectory rpc server: connection wait to %s failed: %s",
                    consumer_alias,
                    exc,
                )
                return
        time.sleep(serve_settle_s)
        try:
            serve(agent, service, consumer_alias)
        except Exception as exc:  # noqa: BLE001
            logger.error("trajectory rpc server crashed: %s", exc, exc_info=True)

    t = threading.Thread(target=_target, daemon=daemon, name="slime-rpc-serve")
    t.start()
    logger.info("trajectory rpc server thread started, exposing to %s", consumer_alias)
    return t


def open_consumer(agent: PeerAgent, producer_alias: str):
    """Return a proxy bound to a remote ``TrajectoryService``."""
    return proxy(agent, producer_alias, TrajectoryService)
