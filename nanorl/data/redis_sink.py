"""Async Redis sink for rollout trajectories.

The rollout produces ~100s of trajectories/sec with full prompt+response text;
we don't want to block decode on Redis roundtrips. This sink runs a single
daemon thread that drains a bounded queue and pipelines XADDs into a stream
with MAXLEN-based eviction (so memory stays capped). On backpressure (queue
full) we silently drop — telemetry is best-effort, never a hard dependency.

Schema (one stream entry per trajectory)::

    XADD nanorl:trajectories *
        group_id <int>
        reward <float>
        reference <str>
        eos <0|1>
        response_len <int>
        prompt <str>
        response <str>
        ts <iso-utc>

Consumers tail with ``XREAD COUNT 100 STREAMS nanorl:trajectories $`` or
batch-replay via ``XRANGE``.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class RedisTrajectorySink:
    """Background writer for trajectories. Lazy-imports redis."""

    def __init__(
        self,
        url: str,
        *,
        key: str = "nanorl:trajectories",
        maxlen: int = 100_000,
        queue_size: int = 10_000,
        batch_size: int = 50,
        timeout: float = 1.0,
    ) -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise RuntimeError("redis-py not installed; pip install redis") from exc
        self._client = redis.Redis.from_url(
            url, socket_timeout=5.0, socket_connect_timeout=5.0
        )
        # Probe connection so caller fails fast if the URL is bad.
        try:
            self._client.ping()
        except Exception as exc:
            raise RuntimeError(f"redis ping failed for {url}: {exc}") from exc

        self._key = key
        self._maxlen = maxlen
        self._batch_size = batch_size
        self._timeout = timeout
        self._q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._dropped = 0
        self._sent = 0
        self._thread = threading.Thread(
            target=self._worker, name="RedisSink", daemon=True
        )
        self._thread.start()
        logger.info(
            "RedisTrajectorySink: writing to %s key=%s maxlen=%d", url, key, maxlen
        )

    def push(self, row: dict[str, Any]) -> None:
        """Enqueue a trajectory row. Drops on backpressure (no exception)."""
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.warning(
                    "RedisTrajectorySink: queue full, dropped %d rows so far",
                    self._dropped,
                )

    def _flatten(self, row: dict[str, Any]) -> dict[str, str]:
        """Serialize values as strings — Redis stream fields are bytes-only."""
        return {
            "group_id": str(row.get("group_id", "")),
            "reward": str(row.get("reward", 0.0)),
            "reference": str(row.get("reference", "")),
            "eos": "1" if row.get("eos") else "0",
            "response_len": str(row.get("response_len", 0)),
            "prompt": (row.get("prompt") or "")[:8000],
            "response": (row.get("response") or "")[:32000],
            "ts": row.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _worker(self) -> None:
        pipe = self._client.pipeline(transaction=False)
        buffered = 0
        while not self._stop.is_set():
            try:
                row = self._q.get(timeout=self._timeout)
            except queue.Empty:
                if buffered:
                    self._safe_execute(pipe)
                    buffered = 0
                continue
            try:
                pipe.xadd(
                    self._key, self._flatten(row), maxlen=self._maxlen, approximate=True
                )
                buffered += 1
            except Exception as exc:  # noqa: BLE001 — never crash rollout
                logger.warning("RedisTrajectorySink: xadd build failed: %s", exc)
                continue
            if buffered >= self._batch_size:
                self._safe_execute(pipe)
                buffered = 0
        # Drain remaining queue at shutdown.
        while True:
            try:
                row = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                pipe.xadd(
                    self._key, self._flatten(row), maxlen=self._maxlen, approximate=True
                )
                buffered += 1
            except Exception:
                continue
        if buffered:
            self._safe_execute(pipe)

    def _safe_execute(self, pipe) -> None:
        try:
            results = pipe.execute()
            self._sent += len(results)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RedisTrajectorySink: pipeline execute failed: %s", exc)

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        try:
            self._client.close()
        except Exception:
            pass
        logger.info(
            "RedisTrajectorySink: closed — sent=%d dropped=%d",
            self._sent,
            self._dropped,
        )
