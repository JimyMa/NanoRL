"""Metrics logger backends. All implement the same minimal Protocol:

    log_step(step: int, metrics: dict[str, float]) -> None
    log_event(name: str, payload: dict) -> None
    close() -> None

The composite delegates to multiple backends; any backend whose
optional dependency is missing degrades to no-op with a warning.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class MetricsLogger(Protocol):
    def log_step(self, step: int, metrics: dict[str, float]) -> None: ...
    def log_event(self, name: str, payload: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class NoopLogger:
    def log_step(self, step, metrics): ...
    def log_event(self, name, payload): ...
    def close(self): ...


class JSONLLogger:
    """Append per-step rows to a JSONL file. The first column is always
    ``step`` (or ``event``); the rest are flat key/value scalars."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", buffering=1)
        logger.info("metrics: JSONL → %s", path)

    def log_step(self, step, metrics):
        row = {"step": step, **{k: _scalar(v) for k, v in metrics.items()}}
        self._fh.write(json.dumps(row) + "\n")

    def log_event(self, name, payload):
        row = {"event": name, **{k: _scalar(v) for k, v in payload.items()}}
        self._fh.write(json.dumps(row) + "\n")

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class WandbLogger:
    """Lazy-imported wandb backend. ``project``/``run_name`` come from cfg."""

    def __init__(
        self, project: str, run_name: str | None = None, config: dict | None = None
    ):
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "wandb backend requested but `wandb` is not installed. "
                "pip install wandb or remove the backend from your cfg."
            ) from exc
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project, name=run_name, config=config or {}, reinit=True
        )
        logger.info("metrics: wandb run=%s project=%s", self._run.name, project)

    def log_step(self, step, metrics):
        self._wandb.log({k: _scalar(v) for k, v in metrics.items()}, step=step)

    def log_event(self, name, payload):
        self._wandb.log({f"event/{name}/{k}": _scalar(v) for k, v in payload.items()})

    def close(self):
        try:
            self._wandb.finish()
        except Exception:
            pass


class TensorBoardLogger:
    """Lazy-imported TensorBoard backend (tensorboard package)."""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "tensorboard backend requested but tensorboard is not installed. "
                "pip install tensorboard or remove the backend from your cfg."
            ) from exc
        from torch.utils.tensorboard import SummaryWriter

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=log_dir)
        logger.info("metrics: TensorBoard → %s", log_dir)

    def log_step(self, step, metrics):
        for k, v in metrics.items():
            try:
                self._writer.add_scalar(k, _scalar(v), step)
            except Exception:
                pass

    def log_event(self, name, payload):
        for k, v in payload.items():
            try:
                self._writer.add_scalar(f"event/{name}/{k}", _scalar(v), 0)
            except Exception:
                pass

    def close(self):
        try:
            self._writer.close()
        except Exception:
            pass


class CompositeLogger:
    """Fan-out to multiple loggers. Failures in one backend never affect
    another (we catch & log, never raise)."""

    def __init__(self, backends: list[MetricsLogger]):
        self._backends = backends

    def log_step(self, step, metrics):
        for b in self._backends:
            try:
                b.log_step(step, metrics)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "metrics backend %s log_step failed: %s", type(b).__name__, exc
                )

    def log_event(self, name, payload):
        for b in self._backends:
            try:
                b.log_event(name, payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "metrics backend %s log_event failed: %s", type(b).__name__, exc
                )

    def close(self):
        for b in self._backends:
            try:
                b.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar(v: Any) -> float:
    """Coerce to a scalar float; non-numeric values fall through unchanged
    so the JSONL backend can still emit them."""
    try:
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except (TypeError, ValueError):
        return v  # type: ignore[return-value]


def build_logger(
    *,
    jsonl_path: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_config: dict | None = None,
    tb_dir: str | None = None,
) -> MetricsLogger:
    """Construct a ``CompositeLogger`` from optional backend specs. If
    nothing is enabled, returns a ``NoopLogger`` rather than raising."""
    backends: list[MetricsLogger] = []
    if jsonl_path:
        backends.append(JSONLLogger(jsonl_path))
    if wandb_project:
        try:
            backends.append(WandbLogger(wandb_project, wandb_run_name, wandb_config))
        except RuntimeError as exc:
            logger.warning("wandb disabled: %s", exc)
    if tb_dir:
        try:
            backends.append(TensorBoardLogger(tb_dir))
        except RuntimeError as exc:
            logger.warning("tensorboard disabled: %s", exc)
    if not backends:
        return NoopLogger()
    return CompositeLogger(backends)
