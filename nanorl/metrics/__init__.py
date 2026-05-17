"""Metrics logging — JSONL / wandb / TensorBoard behind one Protocol,
plus a static-HTML dashboard that consumes the same JSONL schema.

Design: a single ``MetricsLogger`` Protocol with three concrete backends
that lazy-import their dependencies. The ``CompositeLogger`` fans the
same payload to all enabled backends, so the train loop calls
``logger.log_step(step, payload)`` once and doesn't care which sinks
are live.

Backends pick themselves up from ``cfg.metrics.*`` flags in the YAML;
absent dependencies (e.g. ``wandb`` not installed) degrade to no-ops
with a warning, never a hard failure.

The ``dashboard`` submodule is the static-artifact CI-gate complement:
it reads the JSONL produced by ``JSONLLogger`` and emits a
self-contained HTML report with readiness checks.
"""

from .dashboard import (  # noqa: F401
    assess,
    build_dashboard,
    parse_rollout_log,
    parse_train_jsonl,
    render_html,
    RunData,
    StepRecord,
    SyncRecord,
)
from .loggers import (  # noqa: F401
    build_logger,
    CompositeLogger,
    JSONLLogger,
    MetricsLogger,
    NoopLogger,
    TensorBoardLogger,
    WandbLogger,
)
