"""Evaluation module — held-out reward + pass@k harness.

Construction principle: an ``Evaluator`` is decoupled from the
trajectory plane. It takes a ``RolloutEngine`` (or any object with a
``generate(items) -> PendingTrajectory[]`` method) and a verifier, runs
prompts in eval mode, returns aggregate stats. No SlimeRPC / RDMA / Ray
dependencies of its own.

That keeps three call sites possible without code duplication:

1. Standalone ``nanorl eval`` CLI — boots its own RolloutEngine (no
   trajectory publishing, no weight-sync RPC).
2. Periodic eval during training (future) — the train loop spawns an
   evaluator pointing at the same rollout used for trajectories, asks
   it to generate over an eval prompt set with overridden sampling.
3. Tests — pass a stub ``generate``-shaped object.
"""

from .datasets import load_eval_prompts, load_jsonl_eval  # noqa: F401
from .evaluator import EvalConfig, EvalReport, Evaluator, summarize  # noqa: F401
