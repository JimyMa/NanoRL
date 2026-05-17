"""Held-out evaluation harness — answers "is the policy actually getting
better?" with numbers.

Greedy (temperature=0, n=1) gives pass@1 / mean reward.
Sampled (temperature>0, n>1) gives pass@k where any sample passing counts
the prompt as solved.

Decoupled from the rollout's trajectory plane: this just calls the
engine's ``generate`` method on a list of PromptItems with optional
sampling overrides, scores via the engine's verifier, aggregates.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from nanorl.actors.rollout import PromptItem, RolloutEngine
from nanorl.config import SamplingCfg
from nanorl.rl.reward import Verifier

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Sampling overrides for an eval run.

    ``n_samples`` is the number of completions per prompt — combine with
    ``temperature > 0`` for pass@k. Greedy single-sample (``n_samples=1,
    temperature=0``) gives a deterministic pass@1.
    """

    n_samples: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int | None = None
    pass_threshold: float = 0.5  # any reward ≥ this counts as a "pass"


@dataclass
class EvalReport:
    num_prompts: int
    n_samples: int
    mean_reward: float
    median_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    pass_at_1: float
    pass_at_k: float
    elapsed_s: float
    per_prompt: list[dict] = field(default_factory=list)
    samples: list[dict] = field(
        default_factory=list
    )  # short decoded samples for spot checks

    def log(self, log: logging.Logger = logger) -> None:
        log.info(
            "eval: prompts=%d n_samples=%d mean=%.3f median=%.3f std=%.3f "
            "min=%.3f max=%.3f pass@1=%.3f pass@%d=%.3f elapsed=%.1fs",
            self.num_prompts,
            self.n_samples,
            self.mean_reward,
            self.median_reward,
            self.std_reward,
            self.min_reward,
            self.max_reward,
            self.pass_at_1,
            self.n_samples,
            self.pass_at_k,
            self.elapsed_s,
        )

    def to_dict(self) -> dict:
        return {
            "num_prompts": self.num_prompts,
            "n_samples": self.n_samples,
            "mean_reward": self.mean_reward,
            "median_reward": self.median_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "elapsed_s": self.elapsed_s,
            "per_prompt": self.per_prompt,
        }


def summarize(
    items: Sequence[PromptItem],
    rewards_per_prompt: dict[int, list[float]],
    decoded_samples: dict[int, list[str]],
    eval_cfg: EvalConfig,
    elapsed_s: float,
) -> EvalReport:
    """Build an ``EvalReport`` from per-prompt reward lists.

    ``rewards_per_prompt`` keys are ``PromptItem.group_id`` (eval uses
    these as stable prompt ids).
    """
    flat = [r for rs in rewards_per_prompt.values() for r in rs]
    if not flat:
        return EvalReport(
            num_prompts=0,
            n_samples=eval_cfg.n_samples,
            mean_reward=0.0,
            median_reward=0.0,
            std_reward=0.0,
            min_reward=0.0,
            max_reward=0.0,
            pass_at_1=0.0,
            pass_at_k=0.0,
            elapsed_s=elapsed_s,
        )

    pass_at_1_count = 0  # first-sample passes
    pass_at_k_count = 0  # any-sample passes
    per_prompt_rows: list[dict] = []
    for item in items:
        rs = rewards_per_prompt.get(item.group_id, [])
        if not rs:
            continue
        passed_first = rs[0] >= eval_cfg.pass_threshold
        passed_any = any(r >= eval_cfg.pass_threshold for r in rs)
        if passed_first:
            pass_at_1_count += 1
        if passed_any:
            pass_at_k_count += 1
        per_prompt_rows.append(
            {
                "group_id": item.group_id,
                "n": len(rs),
                "mean": statistics.fmean(rs),
                "max": max(rs),
                "passed_first": bool(passed_first),
                "passed_any": bool(passed_any),
                "reference": item.reference,
            }
        )

    seen = len(per_prompt_rows)
    sample_rows = []
    for gid, decoded in list(decoded_samples.items())[:8]:
        sample_rows.append(
            {
                "group_id": gid,
                "first_response": decoded[0][:240] if decoded else "",
                "reward": (
                    rewards_per_prompt[gid][0] if rewards_per_prompt.get(gid) else None
                ),
            }
        )

    return EvalReport(
        num_prompts=seen,
        n_samples=eval_cfg.n_samples,
        mean_reward=statistics.fmean(flat),
        median_reward=statistics.median(flat),
        std_reward=statistics.pstdev(flat) if len(flat) > 1 else 0.0,
        min_reward=min(flat),
        max_reward=max(flat),
        pass_at_1=pass_at_1_count / seen if seen else 0.0,
        pass_at_k=pass_at_k_count / seen if seen else 0.0,
        elapsed_s=elapsed_s,
        per_prompt=per_prompt_rows,
        samples=sample_rows,
    )


class Evaluator:
    """Drive a ``RolloutEngine`` (or a stub) over an eval prompt set.

    The engine's existing ``generate(items)`` returns ``PendingTrajectory``s;
    we inject sampling overrides via ``_eval_sampling_override`` so the
    same NanoDeploy LLM is reused (no second engine to boot). Verifier
    defaults to the engine's bound verifier.
    """

    def __init__(self, engine: RolloutEngine, verifier: Verifier | None = None):
        self._engine = engine
        self._verifier = verifier or engine._verifier  # noqa: SLF001 — owner of engine

    def evaluate(
        self,
        items: Sequence[PromptItem],
        cfg: EvalConfig | None = None,
    ) -> EvalReport:
        eval_cfg = cfg or EvalConfig()
        if not items:
            return summarize(items, {}, {}, eval_cfg, 0.0)

        # Build a SamplingCfg override so the engine emits the right
        # sampling params per request.
        engine_sampling = self._engine._sampling_cfg  # noqa: SLF001
        eval_sampling = SamplingCfg(
            temperature=eval_cfg.temperature,
            top_p=eval_cfg.top_p,
            max_new_tokens=eval_cfg.max_new_tokens or engine_sampling.max_new_tokens,
            n=eval_cfg.n_samples,
        )
        # Swap, generate+score, restore. The engine's `generate` reads
        # `_sampling_cfg` via `_sampling_params()` — easiest hook.
        original = self._engine._sampling_cfg  # noqa: SLF001
        self._engine._sampling_cfg = eval_sampling  # noqa: SLF001

        t0 = time.monotonic()
        try:
            pending = self._engine.generate(items)
            scored = self._engine.score(pending)  # uses engine._verifier
        finally:
            self._engine._sampling_cfg = original  # noqa: SLF001
        elapsed = time.monotonic() - t0

        # Group rewards + decoded responses by prompt id.
        rewards_per_prompt: dict[int, list[float]] = {}
        decoded_per_prompt: dict[int, list[str]] = {}
        idx = 0
        for p, t in zip(pending, scored):
            rewards_per_prompt.setdefault(p.group_id, []).append(t.reward)
            decoded_per_prompt.setdefault(p.group_id, []).append(p.response_text)
            idx += 1

        report = summarize(
            items, rewards_per_prompt, decoded_per_prompt, eval_cfg, elapsed
        )
        report.log()
        return report
