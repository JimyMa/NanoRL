"""Rollout engine: NanoInfra LLM + verifier + SlimeRPC publisher.

This is the rollout actor. It is *not* wrapped in ``@ray.remote`` because
NanoInfra's ``LLM`` already spawns its own per-GPU Ray sub-actors (see
``nanodeploy/engine/ray_executor.py``) — wrapping again would just add a
fan-out hop. When M3 needs the train-side driver to drive the rollout
lifecycle, we can wrap ``RolloutEngine`` in a thin Ray actor or call
``LLM.as_remote(config)`` from the driver.

The pipeline is split into three functions so callers can intercept at
each stage:

    items   ──► generate()  ──► PendingTrajectory[]
                                       │
                                       ▼
                            score(verifier)  ──► Trajectory[]
                                       │
                                       ▼
                                publish()  ──► SlimeRPC consumers

``run()`` is the convenience wrapper that does all three. Tests use the
split methods directly with stub LLMs.
"""

from __future__ import annotations

import logging
import os
import statistics
from dataclasses import dataclass, field
from typing import Sequence as PySeq

from nanorl.config import InferCfg, ModelCfg, SamplingCfg
from nanorl.data.sample import Trajectory
from nanorl.data.trajectory_buffer import TrajectoryService
from nanorl.rl.reward import Verifier

logger = logging.getLogger(__name__)


@dataclass
class PromptItem:
    """One prompt slot for the rollout engine.

    `prompt` is plain text; chat-template application happens here so the
    train side never needs a tokenizer. `reference` is whatever the verifier
    needs to score the response. `group_id` ties the ``sampling.n``
    independent rollouts of the same prompt into a GRPO group.
    """

    prompt: str
    reference: str
    group_id: int


@dataclass
class PendingTrajectory:
    """A finished generation that hasn't been scored yet."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_text: str
    group_id: int
    reference: str
    eos: bool
    # Per-response-token logprobs from the rollout-time policy (length
    # == len(response_ids)) when SamplingCfg.ship_logprobs is True; None
    # otherwise. Forwarded to Trajectory and consumed by the trainer as
    # ``old_logprobs`` in the GRPO importance ratio.
    response_logprobs: list[float] | None = None


@dataclass
class RolloutStats:
    """Summary stats for one round of rollouts."""

    num_trajectories: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    per_group: dict[int, dict[str, float]] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def log(self, log: logging.Logger = logger) -> None:
        log.info(
            "rollout: n=%d mean=%.3f std=%.3f min=%.3f max=%.3f elapsed=%.2fs",
            self.num_trajectories,
            self.mean_reward,
            self.std_reward,
            self.min_reward,
            self.max_reward,
            self.elapsed_s,
        )
        for gid, s in sorted(self.per_group.items()):
            log.info(
                "  group=%d n=%d mean=%.3f std=%.3f",
                gid,
                int(s["n"]),
                s["mean"],
                s["std"],
            )


def _validate_model_path(path: str) -> None:
    """Fast pre-flight check; NanoInfra startup is expensive so we want to
    fail in seconds, not minutes, when the path is wrong."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"model dir does not exist: {path}")
    has_index = os.path.exists(os.path.join(path, "model.safetensors.index.json"))
    has_single = any(f.endswith(".safetensors") for f in os.listdir(path))
    if not (has_index or has_single):
        raise FileNotFoundError(f"no .safetensors files under {path}")


def _build_nano_config(model: ModelCfg, infer: InferCfg, **overrides):
    from nanodeploy.config import Config

    return Config(
        model=model.hf_path,
        attention_tp=infer.attention_tp,
        attention_dp=infer.attention_dp,
        attention_sp=infer.attention_sp,
        ffn_ep=infer.ffn_ep,
        ffn_tp=infer.ffn_tp,
        ffn_dp=infer.ffn_dp,
        max_num_batched_tokens=infer.max_num_batched_tokens,
        max_model_len=infer.max_model_len,
        max_num_seqs=infer.max_num_seqs,
        kvcache_block_size=infer.kvcache_block_size,
        gpu_memory_utilization=infer.gpu_memory_utilization,
        mode=infer.mode,
        loop_count=infer.loop_count,
        executor_backend=infer.executor_backend,
        enforce_eager=infer.enforce_eager,
        trust_remote_code=infer.trust_remote_code,
        use_mega_moe=infer.use_mega_moe,
        num_speculative_tokens=infer.num_speculative_tokens,
        ray_address=infer.ray_address,
        master_address=infer.master_address,
        nanoctrl_address=infer.nanoctrl_address,
        nanoctrl_scope=infer.nanoctrl_scope,
        **overrides,
    )


class RolloutEngine:
    def __init__(
        self,
        model_cfg: ModelCfg,
        infer_cfg: InferCfg,
        sampling_cfg: SamplingCfg,
        verifier: Verifier,
        service: TrajectoryService,
        *,
        startup_validate: bool = True,
    ):
        if startup_validate:
            _validate_model_path(model_cfg.hf_path)

        # Lazy imports — NanoInfra has heavy GPU deps; failing on a CPU dev
        # box should land at construction time, not at import.
        from nanodeploy import Sequence as NanoSequence
        from nanodeploy.llm_component import LLM
        from nanodeploy.sampling_params import SamplingParams
        from transformers import AutoTokenizer

        self._NanoSequence = NanoSequence
        self._SamplingParams = SamplingParams
        self._verifier = verifier
        self._service = service
        self._sampling_cfg = sampling_cfg
        self._model_cfg = model_cfg
        self._infer_cfg = infer_cfg

        tokenizer_path = model_cfg.tokenizer_path or model_cfg.hf_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._llm = LLM(_build_nano_config(model_cfg, infer_cfg))
        self._log_startup_banner()

    # ------------------------------------------------------------------

    def _log_startup_banner(self) -> None:
        infer = self._infer_cfg
        logger.info("=" * 60)
        logger.info("RolloutEngine ready")
        logger.info("  model:             %s", self._model_cfg.hf_path)
        logger.info(
            "  parallelism:       attn_tp=%d attn_dp=%d attn_sp=%d "
            "ffn_tp=%d ffn_ep=%d ffn_dp=%d",
            infer.attention_tp,
            infer.attention_dp,
            infer.attention_sp,
            infer.ffn_tp,
            infer.ffn_ep,
            infer.ffn_dp,
        )
        logger.info("  ray:               %s", infer.ray_address)
        logger.info("  master:            %s", infer.master_address)
        logger.info("  nanoctrl:          %s", infer.nanoctrl_address)
        logger.info("  executor_backend:  %s", infer.executor_backend)
        logger.info(
            "  sampling:          n=%d temperature=%.2f max_new_tokens=%d",
            self._sampling_cfg.n,
            self._sampling_cfg.temperature,
            self._sampling_cfg.max_new_tokens,
        )
        logger.info("=" * 60)

    @property
    def llm(self):
        """The underlying NanoInfra ``LLMComponent``. Exposed so the M3
        weight-sync RPC handler can call ``llm.update_weights(...)``
        without poking at private attrs."""
        return self._llm

    def _encode(self, prompt: str) -> list[int]:
        if self._model_cfg.apply_chat_template and getattr(
            self._tokenizer, "chat_template", None
        ):
            text = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Base models: feed prompt raw. The tokenizer may carry a chat
            # template inherited from an instruct sibling, but the base
            # wasn't trained on the wrapping tokens — feeding them produces
            # incoherent outputs ~half the time. Operator opts in via
            # cfg.model.apply_chat_template=true.
            text = prompt
        return self._tokenizer.encode(text)

    def _sampling_params(self):
        # ``return_completion_logprobs`` opts the worker into the new
        # NanoInfra logprob path (Sampler.forward_with_logprobs ->
        # Sequence.completion_logprobs). Newer NanoInfra builds accept
        # the kwarg; older ones don't — fall back gracefully so a stale
        # build doesn't break the rollout.
        kwargs = dict(
            max_tokens=self._sampling_cfg.max_new_tokens,
            temperature=self._sampling_cfg.temperature,
            ignore_eos=False,
        )
        if getattr(self._sampling_cfg, "ship_logprobs", False):
            kwargs["return_completion_logprobs"] = True
        try:
            return self._SamplingParams(**kwargs)
        except TypeError:
            # NanoInfra build without the new flag — drop it and proceed.
            kwargs.pop("return_completion_logprobs", None)
            return self._SamplingParams(**kwargs)

    # ------------------------------------------------------------------

    def generate(self, items: PySeq[PromptItem]) -> list[PendingTrajectory]:
        """Run ``sampling.n`` rollouts per prompt; return raw completions.

        Does NOT call the verifier and does NOT publish — call ``score`` and
        ``publish`` (or use ``run``) for that.
        """
        if not items:
            return []

        n = self._sampling_cfg.n
        nano_seqs = []
        owners: list[PromptItem] = []
        encoded: list[list[int]] = []
        for item in items:
            ids = self._encode(item.prompt)
            for _ in range(n):
                nano_seqs.append(
                    self._NanoSequence(ids, sampling_params=self._sampling_params())
                )
                owners.append(item)
                encoded.append(ids)

        self._llm.add_request(nano_seqs)
        self._llm.generate(use_tqdm=False)

        out: list[PendingTrajectory] = []
        for seq, item, prompt_ids in zip(nano_seqs, owners, encoded):
            response_ids = list(seq.completion_token_ids)
            response_text = self._tokenizer.decode(
                response_ids, skip_special_tokens=True
            )
            # Read rollout-time per-token logprobs when the patched
            # NanoInfra populated them. Empty (legacy build, or sampling
            # didn't request them) → None so trainer falls back cleanly.
            raw_lp = getattr(seq, "completion_logprobs", None)
            response_logprobs = list(raw_lp) if raw_lp else None
            out.append(
                PendingTrajectory(
                    prompt_ids=list(prompt_ids),
                    response_ids=response_ids,
                    response_text=response_text,
                    group_id=item.group_id,
                    reference=item.reference,
                    eos=bool(getattr(seq, "is_finished", True)),
                    response_logprobs=response_logprobs,
                )
            )
        if getattr(self._sampling_cfg, "ship_logprobs", False):
            with_lp = sum(1 for p in out if p.response_logprobs is not None)
            lp_lens = [
                len(p.response_logprobs or [])
                for p in out
                if p.response_logprobs is not None
            ]
            logger.info(
                "rollout logprobs: %d/%d completions carried logprobs%s",
                with_lp,
                len(out),
                f" len_min={min(lp_lens)} len_max={max(lp_lens)}" if lp_lens else "",
            )
        return out

    def score(self, pending: PySeq[PendingTrajectory]) -> list[Trajectory]:
        """Apply the verifier to each pending trajectory.

        A verifier exception is logged and the sample gets reward 0.0 — we
        keep the pipeline going rather than dropping a whole round on one
        bad rollout.
        """
        out: list[Trajectory] = []
        for p in pending:
            try:
                reward = float(self._verifier.score(p.response_text, p.reference))
            except Exception as exc:  # noqa: BLE001 - verifier code is user-supplied
                logger.warning("verifier raised on group=%d: %s", p.group_id, exc)
                reward = 0.0
            out.append(
                Trajectory(
                    prompt_ids=p.prompt_ids,
                    response_ids=p.response_ids,
                    reward=reward,
                    group_id=p.group_id,
                    eos=p.eos,
                    meta={"reference": p.reference},
                    response_logprobs=p.response_logprobs,
                )
            )
        return out

    def publish(self, trajectories: PySeq[Trajectory]) -> int:
        """Push to the SlimeRPC service buffer; returns new buffered count."""
        return self._service.publish(trajectories)

    # ------------------------------------------------------------------

    def run(
        self,
        items: PySeq[PromptItem],
        *,
        publish: bool = True,
    ) -> tuple[list[Trajectory], RolloutStats]:
        """Generate → score → optionally publish; return trajectories + stats."""
        import time

        t0 = time.monotonic()
        pending = self.generate(items)
        trajectories = self.score(pending)
        if publish:
            self.publish(trajectories)
        stats = _summarize(trajectories, elapsed_s=time.monotonic() - t0)
        return trajectories, stats


def _summarize(trajectories: PySeq[Trajectory], elapsed_s: float = 0.0) -> RolloutStats:
    if not trajectories:
        return RolloutStats(0, 0.0, 0.0, 0.0, 0.0, elapsed_s=elapsed_s)
    rewards = [t.reward for t in trajectories]
    per_group: dict[int, list[float]] = {}
    for t in trajectories:
        per_group.setdefault(t.group_id, []).append(t.reward)
    return RolloutStats(
        num_trajectories=len(trajectories),
        mean_reward=statistics.fmean(rewards),
        std_reward=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        min_reward=min(rewards),
        max_reward=max(rewards),
        per_group={
            gid: {
                "n": float(len(vs)),
                "mean": statistics.fmean(vs),
                "std": statistics.pstdev(vs) if len(vs) > 1 else 0.0,
            }
            for gid, vs in per_group.items()
        },
        elapsed_s=elapsed_s,
    )
