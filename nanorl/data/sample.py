"""Trajectory data carried over SlimeRPC between rollout and train."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Trajectory:
    """A single rollout sample.

    `prompt_ids` and `response_ids` are pre-tokenized so the train side does
    not need a tokenizer. `group_id` ties together samples that share a prompt
    (GRPO needs at least 2 per group to compute advantages).

    `response_logprobs` is the rollout-time logprob of each chosen response
    token (length == len(response_ids)). When present, the trainer uses it
    as ``old_logprobs`` for honest importance-sampling correction. None
    means "rollout did not ship logprobs" — trainer falls back to
    ``current_logprobs.detach()`` for ratios=1.0 behaviour.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    reward: float
    group_id: int
    eos: bool = True
    meta: dict = field(default_factory=dict)
    response_logprobs: list[float] | None = None

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)


@dataclass
class TrajectoryBatch:
    """Padded batch ready for the forward pass.

    Shapes: tokens / response_mask are [B, T]; rewards / group_ids are [B].
    `response_mask` is 1 on response tokens, 0 on prompt or pad — the loss is
    computed only where the mask is 1.

    `response_logprobs` is float32 [B, T-1] aligned with the trainer's
    ``compute_per_token_logprobs(logits, tokens)`` output: index ``t``
    represents the prediction at position ``t`` of token ``t+1``. Padded
    positions hold zeros which the loss-mask zeros out anyway. None when
    no trajectory in the batch has shipped logprobs (all-or-nothing).
    """

    tokens: np.ndarray  # int64
    position_ids: np.ndarray  # int64
    response_mask: np.ndarray  # bool / int8
    rewards: np.ndarray  # float32 [B]
    group_ids: np.ndarray  # int64 [B]
    seq_lengths: np.ndarray  # int64 [B] (real length, pre-pad)
    response_logprobs: np.ndarray | None = None  # float32 [B, T-1] or None

    @classmethod
    def from_trajectories(
        cls,
        trajectories: Sequence[Trajectory],
        pad_id: int = 0,
        max_len: int | None = None,
    ) -> "TrajectoryBatch":
        if not trajectories:
            raise ValueError("empty trajectory list")
        lengths = np.array([t.length for t in trajectories], dtype=np.int64)
        T = int(max_len if max_len is not None else lengths.max())
        B = len(trajectories)

        tokens = np.full((B, T), pad_id, dtype=np.int64)
        response_mask = np.zeros((B, T), dtype=np.int8)
        position_ids = np.zeros((B, T), dtype=np.int64)
        for i, t in enumerate(trajectories):
            seq = (t.prompt_ids + t.response_ids)[:T]
            tokens[i, : len(seq)] = seq
            position_ids[i, : len(seq)] = np.arange(len(seq), dtype=np.int64)
            r_start = min(len(t.prompt_ids), T)
            r_end = min(len(t.prompt_ids) + len(t.response_ids), T)
            response_mask[i, r_start:r_end] = 1

        # All-or-nothing logprobs at batch level. If any trajectory is missing
        # them we pass None to the trainer, which falls back to the
        # ``current_logprobs.detach()`` path for ratios=1.0.
        response_logprobs = None
        if all(t.response_logprobs is not None for t in trajectories):
            response_logprobs = np.zeros((B, T - 1), dtype=np.float32)
            for i, t in enumerate(trajectories):
                # ``compute_per_token_logprobs`` indexes prediction-of-token-t
                # at output position t-1. So a response token at sequence
                # position p is predicted by output position p-1. We fill
                # the mask range shifted left by 1.
                r_start = min(len(t.prompt_ids), T)
                r_end = min(len(t.prompt_ids) + len(t.response_ids), T)
                lo = max(r_start - 1, 0)
                hi = max(r_end - 1, 0)
                # Truncate logprobs to the response window we actually kept
                # after T-clamp (handles max_len truncation).
                kept = (t.response_logprobs or [])[: hi - lo]
                if kept:
                    response_logprobs[i, lo : lo + len(kept)] = kept

        return cls(
            tokens=tokens,
            position_ids=position_ids,
            response_mask=response_mask,
            rewards=np.array([t.reward for t in trajectories], dtype=np.float32),
            group_ids=np.array([t.group_id for t in trajectories], dtype=np.int64),
            seq_lengths=lengths,
            response_logprobs=response_logprobs,
        )
