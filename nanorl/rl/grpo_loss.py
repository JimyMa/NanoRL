"""GRPO loss — vendored from megatron/rl/rl_utils.py:1854.

Pure-functional: no dependence on `get_args()` or megatron.training globals,
so it is safe to call from a Ray actor that initializes its own dist state.

If you change this file, also bump the cross-reference in
``tests/test_grpo_loss.py``.
"""

from __future__ import annotations

import torch


def calculate_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clamp_eps_lower: float,
    clamp_eps_upper: float,
    kl_beta: float,
    entropy_weight: float,
    inference_logprobs: torch.Tensor | None = None,
    is_truncation_coef: float | None = None,
    seq_starts: list | None = None,
    seq_lengths: list | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Per-token GRPO loss + diagnostics.

    Returns: (loss, kl_term, ratios, entropy_term, truncated_above, truncated_below).
    All shapes match ``current_logprobs`` ([B, T] unpacked or [1, bin] packed).
    """
    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"shape mismatch: current={tuple(current_logprobs.shape)} "
            f"old={tuple(old_logprobs.shape)}"
        )

    ratios = (current_logprobs - old_logprobs).exp()
    clamped_ratios = ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)
    truncated_from_above = torch.gt(ratios, 1 + clamp_eps_upper)
    truncated_from_below = torch.lt(ratios, 1 - clamp_eps_lower)

    if seq_starts is not None and seq_lengths is not None:
        bin_size = current_logprobs.shape[1]
        packed_advantages = torch.zeros(
            (1, bin_size), device=current_logprobs.device, dtype=current_logprobs.dtype
        )
        for seq_idx, (start, seq_len) in enumerate(zip(seq_starts, seq_lengths)):
            end = min(start + seq_len - 1, bin_size)
            if end > start:
                packed_advantages[0, start:end] = advantages[seq_idx].item()
        advantages = packed_advantages
    else:
        advantages = advantages.view(-1, 1)

    ref_diff = ref_logprobs - current_logprobs
    kl_term = ref_diff.exp() - ref_diff - 1
    entropy_term = -current_logprobs.exp() * current_logprobs

    is_weights = torch.tensor(1.0, dtype=old_logprobs.dtype, device=old_logprobs.device)
    if inference_logprobs is not None:
        is_weights = (old_logprobs - inference_logprobs).exp()
        if is_truncation_coef is not None:
            is_weights = torch.min(
                is_weights,
                torch.tensor(
                    is_truncation_coef,
                    dtype=old_logprobs.dtype,
                    device=old_logprobs.device,
                ),
            )

    loss = (
        -is_weights * torch.min(ratios * advantages, clamped_ratios * advantages)
        + kl_beta * kl_term
        - entropy_weight * entropy_term
    )

    return (
        loss,
        kl_term,
        ratios,
        entropy_term,
        truncated_from_above,
        truncated_from_below,
    )
