"""Per-token logprobs for an autoregressive LM, no megatron.training globals.

This replaces ``megatron.rl.rl_utils.get_logprobs`` (which reads ``get_args()``).
"""

from __future__ import annotations

import torch


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Memory-efficient ``log_softmax(logits).gather(index)``.

    Equivalent to::

        torch.gather(logits.log_softmax(-1), -1, index.unsqueeze(-1)).squeeze(-1)

    but avoids materializing the full log-softmax. For bf16/fp16 we fall back
    to a row-wise log_softmax because the logsumexp path is numerically
    unstable there. Adapted from TRL.
    """
    if logits.dtype in (torch.float32, torch.float64):
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        return selected_logits - logsumexp_values

    out = []
    for row_logits, row_idx in zip(logits, index):
        row_logps = torch.nn.functional.log_softmax(row_logits, dim=-1)
        out.append(row_logps.gather(dim=-1, index=row_idx.unsqueeze(-1)).squeeze(-1))
    return torch.stack(out)


def compute_per_token_logprobs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Logprobs assigned by `logits` to each *next* token in `tokens`.

    Args:
        logits: [B, T, V] — model output for an input of length T.
        tokens: [B, T]    — same input the logits were computed for.

    Returns:
        [B, T-1] — for each position t in [0, T-1), the log-prob the model
        assigned to ``tokens[:, t+1]`` given context ``tokens[:, :t+1]``.
    """
    return selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])
