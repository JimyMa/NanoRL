"""Group-relative advantages for GRPO.

Each prompt produces a *group* of N rollouts; the advantage is the per-sample
reward standardized within its group. Std is floored to avoid divide-by-zero
when all rollouts in a group score identically.
"""

from __future__ import annotations

import numpy as np


def group_relative_advantages(
    rewards: np.ndarray, group_ids: np.ndarray, std_floor: float = 1e-4
) -> np.ndarray:
    """Standardize rewards within each group.

    Args:
        rewards: [B] float
        group_ids: [B] int — samples sharing a group_id share a prompt
        std_floor: lower bound on the divisor

    Returns:
        [B] float — ``(r - mean_g) / max(std_g, std_floor)``
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    advantages = np.empty_like(rewards)
    for g in np.unique(group_ids):
        mask = group_ids == g
        group_rewards = rewards[mask]
        mean = group_rewards.mean()
        std = max(float(group_rewards.std()), std_floor)
        advantages[mask] = (group_rewards - mean) / std
    return advantages.astype(np.float32)
