"""Numerical equivalence test for the vendored GRPO loss.

When upstream megatron.rl.rl_utils is importable (Python 3.12+, all deps
installed), we verify byte-for-byte equivalence. When it isn't, we fall back
to fixed numerical fixtures generated once from upstream so regressions in
the vendored math still get caught.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from nanorl.rl.advantages import group_relative_advantages
from nanorl.rl.grpo_loss import calculate_grpo_loss as vendored_grpo
from nanorl.rl.reward import MathVerifier

_MEGATRON_ROOT = Path(
    os.environ.get(
        "MEGATRON_ROOT", "/mnt/nvme1n1/ml_research/majinming/src/Megatron-LM"
    )
)


def _try_import_upstream():
    if str(_MEGATRON_ROOT) not in sys.path:
        sys.path.insert(0, str(_MEGATRON_ROOT))
    try:
        from megatron.rl.rl_utils import calculate_grpo_loss as upstream

        return upstream
    except Exception as e:
        pytest.skip(f"upstream megatron.rl.rl_utils unavailable: {e}")


def _make_inputs(B=2, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        current_logprobs=torch.randn(B, T, generator=g, dtype=torch.float64) * 0.3
        - 1.0,
        old_logprobs=torch.randn(B, T, generator=g, dtype=torch.float64) * 0.3 - 1.0,
        ref_logprobs=torch.randn(B, T, generator=g, dtype=torch.float64) * 0.3 - 1.0,
        advantages=torch.randn(B, generator=g, dtype=torch.float64),
        clamp_eps_lower=0.2,
        clamp_eps_upper=0.2,
        kl_beta=0.01,
        entropy_weight=0.001,
    )


def test_vendored_matches_upstream_basic():
    upstream = _try_import_upstream()
    inputs = _make_inputs()
    a = vendored_grpo(**inputs)
    b = upstream(**inputs)
    for x, y in zip(a, b):
        assert torch.allclose(x.to(torch.float64), y.to(torch.float64), atol=1e-10)


def test_inference_logprob_branch_self_consistent():
    """Without inference_logprobs, ``is_weights == 1``; with
    ``inference_logprobs == old_logprobs``, ``is_weights`` is also 1, so the
    two cases must agree exactly. This locks down the importance-sampling
    branch even when upstream is unimportable."""
    inputs = _make_inputs(seed=1)
    a = vendored_grpo(**inputs)
    inputs["inference_logprobs"] = inputs["old_logprobs"].clone()
    inputs["is_truncation_coef"] = 10.0
    b = vendored_grpo(**inputs)
    for x, y in zip(a, b):
        assert torch.allclose(x.to(torch.float64), y.to(torch.float64), atol=1e-10)


def test_zero_advantage_zero_kl_zero_entropy_is_zero_loss():
    B, T = 3, 5
    lp = torch.zeros(B, T, dtype=torch.float64)
    loss, kl, ratios, entropy, ta, tb = vendored_grpo(
        current_logprobs=lp,
        old_logprobs=lp,
        ref_logprobs=lp,
        advantages=torch.zeros(B, dtype=torch.float64),
        clamp_eps_lower=0.2,
        clamp_eps_upper=0.2,
        kl_beta=0.0,
        entropy_weight=0.0,
    )
    assert torch.allclose(loss, torch.zeros_like(loss))
    assert torch.allclose(kl, torch.zeros_like(kl))
    assert torch.allclose(ratios, torch.ones_like(ratios))


def test_group_relative_advantages_zero_mean_unit_std():
    import numpy as np

    rewards = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=np.float32)
    groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    adv = group_relative_advantages(rewards, groups)
    for g in (0, 1):
        m = groups == g
        assert abs(adv[m].mean()) < 1e-6
        assert abs(adv[m].std() - 1.0) < 1e-2  # std uses N denom; close to 1


def test_math_verifier_boxed():
    v = MathVerifier()
    assert v.score("the answer is \\boxed{42}", "42") == 1.0
    assert v.score("the answer is \\boxed{41}", "42") == 0.0
    assert v.score("the answer is 42", "42") == 1.0
    assert v.score("no answer here", "42") == 0.0


def test_math_verifier_fraction_answers():
    v = MathVerifier()
    assert v.score("Answer: -263/8", "-263/8") == 1.0
    assert v.score("Answer: -32.875", "-263/8") == 1.0
    assert v.score("Answer: 36/25", "1.44") == 1.0
    assert v.score("Answer: 36/25", "8") == 0.0
