"""Unit tests for ``nanorl.weights.hf_to_megatron``.

We can't load real Qwen3-4B in CI, but we can:
1. Use the real ``config.json`` to drive ``build_transformer_config`` and
   confirm field values (no GPU needed).
2. Fabricate a small synthetic HF state_dict with the *exact* shapes a real
   Qwen3 dense checkpoint would have, run it through the name-mapper, and
   confirm every Megatron-side parameter name is produced and shaped
   correctly.
3. Verify the QKV reshape is its own inverse (round-trip via the slime
   helper's split logic).

The full GPU sanity check that loads real weights and decodes a coherent
continuation lives in ``scripts/sanity_qwen3_forward.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from nanorl.weights.hf_to_megatron import (
    _fuse_qkv,
    build_transformer_config,
    hf_metadata,
    hf_to_megatron_state_dict,
)


HF_PATH = "/models/model--Qwen-Qwen3-4B-Instruct-2507"


def _has_real_qwen3_config() -> bool:
    return os.path.isfile(os.path.join(HF_PATH, "config.json"))


@pytest.mark.skipif(not _has_real_qwen3_config(), reason="Qwen3-4B config not present")
def test_build_transformer_config_matches_qwen3_4b():
    cfg = build_transformer_config(HF_PATH, bf16=False)
    assert cfg.num_layers == 36
    assert cfg.hidden_size == 2560
    assert cfg.ffn_hidden_size == 9728
    assert cfg.num_attention_heads == 32
    assert cfg.num_query_groups == 8
    assert cfg.kv_channels == 128
    assert cfg.normalization == "RMSNorm"
    assert cfg.qk_layernorm is True
    assert cfg.gated_linear_unit is True
    assert cfg.add_bias_linear is False

    meta = hf_metadata(HF_PATH)
    assert meta["vocab_size"] == 151936
    assert meta["tie_word_embeddings"] is True
    assert meta["rope_base"] == 5000000


def _fake_qwen3_hf_state(
    num_layers=2, hidden=64, ffn=128, n_heads=4, n_kv=2, head_dim=16, vocab=128
):
    """Mimic the *exact* tensor names and shapes of a real Qwen3-dense
    HF checkpoint, just much smaller."""
    sd = {}
    sd["model.embed_tokens.weight"] = torch.randn(vocab, hidden)
    for i in range(num_layers):
        h = f"model.layers.{i}"
        sd[f"{h}.input_layernorm.weight"] = torch.randn(hidden)
        sd[f"{h}.post_attention_layernorm.weight"] = torch.randn(hidden)
        sd[f"{h}.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, hidden)
        sd[f"{h}.self_attn.k_proj.weight"] = torch.randn(n_kv * head_dim, hidden)
        sd[f"{h}.self_attn.v_proj.weight"] = torch.randn(n_kv * head_dim, hidden)
        sd[f"{h}.self_attn.o_proj.weight"] = torch.randn(hidden, n_heads * head_dim)
        sd[f"{h}.self_attn.q_norm.weight"] = torch.randn(head_dim)
        sd[f"{h}.self_attn.k_norm.weight"] = torch.randn(head_dim)
        sd[f"{h}.mlp.gate_proj.weight"] = torch.randn(ffn, hidden)
        sd[f"{h}.mlp.up_proj.weight"] = torch.randn(ffn, hidden)
        sd[f"{h}.mlp.down_proj.weight"] = torch.randn(hidden, ffn)
    sd["model.norm.weight"] = torch.randn(hidden)
    return sd


def test_qkv_fuse_split_roundtrip():
    """Inverse of slime's qwen2 split must match our fuse: fuse then split
    by the slime recipe should recover q/k/v unchanged."""
    n_groups, q_per, head_dim, hidden = 2, 3, 16, 64
    n_heads = n_groups * q_per
    q = torch.randn(n_heads * head_dim, hidden)
    k = torch.randn(n_groups * head_dim, hidden)
    v = torch.randn(n_groups * head_dim, hidden)
    fused = _fuse_qkv(
        q,
        k,
        v,
        num_query_groups=n_groups,
        num_attention_heads=n_heads,
        head_dim=head_dim,
        hidden_size=hidden,
    )
    # slime split (megatron_to_hf/qwen2.py:25-36):
    f4 = fused.view(n_groups, -1, head_dim, hidden)
    q2, k2, v2 = torch.split(f4, [q_per, 1, 1], dim=1)
    assert torch.equal(q2.reshape(-1, hidden), q)
    assert torch.equal(k2.reshape(-1, hidden), k)
    assert torch.equal(v2.reshape(-1, hidden), v)


@pytest.mark.skipif(not _has_real_qwen3_config(), reason="Qwen3-4B config not present")
def test_state_dict_mapper_produces_expected_megatron_names(tmp_path):
    """Run the mapper against a tiny synthetic state_dict but with a real
    Qwen3 config.json and confirm every Megatron name we expect is produced
    with the right shape, and nothing extra slips in."""
    real = json.load(open(os.path.join(HF_PATH, "config.json")))
    overrides = dict(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
    )
    cfg = {**real, **overrides}
    fake_dir = tmp_path / "fake_qwen3"
    fake_dir.mkdir()
    (fake_dir / "config.json").write_text(json.dumps(cfg))

    hf_state = _fake_qwen3_hf_state(
        num_layers=cfg["num_hidden_layers"],
        hidden=cfg["hidden_size"],
        ffn=cfg["intermediate_size"],
        n_heads=cfg["num_attention_heads"],
        n_kv=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        vocab=cfg["vocab_size"],
    )
    out = hf_to_megatron_state_dict(hf_state, str(fake_dir), has_output_layer=False)

    expected_per_layer = {
        "input_layernorm.weight",
        "self_attention.linear_qkv.weight",
        "self_attention.q_layernorm.weight",
        "self_attention.k_layernorm.weight",
        "self_attention.linear_proj.weight",
        "pre_mlp_layernorm.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
    }
    assert "embedding.word_embeddings.weight" in out
    assert "decoder.final_layernorm.weight" in out
    assert "output_layer.weight" not in out  # tied
    for i in range(cfg["num_hidden_layers"]):
        for k in expected_per_layer:
            assert f"decoder.layers.{i}.{k}" in out, f"missing decoder.layers.{i}.{k}"

    # Critical shape checks
    qkv = out["decoder.layers.0.self_attention.linear_qkv.weight"]
    n_groups = cfg["num_key_value_heads"]
    q_per = cfg["num_attention_heads"] // n_groups
    head_dim = cfg["head_dim"]
    assert qkv.shape == (n_groups * (q_per + 2) * head_dim, cfg["hidden_size"])
    fc1 = out["decoder.layers.0.mlp.linear_fc1.weight"]
    assert fc1.shape == (2 * cfg["intermediate_size"], cfg["hidden_size"])  # gate||up
