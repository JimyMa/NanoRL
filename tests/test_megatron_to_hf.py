"""Round-trip equality test for the Megatron→HF gather.

The mapper has to be the byte-exact inverse of ``hf_to_megatron`` — a
synthetic HF state_dict goes ``hf → megatron → hf`` and must come out the
same. If anything is off (transposed split axis, wrong group order,
forgotten chunk), this test catches it before we ship gibberish to a live
NanoInfra engine.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from nanorl.weights.hf_to_megatron import hf_to_megatron_state_dict
from nanorl.weights.megatron_to_hf import (
    _split_qkv,
    _split_swiglu,
    gather_full_state_dict,
)

HF_PATH = "/models/model--Qwen-Qwen3-4B-Instruct-2507"


def _has_real_qwen3_config() -> bool:
    return os.path.isfile(os.path.join(HF_PATH, "config.json"))


def _fake_qwen3_hf_state(
    num_layers=2, hidden=64, ffn=128, n_heads=4, n_kv=2, head_dim=16, vocab=128
):
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


def test_split_qkv_inverts_fuse():
    """Round-trip just the QKV interleave."""
    from nanorl.weights.hf_to_megatron import _fuse_qkv

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
    q2, k2, v2 = _split_qkv(
        fused,
        num_query_groups=n_groups,
        num_attention_heads=n_heads,
        head_dim=head_dim,
        hidden_size=hidden,
    )
    assert torch.equal(q, q2)
    assert torch.equal(k, k2)
    assert torch.equal(v, v2)


def test_split_swiglu_inverts_fuse():
    g, u = torch.randn(128, 64), torch.randn(128, 64)
    fused = torch.cat([g, u], dim=0)
    g2, u2 = _split_swiglu(fused)
    assert torch.equal(g, g2)
    assert torch.equal(u, u2)


@pytest.mark.skipif(not _has_real_qwen3_config(), reason="Qwen3-4B config not present")
def test_hf_to_megatron_to_hf_roundtrip(tmp_path):
    """End-to-end: HF state_dict → megatron-named → ``gather_full_state_dict``
    fed a fake ``model`` whose ``named_parameters`` returns those tensors.
    The result must equal the original HF state_dict byte-for-byte.
    """
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

    hf_in = _fake_qwen3_hf_state(
        num_layers=cfg["num_hidden_layers"],
        hidden=cfg["hidden_size"],
        ffn=cfg["intermediate_size"],
        n_heads=cfg["num_attention_heads"],
        n_kv=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        vocab=cfg["vocab_size"],
    )
    mc = hf_to_megatron_state_dict(hf_in, str(fake_dir), has_output_layer=False)

    # Build a stub model whose named_parameters returns the megatron dict.
    class StubModel(torch.nn.Module):
        def __init__(self, sd):
            super().__init__()
            self._params = {k: torch.nn.Parameter(v.clone()) for k, v in sd.items()}
            for k, v in self._params.items():
                # Register so named_parameters yields them with these names.
                self.register_parameter(k.replace(".", "__"), v)

        def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
            for k, v in self._params.items():
                yield k, v

    stub = StubModel(mc)
    hf_out = gather_full_state_dict(
        stub,
        num_query_groups=cfg["num_key_value_heads"],
        num_attention_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"],
        hidden_size=cfg["hidden_size"],
    )

    assert set(hf_out) == set(hf_in), (
        f"key mismatch:\n  only in out: {set(hf_out) - set(hf_in)}\n"
        f"  only in in:  {set(hf_in) - set(hf_out)}"
    )
    for k in hf_in:
        assert torch.equal(hf_in[k], hf_out[k]), f"value mismatch at {k}"
