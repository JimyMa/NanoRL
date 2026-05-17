"""HF Qwen3 → megatron-core GPTModel weight conversion.

Single highest-risk piece of M1: getting the fused-QKV interleave wrong here
produces silent gibberish. The mapping is the inverse of slime's
``slime/backends/megatron_utils/megatron_to_hf/qwen2.py`` (Qwen3 dense uses
the same shapes as Qwen2 plus per-head q/k norms).

Public surface:
    build_transformer_config(model_cfg, hf_cfg, train_cfg)  -> TransformerConfig
    load_qwen3_hf_into_megatron(model, hf_path)             -> None  (in place)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TransformerConfig builder
# ---------------------------------------------------------------------------


def _read_hf_config(hf_path: str) -> dict[str, Any]:
    with open(os.path.join(hf_path, "config.json")) as f:
        return json.load(f)


def build_transformer_config(hf_path: str, *, bf16: bool = True):
    """Build a megatron-core ``TransformerConfig`` from an HF Qwen3 dir.

    Reads ``config.json`` directly so we don't pay for `transformers` import
    in environments where only the config is needed (e.g. the unit test).
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    hf = _read_hf_config(hf_path)
    if hf.get("model_type") != "qwen3":
        raise ValueError(
            f"expected Qwen3 HF config (model_type='qwen3'), got {hf.get('model_type')!r}"
        )

    num_layers = int(hf["num_hidden_layers"])
    hidden_size = int(hf["hidden_size"])
    ffn_hidden_size = int(hf["intermediate_size"])
    num_attention_heads = int(hf["num_attention_heads"])
    num_query_groups = int(hf["num_key_value_heads"])
    head_dim = int(hf.get("head_dim", hidden_size // num_attention_heads))
    rope_base = int(hf.get("rope_theta", 10000))

    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        kv_channels=head_dim,
        num_query_groups=num_query_groups,
        normalization="RMSNorm",
        layernorm_epsilon=float(hf.get("rms_norm_eps", 1e-6)),
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=bool(bf16),
        params_dtype=torch.bfloat16 if bf16 else torch.float32,
        attention_softmax_in_fp32=False,
        # Full activation checkpointing — at seq=8K with FSDP-8 the FC1
        # output [seq, 1, 2*FFN] saved-for-backward dominates memory
        # (~11 GB across 36 layers). Uniform recompute with one layer per
        # block re-runs each layer's forward during its backward, dropping
        # peak activations from ~26 GB to ~3 GB at the cost of ~30%
        # extra compute. Required for seq>4K on Qwen3-4B + FSDP-8.
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        # piped through to GPTModel separately:
        #   - position_embedding_type="rope"
        #   - rotary_base=rope_base
        #   - share_embeddings_and_output_weights=hf["tie_word_embeddings"]
    )


def hf_metadata(hf_path: str) -> dict[str, Any]:
    """Bits of the HF config that ``GPTModel.__init__`` and the loader need
    that aren't on TransformerConfig itself."""
    hf = _read_hf_config(hf_path)
    return dict(
        vocab_size=int(hf["vocab_size"]),
        max_position_embeddings=int(hf["max_position_embeddings"]),
        rope_base=int(hf.get("rope_theta", 10000)),
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
    )


# ---------------------------------------------------------------------------
# Name mapping (HF → Megatron)
# ---------------------------------------------------------------------------


def _fuse_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_query_groups: int,
    num_attention_heads: int,
    head_dim: int,
    hidden_size: int,
) -> torch.Tensor:
    """Inverse of slime's qwen2 split: reshape per-head HF q/k/v back to
    Megatron's interleaved [groups × (q_per_group + 2) × head_dim, hidden]
    layout. See slime/backends/megatron_utils/megatron_to_hf/qwen2.py:25."""
    value_num_per_group = num_attention_heads // num_query_groups
    q4 = q.reshape(num_query_groups, value_num_per_group, head_dim, hidden_size)
    k4 = k.reshape(num_query_groups, 1, head_dim, hidden_size)
    v4 = v.reshape(num_query_groups, 1, head_dim, hidden_size)
    fused = torch.cat([q4, k4, v4], dim=1)  # [groups, q_per_group+2, head_dim, hidden]
    return fused.reshape(-1, hidden_size).contiguous()


def hf_to_megatron_state_dict(
    hf_state: dict[str, torch.Tensor],
    hf_path: str,
    *,
    has_output_layer: bool,
) -> dict[str, torch.Tensor]:
    """Translate an HF Qwen3 state_dict into Megatron-Core ``GPTModel``
    parameter names (no ``module.module.`` prefix).

    `has_output_layer` should be False when the model was built with
    ``share_embeddings_and_output_weights=True`` (Qwen3 < 30B).
    """
    hf = _read_hf_config(hf_path)
    num_layers = int(hf["num_hidden_layers"])
    hidden_size = int(hf["hidden_size"])
    num_attention_heads = int(hf["num_attention_heads"])
    num_query_groups = int(hf["num_key_value_heads"])
    head_dim = int(hf.get("head_dim", hidden_size // num_attention_heads))

    out: dict[str, torch.Tensor] = {}

    # --- token embedding ------------------------------------------------
    out["embedding.word_embeddings.weight"] = hf_state["model.embed_tokens.weight"]

    # --- per-layer ------------------------------------------------------
    for i in range(num_layers):
        h = f"model.layers.{i}"
        m = f"decoder.layers.{i}"

        # Pre-attention RMSNorm. The Transformer Engine spec fuses this into the QKV linear
        # as `linear_qkv.layer_norm_weight` (TELayerNormColumnParallelLinear).
        out[f"{m}.self_attention.linear_qkv.layer_norm_weight"] = hf_state[
            f"{h}.input_layernorm.weight"
        ]

        # fused qkv weight: [groups × (q_per_group + 2) × head_dim, hidden]
        out[f"{m}.self_attention.linear_qkv.weight"] = _fuse_qkv(
            hf_state[f"{h}.self_attn.q_proj.weight"],
            hf_state[f"{h}.self_attn.k_proj.weight"],
            hf_state[f"{h}.self_attn.v_proj.weight"],
            num_query_groups=num_query_groups,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
        )

        # qk per-head layernorms (Qwen3-specific)
        out[f"{m}.self_attention.q_layernorm.weight"] = hf_state[
            f"{h}.self_attn.q_norm.weight"
        ]
        out[f"{m}.self_attention.k_layernorm.weight"] = hf_state[
            f"{h}.self_attn.k_norm.weight"
        ]

        # attention output projection
        out[f"{m}.self_attention.linear_proj.weight"] = hf_state[
            f"{h}.self_attn.o_proj.weight"
        ]

        # Pre-mlp RMSNorm. The Transformer Engine spec fuses this into the FC1 linear as
        # `linear_fc1.layer_norm_weight` (TELayerNormColumnParallelLinear).
        out[f"{m}.mlp.linear_fc1.layer_norm_weight"] = hf_state[
            f"{h}.post_attention_layernorm.weight"
        ]

        # SwiGLU: gate || up along dim=0 (slime's chunk(2, dim=0) reverses this)
        out[f"{m}.mlp.linear_fc1.weight"] = torch.cat(
            [
                hf_state[f"{h}.mlp.gate_proj.weight"],
                hf_state[f"{h}.mlp.up_proj.weight"],
            ],
            dim=0,
        )
        out[f"{m}.mlp.linear_fc2.weight"] = hf_state[f"{h}.mlp.down_proj.weight"]

    # --- final norm + output layer -------------------------------------
    out["decoder.final_layernorm.weight"] = hf_state["model.norm.weight"]

    if has_output_layer:
        # only when tie_word_embeddings=False (e.g. Qwen3 30B/MoE variants)
        out["output_layer.weight"] = hf_state["lm_head.weight"]
    else:
        if "lm_head.weight" in hf_state:
            logger.debug("ignoring lm_head.weight — model uses tied embeddings")

    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _load_hf_safetensors(hf_path: str) -> dict[str, torch.Tensor]:
    """Load an HF safetensors checkpoint as a state_dict on CPU.

    Uses ``transformers.AutoModelForCausalLM`` so we get the correct
    parameter names without having to parse the index ourselves. The model
    object is dropped immediately after pulling its state_dict, so peak
    RAM is roughly 2× the checkpoint (which Qwen3-4B fits in easily).
    """
    from transformers import AutoModelForCausalLM

    logger.info("loading HF Qwen3 from %s", hf_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    state = {k: v.detach().to(torch.bfloat16) for k, v in hf_model.state_dict().items()}
    del hf_model
    return state


def load_qwen3_hf_into_megatron(model, hf_path: str) -> None:
    """Load a Qwen3 HF checkpoint into a constructed Megatron ``GPTModel``.

    `model` must be the bare GPTModel (not wrapped in DDP / MegatronModule
    parent), built with the ``TransformerConfig`` returned by
    ``build_transformer_config(hf_path)``.
    """
    has_output_layer = any(
        name.endswith("output_layer.weight") for name, _ in model.named_parameters()
    )

    hf_state = _load_hf_safetensors(hf_path)
    remapped = hf_to_megatron_state_dict(
        hf_state, hf_path, has_output_layer=has_output_layer
    )

    # `lm_head.weight` should not appear in `remapped` for tied-embedding
    # models — the GPTModel re-uses `embedding.word_embeddings.weight` for
    # the LM head, and supplying a duplicate would silently override the
    # tie. This is the safety belt referenced in plan-doc "Open risks #2".
    assert (
        has_output_layer or "output_layer.weight" not in remapped
    ), "tied-embedding model must not receive an output_layer.weight"

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # We tolerate a small set of expected gaps:
    #   - extra_state buffers (Megatron's Transformer Engine-compat metadata)
    #   - the tied lm-head when share_embeddings_and_output_weights=True
    soft_missing = [
        k
        for k in missing
        if not (k.endswith("_extra_state") or k == "output_layer.weight")
    ]
    soft_unexpected = list(unexpected)
    if soft_missing or soft_unexpected:
        raise RuntimeError(
            f"HF→Megatron load mismatch.\n"
            f"  missing ({len(soft_missing)}): {soft_missing[:8]}\n"
            f"  unexpected ({len(soft_unexpected)}): {soft_unexpected[:8]}"
        )
    logger.info(
        "HF→Megatron load OK: %d tensors transferred, %d benign missing keys ignored",
        len(remapped),
        len(missing) - len(soft_missing),
    )
