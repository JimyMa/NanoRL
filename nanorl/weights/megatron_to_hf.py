"""Inverse of ``hf_to_megatron`` — gather a megatron-core ``GPTModel``'s
parameter tree as HF-named full tensors.

For TP=1 PP=1 EP=1 (M1 baseline) this is purely a name + reshape
transform: each Megatron parameter on this rank already holds the full
tensor. For TP/PP/EP > 1 (deferred), we'd inject collective gathers here.

The shipped tensors are what a fresh ``transformers.AutoModelForCausalLM``
checkpoint of Qwen3 dense would expose — the same names NanoInfra's
``model_runner.self.model`` indexes by, so each NanoInfra worker can call
its existing per-parameter ``weight_loader`` to slice for its own TP rank.
"""

from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)


_LAYER_RE = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")


def _split_qkv(
    fused: torch.Tensor,
    *,
    num_query_groups: int,
    num_attention_heads: int,
    head_dim: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of ``nanorl.weights.hf_to_megatron._fuse_qkv``.

    Megatron stores ``linear_qkv.weight`` as
    ``[groups × (q_per_group + 2) × head_dim, hidden]`` (see slime's
    qwen2 mapper for the canonical view recipe). Split it back into HF's
    per-projection ``q_proj``, ``k_proj``, ``v_proj``.
    """
    value_num_per_group = num_attention_heads // num_query_groups
    f4 = fused.view(num_query_groups, value_num_per_group + 2, head_dim, hidden_size)
    q4, k4, v4 = torch.split(f4, [value_num_per_group, 1, 1], dim=1)
    q = q4.reshape(num_attention_heads * head_dim, hidden_size).contiguous()
    k = k4.reshape(num_query_groups * head_dim, hidden_size).contiguous()
    v = v4.reshape(num_query_groups * head_dim, hidden_size).contiguous()
    return q, k, v


def _split_swiglu(fc1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``torch.cat([gate, up], dim=0)`` — HF stores them split."""
    gate, up = fc1.chunk(2, dim=0)
    return gate.contiguous(), up.contiguous()


def _materialize(param: torch.Tensor) -> torch.Tensor:
    """Return a regular full Tensor for ``param``, all-gathering across the
    DP mesh if it's a Megatron-FSDP uneven DTensor.

    Megatron-FSDP packs multiple parameters into a shared flat buffer per
    FSDP unit and stores each param as an *uneven* DTensor (different ranks
    hold different-sized slices). The standard ``DTensor.full_tensor()``
    returns a per-rank padded view, NOT the global tensor we want. The
    correct API is
    ``megatron_fsdp.uneven_dtensor.uneven_dtensor_to_full_tensor`` which
    gathers chunk metadata across ranks and reassembles the original
    global shape.

    For DDP / single-rank (non-DTensor params) this is just ``.detach()``.
    """
    # Detect a DTensor (covers both stock PyTorch and Megatron-FSDP variants).
    try:
        from torch.distributed.tensor import DTensor as _DTensor
    except Exception:
        _DTensor = None  # type: ignore

    if _DTensor is not None and isinstance(param, _DTensor):
        try:
            from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
                uneven_dtensor_to_full_tensor,
            )
        except Exception:
            return param.full_tensor()  # last-resort, may be wrong shape
        return uneven_dtensor_to_full_tensor(param)
    return param.detach()


def _maybe_unshard_fsdp(model: torch.nn.Module) -> object | None:
    """Detect FSDP wrapping. Returns the wrapper (so caller can re-shard
    after walk) or None.

    No actual unshard happens here — the per-param ``full_tensor()`` call
    inside ``_gather_walk`` does the all-gather lazily for each parameter.
    Doing it lazily means we never hold the entire materialized state on a
    non-zero rank: the temporary tensor goes out of scope after the rank-0
    register step picks it up (rank-0-only filtering is in
    ``gather_and_publish``).
    """
    try:
        from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import (
            MegatronFSDP,
        )
    except Exception:
        return None
    return model if isinstance(model, MegatronFSDP) else None


def _maybe_reshard_fsdp(fsdp_model) -> None:
    if fsdp_model is None:
        return
    # ``full_tensor()`` returns a fresh tensor; the original DTensor params
    # are unaffected. Nothing to re-shard.
    return


def gather_full_state_dict(
    model: torch.nn.Module,
    *,
    num_query_groups: int,
    num_attention_heads: int,
    head_dim: int,
    hidden_size: int,
) -> dict[str, torch.Tensor]:
    """Walk the Megatron model's params and produce HF-named full tensors.

    Handles both DDP-wrapped (each param is full on this rank) and
    Megatron-FSDP-wrapped (each rank holds 1/N of every param; we trigger
    an all-gather first and re-shard after). For TP/PP > 1 there'd be an
    additional collective step before the walk; that's deferred work.

    Parameters NanoInfra doesn't recognize (e.g. ``output_layer.weight``
    when the model uses tied embeddings, or any ``_extra_state`` buffer)
    are simply not produced — the rollout side's
    ``apply_named_tensors_in_place`` skips unknowns silently anyway.
    """
    fsdp_handle = _maybe_unshard_fsdp(model)
    try:
        return _gather_walk(
            model,
            num_query_groups=num_query_groups,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
        )
    finally:
        _maybe_reshard_fsdp(fsdp_handle)


def _gather_walk(
    model: torch.nn.Module,
    *,
    num_query_groups: int,
    num_attention_heads: int,
    head_dim: int,
    hidden_size: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    seen_module_prefix = False  # detect Megatron-FSDP / DDP wrapper

    for full_name, param in model.named_parameters():
        # Strip the DDP/FSDP wrapper prefix if present. The NanoRL
        # TrainActor wraps GPTModel in megatron-core's DDP (or FSDP), so
        # names look like "module.embedding.word_embeddings.weight" when
        # iterating the wrapper. ``model`` may already be unwrapped by
        # the caller (e.g. ``model.module``), so handle both.
        name = full_name
        if name.startswith("module."):
            name = name[len("module.") :]
            seen_module_prefix = True

        # Materialize: if `param` is a DTensor (FSDP path), this all-gathers
        # across the DP mesh and returns a plain Tensor. If it's a regular
        # Tensor (DDP / single-rank path), this just returns it.
        full = _materialize(param)

        if name == "embedding.word_embeddings.weight":
            out["model.embed_tokens.weight"] = full
            continue

        if name == "decoder.final_layernorm.weight":
            out["model.norm.weight"] = full
            continue

        if name == "output_layer.weight":
            out["lm_head.weight"] = full
            continue

        m = _LAYER_RE.match(name)
        if m is None:
            logger.debug("skipping parameter (unrecognized): %s", name)
            continue
        i, rest = int(m.group(1)), m.group(2)
        h = f"model.layers.{i}"

        if rest == "input_layernorm.weight":
            out[f"{h}.input_layernorm.weight"] = full
        elif rest == "pre_mlp_layernorm.weight":
            out[f"{h}.post_attention_layernorm.weight"] = full
        elif rest == "self_attention.linear_qkv.weight":
            q, k, v = _split_qkv(
                full,
                num_query_groups=num_query_groups,
                num_attention_heads=num_attention_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
            )
            out[f"{h}.self_attn.q_proj.weight"] = q
            out[f"{h}.self_attn.k_proj.weight"] = k
            out[f"{h}.self_attn.v_proj.weight"] = v
        elif rest == "self_attention.linear_proj.weight":
            out[f"{h}.self_attn.o_proj.weight"] = full
        elif rest == "self_attention.q_layernorm.weight":
            out[f"{h}.self_attn.q_norm.weight"] = full
        elif rest == "self_attention.k_layernorm.weight":
            out[f"{h}.self_attn.k_norm.weight"] = full
        elif rest == "mlp.linear_fc1.weight":
            gate, up = _split_swiglu(full)
            out[f"{h}.mlp.gate_proj.weight"] = gate
            out[f"{h}.mlp.up_proj.weight"] = up
        elif rest == "mlp.linear_fc2.weight":
            out[f"{h}.mlp.down_proj.weight"] = full
        else:
            logger.debug("skipping parameter (unmapped layer suffix): %s", name)

    if seen_module_prefix:
        logger.info(
            "gather_full_state_dict: stripped 'module.' prefix; produced %d tensors",
            len(out),
        )
    else:
        logger.info("gather_full_state_dict: produced %d tensors", len(out))
    return out
