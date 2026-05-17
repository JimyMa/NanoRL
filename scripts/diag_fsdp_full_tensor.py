#!/usr/bin/env python3
"""Diagnose what MegatronFSDP DTensor.full_tensor() returns for QKV.

Symptom from the M3 FSDP smoke: after fully_shard with world_size=2 on
Qwen3-4B, calling ``param.full_tensor()`` on
``decoder.layers.0.self_attention.linear_qkv.weight`` returns a tensor
with 30,351,360 elements — but the declared shape is [6144, 2560]
(15,728,640 elements). This script probes the actual shape so we can
fix the gather walk.

Usage:
    torchrun --nproc_per_node=2 scripts/diag_fsdp_full_tensor.py
"""
from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    print(f"[rank {rank}] world={world} local={local_rank}")

    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed.fsdp.src.megatron_fsdp import fully_shard
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)

    from nanorl.weights.hf_to_megatron import build_transformer_config, hf_metadata

    HF = "/models/model--Qwen-Qwen3-4B-Instruct-2507"
    tcfg = build_transformer_config(HF, bf16=True)
    meta = hf_metadata(HF)
    spec = get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm")

    gpt = (
        GPTModel(
            config=tcfg,
            transformer_layer_spec=spec,
            vocab_size=meta["vocab_size"],
            max_sequence_length=2048,
            share_embeddings_and_output_weights=meta["tie_word_embeddings"],
            position_embedding_type="rope",
            rotary_base=meta["rope_base"],
        )
        .cuda()
        .to(torch.bfloat16)
    )

    base_optim = torch.optim.Adam(gpt.parameters(), lr=1e-6)
    model, optim = fully_shard(
        module=gpt,
        optimizer=base_optim,
        fsdp_unit_modules=[
            "megatron.core.transformer.transformer_layer.TransformerLayer"
        ],
        zero_dp_strategy="optim_grads_params",
    )

    # Pick a few representative params and print their shape and full_tensor shape.
    interesting = [
        "module.embedding.word_embeddings.weight",
        "module.decoder.layers.0.input_layernorm.weight",
        "module.decoder.layers.0.self_attention.linear_qkv.weight",
        "module.decoder.layers.0.self_attention.linear_proj.weight",
        "module.decoder.layers.0.mlp.linear_fc1.weight",
        "module.decoder.layers.0.mlp.linear_fc2.weight",
    ]
    name_to_p = dict(model.named_parameters())
    for nm in interesting:
        p = name_to_p.get(nm)
        if p is None:
            print(f"[rank {rank}] {nm}: <MISSING>")
            continue
        is_dt = (
            hasattr(p, "full_tensor")
            and not isinstance(p, torch.Tensor)
            or "DTensor" in type(p).__name__
        )
        local_shape = tuple(p.shape) if hasattr(p, "shape") else "<?>"
        local_numel = p.numel() if hasattr(p, "numel") else -1
        print(
            f"[rank {rank}] {nm}: type={type(p).__name__} declared_shape={local_shape} numel={local_numel} dtensor={is_dt}"
        )
        if hasattr(p, "full_tensor"):
            try:
                ft = p.full_tensor()
                print(
                    f"[rank {rank}]   full_tensor: type={type(ft).__name__} shape={tuple(ft.shape)} numel={ft.numel()}"
                )
            except Exception as e:
                print(f"[rank {rank}]   full_tensor FAILED: {type(e).__name__}: {e}")
        if hasattr(p, "to_local"):
            try:
                tl = p.to_local()
                print(
                    f"[rank {rank}]   to_local:    type={type(tl).__name__} shape={tuple(tl.shape)} numel={tl.numel()}"
                )
            except Exception as e:
                print(f"[rank {rank}]   to_local FAILED: {type(e).__name__}: {e}")

    # Also report what megatron-fsdp's preferred state_dict looks like
    print(f"[rank {rank}] testing model.state_dict()...")
    sd = model.state_dict()
    if rank == 0:
        for k in interesting:
            v = sd.get(k)
            if v is None:
                print(f"[state_dict] {k}: <MISSING>")
                continue
            print(
                f"[state_dict] {k}: type={type(v).__name__} "
                f"shape={tuple(v.shape)} numel={v.numel()} "
                f"dtype={v.dtype}"
            )
    dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
