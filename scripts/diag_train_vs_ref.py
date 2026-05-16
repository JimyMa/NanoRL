#!/usr/bin/env python3
"""Diagnose the M3e KL blowup.

At step 0 the trainable model (DDP-wrapped GPTModel with HF weights) and
the reference model (bare GPTModel with the same HF weights) should
produce byte-identical logits on the same input. If they don't, our KL
term computes ``exp(big_diff) - big_diff - 1`` and we get the runaway
losses we saw (kl_mean = 1e3 → 1e8 over 6 steps).

Run this on 1 GPU; it builds both models and compares logits on a small
random batch.
"""
from __future__ import annotations

import os
import sys

import torch


def main() -> int:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29510")

    import torch.distributed as dist

    dist.init_process_group(backend="nccl", rank=0, world_size=1)

    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.distributed import (
        DistributedDataParallel as DDP,
        DistributedDataParallelConfig as DDPCfg,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)

    from nanorl.rl.logprobs import compute_per_token_logprobs
    from nanorl.weights.hf_to_megatron import (
        build_transformer_config,
        hf_metadata,
        load_qwen3_hf_into_megatron,
    )

    HF = "/models/model--Qwen-Qwen3-4B-Instruct-2507"
    tcfg = build_transformer_config(HF, bf16=True)
    meta = hf_metadata(HF)
    spec = get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm")

    def make():
        m = (
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
        load_qwen3_hf_into_megatron(m, HF)
        return m

    print("[diag] building trainable (DDP-wrapped)...")
    bare_train = make()
    ddp_cfg = DDPCfg(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        bucket_size=None,
    )
    train_model = DDP(config=tcfg, ddp_config=ddp_cfg, module=bare_train)
    train_model.eval()

    print("[diag] building reference (bare)...")
    ref_model = make()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # Same batch.
    torch.manual_seed(123)
    B, T = 2, 32
    tokens = torch.randint(0, 1000, (B, T), device="cuda", dtype=torch.long)
    position_ids = torch.arange(T, device="cuda").unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        train_logits = train_model(tokens, position_ids, attention_mask=None)
        ref_logits = ref_model(tokens, position_ids, attention_mask=None)

    print(
        "[diag] train logits shape:",
        tuple(train_logits.shape),
        "dtype:",
        train_logits.dtype,
    )
    print(
        "[diag] ref   logits shape:",
        tuple(ref_logits.shape),
        "dtype:",
        ref_logits.dtype,
    )
    print(
        "[diag] max abs diff (no_grad vs no_grad):",
        (train_logits - ref_logits).abs().max().item(),
    )

    # Now in *gradient* mode for the trainable.
    train_model.train()
    grad_logits = train_model(tokens, position_ids, attention_mask=None)
    print("[diag] grad-mode train logits dtype:", grad_logits.dtype)
    print(
        "[diag] max abs diff (grad vs no_grad train):",
        (grad_logits - train_logits).abs().max().item(),
    )
    print(
        "[diag] max abs diff (grad train vs no_grad ref):",
        (grad_logits - ref_logits).abs().max().item(),
    )

    train_lp = compute_per_token_logprobs(train_logits, tokens)
    ref_lp = compute_per_token_logprobs(ref_logits, tokens)
    grad_lp = compute_per_token_logprobs(grad_logits, tokens)
    print("[diag] logprob diffs:")
    print("  no_grad vs no_grad:", (train_lp - ref_lp).abs().max().item())
    print("  grad vs no_grad train:", (grad_lp - train_lp).abs().max().item())
    print("  grad vs no_grad ref:", (grad_lp - ref_lp).abs().max().item())

    # Kl term that gets fed to grpo_loss in this scenario:
    ref_diff = ref_lp - grad_lp
    kl = ref_diff.exp() - ref_diff - 1
    print(
        "[diag] KL(grad current || no_grad ref):  max=%.4f  mean=%.4f"
        % (
            kl.abs().max().item(),
            kl.abs().mean().item(),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
