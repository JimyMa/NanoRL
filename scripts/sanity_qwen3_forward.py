#!/usr/bin/env python3
"""One-shot Qwen3-4B HF→Megatron forward sanity check.

Critical for M1: if the fused-QKV interleave or any other Megatron name
mapping is wrong, ``transformers.AutoModelForCausalLM`` greedy-decodes the
prompt fine but our ``GPTModel`` returns gibberish. We catch that here in
~30 seconds before sinking time into TrainActor wiring.

Runs on a single GPU. Loads Qwen3-4B from HF, builds the matching Megatron
``GPTModel``, copies weights via ``load_qwen3_hf_into_megatron``, then does
both:

  1) Greedy-decodes a short continuation and prints it (visual sanity).
  2) For a fixed prompt, computes the top-1 token's logprob under both the
     HF model and the Megatron model. The two should agree to within ~5e-2
     in BF16 — wider than fp32 fortunately because of accumulated norm
     differences (Apex vs Torch Norm, different reduction orders).

Usage:
    python scripts/sanity_qwen3_forward.py            # default Qwen3-4B
    python scripts/sanity_qwen3_forward.py --hf-path /path/to/other
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", default="/models/model--Qwen-Qwen3-4B-Instruct-2507")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--logprob-tol", type=float, default=5e-2)
    args = ap.parse_args()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")

    import torch.distributed as dist

    dist.init_process_group(backend="nccl", rank=0, world_size=1)

    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)

    from nanorl.weights.hf_to_megatron import (
        build_transformer_config,
        hf_metadata,
        load_qwen3_hf_into_megatron,
    )

    print(f"[sanity] building Megatron GPTModel from {args.hf_path}")
    tcfg = build_transformer_config(args.hf_path, bf16=True)
    meta = hf_metadata(args.hf_path)
    spec = get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm")
    m = (
        GPTModel(
            config=tcfg,
            transformer_layer_spec=spec,
            vocab_size=meta["vocab_size"],
            max_sequence_length=meta["max_position_embeddings"],
            share_embeddings_and_output_weights=meta["tie_word_embeddings"],
            position_embedding_type="rope",
            rotary_base=meta["rope_base"],
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    print(f"[sanity] {sum(p.numel() for p in m.parameters())/1e9:.2f}B params")

    print("[sanity] loading HF weights into Megatron...")
    load_qwen3_hf_into_megatron(m, args.hf_path)

    print("[sanity] loading reference HF model for cross-check...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_path)
    hf = AutoModelForCausalLM.from_pretrained(
        args.hf_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # ---- Greedy continuation from Megatron ------------------------------
    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = torch.tensor([tok.encode(text)], device="cuda", dtype=torch.long)
    with torch.no_grad():
        cur = ids
        for _ in range(args.max_new_tokens):
            position_ids = torch.arange(cur.shape[1], device="cuda").unsqueeze(0)
            logits = m(cur, position_ids, attention_mask=None)
            # Megatron returns [B, T, V] when no `labels` are passed.
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_id], dim=1)
            if next_id.item() == tok.eos_token_id:
                break
        generated = tok.decode(cur[0, ids.shape[1] :], skip_special_tokens=True)
    print(f"[sanity] Megatron greedy continuation: {generated!r}")

    # ---- Top-1 token agreement vs HF ------------------------------------
    with torch.no_grad():
        hf_out = hf(ids).logits[:, -1, :]
        position_ids = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
        mc_out = m(ids, position_ids, attention_mask=None)[:, -1, :]
        hf_top1 = hf_out.argmax(dim=-1).item()
        mc_top1 = mc_out.argmax(dim=-1).item()
        hf_lp = torch.log_softmax(hf_out.float(), dim=-1).max().item()
        mc_lp = torch.log_softmax(mc_out.float(), dim=-1).max().item()
        diff = abs(hf_lp - mc_lp)
    print(f"[sanity] HF top-1: {hf_top1} ({tok.decode([hf_top1])!r}) lp={hf_lp:.4f}")
    print(f"[sanity] MC top-1: {mc_top1} ({tok.decode([mc_top1])!r}) lp={mc_lp:.4f}")
    print(f"[sanity] |Δ logprob| = {diff:.4f} (tol={args.logprob_tol})")
    if hf_top1 != mc_top1:
        print(
            "[sanity] FAIL: top-1 token disagrees — QKV interleave or other mapping is wrong"
        )
        return 2
    if diff > args.logprob_tol:
        print(
            "[sanity] WARN: logprob gap exceeds tol — investigate before relying on this loader"
        )
        return 3
    print("[sanity] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
