#!/usr/bin/env python3
"""One-shot sanity proof for ``LLMEngine.update_weights`` (NanoDeploy patch).

This is the M3a guardrail. It verifies two claims:

1. **In-place apply preserves CUDA graphs.** Calling ``update_weights({})``
   touches no parameters at all; the next greedy decode of the same prompt
   must be byte-identical to the pre-call decode. If it isn't, something
   in the apply path is rebinding storages and we cannot rely on
   ``param.data.copy_`` to keep captured graphs valid.

2. **The apply path actually writes into the live model.** We grab one
   HF parameter (``model.layers.0.input_layernorm.weight``), zero it,
   wrap it as a one-element ``named_tensors`` dict, and call
   ``update_weights``. The next greedy decode must change. If it doesn't,
   the named-tensor path isn't actually reaching the model — even though
   the no-op test passed.

Pre-reqs: NanoCtrl on http://10.102.97.179:3000, the shared Ray cluster,
and 4 GPUs on the configured ``master_address``. This script piggybacks
on ``nanorl/configs/qwen3_4b_grpo.yaml`` for all engine settings.

Usage:
    python scripts/sanity_apply_weight_update.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import torch


def _greedy_decode(llm, tokenizer, prompt_ids, max_new_tokens):
    from nanodeploy import Sequence
    from nanodeploy.sampling_params import SamplingParams

    seq = Sequence(
        list(prompt_ids),
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, ignore_eos=False
        ),
    )
    llm.add_request([seq])
    llm.generate(use_tqdm=False)
    return list(seq.completion_token_ids)


def main() -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("m3a-sanity")

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="nanorl/configs/qwen3_4b_grpo.yaml")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument(
        "--zero-param",
        default="model.layers.0.input_layernorm.weight",
        help="HF param name to zero out for the change-detect test",
    )
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nanodeploy.llm_component import LLM
    from nanorl.actors.rollout import _build_nano_config
    from nanorl.config import NanoRLCfg
    from transformers import AutoTokenizer

    cfg = NanoRLCfg.from_yaml(args.cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_path or cfg.model.hf_path
    )

    log.info("booting NanoDeploy LLM (this takes ~90s on Qwen3-4B)...")
    llm = LLM(_build_nano_config(cfg.model, cfg.infer))
    log.info("LLM ready")

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(text)
    log.info("prompt encoded: %d tokens", len(prompt_ids))

    # ---- 1) Baseline greedy decode ------------------------------------
    out_a = _greedy_decode(llm, tokenizer, prompt_ids, args.max_new_tokens)
    log.info("baseline output: %r", tokenizer.decode(out_a, skip_special_tokens=True))

    # ---- 2) No-op update_weights, expect byte-identical ---------------
    log.info("calling update_weights({}) (no-op)")
    llm.update_weights({})
    out_b = _greedy_decode(llm, tokenizer, prompt_ids, args.max_new_tokens)
    log.info("post-noop output: %r", tokenizer.decode(out_b, skip_special_tokens=True))
    if out_a != out_b:
        log.error(
            "FAIL: noop update_weights changed greedy decode\n"
            "  before: %s\n  after:  %s",
            out_a,
            out_b,
        )
        log.error(
            "  this means the apply path is rebinding storages or "
            "CUDA graphs are being recaptured implicitly"
        )
        return 2
    log.info("OK: noop update_weights leaves greedy decode byte-identical")

    # ---- 3) Real update: zero one layer, expect a change --------------
    # We need to discover the parameter's full shape, which lives on a
    # worker. Fetch any worker's view of it via collective_rpc — they all
    # share the same architecture.
    def _get_full_shape(named):
        """Return [shape, dtype] for `named` on rank 0; this is the *unsharded*
        HF shape because the train-side gather always sends full tensors."""
        # Cheap: just construct what HF would have here. We know the model
        # config from cfg; for `model.layers.0.input_layernorm.weight` the
        # shape is [hidden_size].
        import json

        with open(os.path.join(cfg.model.hf_path, "config.json")) as f:
            hf = json.load(f)
        if named.endswith("input_layernorm.weight") or named.endswith(
            "post_attention_layernorm.weight"
        ):
            return (hf["hidden_size"],), torch.bfloat16
        raise NotImplementedError(f"don't know shape for {named}; pick a layernorm")

    shape, dtype = _get_full_shape(args.zero_param)
    zero_tensor = torch.zeros(shape, dtype=dtype)
    log.info("zeroing %s (shape=%s)", args.zero_param, shape)
    llm.update_weights({args.zero_param: zero_tensor})
    out_c = _greedy_decode(llm, tokenizer, prompt_ids, args.max_new_tokens)
    log.info("post-zero output: %r", tokenizer.decode(out_c, skip_special_tokens=True))
    if out_b == out_c:
        log.error(
            "FAIL: zeroing %s did not change greedy decode\n"
            "  this means the named-tensor path didn't reach the live model",
            args.zero_param,
        )
        return 3
    log.info("OK: real update_weights changes greedy decode")
    log.info("M3a sanity PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
