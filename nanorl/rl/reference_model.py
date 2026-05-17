"""Frozen reference model holder.

The KL term in GRPO needs ``ref_logprobs`` from a stable reference policy.
The canonical recipe (see Anthropic / DeepSeek RLHF) keeps a copy of the
initial pretrained policy frozen for the entire run. Even when group
rewards saturate (every member of a GRPO group scores identically), the
KL term still provides a non-zero gradient signal — that's the whole
point of ``kl_beta``.

This module owns the construction and a thin no_grad logprob helper.
TrainActor instantiates it after loading its trainable model from the
same HF checkpoint, so the ref starts at the exact same point in
parameter space as ``pi_theta``.
"""

from __future__ import annotations

import logging

import torch

from nanorl.config import ModelCfg, TrainCfg
from nanorl.rl.logprobs import compute_per_token_logprobs

logger = logging.getLogger(__name__)


class ReferenceModel:
    """A frozen Qwen3 ``GPTModel`` for reference-policy logprob queries.

    Built with the same ``TransformerConfig`` and HF init as the trainable
    model. ``param.requires_grad_(False)`` and ``model.eval()`` so no grad
    bookkeeping fires; we still keep it on GPU for fast no_grad forward.

    Reference weights are *not* part of the M3 weight sync — they stay at
    the HF init for the lifetime of the job.
    """

    def __init__(self, model_cfg: ModelCfg, train_cfg: TrainCfg):
        # Lazy imports — avoid pulling Megatron into every CLI process.
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel

        from nanorl.weights.hf_to_megatron import (
            build_transformer_config,
            hf_metadata,
            load_qwen3_hf_into_megatron,
        )

        logger.info("building reference GPTModel from %s", model_cfg.hf_path)
        tcfg = build_transformer_config(model_cfg.hf_path, bf16=train_cfg.bf16)
        meta = hf_metadata(model_cfg.hf_path)
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        self.model = (
            GPTModel(
                config=tcfg,
                transformer_layer_spec=spec,
                vocab_size=meta["vocab_size"],
                max_sequence_length=train_cfg.seq_len,
                share_embeddings_and_output_weights=meta["tie_word_embeddings"],
                position_embedding_type="rope",
                rotary_base=meta["rope_base"],
            )
            .cuda()
            .to(torch.bfloat16 if train_cfg.bf16 else torch.float32)
        )
        load_qwen3_hf_into_megatron(self.model, model_cfg.hf_path)
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Stay in train() mode so the forward kernel selection matches
        # the trainable model's gradient-mode forward. Megatron-Core's
        # train/eval branch picks different attention kernels in bf16,
        # which can produce per-token logit drifts of ~10 between modes
        # — that nukes the KL term unless both forwards take the same
        # path. requires_grad=False already guarantees no gradient
        # bookkeeping, so train() here is purely a kernel-routing flag.
        self.model.train()
        logger.info(
            "reference model frozen (in train() for kernel parity), %d params",
            sum(p.numel() for p in self.model.parameters()),
        )

    def logprobs(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token logprobs under the frozen reference policy.

        Returns ``[B, T-1]`` — same shape as
        ``compute_per_token_logprobs`` produces under the trainable model,
        so the GRPO loss can subtract the two directly.
        """
        with torch.no_grad():
            logits = self.model(tokens, position_ids, attention_mask=None)
        return compute_per_token_logprobs(logits, tokens).detach()
