"""Megatron-core TrainActor — single-rank DDP and multi-rank FSDP.

Pulls trajectories over SlimeRPC, runs a GRPO step, applies the optimizer.
``kl_beta=0`` for now; reference-model + KL ride alongside M3 weight sync.

Single-rank baseline (M1):
    World size 1, megatron-core ``DistributedDataParallel`` wrap, plain
    Megatron optimizer.

Multi-rank FSDP (M3+):
    World size ≥ 2, ``megatron.core.distributed.fsdp.fully_shard`` wraps
    the model + optimizer (each rank holds 1/N of every param).
    Trajectory pulling and the weight-sync RPC happen only on rank 0; the
    batch is broadcast across ranks via ``torch.distributed.broadcast``
    so every rank runs the same forward/backward in lockstep.
"""

from __future__ import annotations

import json

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nanorl.config import NanoRLCfg
from nanorl.data.data_loader import TrajectoryClient, TrajectoryClientCfg
from nanorl.data.sample import TrajectoryBatch
from nanorl.rl.advantages import group_relative_advantages
from nanorl.rl.grpo_loss import calculate_grpo_loss
from nanorl.rl.logprobs import compute_per_token_logprobs
from nanorl.rl.reference_model import ReferenceModel

logger = logging.getLogger(__name__)


@dataclass
class TrainStats:
    step: int
    loss: float
    kl_mean: float  # KL(ref || cur) averaged over response tokens (only when kl_beta>0)
    kl_to_old: float  # KL(πθ_old || πθ) approx — off-policy distance for one optim step
    ratios_mean: float  # exp(cur - old), masked by response_mask, averaged
    ratios_max: float
    truncated_above_rate: float  # fraction of response tokens hitting upper clip
    truncated_below_rate: float  # fraction hitting lower clip
    entropy_mean: float  # per-token entropy of πθ on response tokens
    old_logprobs_present: bool  # true when rollout-side response_logprobs arrived
    logprob_to_old_mean: float  # masked mean |current_logprobs - old_logprobs|
    logprob_to_old_max: float  # masked max |current_logprobs - old_logprobs|
    old_logprobs_abs_diff_mean: float  # masked mean |current_logprobs - old_logprobs|
    old_logprobs_abs_diff_max: float  # masked max |current_logprobs - old_logprobs|
    grad_norm: float  # total parameter gradient L2 norm (post-clip if clipped)
    mean_reward: float
    mean_advantage: float
    advantage_abs_mean: float
    advantage_std: float
    advantage_nonzero_rate: float
    response_tokens: int
    response_length_mean: float
    elapsed_s: float


def _is_rank_zero() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


class TrainActor:
    def __init__(
        self,
        cfg: NanoRLCfg,
        *,
        producer_alias: str | None = None,
        consumer_alias: str | None = None,
        master_port: int = 29500,
    ):
        self.cfg = cfg
        if cfg.train.tp != 1 or cfg.train.pp != 1 or cfg.train.ep != 1:
            raise NotImplementedError(
                "TrainActor M1 only supports TP=1 PP=1 EP=1 "
                f"(got tp={cfg.train.tp} pp={cfg.train.pp} ep={cfg.train.ep})"
            )

        env_world = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank = int(os.environ.get("RANK", "0"))
        env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if cfg.train.fsdp and env_world < 2:
            raise RuntimeError(
                f"cfg.train.fsdp=True requires WORLD_SIZE>=2 (got {env_world}). "
                "Launch via torchrun or via a Ray placement group with one "
                "TrainActor per GPU."
            )

        os.environ.setdefault("RANK", str(env_rank))
        os.environ.setdefault("WORLD_SIZE", str(env_world))
        os.environ.setdefault("LOCAL_RANK", str(env_local_rank))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(master_port))

        # Device binding: torchrun sets CUDA_VISIBLE_DEVICES per rank,
        # in which case every rank sees only one GPU as cuda:0. Otherwise
        # bind by LOCAL_RANK so siblings on the same node don't collide.
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and len(cuda_visible.split(",")) == 1:
            torch.cuda.set_device(0)
            self._cuda_device = torch.device("cuda:0")
        else:
            torch.cuda.set_device(env_local_rank)
            self._cuda_device = torch.device(f"cuda:{env_local_rank}")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=(
                    f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
                ),
                world_size=env_world,
                rank=env_rank,
            )

        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._is_rank0 = self._rank == 0

        from megatron.core import parallel_state, tensor_parallel
        from megatron.core.distributed import (
            DistributedDataParallel as DDP,
            DistributedDataParallelConfig as DDPCfg,
        )
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

        from nanorl.weights.hf_to_megatron import (
            build_transformer_config,
            hf_metadata,
            load_qwen3_hf_into_megatron,
        )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=cfg.train.tp,
            pipeline_model_parallel_size=cfg.train.pp,
            expert_model_parallel_size=cfg.train.ep,
        )
        tensor_parallel.model_parallel_cuda_manual_seed(0)

        logger.info(
            "[rank %d/%d] building GPTModel from %s",
            self._rank,
            self._world_size,
            cfg.model.hf_path,
        )
        self._tcfg = build_transformer_config(cfg.model.hf_path, bf16=cfg.train.bf16)
        meta = hf_metadata(cfg.model.hf_path)
        self._meta = meta
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        gpt = (
            GPTModel(
                config=self._tcfg,
                transformer_layer_spec=spec,
                vocab_size=meta["vocab_size"],
                max_sequence_length=cfg.train.seq_len,
                share_embeddings_and_output_weights=meta["tie_word_embeddings"],
                position_embedding_type="rope",
                rotary_base=meta["rope_base"],
            )
            .cuda()
            .to(torch.bfloat16 if cfg.train.bf16 else torch.float32)
        )
        load_qwen3_hf_into_megatron(gpt, cfg.model.hf_path)

        self._fsdp = bool(cfg.train.fsdp)
        if self._fsdp:
            from megatron.core.distributed.fsdp.src.megatron_fsdp import fully_shard

            opt_cfg = cfg.train.optimizer
            base_optimizer = torch.optim.Adam(
                gpt.parameters(),
                lr=opt_cfg.lr,
                betas=opt_cfg.betas,
                eps=1e-8,
                weight_decay=opt_cfg.weight_decay,
            )
            self.model, self.optimizer = fully_shard(
                module=gpt,
                optimizer=base_optimizer,
                fsdp_unit_modules=[
                    "megatron.core.transformer.transformer_layer.TransformerLayer",
                ],
                zero_dp_strategy=cfg.train.fsdp_sharding_strategy,
            )
            self.model.train()
            logger.info(
                "[rank %d] FSDP enabled: zero_dp=%s world_size=%d unit=TransformerLayer",
                self._rank,
                cfg.train.fsdp_sharding_strategy,
                self._world_size,
            )
        else:
            ddp_cfg = DDPCfg(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                use_distributed_optimizer=False,
                bucket_size=None,
            )
            self.model = DDP(config=self._tcfg, ddp_config=ddp_cfg, module=gpt)
            self.model.train()
            self._opt_cfg = OptimizerConfig(
                optimizer=cfg.train.optimizer.name,
                lr=cfg.train.optimizer.lr,
                adam_beta1=cfg.train.optimizer.betas[0],
                adam_beta2=cfg.train.optimizer.betas[1],
                adam_eps=1e-8,
                weight_decay=cfg.train.optimizer.weight_decay,
                bf16=cfg.train.bf16,
                params_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
                clip_grad=1.0,
                use_distributed_optimizer=False,
            )
            self.optimizer = get_megatron_optimizer(self._opt_cfg, [self.model])

        # Reference model only on rank 0 (it's only consulted there for
        # ref_logprobs broadcast). With kl_beta=0 we skip entirely.
        self._ref: ReferenceModel | None = None
        if cfg.rl.kl_beta > 0.0 and self._is_rank0:
            self._ref = ReferenceModel(cfg.model, cfg.train)

        self._step = 0

        # Trajectory client + PeerAgent only on rank 0. Other ranks have
        # no SlimeRPC connection; they receive the batch via NCCL
        # broadcast in train_step. Built AFTER model init because dlslime's
        # PeerAgent setup depends on a fully-initialised CUDA + NCCL
        # context — earlier attempts to move this before the model load
        # caused `connection.wait` to time out (qp_ready handshake never
        # completes). Producer-side ``_wait_for_mr`` is bumped via
        # ``DLSLIME_MR_WAIT_TIMEOUT_S`` to cover the model-init runtime.
        self._client: TrajectoryClient | None = None
        self._peer = None
        self._consumer_alias: str | None = None
        self._producer_alias: str | None = None
        if self._is_rank0:
            self._client = self._build_trajectory_client(producer_alias, consumer_alias)

    # ------------------------------------------------------------------

    def _build_trajectory_client(
        self, producer_alias: str | None, consumer_alias: str | None
    ) -> TrajectoryClient:
        from dlslime import PeerAgent

        cfg = self.cfg
        prod = producer_alias or f"{cfg.dlslime.rollout_alias_prefix}:0"
        cons = consumer_alias or f"{cfg.dlslime.train_alias_prefix}:0"
        self._consumer_alias = cons
        self._producer_alias = prod

        self._peer = PeerAgent(nanoctrl_url=cfg.dlslime.nanoctrl_url, alias=cons)
        connection = self._peer.connect_to(
            prod,
            ib_port=cfg.dlslime.ib_port,
            qp_num=cfg.dlslime.qp_num,
        )
        connection.wait(timeout=600.0)
        time.sleep(0.2)
        client = TrajectoryClient(
            self._peer,
            TrajectoryClientCfg(
                producer_alias=prod,
                pull_size=cfg.train.global_batch_size,
                prefetch_depth=2,
                pull_timeout_s=60.0,
            ),
        )
        logger.info("[rank 0] TrainActor SlimeRPC consumer ready: %s ← %s", cons, prod)
        return client

    # ------------------------------------------------------------------
    # Batch fetch + broadcast across ranks
    # ------------------------------------------------------------------

    def _fetch_or_recv_batch(self) -> tuple[TrajectoryBatch, torch.Tensor]:
        """Rank 0 pulls from SlimeRPC + group-relative advantage, then
        broadcasts everything to other ranks. Returns (batch, advantages)
        with advantages on this rank's CUDA device.

        For FSDP every rank needs the same tensors so the forward+backward
        is in lockstep. For DDP at world_size=1 this is a single-rank no-op.
        """
        cfg = self.cfg

        # Rank 0: pull, build numpy tensors.
        if self._is_rank0:
            batch = self._client.next_batch(
                cfg.train.global_batch_size, pad_id=self._meta["vocab_size"] - 1
            )
            advantages_np = group_relative_advantages(
                batch.rewards, batch.group_ids
            ).astype(np.float32)
        else:
            batch = None
            advantages_np = None

        if self._world_size == 1:
            adv = torch.from_numpy(advantages_np).to(
                self._cuda_device, dtype=torch.bfloat16
            )
            return batch, adv

        # All ranks: broadcast shape metadata first, then payload. The third
        # element is 1 iff the rank-0 batch carries response_logprobs (Phase
        # 1 of ref-on-infer-node design); other ranks then know whether to
        # allocate the receive buffer.
        meta_t = torch.zeros(3, dtype=torch.int64, device=self._cuda_device)
        if self._is_rank0:
            B, T = int(batch.tokens.shape[0]), int(batch.tokens.shape[1])
            meta_t[0] = B
            meta_t[1] = T
            meta_t[2] = 1 if batch.response_logprobs is not None else 0
        torch.distributed.broadcast(meta_t, src=0)
        B, T = int(meta_t[0].item()), int(meta_t[1].item())
        has_logprobs = bool(meta_t[2].item())

        def _alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=self._cuda_device)

        tokens_t = _alloc((B, T), torch.int64)
        position_ids_t = _alloc((B, T), torch.int64)
        response_mask_t = _alloc((B, T), torch.int64)
        rewards_t = _alloc((B,), torch.float32)
        group_ids_t = _alloc((B,), torch.int64)
        advantages_t = _alloc((B,), torch.float32)
        logprobs_t = _alloc((B, T - 1), torch.float32) if has_logprobs else None
        if self._is_rank0:
            tokens_t.copy_(torch.from_numpy(batch.tokens).cuda())
            position_ids_t.copy_(torch.from_numpy(batch.position_ids).cuda())
            response_mask_t.copy_(
                torch.from_numpy(batch.response_mask.astype(np.int64)).cuda()
            )
            rewards_t.copy_(torch.from_numpy(batch.rewards).cuda())
            group_ids_t.copy_(torch.from_numpy(batch.group_ids).cuda())
            advantages_t.copy_(torch.from_numpy(advantages_np).cuda())
            if has_logprobs:
                logprobs_t.copy_(torch.from_numpy(batch.response_logprobs).cuda())
        broadcast_list = [
            tokens_t,
            position_ids_t,
            response_mask_t,
            rewards_t,
            group_ids_t,
            advantages_t,
        ]
        if has_logprobs:
            broadcast_list.append(logprobs_t)
        for t in broadcast_list:
            torch.distributed.broadcast(t, src=0)

        # Reconstruct a TrajectoryBatch on every rank (numpy on CPU; cheap
        # for the small numeric sidecars and lets _forward_step share its
        # current code path).
        batch_local = TrajectoryBatch(
            tokens=tokens_t.cpu().numpy(),
            position_ids=position_ids_t.cpu().numpy(),
            response_mask=response_mask_t.cpu().numpy().astype(np.int8),
            rewards=rewards_t.cpu().numpy(),
            group_ids=group_ids_t.cpu().numpy(),
            seq_lengths=np.full((B,), T, dtype=np.int64),  # not used downstream
            response_logprobs=(logprobs_t.cpu().numpy() if has_logprobs else None),
        )
        adv_bf = advantages_t.to(dtype=torch.bfloat16)
        return batch_local, adv_bf

    # ------------------------------------------------------------------

    def _logprobs_under_current_model(self, batch: TrajectoryBatch) -> torch.Tensor:
        """Per-token logprobs under the current model, snapshot for ``old``.

        Critically runs the forward with ``torch.enable_grad()`` so Transformer Engine
        flash-attn picks the same backward-derivable kernel as the loss
        forward later in the same step. Under ``no_grad`` Transformer Engine picks a
        forward-only kernel that disagrees with the grad-mode one by ~10
        nats per token — exponentiates to ratios = 1e9, kills GRPO. We
        immediately ``.detach()`` the output and ``del`` the activation
        graph so this costs only a forward, no backward retention.
        """
        tokens = torch.from_numpy(batch.tokens).cuda()
        position_ids = torch.from_numpy(batch.position_ids).cuda()
        with torch.enable_grad():
            logits = self.model(tokens, position_ids, attention_mask=None)
            lp = compute_per_token_logprobs(logits, tokens).detach()
        # Drop the autograd graph so the saved activations for backward are
        # freed before the loss forward starts (otherwise we'd hold two full
        # forward graphs in memory at once).
        del logits
        return lp

    def _forward_step(self, batch, old_logprobs, advantages, ref_logprobs):
        tokens = torch.from_numpy(batch.tokens).cuda()
        position_ids = torch.from_numpy(batch.position_ids).cuda()
        loss_mask = torch.from_numpy(batch.response_mask[:, 1:]).cuda().float()
        kl_beta = float(self.cfg.rl.kl_beta)

        def forward_step_func(_data_iter, model):
            logits = model(tokens, position_ids, attention_mask=None)
            current_logprobs = compute_per_token_logprobs(logits, tokens)
            # Derive old_logprobs from the SAME forward, not from a separate
            # no_grad pass. The kernel-parity gap between two Transformer Engine flash-attn
            # forwards (even on the same input + same dtype) is ~10 nats per
            # token; running the loss with that gap gives ratios = 1e9 and
            # nukes GRPO. Sharing one forward gives ratios identically 1.0,
            # which is what GRPO expects when old=current (the policy hasn't
            # drifted yet, only the optimizer step about to fire will). The
            # `old_logprobs` argument is preserved for callers that DO want
            # to plumb in a true old-policy snapshot (M3-future: ship from
            # rollout side); when nothing is supplied, fall back to detach.
            effective_old = (
                old_logprobs if old_logprobs is not None else current_logprobs.detach()
            )
            # ref_logprobs only matters when kl_beta > 0; absent a real
            # reference model, fall back to the same tensor as old so the
            # KL term is identically zero (and the multiply-by-zero kl_beta
            # in calculate_grpo_loss makes the substitution irrelevant).
            effective_ref = ref_logprobs if ref_logprobs is not None else effective_old

            def loss_func(_output_tensor):
                per_token, kl, ratios, ent, trunc_a, trunc_b = calculate_grpo_loss(
                    current_logprobs=current_logprobs,
                    old_logprobs=effective_old,
                    ref_logprobs=effective_ref,
                    advantages=advantages,
                    clamp_eps_lower=self.cfg.rl.clamp_eps_lower,
                    clamp_eps_upper=self.cfg.rl.clamp_eps_upper,
                    kl_beta=kl_beta,
                    entropy_weight=self.cfg.rl.entropy_weight,
                )
                masked = per_token * loss_mask
                denom = loss_mask.sum().clamp_min(1.0)
                scalar = masked.sum() / denom
                kl_scalar = (kl * loss_mask).sum() / denom

                # Tier-2 health signals — all masked to response tokens.
                # ``kl_to_old`` uses the same Schulman approx but on
                # (current - old) so it tells us how far one optim step
                # moved the policy. ``ratios = exp(cur - old)`` already.
                old_diff = current_logprobs - effective_old
                old_abs_diff = old_diff.abs() * loss_mask
                logprob_to_old_mean = old_abs_diff.sum().detach() / denom
                logprob_to_old_max = old_abs_diff.max().detach()
                self._maybe_dump_logprob_parity(
                    batch=batch,
                    current_logprobs=current_logprobs,
                    old_logprobs=old_logprobs,
                    loss_mask=loss_mask,
                )
                kl_to_old = ((old_diff.exp() - old_diff - 1) * loss_mask).sum() / denom
                ratios_max = (ratios * loss_mask).max().detach()
                trunc_a_rate = (trunc_a.float() * loss_mask).sum() / denom
                trunc_b_rate = (trunc_b.float() * loss_mask).sum() / denom
                entropy_scalar = (ent * loss_mask).sum() / denom
                return scalar, {
                    "loss": scalar.detach(),
                    "kl_mean": kl_scalar.detach(),
                    "kl_to_old": kl_to_old.detach(),
                    "ratios_mean": (ratios * loss_mask).sum().detach() / denom,
                    "ratios_max": ratios_max,
                    "truncated_above_rate": trunc_a_rate.detach(),
                    "truncated_below_rate": trunc_b_rate.detach(),
                    "entropy_mean": entropy_scalar.detach(),
                    "old_logprobs_present": torch.tensor(
                        old_logprobs is not None,
                        device=current_logprobs.device,
                        dtype=torch.bool,
                    ),
                    "logprob_to_old_mean": logprob_to_old_mean,
                    "logprob_to_old_max": logprob_to_old_max,
                    "old_logprobs_abs_diff_mean": logprob_to_old_mean,
                    "old_logprobs_abs_diff_max": logprob_to_old_max,
                    "response_tokens": denom.detach(),
                }

            return logits, loss_func

        return forward_step_func

    # ------------------------------------------------------------------

    def _maybe_dump_logprob_parity(
        self,
        *,
        batch: TrajectoryBatch,
        current_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor | None,
        loss_mask: torch.Tensor,
    ) -> None:
        """Dump a tiny per-token current-vs-rollout logprob probe.

        Enable with ``NANORL_DEBUG_LOGPROB_PARITY=1``. This is intentionally
        rank-0 only and bounded so a long run cannot flood disk.
        """
        if (
            not self._is_rank0
            or old_logprobs is None
            or os.environ.get("NANORL_DEBUG_LOGPROB_PARITY", "").lower()
            not in {"1", "true", "yes", "on"}
        ):
            return

        max_steps = int(os.environ.get("NANORL_DEBUG_LOGPROB_PARITY_STEPS", "2"))
        if self._step >= max_steps:
            return

        max_samples = int(os.environ.get("NANORL_DEBUG_LOGPROB_PARITY_SAMPLES", "2"))
        max_tokens = int(os.environ.get("NANORL_DEBUG_LOGPROB_PARITY_TOKENS", "64"))
        path = os.environ.get(
            "NANORL_DEBUG_LOGPROB_PARITY_PATH",
            "/tmp/nanorl_logprob_parity.jsonl",
        )

        cur = current_logprobs.detach().float().cpu().numpy()
        old = old_logprobs.detach().float().cpu().numpy()
        mask = loss_mask.detach().bool().cpu().numpy()
        tokens = batch.tokens
        response_mask = batch.response_mask.astype(bool)

        shift_mae = {}
        for shift in (-2, -1, 0, 1, 2):
            total = 0.0
            count = 0
            for i in range(mask.shape[0]):
                idx = np.flatnonzero(mask[i])
                if shift < 0:
                    idx = idx[idx >= -shift]
                elif shift > 0:
                    idx = idx[idx + shift < cur.shape[1]]
                if idx.size == 0:
                    continue
                total += float(np.abs(cur[i, idx + shift] - old[i, idx]).sum())
                count += int(idx.size)
            shift_mae[str(shift)] = total / max(count, 1)

        samples = []
        for i in range(min(max_samples, mask.shape[0])):
            idx = np.flatnonzero(mask[i])
            prompt_len = (
                int(np.argmax(response_mask[i])) if response_mask[i].any() else 0
            )
            response_len = int(response_mask[i].sum())
            token_rows = []
            for j in idx[:max_tokens]:
                c = float(cur[i, j])
                o = float(old[i, j])
                token_rows.append(
                    {
                        "logprob_index": int(j),
                        "token_pos": int(j + 1),
                        "token_id": int(tokens[i, j + 1]),
                        "current_logprob": c,
                        "old_logprob": o,
                        "diff": c - o,
                        "ratio": float(np.exp(c - o)),
                    }
                )
            samples.append(
                {
                    "batch_index": i,
                    "group_id": int(batch.group_ids[i]),
                    "reward": float(batch.rewards[i]),
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "tokens": token_rows,
                }
            )

        row = {
            "step": int(self._step),
            "shift_mae": shift_mae,
            "samples": samples,
        }
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to dump logprob parity probe: %s", exc)

    def train_step(self) -> TrainStats:
        from megatron.core.pipeline_parallel import get_forward_backward_func

        t0 = time.monotonic()
        batch, advantages = self._fetch_or_recv_batch()
        # ``old_logprobs`` for the importance ratio. Phase-1 source: shipped
        # from the rollout side via ``Trajectory.response_logprobs`` so
        # ``ratios = exp(current - old)`` reflects real policy drift since
        # the last weight sync (was identically 1.0 with the train-side
        # ``current_logprobs.detach()`` fallback). When unavailable (older
        # rollouts, or ``cfg.sampling.ship_logprobs=false``) we fall back
        # to that detach path inside ``_forward_step``.
        if batch.response_logprobs is not None:
            old_logprobs = torch.from_numpy(batch.response_logprobs).to(
                self._cuda_device, dtype=torch.bfloat16
            )
        else:
            old_logprobs = None
        if self._ref is not None:
            tokens = torch.from_numpy(batch.tokens).cuda()
            position_ids = torch.from_numpy(batch.position_ids).cuda()
            ref_logprobs = self._ref.logprobs(tokens, position_ids)
        else:
            ref_logprobs = None

        forward_step_func = self._forward_step(
            batch, old_logprobs, advantages, ref_logprobs
        )
        forward_backward = get_forward_backward_func()

        self.optimizer.zero_grad()
        losses_reduced = forward_backward(
            forward_step_func=forward_step_func,
            data_iterator=iter([None]),
            model=[self.model],
            num_microbatches=1,
            seq_length=batch.tokens.shape[1],
            micro_batch_size=batch.tokens.shape[0],
            forward_only=False,
        )
        # FSDP's wrapped optimizer.step returns None; megatron's returns a
        # 3-tuple. Tolerate both. For grad-norm we compute it ourselves
        # (post-step but pre-zero) so the value is consistent across
        # backends — neither wrapper exposes it cleanly.
        grad_norm = self._global_grad_norm()
        ret = self.optimizer.step()
        if ret is None or not isinstance(ret, tuple):
            update_successful = True
        else:
            update_successful = ret[0]

        d = losses_reduced[0] if losses_reduced else {}
        loss_val = float(d.get("loss", torch.tensor(float("nan"))).item())
        kl_val = float(d.get("kl_mean", torch.tensor(0.0)).item())
        kl_to_old_val = float(d.get("kl_to_old", torch.tensor(0.0)).item())
        ratios_mean = float(d.get("ratios_mean", torch.tensor(1.0)).item())
        ratios_max = float(d.get("ratios_max", torch.tensor(1.0)).item())
        trunc_a = float(d.get("truncated_above_rate", torch.tensor(0.0)).item())
        trunc_b = float(d.get("truncated_below_rate", torch.tensor(0.0)).item())
        entropy_mean = float(d.get("entropy_mean", torch.tensor(0.0)).item())
        old_logprobs_present = bool(
            d.get("old_logprobs_present", torch.tensor(False)).item()
        )
        old_abs_diff_mean = float(
            d.get(
                "logprob_to_old_mean",
                d.get("old_logprobs_abs_diff_mean", torch.tensor(0.0)),
            ).item()
        )
        old_abs_diff_max = float(
            d.get(
                "logprob_to_old_max",
                d.get("old_logprobs_abs_diff_max", torch.tensor(0.0)),
            ).item()
        )
        ntok = int(d.get("response_tokens", torch.tensor(0)).item())

        # Response-length stat: mean of per-row response_mask sum.
        rmask = batch.response_mask
        if rmask.size:
            response_length_mean = float(rmask.sum(axis=1).mean())
        else:
            response_length_mean = 0.0
        adv_float = advantages.float()
        advantage_abs_mean = float(adv_float.abs().mean().item())
        advantage_std = float(adv_float.std(unbiased=False).item())
        advantage_nonzero_rate = float((adv_float.abs() > 1e-6).float().mean().item())

        stats = TrainStats(
            step=self._step,
            loss=loss_val,
            kl_mean=kl_val,
            kl_to_old=kl_to_old_val,
            ratios_mean=ratios_mean,
            ratios_max=ratios_max,
            truncated_above_rate=trunc_a,
            truncated_below_rate=trunc_b,
            entropy_mean=entropy_mean,
            old_logprobs_present=old_logprobs_present,
            logprob_to_old_mean=old_abs_diff_mean,
            logprob_to_old_max=old_abs_diff_max,
            old_logprobs_abs_diff_mean=old_abs_diff_mean,
            old_logprobs_abs_diff_max=old_abs_diff_max,
            grad_norm=grad_norm,
            mean_reward=float(np.mean(batch.rewards)),
            mean_advantage=float(adv_float.mean().item()),
            advantage_abs_mean=advantage_abs_mean,
            advantage_std=advantage_std,
            advantage_nonzero_rate=advantage_nonzero_rate,
            response_tokens=ntok,
            response_length_mean=response_length_mean,
            elapsed_s=time.monotonic() - t0,
        )
        self._step += 1
        return stats

    def _global_grad_norm(self) -> float:
        """Total L2 norm of all gradients on this rank, post-backward.

        Megatron's DDP keeps the FP32 reduced gradient on ``param.main_grad``
        (the bucket-flat buffer); ``param.grad`` is None. FSDP and vanilla
        PyTorch use ``param.grad`` (possibly a DTensor shard). Read whichever
        is populated, then sum-reduce the squared norm across ranks for FSDP.
        """
        sq = torch.zeros(1, device=self._cuda_device)
        for p in self.model.parameters():
            g = getattr(p, "main_grad", None)
            if g is None:
                g = p.grad
            if g is None:
                continue
            if hasattr(g, "to_local"):
                g = g.to_local()
            sq += g.float().pow(2).sum()
        if (
            self._fsdp
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.all_reduce(sq, op=torch.distributed.ReduceOp.SUM)
        return float(sq.sqrt().item())

    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                if self._peer is not None:
                    self._peer.shutdown()

    # ------------------------------------------------------------------
    # M3: weight sync
    # ------------------------------------------------------------------

    def gather_and_publish(self, version: int) -> dict | None:
        """Collective gather (every rank participates), rank 0 publishes."""
        import pickle

        from dlslime.rpc import proxy as _proxy

        from nanorl.data.trajectory_buffer import TrajectoryService
        from nanorl.weights.megatron_to_hf import gather_full_state_dict
        from nanorl.weights.transport import WeightTransportTrain

        if self._fsdp:
            target = self.model
        else:
            target = self.model.module if hasattr(self.model, "module") else self.model

        named = gather_full_state_dict(
            target,
            num_query_groups=self._tcfg.num_query_groups,
            num_attention_heads=self._tcfg.num_attention_heads,
            head_dim=self._tcfg.kv_channels,
            hidden_size=self._tcfg.hidden_size,
        )

        if not self._is_rank0:
            return None

        if not hasattr(self, "_weight_sender"):
            self._weight_sender = WeightTransportTrain(self._peer)
            self._weight_proxy = _proxy(
                self._peer,
                self._producer_alias,
                TrajectoryService,
            )

        manifest = self._weight_sender.register(version, named)
        try:
            fut = self._weight_proxy.apply_weight_update(pickle.dumps(manifest))
            result_blob = fut.wait(timeout=600.0)
            result = pickle.loads(result_blob)
        finally:
            self._weight_sender.unregister(version)
        logger.info("weight sync complete: %s", result)
        return result

    def save_hf_checkpoint(self, path: str, *, step: int | None = None) -> dict | None:
        """Collectively gather weights and save a HF-style checkpoint on rank 0."""
        from safetensors.torch import save_file

        from nanorl.weights.megatron_to_hf import gather_full_state_dict

        if self._fsdp:
            target = self.model
        else:
            target = self.model.module if hasattr(self.model, "module") else self.model

        named = gather_full_state_dict(
            target,
            num_query_groups=self._tcfg.num_query_groups,
            num_attention_heads=self._tcfg.num_attention_heads,
            head_dim=self._tcfg.kv_channels,
            hidden_size=self._tcfg.hidden_size,
        )

        if not self._is_rank0:
            return None

        out_dir = Path(path)
        tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        src_dir = Path(self.cfg.model.hf_path)
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, tmp_dir / name)

        save_file(
            {k: v.detach().cpu().contiguous() for k, v in named.items()},
            str(tmp_dir / "model.safetensors"),
            metadata={"format": "pt"},
        )
        metadata = {
            "step": int(self._step if step is None else step),
            "source_hf_path": str(src_dir),
            "format": "hf_safetensors",
            "n_tensors": len(named),
        }
        (tmp_dir / "nanorl_checkpoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp_dir, out_dir)
        logger.info("saved HF checkpoint: %s", metadata | {"path": str(out_dir)})
        return metadata | {"path": str(out_dir)}
