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

import logging
import os
import time
from dataclasses import dataclass

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
    kl_mean: float
    mean_reward: float
    mean_advantage: float
    response_tokens: int
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
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
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
        spec = get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm")
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
        # broadcast in train_step.
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

        # All ranks: broadcast shape metadata first, then payload.
        meta_t = torch.zeros(2, dtype=torch.int64, device=self._cuda_device)
        if self._is_rank0:
            B, T = int(batch.tokens.shape[0]), int(batch.tokens.shape[1])
            meta_t[0] = B
            meta_t[1] = T
        torch.distributed.broadcast(meta_t, src=0)
        B, T = int(meta_t[0].item()), int(meta_t[1].item())

        def _alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=self._cuda_device)

        tokens_t = _alloc((B, T), torch.int64)
        position_ids_t = _alloc((B, T), torch.int64)
        response_mask_t = _alloc((B, T), torch.int64)
        rewards_t = _alloc((B,), torch.float32)
        group_ids_t = _alloc((B,), torch.int64)
        advantages_t = _alloc((B,), torch.float32)
        if self._is_rank0:
            tokens_t.copy_(torch.from_numpy(batch.tokens).cuda())
            position_ids_t.copy_(torch.from_numpy(batch.position_ids).cuda())
            response_mask_t.copy_(
                torch.from_numpy(batch.response_mask.astype(np.int64)).cuda()
            )
            rewards_t.copy_(torch.from_numpy(batch.rewards).cuda())
            group_ids_t.copy_(torch.from_numpy(batch.group_ids).cuda())
            advantages_t.copy_(torch.from_numpy(advantages_np).cuda())
        for t in (
            tokens_t,
            position_ids_t,
            response_mask_t,
            rewards_t,
            group_ids_t,
            advantages_t,
        ):
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
        )
        adv_bf = advantages_t.to(dtype=torch.bfloat16)
        return batch_local, adv_bf

    # ------------------------------------------------------------------

    def _logprobs_under_current_model(self, batch: TrajectoryBatch) -> torch.Tensor:
        tokens = torch.from_numpy(batch.tokens).cuda()
        position_ids = torch.from_numpy(batch.position_ids).cuda()
        with torch.no_grad():
            logits = self.model(tokens, position_ids, attention_mask=None)
        return compute_per_token_logprobs(logits, tokens).detach()

    def _forward_step(self, batch, old_logprobs, advantages, ref_logprobs):
        tokens = torch.from_numpy(batch.tokens).cuda()
        position_ids = torch.from_numpy(batch.position_ids).cuda()
        loss_mask = torch.from_numpy(batch.response_mask[:, 1:]).cuda().float()
        kl_beta = float(self.cfg.rl.kl_beta)

        def forward_step_func(_data_iter, model):
            logits = model(tokens, position_ids, attention_mask=None)
            current_logprobs = compute_per_token_logprobs(logits, tokens)

            def loss_func(_output_tensor):
                per_token, kl, ratios, ent, _ta, _tb = calculate_grpo_loss(
                    current_logprobs=current_logprobs,
                    old_logprobs=old_logprobs,
                    ref_logprobs=ref_logprobs,
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
                return scalar, {
                    "loss": scalar.detach(),
                    "kl_mean": kl_scalar.detach(),
                    "ratios_mean": (ratios * loss_mask).sum().detach() / denom,
                    "response_tokens": denom.detach(),
                }

            return logits, loss_func

        return forward_step_func

    # ------------------------------------------------------------------

    def train_step(self) -> TrainStats:
        from megatron.core.pipeline_parallel import get_forward_backward_func

        t0 = time.monotonic()
        batch, advantages = self._fetch_or_recv_batch()
        old_logprobs = self._logprobs_under_current_model(batch)

        if self._ref is not None:
            tokens = torch.from_numpy(batch.tokens).cuda()
            position_ids = torch.from_numpy(batch.position_ids).cuda()
            ref_logprobs = self._ref.logprobs(tokens, position_ids)
        else:
            ref_logprobs = old_logprobs

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
        # 3-tuple. Tolerate both.
        ret = self.optimizer.step()
        if ret is None or not isinstance(ret, tuple):
            update_successful = True
        else:
            update_successful = ret[0]

        d = losses_reduced[0] if losses_reduced else {}
        loss_val = float(d.get("loss", torch.tensor(float("nan"))).item())
        kl_val = float(d.get("kl_mean", torch.tensor(0.0)).item())
        ntok = int(d.get("response_tokens", torch.tensor(0)).item())
        stats = TrainStats(
            step=self._step,
            loss=loss_val,
            kl_mean=kl_val,
            mean_reward=float(np.mean(batch.rewards)),
            mean_advantage=float(advantages.float().mean().item()),
            response_tokens=ntok,
            elapsed_s=time.monotonic() - t0,
        )
        self._step += 1
        return stats

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
