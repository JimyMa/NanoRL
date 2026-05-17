# Architecture

NanoRL is the assembly. Every component is reused; only the integration glue is new.

## System diagram

```
                  ┌─────────────────────────────────────────────┐
                  │       nanorl train / train-ray              │
                  │   - driver loop                             │
                  │   - Ray placement for TrainActors           │
                  │   - weight-sync / save barriers             │
                  └──────────────┬──────────────────────────────┘
                                 │
       ┌───────────────────────┼─────────────────────────┐
       │                       │                         │
       ▼                       ▼                         ▼
  RolloutEngine           TrainActor (rank 0..N-1)  NanoCtrl + Redis
  (NanoDeploy LLM,        (megatron-core,           (peer registry,
   apply_weight_update,        DDP @ N=1 or              MR table,
   pull_and_apply,             FSDP/ZeRO-3 @ N≥2)        manifest blobs)
   rollout logprobs)          ↑                          ↑
       │                       │                         │
       └────── DLSlime ────────┴─────────────────────────┘
       SlimeRPC: trajectories + old_logprobs (rollout → train, M2)
       SlimeRPC: control RPCs  (apply_weight_update, M3)
       PeerAgent.read: weights (each NanoDeploy worker pulls direct, M3)
```

## Roles

### Control plane — Ray + NanoCtrl

- **Ray** owns process placement, NanoDeploy worker lifecycle, TrainActor placement (`train-ray`), and inter-node scheduling.
- **NanoCtrl + Redis** is the DLSlime peer registry: every PeerAgent (one per train rank, one per NanoDeploy worker, one per rollout driver) registers an alias and an RDMA memory-region table; remote peers look them up to bootstrap connections.

### Data plane — DLSlime

- **SlimeRPC** carries trajectories from rollout → train and the control-plane `apply_weight_update` call from train → rollout. Trajectories can include rollout-time sampled-token logprobs. Wire contract: `nanorl/data/trajectory_buffer.py:TrajectoryService`.
- **PeerAgent.endpoint.read** carries weight tensors from train → NanoDeploy workers as raw RDMA reads. Each worker uses its *own* PeerAgent (the one already started for KV-cache migration in `nanodeploy/context/cache.py:start_peer_agent`) so all workers pull in parallel.

### Training — megatron-core

NanoRL avoids `megatron.training`'s argparse + torchrun assumptions and uses `megatron.core` building blocks directly:

- `megatron.core.parallel_state.initialize_model_parallel(...)`
- `megatron.core.models.gpt.GPTModel` with `get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)`
- `megatron.core.pipeline_parallel.get_forward_backward_func(...)`
- `megatron.core.optimizer.get_megatron_optimizer(...)` for DDP, plain `torch.optim.Adam` + `megatron.core.distributed.fsdp.fully_shard` for FSDP
- `megatron.core.distributed.DistributedDataParallel` (DDP) or `megatron.core.distributed.fsdp.MegatronFSDP` (ZeRO-3)
- `megatron.core.distributed.fsdp.uneven_dtensor.uneven_dtensor_to_full_tensor` for the FSDP gather

The GRPO loss math (`nanorl/rl/grpo_loss.py`) is vendored byte-for-byte from `megatron/rl/rl_utils.py:1854` and is pure-functional. The per-token logprob helper there reads `get_args()`, so we replace it with `nanorl/rl/logprobs.py`. In the off-policy path, rollout-time logprobs are used as `old_logprobs`; otherwise the trainer falls back to `current_logprobs.detach()`.

### Inference — NanoDeploy

The rollout actor is a thin wrapper over `nanodeploy.llm_component.LLM`. NanoDeploy spawns its own per-GPU Ray sub-actors (`ModelRunner`s), while NanoRL's train side uses its own Ray actors in `train-ray`. The M3 patch added two methods:

- `ModelRunner.apply_weight_update(named_tensors)` — in-place copy via the parameter's existing `weight_loader` callback (handles TP shard slicing automatically). In-place semantics preserve captured CUDA-graph addresses.
- `ModelRunner.pull_and_apply_weights(manifest_blob, train_alias)` — uses the worker's own PeerAgent to RDMA-read the manifest entries directly, skipping the Ray fan-out from the rollout driver.
- `SamplingParams(return_completion_logprobs=True)` / sampler plumbing — returns sampled-token logprobs so NanoRL can compute off-policy ratios on the train side.

## Topology

For v1 we run **disaggregated**: train ranks are strict-packed by Ray on `TRAIN_IP`, while NanoDeploy workers run on the rollout `master_address` (for example `.183`). The weight-sync interface is generic enough that co-located mode (same GPUs, time-sliced) can drop in later.

## Three milestones, all green

|     | What it proves                                              | Status                              |
| --- | ----------------------------------------------------------- | ----------------------------------- |
| M1  | Train pulls trajectories over SlimeRPC and runs a GRPO step | ✅ DDP single-rank, FSDP multi-rank |
| M2  | Rollout generates+scores+publishes trajectories             | ✅                                  |
| M3  | Train↔rollout weight sync; full GRPO loop                   | ✅ DDP, ✅ FSDP (2-rank ZeRO-3)     |

The performance optimization ("each NanoDeploy worker pulls direct via its own PeerAgent") cut sync wall time **13×** vs the original Ray-fan-out design.
