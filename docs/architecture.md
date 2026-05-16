# Architecture

NanoRL is the assembly. Every component is reused; only the integration glue is new.

## System diagram

```
                  ┌─────────────────────────────────────────────┐
                  │              nanorl train                   │
                  │   (driver: rank 0 of N FSDP/DDP train ranks)│
                  │   - global step loop                        │
                  │   - weight-sync barrier                     │
                  └──────────────┬──────────────────────────────┘
                                 │
       ┌───────────────────────┼─────────────────────────┐
       │                       │                         │
       ▼                       ▼                         ▼
  RolloutEngine           TrainActor (rank 0..N-1)  NanoCtrl + Redis
  (NanoInfra LLM,         (megatron-core,           (peer registry,
   patched with                DDP @ N=1 or              MR table,
   apply_weight_update         FSDP/ZeRO-3 @ N≥2)        manifest blobs)
   + pull_and_apply)          ↑                          ↑
       │                       │                         │
       └────── DLSlime ────────┴─────────────────────────┘
       SlimeRPC: trajectories  (rollout → train, M2)
       SlimeRPC: control RPCs  (apply_weight_update, M3)
       PeerAgent.read: weights (each NanoInfra worker pulls direct, M3)
```

## Roles

### Control plane — Ray + NanoCtrl

- **Ray** owns process placement, NanoInfra worker lifecycle, and inter-node scheduling.
- **NanoCtrl + Redis** is the DLSlime peer registry: every PeerAgent (one per train rank, one per NanoInfra worker, one per rollout driver) registers an alias and an RDMA memory-region table; remote peers look them up to bootstrap connections.

### Data plane — DLSlime

- **SlimeRPC** carries trajectories from rollout → train and the control-plane `apply_weight_update` call from train → rollout. Wire contract: `nanorl/data/trajectory_buffer.py:TrajectoryService`.
- **PeerAgent.endpoint.read** carries weight tensors from train → NanoInfra workers as raw RDMA reads. Each worker uses its *own* PeerAgent (the one already started for KV-cache migration in `nanodeploy/context/cache.py:start_peer_agent`) so all workers pull in parallel.

### Training — megatron-core

NanoRL avoids `megatron.training`'s argparse + torchrun assumptions and uses `megatron.core` building blocks directly:

- `megatron.core.parallel_state.initialize_model_parallel(...)`
- `megatron.core.models.gpt.GPTModel` with `get_gpt_layer_local_spec(qk_layernorm=True, normalization="RMSNorm")` (no Transformer Engine dep)
- `megatron.core.pipeline_parallel.get_forward_backward_func(...)`
- `megatron.core.optimizer.get_megatron_optimizer(...)` for DDP, plain `torch.optim.Adam` + `megatron.core.distributed.fsdp.fully_shard` for FSDP
- `megatron.core.distributed.DistributedDataParallel` (DDP) or `megatron.core.distributed.fsdp.MegatronFSDP` (ZeRO-3)
- `megatron.core.distributed.fsdp.uneven_dtensor.uneven_dtensor_to_full_tensor` for the FSDP gather

The GRPO loss math (`nanorl/rl/grpo_loss.py`) is vendored byte-for-byte from `megatron/rl/rl_utils.py:1854` and is pure-functional. The per-token logprob helper there reads `get_args()`, so we replace it with `nanorl/rl/logprobs.py`.

### Inference — NanoInfra

The rollout actor is a thin wrapper over `nanodeploy.llm_component.LLM`. NanoInfra spawns its own per-GPU Ray sub-actors (`ModelRunner`s) so NanoRL doesn't add a second `@ray.remote` layer. The M3 patch added two methods:

- `ModelRunner.apply_weight_update(named_tensors)` — in-place copy via the parameter's existing `weight_loader` callback (handles TP shard slicing automatically). In-place semantics preserve captured CUDA-graph addresses.
- `ModelRunner.pull_and_apply_weights(manifest_blob, train_alias)` — uses the worker's own PeerAgent to RDMA-read the manifest entries directly, skipping the Ray fan-out from the rollout driver.

## Topology

For v1 we run **disaggregated**: train ranks on one node (`.179`), NanoInfra workers on another (`.183`). The weight-sync interface is generic enough that co-located mode (same GPUs, time-sliced) can drop in later.

## Three milestones, all green

|     | What it proves                                              | Status                              |
| --- | ----------------------------------------------------------- | ----------------------------------- |
| M1  | Train pulls trajectories over SlimeRPC and runs a GRPO step | ✅ DDP single-rank, FSDP multi-rank |
| M2  | Rollout generates+scores+publishes trajectories             | ✅                                  |
| M3  | Train↔rollout weight sync; full GRPO loop                   | ✅ DDP, ✅ FSDP (2-rank ZeRO-3)     |

The performance optimization ("each NanoInfra worker pulls direct via its own PeerAgent") cut sync wall time **13×** vs the original Ray-fan-out design.
