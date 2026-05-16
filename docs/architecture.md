# Architecture

NanoRL is the assembly: every component is reused; only the integration glue is new.

## System diagram

```
                     ┌────────────────────────────────────┐
                     │   Ray Driver  (nanorl.cli)         │
                     │   - placement groups               │
                     │   - actor lifecycle                │
                     │   - global step loop               │
                     └────────────┬───────────────────────┘
                                  │
        ┌─────────────────────────┼────────────────────────────┐
        │                         │                            │
        ▼                         ▼                            ▼
   RolloutActor                TrainActor                NanoCtrl + Redis
   (NanoInfra LLM,             (megatron-core,           (peer registry,
    attn_tp/ffn_ep)              TP/PP/EP per cfg)         RDMA MR table)
        │                         │                            │
        └────── DLSlime ──────────┴────────────────────────────┘
        SlimeRPC: trajectories  (rollout → train, M2/M1)
        PeerAgent.read: weights (train  → rollout, M3)
```

## Roles

### Control plane — Ray + NanoCtrl

- **Ray** owns process placement, actor lifecycle, and inter-node scheduling.
- **NanoCtrl + Redis** is the DLSlime peer registry: every PeerAgent registers an alias and an RDMA memory-region (MR) table; remote peers look them up to bootstrap connections.

### Data plane — DLSlime

- **SlimeRPC** (RPC-of-bytes over registered RDMA buffers) carries trajectories from rollout to train. The wire contract lives in `nanorl/data/trajectory_buffer.py:TrajectoryService`.
- **PeerAgent + RDMAEndpoint** carries weight tensors from train to rollout (M3). NanoRL registers the gathered (post-TP/PP/EP) tensors as MRs; the rollout side reads them in place.

### Training — megatron-core (not megatron.training)

NanoRL avoids `megatron.training`'s argparse + torchrun assumptions by calling `megatron.core` building blocks directly:

- `megatron.core.parallel_state.initialize_model_parallel(...)`
- `megatron.core.models.gpt.GPTModel`
- `megatron.core.pipeline_parallel.get_forward_backward_func(...)`
- `megatron.core.optimizer.get_megatron_optimizer(...)`
- `megatron.core.dist_checkpointing.{save,load}`
- `megatron.core.tensor_parallel.mappings.gather_from_tensor_model_parallel_region`

The GRPO loss math (`nanorl/rl/grpo_loss.py`) is vendored from `megatron/rl/rl_utils.py:1854` — byte-equivalent and pure-functional. The per-token logprob helper there reads `get_args()`, so we replace it with `nanorl/rl/logprobs.py`.

### Inference — NanoInfra

The rollout actor is a thin wrapper over `nanodeploy.llm_component.LLM`. NanoInfra spawns its own per-GPU Ray sub-actors (`ModelRunner`s) so NanoRL doesn't need a second layer of `@ray.remote`. The known gap — no in-place weight reload — is addressed in M3 by patching `ModelRunner.update_weights(named_tensors)`.

## Why these components

| Need                 | Choice                        | Why not alternative                                                                                                                                           |
| -------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trajectory transport | SlimeRPC                      | gRPC needs serialization on every byte; SlimeRPC reuses RDMA-registered buffers and gives us zero-copy via `@method(raw=True, inplace=True)` when we need it. |
| Weight transport     | DLSlime PeerAgent (raw RDMA)  | NCCL would force train and infer into the same process group; DLSlime decouples them.                                                                         |
| Training             | megatron-core                 | Already supports TP/PP/EP/SP and has a battle-tested distributed optimizer + dist-checkpointing. We borrow the building blocks but not the training script.   |
| Inference            | NanoInfra                     | Native EP for MoE, DLSlime executor backend means trajectories can stream over the same fabric, runs alongside Ray.                                           |
| Reward               | Pluggable `Verifier` protocol | Math/code rules first; reward models stay an option.                                                                                                          |

## Topology

For v1 we run **disaggregated**: train actors and rollout actors live on different Ray placement-group bundles. The weight-sync interface (`nanorl/weights/`, M3) is generic enough that the same code can run **co-located** later — that's a memory-eviction problem, not an interface problem.

## Three milestones

|     | What it proves                                              | What's left out                                                |
| --- | ----------------------------------------------------------- | -------------------------------------------------------------- |
| M1  | Train pulls trajectories over SlimeRPC and runs a GRPO step | No real rollout (we use a `fake_trajectory_server.py` fixture) |
| M2  | Rollout generates+scores+publishes trajectories             | No train (we use `fake_train_consumer.py` to pull)             |
| M3  | Train↔rollout weight sync; full GRPO loop                   | —                                                              |

Each is independently runnable end-to-end and has its own pass criteria; see the plan in `~/.claude/plans/fixed-eager-stroustrup.md`.
