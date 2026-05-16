# Training (M1 + M3)

NanoRL's training side has two recipes, both running on real Qwen3-4B with the M2 rollout pipeline as the trajectory source.

## DDP single-rank (M1 / M3 baseline)

`nanorl train` (no torchrun, world_size=1) builds a megatron-core `GPTModel` wrapped in `megatron.core.distributed.DistributedDataParallel`. Each parameter is a regular `torch.Tensor` on this rank's GPU.

```bash
bash scripts/m3_smoke.sh
```

Smoke output (the canonical pass):

```
PASS: 5 train steps, 2 weight syncs, avg manifest size=398 tensors
```

Per sync:

|         |                                                                                  |
| ------- | -------------------------------------------------------------------------------- |
| Tensors | 398 (Qwen3-4B HF parameter count, minus tied lm_head and `_extra_state` buffers) |
| pull_s  | ~3 s (4 NanoInfra workers RDMA-read in parallel)                                 |
| apply_s | ~0.85 s (in-place `param.data.copy_`, no CUDA-graph recapture)                   |
| Wall    | ~5 s                                                                             |

## FSDP multi-rank (M3+ ZeRO-3)

`cfg.train.fsdp = true` plus `WORLD_SIZE >= 2` switches to `megatron.core.distributed.fsdp.fully_shard`. Each parameter is sharded across the data-parallel mesh as an *uneven DTensor* (Megatron-FSDP's variant — different ranks may hold different-sized slices).

```bash
bash scripts/m3_fsdp_smoke.sh        # default 2 GPUs, GPU 6+7
NPROC=4 bash scripts/m3_fsdp_smoke.sh  # 4-rank
```

Smoke output (5 train steps × 2 ranks, 2 syncs):

```
{"step": 0, ...}    (rank 0)
{"step": 0, ...}    (rank 1, same loss/kl, same elapsed_s)
{"event": "weight_sync", "version": 2, "n_tensors": 398, "wall_s": 8.7, ...}
... 5 steps, 2 syncs ...
trainer exit=0
```

### Per-sync timing comparison

|                          | DDP      | FSDP-2                              |
| ------------------------ | -------- | ----------------------------------- |
| Train step               | 0.30 s   | 0.30 s                              |
| Weight gather collective | 0 s      | **6 s** (uneven-DTensor all-gather) |
| RDMA pull (rollout side) | 1.5 s    | 1.5 s                               |
| Apply                    | 0.85 s   | 1.5 s                               |
| **Total per sync**       | **~5 s** | **~9 s**                            |

The 6 s FSDP gather collective is `uneven_dtensor_to_full_tensor` running over 398 parameters. We could batch this (one collective per FSDP unit, ~36 round-trips total instead of 398) — deferred optimization.

## Multi-rank semantics

Under FSDP all ranks must enter the same collectives in the same order:

- **Trajectory pulling** is rank-0-only. Rank 0 fetches a `TrajectoryBatch` over SlimeRPC, then broadcasts `tokens`, `position_ids`, `response_mask`, `rewards`, `group_ids`, `advantages` to other ranks via `torch.distributed.broadcast`.
- **Forward + backward** runs collectively (FSDP all-gathers params during forward, re-shards after, all-reduces grads in backward).
- **Optimizer step** is local on each rank (each rank only owns its 1/N shard of every param).
- **Weight gather** every rank participates in `uneven_dtensor_to_full_tensor` (it's an all-gather), but only rank 0 keeps the result and registers MRs. Non-zero ranks return `None` from `gather_and_publish`.
- **Weight publish RPC** is rank-0-only.

## Weight sync internals

The fast path (`LLMComponent.pull_and_apply_weights`) is what makes weight_sync_every=1 practical. For an 8 GB Qwen3-4B sync:

1. Train rank 0 calls `gather_full_state_dict` → 398 HF-named full tensors on CPU
2. `WeightTransportTrain.register(version=N, named_tensors)` → 398 RDMA MRs published under the train alias
3. Rank 0 RPCs the rollout side: `apply_weight_update(manifest_blob)`
4. Rollout's `TrajectoryService.apply_weight_update` forwards the manifest to `LLMComponent.pull_and_apply_weights`
5. `LLMComponent.pull_and_apply_weights` fan-outs to 4 NanoInfra workers via `executor.collective_rpc("pull_and_apply_weights", ...)`
6. Each NanoInfra worker uses its own PeerAgent (already started for KV-cache migration) to RDMA-read every entry **in parallel** with the other workers
7. Each worker calls `apply_named_tensors_in_place(self.model, named_tensors)` — the parameter's existing `weight_loader` callback handles TP shard slicing, in-place `.copy_()` preserves CUDA-graph addresses
8. Counts return up the chain; train rank 0 logs and unregisters MRs

The original design had the rollout driver pull all 8 GB to CPU and then Ray-RPC the dict to each of 4 workers (cross-host serialization). That was 65 s per sync. Worker-direct RDMA cut it to 5 s.

## Configuration

`cfg.train.*` fields (`nanorl/config.py:TrainCfg`):

| Field                                    | DDP                        | FSDP                                                                       |
| ---------------------------------------- | -------------------------- | -------------------------------------------------------------------------- |
| `tp` / `pp` / `ep`                       | 1 / 1 / 1 (only supported) | 1 / 1 / 1                                                                  |
| `fsdp`                                   | `false`                    | `true`                                                                     |
| `fsdp_sharding_strategy`                 | n/a                        | `optim_grads_params` (ZeRO-3, default), `optim_grads`, `optim`, `no_shard` |
| `world_size`                             | 1                          | 2+                                                                         |
| `micro_batch_size`                       | 1                          | 1                                                                          |
| `global_batch_size`                      | matches `sampling.n`       | matches `sampling.n`                                                       |
| `optimizer.{name,lr,betas,weight_decay}` | usual                      | usual                                                                      |
| `seq_len`                                | max sequence               | max sequence                                                               |
| `bf16`                                   | `true`                     | `true`                                                                     |

`cfg.rl.kl_beta` defaults to **0** because of an unresolved kernel-parity issue between gradient-mode and no_grad-mode SDPA in BF16 — see `docs/troubleshooting.md`.

## Known limitations

- **TP / PP > 1 train side not supported.** The gather walk assumes the parameter list yielded by `named_parameters()` covers the full set. TP/PP would add additional collectives.
- **MoE not supported.** Qwen3.5-35B-A3B has expert-routed FFNs; the gather walk doesn't unfuse expert tensors.
- **Reference KL term doesn't help yet.** Wired but `kl_beta=0` until SDPA kernel parity is solved.
- **No checkpoint save/resume.**
- **`scripts/diag_fsdp_full_tensor.py`** reproduces the per-rank uneven-DTensor shape mismatch in ~30 s if you ever change the gather code and want to re-verify the `uneven_dtensor_to_full_tensor` path.
