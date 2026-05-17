# Training (M1 + M3)

NanoRL's training side is Ray-managed. `nanorl train` launches TrainActor
workers as Ray actors on the requested `TRAIN_IP`; the shell that starts the
driver does not have to be the training node.

## DDP single-rank (M1 / M3 baseline)

`nanorl train --nproc 1` builds a megatron-core `GPTModel` wrapped in
`megatron.core.distributed.DistributedDataParallel`. Each parameter is a regular
`torch.Tensor` on this rank's GPU.

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
| pull_s  | ~3 s (4 NanoDeploy workers RDMA-read in parallel)                                |
| apply_s | ~0.85 s (in-place `param.data.copy_`, no CUDA-graph recapture)                   |
| Wall    | ~5 s                                                                             |

The smoke scripts use `nanorl train`, so the shell can run on a driver node
while Ray places the TrainActor on the requested `TRAIN_IP`.

## FSDP multi-rank (M3+ ZeRO-3)

`cfg.train.fsdp = true` plus `WORLD_SIZE >= 2` switches to `megatron.core.distributed.fsdp.fully_shard`. Each parameter is sharded across the data-parallel mesh as an *uneven DTensor* (Megatron-FSDP's variant — different ranks may hold different-sized slices).

```bash
TRAIN_IP=10.102.98.166 NPROC=8 bash scripts/m3_fsdp_smoke.sh
```

Smoke output (5 train steps × 2 ranks, 2 syncs):

```
{"step": 0, ...}    (rank 0)
{"step": 0, ...}    (rank 1, same loss/kl, same elapsed_s)
{"event": "weight_sync", "version": 2, "n_tensors": 398, "wall_s": 8.7, ...}
... 5 steps, 2 syncs ...
trainer exit=0
```

The train process is Ray-managed: `scripts/m3_fsdp_smoke.sh` starts rollout on
the rollout node, waits for the first batch, then launches `nanorl.cli train`
with a strict-pack placement group on `TRAIN_IP`.

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
- **Checkpoint save** is collective. Every rank participates in the FSDP gather; rank 0 writes the HF-format files.

## Off-policy logprobs

When `sampling.ship_logprobs: true`, rollout requests `return_completion_logprobs` from NanoDeploy. Each `Trajectory` carries the logprob of every sampled response token under the rollout-time policy. `TrainActor` consumes those as `old_logprobs`:

- `ratios = exp(current_logprobs - old_logprobs)` is no longer identically 1.
- `truncated_above_rate` and `truncated_below_rate` now reflect actual off-policy clipping.
- `logprob_to_old_mean` / `logprob_to_old_max` report masked absolute logprob drift.
- `kl_to_old` is the monitored off-policy distance and is the useful KL-like number for this path.

If rollout logprobs are absent (old NanoDeploy build, `--no-ship-logprobs`, or `sampling.ship_logprobs: false`), NanoRL falls back to train-side `current_logprobs.detach()`. That fallback is still useful for parity tests, but it makes ratios 1 by construction.

For base models set `model.apply_chat_template: false`. Some Qwen base checkpoints ship a tokenizer chat template inherited from instruct siblings; feeding those wrapper tokens to the base model can make outputs incoherent.

## Checkpoint save

`nanorl train` supports HF-format checkpoint export:

```bash
python -m nanorl.cli train ... \
  --save-dir /tmp/nanorl_ckpts/my_run \
  --save-every 50 \
  --save-final
```

Each checkpoint writes:

- `step_XXXXXX/model.safetensors`
- tokenizer/config files copied from `cfg.model.hf_path`
- `step_XXXXXX/nanorl_checkpoint.json`

The path is local to the Ray train node (`--train-ip`), not necessarily the
driver node. This is a weights-only export path; optimizer state, dataloader
position, and RNG state are not restored yet.

## Weight sync internals

The fast path (`LLMComponent.pull_and_apply_weights`) is what makes weight_sync_every=1 practical. For an 8 GB Qwen3-4B sync:

1. Train rank 0 calls `gather_full_state_dict` → 398 HF-named full tensors on CPU
2. `WeightTransportTrain.register(version=N, named_tensors)` → 398 RDMA MRs published under the train alias
3. Rank 0 RPCs the rollout side: `apply_weight_update(manifest_blob)`
4. Rollout's `TrajectoryService.apply_weight_update` forwards the manifest to `LLMComponent.pull_and_apply_weights`
5. `LLMComponent.pull_and_apply_weights` fan-outs to 4 NanoDeploy workers via `executor.collective_rpc("pull_and_apply_weights", ...)`
6. Each NanoDeploy worker uses its own PeerAgent (already started for KV-cache migration) to RDMA-read every entry **in parallel** with the other workers
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
| `off_policy_iters`                       | 1                          | 1                                                                          |

Other config fields that matter for the current training path:

| Field                       | Meaning                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| `model.apply_chat_template` | Whether rollout wraps prompts in the tokenizer chat template. Use `false` for base models. |
| `sampling.ship_logprobs`    | Ship rollout-time sampled-token logprobs to train as `old_logprobs`.                       |
| `rl.clamp_eps_lower/upper`  | GRPO ratio clipping range.                                                                 |
| `rl.kl_beta`                | Reference-model KL coefficient. Defaults to `0.0` in the FSDP smoke config.                |

`cfg.rl.kl_beta` defaults to **0** because of an unresolved kernel-parity issue between gradient-mode and no_grad-mode SDPA in BF16 — see `docs/troubleshooting.md`.

## Known limitations

- **TP / PP > 1 train side not supported.** The gather walk assumes the parameter list yielded by `named_parameters()` covers the full set. TP/PP would add additional collectives.
- **MoE not supported.** Qwen3.5-35B-A3B has expert-routed FFNs; the gather walk doesn't unfuse expert tensors.
- **Reference KL term doesn't help yet.** Wired but `kl_beta=0` until SDPA kernel parity is solved. Use `kl_to_old` for off-policy monitoring.
- **No checkpoint resume yet.** HF-format checkpoint save is implemented; optimizer/RNG resume is not.
- **`scripts/diag_fsdp_full_tensor.py`** reproduces the per-rank uneven-DTensor shape mismatch in ~30 s if you ever change the gather code and want to re-verify the `uneven_dtensor_to_full_tensor` path.
