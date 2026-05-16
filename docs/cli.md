# CLI reference

Entrypoint:

```bash
python -m nanorl.cli {rollout-only,train-only,train} ...
```

Or, after `pip install -e .`, the same as `nanorl {...}`.

| Subcommand     | Status | Purpose                                                                                                                                                                  |
| -------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `rollout-only` | ✅     | Generate trajectories with NanoInfra, score with verifier, optionally publish over SlimeRPC. M2 entry point.                                                             |
| `train-only`   | ✅     | Drive a megatron-core TrainActor against an externally-running rollout. M1 entry point — no weight sync.                                                                 |
| `train`        | ✅     | Full M3 loop: same as `train-only` plus periodic `gather_and_publish` to push trained weights into the running rollout. Supports both DDP and FSDP via `cfg.train.fsdp`. |

Set log level with `NANORL_LOG_LEVEL` (default `INFO`).

______________________________________________________________________

## `nanorl rollout-only`

Stand up a `RolloutEngine` (NanoInfra `LLM` + verifier) and run one or more rollout rounds. Trajectories are scored, optionally appended to a JSONL for offline inspection, and optionally published over SlimeRPC for a downstream consumer.

### Required

| Flag             | Type  | Notes                                                                                |
| ---------------- | ----- | ------------------------------------------------------------------------------------ |
| `--cfg PATH`     | yaml  | Loads into `nanorl.config.NanoRLCfg`.                                                |
| `--prompts PATH` | jsonl | One `{"prompt", "reference", "group_id"}` per line. Bad rows skipped with a warning. |

### Generation

| Flag                 | Default   | Notes                                                                 |
| -------------------- | --------- | --------------------------------------------------------------------- |
| `--rounds N`         | `1`       | Each round runs all prompts × `sampling.n` rollouts.                  |
| `--limit-prompts N`  | `0` (off) | Take only the first N prompts.                                        |
| `--seed N`           | none      | Seeds Python `random` and `numpy.random`. Doesn't seed CUDA samplers. |
| `--top-p F`          | from cfg  | Override `sampling.top_p`.                                            |
| `--max-new-tokens N` | from cfg  | Override `sampling.max_new_tokens`.                                   |

### Output

| Flag                | Default | Notes                                       |
| ------------------- | ------- | ------------------------------------------- |
| `--save-jsonl PATH` | none    | Append every `Trajectory` to disk.          |
| `--no-rpc`          | off     | Don't start SlimeRPC server.                |
| `--dry-run`         | off     | Validate + exit before launching NanoInfra. |

### SlimeRPC + weight-sync wiring

| Flag                      | Default   | Notes                                                                                                         |
| ------------------------- | --------- | ------------------------------------------------------------------------------------------------------------- |
| `--producer-alias S`      | from cfg  | This rollout's NanoCtrl alias.                                                                                |
| `--consumer-alias S`      | from cfg  | The downstream train actor's alias.                                                                           |
| `--serve-forever`         | off       | Idle after `--rounds`, keep the SlimeRPC server up so a `nanorl train` driver can call `apply_weight_update`. |
| `--stop-after-buffered N` | `0` (off) | Exit when the SlimeRPC queue holds ≥ N.                                                                       |

When SlimeRPC is enabled (no `--no-rpc`), the rollout side also exposes the M3 `apply_weight_update(manifest_blob)` RPC that the train side calls during `gather_and_publish`. No extra flag is needed; the wiring activates automatically.

______________________________________________________________________

## `nanorl train-only`

Drive a single-rank megatron-core TrainActor (DDP) against an externally-running rollout. **No weight sync** — equivalent to M1.

| Flag                 | Default  | Notes                                |
| -------------------- | -------- | ------------------------------------ |
| `--cfg PATH`         | required | YAML config                          |
| `--steps N`          | `10`     | Number of GRPO steps                 |
| `--producer-alias S` | from cfg | Rollout to pull trajectories from    |
| `--consumer-alias S` | from cfg | This train's alias                   |
| `--master-port N`    | `29500`  | torch.distributed init port          |
| `--log-jsonl PATH`   | none     | Per-step `TrainStats` JSONL          |
| `--dry-run`          | off      | Build config, exit before TrainActor |

______________________________________________________________________

## `nanorl train`

Full M3 loop: train + periodic weight sync. Supports both DDP single-rank and FSDP multi-rank (set `cfg.train.fsdp = true` and launch with `torchrun --nproc_per_node=N`).

| Flag                    | Default  | Notes                                                                         |
| ----------------------- | -------- | ----------------------------------------------------------------------------- |
| `--cfg PATH`            | required | YAML config                                                                   |
| `--steps N`             | `10`     | Number of GRPO steps                                                          |
| `--weight-sync-every N` | from cfg | Sync interval (overrides `cfg.weight_sync_every`; `0` disables)               |
| `--producer-alias S`    | from cfg | Rollout to pull trajectories from                                             |
| `--consumer-alias S`    | from cfg | This train's alias (used as alias prefix; per-rank suffixes added under FSDP) |
| `--master-port N`       | `29500`  | Single-rank only — torchrun sets it under FSDP                                |
| `--log-jsonl PATH`      | none     | Per-step `TrainStats` + `weight_sync` events                                  |
| `--dry-run`             | off      | Build config, exit before TrainActor                                          |

### Examples

**DDP single-rank (M3 baseline):**

```bash
python -m nanorl.cli train \
  --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --steps 10 --weight-sync-every 2 \
  --producer-alias rollout:0 --consumer-alias train:0 \
  --log-jsonl /tmp/train.jsonl
```

**FSDP 2-rank (ZeRO-3):**

```bash
CUDA_VISIBLE_DEVICES=6,7 \
PYTHONPATH=/mnt/nvme1n1/ml_research/majinming/src/Megatron-LM \
torchrun --nproc_per_node=2 --master_port=29600 \
  -m nanorl.cli train \
    --cfg nanorl/configs/qwen3_4b_grpo_fsdp.yaml \
    --steps 5 --weight-sync-every 2 \
    --producer-alias rollout:0 --consumer-alias train:0
```

Or just `bash scripts/m3_fsdp_smoke.sh` which orchestrates both producer and trainer.

______________________________________________________________________

## Logging

`NANORL_LOG_LEVEL=DEBUG` to see SlimeRPC / RDMA QP setup chatter. NanoInfra and dlslime have their own loggers not controlled by this env var.

## Exit codes

| Code    | Meaning                                                           |
| ------- | ----------------------------------------------------------------- |
| `0`     | Success.                                                          |
| `1`     | Subcommand failed.                                                |
| `2`     | No valid prompts in the JSONL (rollout-only).                     |
| `3`     | NaN loss detected (train / train-only).                           |
| nonzero | Uncaught exception. SIGINT triggers a clean shutdown with code 0. |
