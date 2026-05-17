# CLI reference

Entrypoints:

```bash
python -m nanorl.cli {rollout-only,train-only,train,train-ray,consume-ray} ...
nanorl-dashboard --train-jsonl /tmp/nanorl_smoke/m3_train.jsonl
```

Or, after `pip install -e .`, the same as `nanorl {...}` and
`nanorl-dashboard ...`.

| Command            | Status | Purpose                                                                                                                                                                   |
| ------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rollout-only`     | ✅     | Generate trajectories with NanoDeploy, score with verifier, optionally publish over SlimeRPC. M2 entry point. Can also run held-out eval and ship rollout-time logprobs.  |
| `train-only`       | ✅     | Drive a megatron-core TrainActor against an externally-running rollout. M1 entry point — no weight sync.                                                                  |
| `train`            | ✅     | Direct local/`torchrun` trainer. Same M3 loop as `train-only` plus periodic `gather_and_publish`, but the train process runs where the command or `torchrun` is launched. |
| `train-ray`        | ✅     | Ray-managed trainer. Launch TrainActor workers as Ray actors on a chosen train node; preferred because training GPUs are placed by Ray.                                   |
| `consume-ray`      | ✅     | Run the M2 fake trajectory consumer as a Ray actor on a chosen node. Useful for placement and SlimeRPC checks.                                                            |
| `nanorl-dashboard` | ✅     | Static HTML dashboard for `train --log-jsonl` output and optional rollout logs.                                                                                           |

Set log level with `NANORL_LOG_LEVEL` (default `INFO`).

______________________________________________________________________

## `nanorl rollout-only`

Stand up a `RolloutEngine` (NanoDeploy `LLM` + verifier) and run one or more rollout rounds. Trajectories are scored, optionally appended to a JSONL for offline inspection, and optionally published over SlimeRPC for a downstream consumer.

### Required

| Flag             | Type  | Notes                                                                                |
| ---------------- | ----- | ------------------------------------------------------------------------------------ |
| `--cfg PATH`     | yaml  | Loads into `nanorl.config.NanoRLCfg`.                                                |
| `--prompts PATH` | jsonl | One `{"prompt", "reference", "group_id"}` per line. Bad rows skipped with a warning. |

### Generation

| Flag                                     | Default   | Notes                                                                                                       |
| ---------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------- |
| `--rounds N`                             | `1`       | Each round runs all prompts × `sampling.n` rollouts.                                                        |
| `--limit-prompts N`                      | `0` (off) | Take only the first N prompts.                                                                              |
| `--seed N`                               | none      | Seeds Python `random` and `numpy.random`. Doesn't seed CUDA samplers.                                       |
| `--top-p F`                              | from cfg  | Override `sampling.top_p`.                                                                                  |
| `--max-new-tokens N`                     | from cfg  | Override `sampling.max_new_tokens`.                                                                         |
| `--ship-logprobs` / `--no-ship-logprobs` | from cfg  | Override `sampling.ship_logprobs`. When enabled, rollout-time logprobs are sent to train as `old_logprobs`. |

### Output

| Flag                          | Default | Notes                                                                              |
| ----------------------------- | ------- | ---------------------------------------------------------------------------------- |
| `--save-jsonl PATH`           | none    | Append every `Trajectory` to disk.                                                 |
| `--print-traj-every N`        | `0`     | Every N rounds, log decoded sample trajectories for spot-checking.                 |
| `--print-traj-n N`            | `2`     | Number of trajectories to print per `--print-traj-every` tick.                     |
| `--eval-prompts PATH-or-NAME` | none    | Held-out prompt set; accepts a JSONL path or bundled names like `sample` / `aime`. |
| `--eval-every N`              | `20`    | Rollout rounds between held-out eval passes.                                       |
| `--eval-limit-prompts N`      | `0`     | Cap eval prompt count; `0` means all.                                              |
| `--eval-jsonl PATH`           | none    | Append per-eval summary rows (`round`, `mean_reward`, `pos_count`).                |
| `--no-rpc`                    | off     | Don't start SlimeRPC server.                                                       |
| `--dry-run`                   | off     | Validate + exit before launching NanoDeploy.                                       |

### SlimeRPC + weight-sync wiring

| Flag                      | Default               | Notes                                                                                                         |
| ------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--producer-alias S`      | from cfg              | This rollout's NanoCtrl alias.                                                                                |
| `--consumer-alias S`      | from cfg              | The downstream train actor's alias.                                                                           |
| `--serve-forever`         | off                   | Idle after `--rounds`, keep the SlimeRPC server up so a `nanorl train` driver can call `apply_weight_update`. |
| `--stop-after-buffered N` | `0` (off)             | Exit when the SlimeRPC queue holds ≥ N.                                                                       |
| `--redis-url URL`         | none                  | Optional async sink: stream each trajectory to a Redis stream (`XADD`).                                       |
| `--redis-key KEY`         | `nanorl:trajectories` | Redis stream key.                                                                                             |
| `--redis-maxlen N`        | `100000`              | Approximate stream length cap (`XADD MAXLEN ~`).                                                              |

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
| `--wandb-project S`  | none     | Enable W&B logging                   |
| `--wandb-run-name S` | none     | W&B run name                         |
| `--tb-dir PATH`      | none     | Enable TensorBoard event logging     |
| `--dry-run`          | off      | Build config, exit before TrainActor |

______________________________________________________________________

## `nanorl train`

Full M3 loop: train + periodic weight sync. This is the direct local/`torchrun`
path, not the Ray-managed train path. Supports both DDP single-rank and FSDP
multi-rank (set `cfg.train.fsdp = true` and launch with
`torchrun --nproc_per_node=N`).

| Flag                    | Default  | Notes                                                                         |
| ----------------------- | -------- | ----------------------------------------------------------------------------- |
| `--cfg PATH`            | required | YAML config                                                                   |
| `--steps N`             | `10`     | Number of GRPO steps                                                          |
| `--weight-sync-every N` | from cfg | Sync interval (overrides `cfg.weight_sync_every`; `0` disables)               |
| `--producer-alias S`    | from cfg | Rollout to pull trajectories from                                             |
| `--consumer-alias S`    | from cfg | This train's alias (used as alias prefix; per-rank suffixes added under FSDP) |
| `--master-port N`       | `29500`  | Single-rank only — torchrun sets it under FSDP                                |
| `--log-jsonl PATH`      | none     | Per-step `TrainStats` + `weight_sync` events                                  |
| `--wandb-project S`     | none     | Enable W&B logging                                                            |
| `--wandb-run-name S`    | none     | W&B run name                                                                  |
| `--tb-dir PATH`         | none     | Enable TensorBoard event logging                                              |
| `--save-dir PATH`       | none     | Write HF-format checkpoints under `step_XXXXXX/`                              |
| `--save-every N`        | `0`      | Save every N steps; `0` disables periodic saves                               |
| `--save-final`          | off      | Save one final checkpoint during shutdown                                     |
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

For day-to-day FSDP runs, prefer `train-ray` or
`bash scripts/m3_fsdp_smoke.sh`, which places the trainer through Ray.

______________________________________________________________________

## `nanorl train-ray`

Launch one TrainActor Ray worker per GPU on a selected train node, packed into one Ray placement group. This is the preferred M3 FSDP launch path: the command can be run from a driver node, while the trainer itself runs on `--train-ip`.

| Flag                    | Default         | Notes                                                              |
| ----------------------- | --------------- | ------------------------------------------------------------------ |
| `--cfg PATH`            | required        | YAML config                                                        |
| `--steps N`             | `10`            | Number of GRPO steps                                               |
| `--weight-sync-every N` | from cfg        | Sync interval (overrides `cfg.weight_sync_every`; `0` disables)    |
| `--producer-alias S`    | from cfg        | Rollout to pull trajectories from                                  |
| `--consumer-alias S`    | from cfg        | This train's alias                                                 |
| `--master-port N`       | `29500`         | `MASTER_PORT` used by Ray TrainActors                              |
| `--train-ip IP`         | `10.102.98.154` | Ray node address where TrainActors should be packed                |
| `--nproc N`             | cfg world size  | Number of Ray TrainActor workers / GPUs                            |
| `--ray-address ADDR`    | cfg Ray addr    | Ray address for the driver                                         |
| `--log-jsonl PATH`      | none            | Per-step `TrainStats` + `weight_sync` events                       |
| `--wandb-project S`     | none            | Enable W&B logging                                                 |
| `--wandb-run-name S`    | none            | W&B run name                                                       |
| `--tb-dir PATH`         | none            | Enable TensorBoard event logging                                   |
| `--save-dir PATH`       | none            | Write HF-format checkpoints under `step_XXXXXX/` on the train node |
| `--save-every N`        | `0`             | Save every N steps; `0` disables periodic saves                    |
| `--save-final`          | off             | Save one final checkpoint during shutdown                          |
| `--dry-run`             | off             | Validate placement/config and exit before launching TrainActors    |

Example:

```bash
python -m nanorl.cli train-ray \
  --cfg nanorl/configs/qwen3_4b_grpo_fsdp.yaml \
  --steps 500 --weight-sync-every 2 \
  --producer-alias rollout:run --consumer-alias train:run \
  --train-ip 10.102.98.166 --nproc 8 \
  --log-jsonl /tmp/nanorl_run/train.jsonl \
  --tb-dir /tmp/nanorl_run/tb \
  --save-dir /tmp/nanorl_ckpts/run \
  --save-every 50 --save-final
```

In Ray mode checkpoint paths are local to the train node where the worker runs.

______________________________________________________________________

## `nanorl consume-ray`

Run the M2 fake consumer as a Ray actor on a selected node. It is mainly for confirming Ray placement and SlimeRPC connectivity before starting a real trainer.

| Flag                 | Default         | Notes                                        |
| -------------------- | --------------- | -------------------------------------------- |
| `--cfg PATH`         | required        | YAML config                                  |
| `--producer-alias S` | `rollout:0`     | Rollout service to pull from                 |
| `--consumer-alias S` | `train:0`       | Consumer alias                               |
| `--batches N`        | `3`             | Number of batches to pull                    |
| `--batch-size N`     | `8`             | Requested trajectories per batch             |
| `--consumer-ip IP`   | `10.102.98.154` | Ray node where the consumer actor should run |
| `--ray-address ADDR` | cfg Ray addr    | Ray address for the driver                   |

______________________________________________________________________

## Logging

`NANORL_LOG_LEVEL=DEBUG` to see SlimeRPC / RDMA QP setup chatter. NanoDeploy and dlslime have their own loggers not controlled by this env var.

______________________________________________________________________

## Dashboard

Generate a static readiness dashboard from a smoke run:

```bash
nanorl-dashboard \
  --train-jsonl /tmp/nanorl_smoke/m3_train.jsonl \
  --producer-log /tmp/nanorl_smoke/m3_producer.log \
  --expect-sync \
  --out /tmp/nanorl_smoke/dashboard.html
```

The HTML is self-contained and reports finite loss, loss trend, reward
variance, weight-sync count, loaded tensor counts, skipped tensors, and
step/sync timings.

## Exit codes

| Code    | Meaning                                                           |
| ------- | ----------------------------------------------------------------- |
| `0`     | Success.                                                          |
| `1`     | Subcommand failed.                                                |
| `2`     | No valid prompts in the JSONL (rollout-only).                     |
| `3`     | NaN loss detected (train / train-only).                           |
| nonzero | Uncaught exception. SIGINT triggers a clean shutdown with code 0. |
