# CLI reference

Entrypoint:

```bash
python -m nanorl.cli {rollout-only,train-only,train} ...
```

Or, after `pip install -e .`, the same as `nanorl {...}`.

| Subcommand     | Status         | Purpose                                                                                      |
| -------------- | -------------- | -------------------------------------------------------------------------------------------- |
| `rollout-only` | ✅ shipping    | Generate trajectories with NanoInfra, score with verifier, optionally publish over SlimeRPC. |
| `train-only`   | 🚧 placeholder | Drive a megatron-core TrainActor against a SlimeRPC trajectory source. Lands with M1.        |
| `train`        | 🚧 placeholder | Full GRPO loop with weight sync. Lands with M3.                                              |

Set log level with `NANORL_LOG_LEVEL` (default `INFO`).

______________________________________________________________________

## `nanorl rollout-only`

Stand up a `RolloutEngine` (NanoInfra `LLM` + verifier) and run one or more rollout rounds. Trajectories are scored, optionally appended to a JSONL for offline inspection, and optionally published over SlimeRPC for a downstream consumer.

### Required

| Flag             | Type  | Notes                                                                                    |
| ---------------- | ----- | ---------------------------------------------------------------------------------------- |
| `--cfg PATH`     | yaml  | Loads into `nanorl.config.NanoRLCfg`. See `nanorl/configs/qwen3_4b_grpo.yaml`.           |
| `--prompts PATH` | jsonl | One `{"prompt", "reference", "group_id"}` per line. Bad rows are skipped with a warning. |

### Generation

| Flag                 | Default   | Notes                                                                              |
| -------------------- | --------- | ---------------------------------------------------------------------------------- |
| `--rounds N`         | `1`       | Number of rollout rounds; each round runs all prompts × `sampling.n` rollouts.     |
| `--limit-prompts N`  | `0` (off) | Take only the first N prompts. Useful for fast smoke runs.                         |
| `--seed N`           | none      | Seeds Python `random` and `numpy.random`. Does not seed NanoInfra's CUDA samplers. |
| `--top-p F`          | from cfg  | Override `sampling.top_p`.                                                         |
| `--max-new-tokens N` | from cfg  | Override `sampling.max_new_tokens`.                                                |

### Output

| Flag                | Default | Notes                                                                                                   |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `--save-jsonl PATH` | none    | Append every `Trajectory` (one per line) to this file as JSON. Created if missing; parent dirs created. |
| `--no-rpc`          | off     | Do not start a SlimeRPC server. Useful for local smoke or for jobs that only want the JSONL output.     |
| `--dry-run`         | off     | Load + validate config and prompts, then exit before launching NanoInfra. Useful for CI / pre-flight.   |

### SlimeRPC

| Flag                      | Default   | Notes                                                                                             |
| ------------------------- | --------- | ------------------------------------------------------------------------------------------------- |
| `--producer-alias S`      | from cfg  | This rollout's NanoCtrl alias.                                                                    |
| `--consumer-alias S`      | from cfg  | The downstream train actor's alias to connect to.                                                 |
| `--serve-forever`         | off       | After `--rounds` complete, idle and keep the SlimeRPC server up so consumers can finish draining. |
| `--stop-after-buffered N` | `0` (off) | Stop the round loop early once the SlimeRPC queue holds at least N trajectories.                  |

### Examples

**Local smoke (no SlimeRPC, no consumer needed):**

```bash
python -m nanorl.cli rollout-only \
  --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl \
  --rounds 1 --no-rpc
```

**Two-process integration:**

```bash
# producer
python -m nanorl.cli rollout-only \
  --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl \
  --rounds 3 --serve-forever \
  --producer-alias rollout:42 --consumer-alias train:42

# consumer (separate shell)
python scripts/fake_train_consumer.py \
  --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --producer-alias rollout:42 --consumer-alias train:42 \
  --batches 3 --batch-size 4
```

Or in one command via `bash scripts/m2_smoke.sh`, which uses timestamped aliases to avoid collisions with leftover NanoCtrl state.

**Save trajectories for offline analysis:**

```bash
python -m nanorl.cli rollout-only \
  --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl \
  --rounds 5 --no-rpc \
  --save-jsonl /tmp/nanorl_trajs.jsonl
```

Each JSONL line:

```json
{"prompt_ids": [...], "response_ids": [...], "reward": 1.0, "group_id": 0, "eos": true, "meta": {"reference": "2"}}
```

**CI / pre-flight only (no GPUs needed):**

```bash
python -m nanorl.cli rollout-only \
  --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl \
  --dry-run
```

______________________________________________________________________

## Logging

`NANORL_LOG_LEVEL=DEBUG` to see SlimeRPC / RDMA QP setup chatter. Note that NanoInfra and dlslime have their own loggers that don't honor this env var; their levels are controlled by their own configuration.

## Exit codes

| Code    | Meaning                                                                          |
| ------- | -------------------------------------------------------------------------------- |
| `0`     | Success.                                                                         |
| `1`     | Subcommand placeholder (e.g. `train-only` before M1 lands).                      |
| `2`     | No valid prompts in the JSONL (after warnings about bad rows).                   |
| nonzero | Uncaught exception; see traceback. SIGINT triggers a clean shutdown with code 0. |
