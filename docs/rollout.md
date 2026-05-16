# Rollout (M2)

The `rollout-only` subcommand stands up a NanoInfra inference engine, generates `n` rollouts per prompt, scores each with a verifier, and either prints results (`--no-rpc`) or publishes them to a SlimeRPC consumer.

Every CLI flag is enumerated in `docs/cli.md`. This page focuses on the *flow*: what the engine does each round, what the wire format looks like, and what to expect when it works.

## Files

| File                                                 | Role                                                                                              |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `nanorl/actors/rollout.py`                           | `RolloutEngine`: wraps NanoInfra `LLM`, drives generation, calls verifier, populates `Trajectory` |
| `nanorl/cli.py:_rollout_only`                        | CLI handler                                                                                       |
| `nanorl/data/trajectory_buffer.py:TrajectoryService` | SlimeRPC service the consumer pulls from                                                          |
| `nanorl/data/data_loader.py:TrajectoryClient`        | Pull side; used by `fake_train_consumer.py`                                                       |
| `nanorl/rl/reward.py:MathVerifier`                   | Default verifier (extracts `\boxed{...}` or trailing number)                                      |
| `nanorl/configs/qwen3_4b_grpo.yaml`                  | Reference YAML for Qwen3-4B-Instruct                                                              |
| `nanorl/configs/sample_prompts.jsonl`                | Example JSONL of arithmetic prompts                                                               |
| `scripts/m2_smoke.sh`                                | Single-command end-to-end smoke                                                                   |
| `scripts/fake_train_consumer.py`                     | Standalone SlimeRPC consumer for ad-hoc testing                                                   |

## Prompts JSONL

One JSON object per line. Required fields: `prompt`, `reference`, `group_id`. The chat template (if the tokenizer has one) is applied automatically — pass the *user message*, not the templated text.

```json
{"prompt": "What is 7 * 8? Answer in \\boxed{...}.", "reference": "56", "group_id": 1}
```

`group_id` is the GRPO group key. All `n` rollouts of the same prompt share the same `group_id`, which is what `nanorl/rl/advantages.py:group_relative_advantages` will use on the train side. Bad rows (malformed JSON, missing `prompt`, non-int `group_id`) are skipped with a warning so a single typo doesn't tank a long file.

## What a round does

For each round (controlled by `--rounds`):

1. Tokenize every prompt (apply chat template if the tokenizer has one)
2. Submit `len(prompts) × sampling.n` `nanodeploy.Sequence` requests
3. Drive `LLM.generate()` until all sequences finish
4. Decode each completion, score with the verifier, build a `Trajectory`
5. (unless `--no-rpc`) push the batch into `TrajectoryService.publish` for downstream consumers
6. (if `--save-jsonl PATH`) append every trajectory to disk as one JSON object per line

Throughput, queueing, and per-sequence timing are logged by NanoInfra at INFO. Per-group reward stats are logged by NanoRL.

## Configuration knobs that matter

The YAML's `infer:` section maps 1:1 to `nanodeploy.config.Config`:

| Field                           | What it does                                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `attention_tp`, `ffn_tp`        | Tensor parallelism                                                                                                    |
| `attention_dp`, `ffn_dp`        | DP for attention/FFN                                                                                                  |
| `ffn_ep`                        | Expert parallelism (set ≥ 2 for MoE)                                                                                  |
| `gpu_memory_utilization`        | Fraction of GPU memory the engine grabs. Lower (e.g. `0.12`) when sharing a host.                                     |
| `mode`                          | `prefill` / `decode` / `hybrid`. M2 uses `hybrid`.                                                                    |
| `executor_backend`              | `ray` (Ray collectives) or `dlslime` (RDMA collectives). Use `dlslime` to share the fabric with the trajectory plane. |
| `ray_address`, `master_address` | The shared Ray cluster + the node to schedule rollout workers on.                                                     |
| `nanoctrl_address`              | Where to register PeerAgents (must match the SlimeRPC plane's URL).                                                   |

Sampling knobs in `sampling:`:

| Field            | Notes                                        |
| ---------------- | -------------------------------------------- |
| `n`              | Rollouts per prompt; equals `rl.group_size`. |
| `temperature`    | `0` for deterministic; the smoke uses `0.7`. |
| `top_p`          | Override per-run with `--top-p`.             |
| `max_new_tokens` | Override per-run with `--max-new-tokens`.    |

## Adding a new verifier

A verifier is anything matching the `Verifier` protocol:

```python
class Verifier(Protocol):
    def score(self, response: str, reference: str) -> float: ...
```

To wire one in: import it from `nanorl/rl/reward.py`, instantiate it in `_rollout_only` (`nanorl/cli.py`), and pass it to `RolloutEngine(...)`. There is no `--verifier` CLI flag yet (tracked as a productionization gap); for now it's a one-line code edit.

For a heavier verifier (LLM-as-judge, code execution sandbox), keep `score` synchronous and parallelize across trajectories at the call site — the engine's `score` method is already a separate stage from `generate`.

## End-to-end smoke (`scripts/m2_smoke.sh`)

```bash
bash scripts/m2_smoke.sh
```

This script:

1. Generates timestamped aliases (`rollout:<epoch>` / `train:<epoch>`) so a `kill -9`'d previous run doesn't collide with `409 Conflict` on alias registration.
2. Starts the producer with `--rounds 3 --serve-forever`.
3. Polls the producer log until the first round completes (~90s on Qwen3-4B; most of that is NanoInfra's cudagraph capture).
4. Sleeps 3s for the producer's `serve_settle_s` to elapse.
5. Runs the consumer: 3 batches × 4 trajectories.
6. Sends SIGINT to the producer, falls back to SIGKILL after 5s.

Successful output ends with three `batch=` lines:

```
batch=0 B=4 T=37 mean_reward=1.000 wait=0.0s sample='user\nWhat is 1 + 1?...assistant\n1 + 1 = 2\n\n\boxed{2}'
batch=1 B=4 T=46 mean_reward=1.000 wait=0.0s sample='user\nWhat is 7 * 8?...assistant\n7 * 8 = 56\n\n\boxed{56}'
batch=2 B=4 T=87 mean_reward=1.000 wait=0.0s sample='user\nWhat is 144 / 12?...'
```

That's the only proof we currently have that the cross-process flow works — the unit tests don't simulate the second process.

## Common flag patterns

**Local-only smoke (no consumer needed):**

```bash
python -m nanorl.cli rollout-only --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl --rounds 1 --no-rpc
```

**Save trajectories for offline analysis:**

```bash
python -m nanorl.cli rollout-only --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl --rounds 5 --no-rpc \
  --save-jsonl /tmp/nanorl_trajs.jsonl
```

**CI / pre-flight (no GPU, no NanoInfra startup):**

```bash
python -m nanorl.cli rollout-only --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl --dry-run
```

**Stop after the consumer has drained N trajectories:**

```bash
python -m nanorl.cli rollout-only --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl --rounds 100 \
  --stop-after-buffered 0    # actually never — see note below
```

(`--stop-after-buffered N` exits when the producer's outstanding queue is ≥ N. To stop on consumer drain rather than producer buffering, you'd extend `RolloutEngine` with a callback — not built yet.)

## SlimeRPC pitfalls (inline)

When two processes try to talk over SlimeRPC, three subtle ordering rules have bitten us. NanoRL's code already handles them; this is for anyone writing new SlimeRPC services.

1. **`serve()` must run after `connect_to(...).wait()` returns.** Otherwise dlslime raises `ValueError("requires a connected endpoint")`. `nanorl/data/trajectory_buffer.py:run_rpc_server` accepts the `Connection` object and waits inside its serve thread.
2. **Sleep ~200ms between connection-up and the first remote send.** Otherwise the first WR can hit `IBV_WC_RETRY_EXC_ERR` (Vendor Err 129). NanoRL adds `serve_settle_s=0.2` (producer) and `initial_settle_s=0.2` (consumer).
3. **Both sides must register with the SAME NanoCtrl URL string.** Mixing `127.0.0.1:3000` and `10.102.97.179:3000` causes mailbox-MR lookups to fail silently. Use the LAN IP everywhere.

Bonus: if a process is `kill -9`'d, its alias stays registered with NanoCtrl until the heartbeat TTL expires (~30s). Use unique aliases per run, or wait, or POST to NanoCtrl `/cleanup`. See `docs/troubleshooting.md`.
