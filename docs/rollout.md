# Rollout (M2)

The `rollout-only` subcommand stands up a NanoInfra inference engine, generates `n` rollouts per prompt, scores each with a verifier, and either prints results (`--no-rpc`) or publishes them to a SlimeRPC consumer.

In M3 the rollout-only process *also* exposes the `apply_weight_update` RPC that the train side calls — same SlimeRPC service, additional method.

## Files

| File                                                 | Role                                                                      |
| ---------------------------------------------------- | ------------------------------------------------------------------------- |
| `nanorl/actors/rollout.py`                           | `RolloutEngine`: wraps NanoInfra `LLM`, drives generation, calls verifier |
| `nanorl/cli.py:_rollout_only`                        | CLI handler                                                               |
| `nanorl/data/trajectory_buffer.py:TrajectoryService` | SlimeRPC service: trajectory pull + M3 weight-update RPC                  |
| `nanorl/data/data_loader.py:TrajectoryClient`        | Pull side; used by `fake_train_consumer.py` and `TrainActor`              |
| `nanorl/rl/reward.py:MathVerifier`                   | Default verifier (`\boxed{...}` or trailing number)                       |
| `nanorl/configs/qwen3_4b_grpo.yaml`                  | DDP variant                                                               |
| `nanorl/configs/qwen3_4b_grpo_fsdp.yaml`             | FSDP variant (rollout config identical)                                   |
| `nanorl/configs/sample_prompts.jsonl`                | Bundled arithmetic prompts                                                |
| `scripts/m2_smoke.sh`                                | rollout-only + fake_train_consumer                                        |
| `scripts/fake_train_consumer.py`                     | Standalone SlimeRPC consumer                                              |

## Prompts JSONL

One JSON object per line. Required: `prompt`, `reference`, `group_id`. Chat template is applied automatically — pass user message text, not templated text. Bad rows are skipped with a warning.

```json
{"prompt": "What is 7 * 8? Answer in \\boxed{...}.", "reference": "56", "group_id": 1}
```

`group_id` is the GRPO group key. All `n` rollouts of the same prompt share `group_id`; `nanorl/rl/advantages.py:group_relative_advantages` standardizes rewards within each group.

## What a round does

1. Tokenize every prompt (apply chat template if available)
2. Submit `len(prompts) × sampling.n` `nanodeploy.Sequence` requests
3. Drive `LLM.generate()` until all sequences finish
4. Decode each completion, score with verifier, build `Trajectory`
5. (unless `--no-rpc`) push into `TrajectoryService.publish` for downstream consumers
6. (if `--save-jsonl PATH`) append each trajectory to disk

Throughput, queueing, per-sequence timing logged by NanoInfra at INFO. Per-group reward stats logged by NanoRL.

## Configuration knobs that matter

The YAML's `infer:` section maps 1:1 to `nanodeploy.config.Config`:

| Field                              | What it does                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| `attention_tp`, `ffn_tp`           | Tensor parallelism (4 in the default Qwen3-4B config)                                      |
| `attention_dp`, `ffn_dp`, `ffn_ep` | DP/EP for attention/FFN; set EP for MoE                                                    |
| `gpu_memory_utilization`           | GPU memory fraction. Use ~0.12 when sharing a host.                                        |
| `mode`                             | `prefill` / `decode` / `hybrid`                                                            |
| `executor_backend`                 | `ray` (Ray collectives) or `dlslime` (RDMA collectives — required for M3 weight-pull path) |
| `ray_address`, `master_address`    | Shared Ray cluster + node to schedule rollout workers on                                   |
| `nanoctrl_address`                 | Where to register PeerAgents (must match SlimeRPC plane URL)                               |

## Smoke

```bash
bash scripts/m2_smoke.sh
```

Three batches × four trajectories pulled over RDMA, math verifier scores them. Final lines:

```
batch=0 B=4 T=37 mean_reward=1.000 sample='user\nWhat is 1 + 1?...assistant\n1 + 1 = 2\n\n\boxed{2}'
batch=1 B=4 T=46 mean_reward=1.000 sample='...What is 7 * 8?...\n7 * 8 = 56\n\n\boxed{56}'
batch=2 B=4 T=87 mean_reward=1.000 sample='...144 \div 12...'
```

The four bundled prompts saturate (mean_reward=1.000) — this is expected; harder prompts are the path to seeing reward variance.

## Adding a new verifier

A verifier is anything matching the `Verifier` protocol:

```python
class Verifier(Protocol):
    def score(self, response: str, reference: str) -> float: ...
```

Wire it in: import from `nanorl/rl/reward.py`, instantiate in `_rollout_only` (`nanorl/cli.py`), pass to `RolloutEngine(...)`. There is no `--verifier` CLI flag yet (productionization gap).

## SlimeRPC pitfalls

When two processes talk SlimeRPC, three subtle ordering rules have bitten us. NanoRL handles them; this is for anyone writing new SlimeRPC services.

1. **`serve()` must run after `connect_to(...).wait()` returns.** Otherwise dlslime raises `ValueError("requires a connected endpoint")`. `nanorl/data/trajectory_buffer.py:run_rpc_server` waits inside its serve thread.
2. **Sleep ~200 ms between connection-up and the first remote send.** Otherwise `IBV_WC_RETRY_EXC_ERR` (Vendor Err 129). NanoRL adds `serve_settle_s=0.2` and `initial_settle_s=0.2`.
3. **Both sides must register with the same NanoCtrl URL string.** Mixing `127.0.0.1:3000` and `10.102.97.179:3000` causes mailbox-MR lookups to fail silently. Use the LAN IP everywhere.

If a process is `kill -9`'d, its alias stays in NanoCtrl until the heartbeat TTL (~30 s). Use unique aliases per run (timestamps), wait, or POST `/cleanup`.
