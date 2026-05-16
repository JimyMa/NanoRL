# NanoRL

Ray-orchestrated reinforcement learning for large language models. The first pipeline is **GRPO**, with **megatron-core** doing the training, **NanoInfra** doing the rollouts, and **DLSlime** moving both trajectories and weight tensors.

This repo is in active development. The rollout pipeline (M2) is end-to-end working on real RDMA; the train pipeline (M1) is **not started** yet.

## Status

| Milestone                       | What it proves                                                                                                            | State                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| M2 — RolloutActor + dataloader  | NanoInfra serves rollouts, math verifier scores, samples published via SlimeRPC, train-side `TrajectoryClient` pulls them | ✅ done; reproduce with `bash scripts/m2_smoke.sh` |
| M1 — TrainActor + dataloader    | megatron-core forward/backward + GRPO loss against trajectories pulled over SlimeRPC                                      | not started                                        |
| M3 — Full loop with weight sync | DLSlime ships gathered weights from train to infer each step                                                              | not started                                        |

## Quick start

Pre-reqs (already running on this cluster — see `docs/install.md` if any are missing):

- NanoCtrl on `http://10.102.97.179:3000` + Redis on `127.0.0.1:6379`
- A Ray cluster reachable at `10.102.97.179:7078`
- RDMA HCAs visible under `/sys/class/infiniband`
- Free GPUs on the configured `master_address` host (default `10.102.97.183`)

Install editable:

```bash
pip install -e .
```

Local rollout smoke (no SlimeRPC, prints rewards):

```bash
python -m nanorl.cli rollout-only \
  --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl \
  --rounds 1 --no-rpc
```

Full producer ↔ consumer end-to-end:

```bash
bash scripts/m2_smoke.sh
```

## Tests

```bash
pytest tests/                        # 10 unit + 1 RDMA loopback (skipped if no HCA)
pytest tests/test_grpo_loss.py       # vendored GRPO math byte-equiv to upstream
pytest tests/test_slime_rpc_loopback.py  # real RDMA → NanoCtrl → Redis trajectory roundtrip
```

The cross-process trajectory flow is **not** in pytest — `bash scripts/m2_smoke.sh` is the only way we currently exercise it.

## Repository layout

```
nanorl/
  cli.py                 CLI entrypoint (rollout-only ✅, train-only / train 🚧)
  config.py              pydantic schemas; loaded from YAML
  actors/rollout.py      RolloutEngine: NanoInfra LLM + verifier + publisher
  actors/train.py        TrainActor: megatron-core (M1, not started)
  data/sample.py         Trajectory / TrajectoryBatch (with padding)
  data/trajectory_buffer.py  SlimeRPC TrajectoryService (producer)
  data/data_loader.py    SlimeRPC TrajectoryClient (consumer, with prefetch + backoff)
  rl/grpo_loss.py        Vendored byte-for-byte from megatron/rl/rl_utils.py
  rl/logprobs.py         Per-token logprobs without megatron.training globals
  rl/advantages.py       Group-relative advantages
  rl/reward.py           Verifier protocol + a tiny math verifier
  weights/               Train→infer weight gather/transport (M3, empty)
  configs/*.yaml         Reference configs (Qwen3-4B GRPO included)

scripts/
  fake_train_consumer.py    Pulls from a running rollout-only over SlimeRPC
  m2_smoke.sh               Single-command end-to-end smoke test

tests/                   Unit + integration tests
docs/                    Walkthroughs (see below)
```

## Documentation

| Doc                       | When to read                                                              |
| ------------------------- | ------------------------------------------------------------------------- |
| `docs/install.md`         | Setting up a new host or debugging missing pre-reqs                       |
| `docs/architecture.md`    | Understanding how Ray, NanoInfra, megatron-core, and DLSlime fit together |
| `docs/cli.md`             | Every CLI flag with examples                                              |
| `docs/rollout.md`         | M2 walkthrough — config, JSONL format, smoke output                       |
| `docs/data_plane.md`      | SlimeRPC trajectory contract                                              |
| `docs/troubleshooting.md` | Failures we have actually hit and how we fixed them                       |

The approved implementation plan lives at `~/.claude/plans/fixed-eager-stroustrup.md`.
