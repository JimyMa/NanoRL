# NanoRL

Ray-orchestrated reinforcement learning for large language models. **Off-policy GRPO** with [**megatron-core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) training (DDP single-rank or FSDP/ZeRO-3 multi-rank), [**NanoDeploy**](https://github.com/DeepLink-org/NanoDeploy) rollouts, and [**DLSlime**](https://github.com/DeepLink-org/NanoDeploy) moving trajectories, rollout-time logprobs, and weight tensors over RDMA.

## Status

| Milestone                              | What it proves                                                                                            | State                              |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **M1** — TrainActor + dataloader       | Ray-managed megatron-core TrainActor pulls trajectories over SlimeRPC and runs GRPO                       | ✅ `bash scripts/m1_smoke.sh`      |
| **M2** — RolloutActor + dataloader     | NanoDeploy serves rollouts, math verifier scores, samples + optional logprobs ship over RDMA              | ✅ `bash scripts/m2_smoke.sh`      |
| **M3** — Train↔rollout weight sync     | DDP train side gathers, ships 8 GB Qwen3-4B → NanoDeploy workers via parallel RDMA pull                   | ✅ `bash scripts/m3_smoke.sh`      |
| **M3+FSDP** — multi-rank ZeRO-3 + sync | Ray-managed multi-rank FSDP train on a selected node, uneven-DTensor gather, weight sync, checkpoint save | ✅ `bash scripts/m3_fsdp_smoke.sh` |

The full off-policy GRPO loop runs end-to-end on Qwen3-4B. Rollout-side logprobs can be used as `old_logprobs` for the importance ratio, so ratios and clipping reflect real policy drift between weight syncs.

## Quick start

Pre-reqs (already running on this cluster — see `docs/install.md` if any are missing):

- NanoCtrl on `http://10.102.97.179:3000` + Redis on `127.0.0.1:6379`
- Ray cluster reachable at `10.102.97.179:7078`
- RDMA HCAs visible under `/sys/class/infiniband`
- Free GPUs on the configured `master_address` host for rollout and on `TRAIN_IP` for train. The smoke scripts launch both sides under Ray; the shell host is only the driver.

```bash
pip install -e .
```

**Local rollout smoke** (no SlimeRPC, prints rewards):

```bash
python -m nanorl.cli rollout-only --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl --rounds 1 --no-rpc
```

**Full M3 loop** (rollout on `.183`, Ray TrainActor on the configured train node, 5 GRPO steps with 2 weight syncs):

```bash
bash scripts/m3_smoke.sh
```

**Multi-rank FSDP** (rollout on `.183`, ZeRO-3 train packed by Ray on `TRAIN_IP`):

```bash
TRAIN_IP=10.102.98.166 NPROC=8 bash scripts/m3_fsdp_smoke.sh
```

Useful smoke-script overrides:

```bash
PROMPTS=/tmp/train.jsonl EVAL_PROMPTS=/tmp/eval.jsonl \
STEPS=500 EVAL_EVERY=25 SYNC_EVERY=2 \
SAVE_DIR=/tmp/nanorl_ckpts/run SAVE_EVERY=50 SAVE_FINAL=1 \
TRAIN_IP=10.102.98.166 NPROC=8 \
bash scripts/m3_fsdp_smoke.sh
```

**Dashboard** for a smoke run:

```bash
nanorl-dashboard \
  --train-jsonl /tmp/nanorl_smoke/m3_train.jsonl \
  --producer-log /tmp/nanorl_smoke/m3_producer.log \
  --expect-sync \
  --out /tmp/nanorl_smoke/dashboard.html
```

The generated HTML is static and highlights finite loss, loss trend, reward
variance, weight-sync health, loaded tensor counts, and timing.

## Rollout-side logprobs

`sampling.ship_logprobs: true` asks NanoDeploy to return per-response-token logprobs for sampled tokens. NanoRL stores them on each `Trajectory` and the trainer consumes them as `old_logprobs`, so:

- `old_lp=True` in train logs means the inference-side logprobs arrived.
- `logprob_to_old=mean/max` tracks policy drift from the rollout policy to the current train policy.
- `kl_to_old` is the off-policy distance actually used for monitoring. The older `kl`/`kl_mean` field is reference-model KL and remains a diagnostic unless `rl.kl_beta > 0`.

Use `--no-ship-logprobs` on `rollout-only` to fall back to train-side `current_logprobs.detach()`, which makes ratios equal 1 by construction and is useful for parity debugging.

## Checkpoints

`nanorl train` and `nanorl train-ray` can save HF-format checkpoints:

```bash
python -m nanorl.cli train-ray ... \
  --save-dir /tmp/nanorl_ckpts/my_run \
  --save-every 50 \
  --save-final
```

Each save writes `step_XXXXXX/model.safetensors`, tokenizer/config files copied from the source HF directory, and `nanorl_checkpoint.json`. In Ray mode the path is local to the train node where the Ray TrainActor runs.

## Operational numbers

Qwen3-4B-Instruct, 1 train rank (DDP) + 4 NanoDeploy workers (TP=4 attn, ffn_tp=4):

|                                           | DDP train                  | 2-rank FSDP train               |
| ----------------------------------------- | -------------------------- | ------------------------------- |
| Train step                                | 0.30 s                     | 0.30 s                          |
| Weight gather (collective)                | 0 s (params already full)  | 6 s (uneven-DTensor all-gather) |
| Weight RDMA pull (rollout side)           | 1.5 s (4 workers parallel) | 1.5 s                           |
| Apply (in-place copy, no graph recapture) | 0.85 s                     | 1.5 s                           |
| **Total per sync**                        | **~5 s**                   | **~9 s**                        |
| Manifest size                             | 398 HF tensors             | 398 HF tensors                  |

8 GB transferred per sync; previous Ray-RPC fan-out was ~65 s — direct worker-side RDMA cut it 13×.

## Tests

```bash
pytest tests/                        # 18 unit + 1 RDMA loopback (skipped without HCA)
pytest tests/test_grpo_loss.py       # vendored GRPO math byte-equal to upstream
pytest tests/test_slime_rpc_loopback.py  # real RDMA → NanoCtrl → Redis trajectory roundtrip
pytest tests/test_megatron_to_hf.py  # HF↔Megatron name-map round-trip
pytest tests/test_weight_manifest.py # 2-process RDMA weight transport
```

The four cross-process smokes (`m1`, `m2`, `m3`, `m3_fsdp`) are the integration coverage; pytest covers the math and the per-component contracts.

## Repository layout

```
nanorl/
  cli.py                    rollout-only ✅, train-only ✅, train ✅, train-ray ✅, consume-ray ✅
  config.py                 pydantic schemas; loaded from YAML
  actors/
    train.py                TrainActor: megatron-core (DDP or FSDP), GRPO step, weight gather, HF save
    rollout.py              RolloutEngine: NanoDeploy LLM + verifier + publisher + rollout logprobs
  data/
    sample.py               Trajectory / TrajectoryBatch
    trajectory_buffer.py    SlimeRPC TrajectoryService (producer + apply_weight_update RPC)
    data_loader.py          SlimeRPC TrajectoryClient (consumer, prefetch + backoff)
  weights/
    hf_to_megatron.py       HF Qwen3 → megatron-core GPTModel (QKV/SwiGLU fuse, qk-layernorm)
    megatron_to_hf.py       Inverse: walks GPTModel params, materializes DTensors via
                            uneven_dtensor_to_full_tensor for FSDP
    transport.py            DLSlime weight MR registration + RDMA pull
  rl/
    grpo_loss.py            Vendored byte-equal from megatron/rl/rl_utils.py:1854
    logprobs.py             Per-token logprobs without megatron.training globals
    advantages.py           Group-relative
    reward.py               Verifier protocol + math verifier
    reference_model.py      Frozen GPTModel for KL term (kl_beta=0 default — see kl note)
  configs/
    qwen3_4b_grpo.yaml      DDP single-rank baseline
    qwen3_4b_grpo_fsdp.yaml ZeRO-3 multi-rank variant
    datasets/
      weird_algebra_train192.jsonl, weird_algebra_test64.jsonl
                              fixed 192/64 algebra split used for held-out validation

scripts/
  m1_smoke.sh, m2_smoke.sh, m3_smoke.sh, m3_fsdp_smoke.sh   end-to-end smokes
  fake_train_consumer.py    pulls from a running rollout-only over SlimeRPC
  sanity_apply_weight_update.py  one-shot: NanoDeploy patch in/out check
  sanity_qwen3_forward.py   HF↔Megatron logit cross-check (Δ logprob ≈ 4e-4)
  diag_train_vs_ref.py      reproduces the kl-kernel-parity issue (kl_beta=0 cause)
  diag_fsdp_full_tensor.py  reproduces the per-rank uneven-DTensor shape mismatch

NanoDeploy patches (in /mnt/nvme1n1/ml_research/majinming/src/NanoDeploy/NanoDeploy):
  nanodeploy/worker/weight_update.py         apply_named_tensors_in_place helper
  nanodeploy/worker/pull_weights.py          worker-direct RDMA pull (the 13× speedup)
  nanodeploy/engine/weight_sync.py           engine fan-out wrapper
  + thin delegating methods on ModelRunner and LLMEngine
```

## Documentation

| Doc                                | Read when                                                                           |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| `docs/install.md`                  | Setting up a new host or debugging missing pre-reqs                                 |
| `docs/architecture.md`             | How Ray, NanoDeploy, megatron-core, DLSlime fit together                            |
| `docs/cli.md`                      | Every CLI flag with examples                                                        |
| `docs/rollout.md`                  | M2 walkthrough — config, JSONL format, smoke output                                 |
| `docs/training.md`                 | M1/M3 walkthrough — Ray TrainActors, DDP/FSDP recipes, weight sync, checkpoint save |
| `docs/data_plane.md`               | SlimeRPC trajectory contract, raw-RDMA weight transport                             |
| `docs/weird_algebra_validation.md` | Fixed generated algebra split and the verified Qwen3-4B FSDP improvement run        |
| `docs/troubleshooting.md`          | Failures we have actually hit and how we fixed them                                 |

## Known limitations

- **`kl_beta = 0` by default.** Reference KL works mathematically but PyTorch's gradient-mode SDPA picks different attention kernels than no_grad mode, drifting per-token logprobs by ~5 in BF16 on Qwen3. KL term blows up. Pin a deterministic SDPA backend to re-enable. Reproducible via `scripts/diag_train_vs_ref.py`.
- **TP > 1 / PP > 1 train side not supported yet** (only TP=1 PP=1 EP=1; FSDP at world_size > 1 is the multi-rank story today).
- **MoE Qwen3.5-35B-A3B not wired** — the gather walk and shapes assume dense Qwen3.
- **Single math verifier** — no `--verifier` flag yet.
- **Resume is not implemented yet.** Checkpoint save writes HF-format weights for evaluation/export, but there is no optimizer/RNG restore path yet.
- **Bundled prompts are smoke tests, not learning benchmarks.** Use a harder train/eval JSONL pair to see reward variance and real policy movement.
