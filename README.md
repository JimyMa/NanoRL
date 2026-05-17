<h1 align="center">NanoRL</h1>

<h3 align="center">Ray-Orchestrated Off-Policy RL for Large Language Models</h3>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_zh.md">中文</a>
</p>

NanoRL is a training-inference co-designed RL framework for large language
models. It connects
[**megatron-core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)
training, [**NanoDeploy**](https://github.com/DeepLink-org/NanoDeploy)
inference, and [**DLSlime**](https://github.com/DeepLink-org/NanoDeploy)
transport into an off-policy GRPO loop with rollout-side logprobs, Ray-managed
train actors, FSDP/ZeRO-3, RDMA weight sync, held-out eval, and HF checkpoint
export.

## What You Can Run

| Path                     | What it shows                                                                                                             | Entry point                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Training example**     | A Ray-managed megatron-core TrainActor consuming trajectories and running GRPO steps                                      | `bash scripts/m1_smoke.sh`                                       |
| **Inference / rollout**  | NanoDeploy rollout workers generate math trajectories, verify rewards, and optionally ship sampled-token logprobs         | `python -m nanorl.cli rollout ...` or `bash scripts/m2_smoke.sh` |
| **RL training practice** | The complete off-policy GRPO loop: rollout logprobs as `old_logprobs`, FSDP train, weight sync, eval, and checkpoint save | `bash scripts/m3_fsdp_smoke.sh`                                  |

The third path is the main validated workflow. On the bundled
`nanorl_weird_algebra_v1` split, Qwen3-4B FSDP training improved held-out sampled
reward from `0.4023` to `0.5625` in a 500-step smoke run. See
`docs/weird_algebra_validation.md` for the dataset, command, and caveats.

## Quick start

Pre-reqs (already running on this cluster — see `docs/install.md` if any are missing):

- NanoCtrl on `http://10.102.97.179:3000` + Redis on `127.0.0.1:6379`
- Ray cluster reachable at `10.102.97.179:7078`
- RDMA HCAs visible under `/sys/class/infiniband`
- Free GPUs on the configured `master_address` host for rollout and on `TRAIN_IP` for train. The smoke scripts launch both sides under Ray; the shell host is only the driver.

```bash
pip install -e .
```

### 1. Training Example

Run the minimal training-side example when you want to check that the
megatron-core TrainActor, dataloader, and GRPO step are wired correctly:

```bash
NANORL_LOG_LEVEL=INFO STEPS=5 TRAIN_GPU=0 bash scripts/m1_smoke.sh
```

This is a developer smoke test for the train actor. It is useful before touching
distributed rollout or weight sync.

### 2. Inference / Rollout

Run rollout when you want to inspect generated trajectories and verifier
rewards without updating weights:

```bash
python -m nanorl.cli rollout --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl --rounds 1 --no-rpc
```

To exercise the distributed rollout service path, use:

```bash
bash scripts/m2_smoke.sh
```

Rollout-side logprobs are enabled by default in the GRPO configs. Use
`--no-ship-logprobs` for parity debugging when you intentionally want ratios to
be 1.

### 3. RL Training Practice

This is the recommended end-to-end path. It launches NanoDeploy rollout workers,
places Ray TrainActors on `TRAIN_IP`, trains Qwen3-4B with FSDP/ZeRO-3, syncs
weights back to inference, evaluates on the held-out split, and saves HF-format
checkpoints:

```bash
LOG_DIR=/tmp/nanorl_weird_algebra_m3_fsdp
mkdir -p "$LOG_DIR"

NANORL_LOG_LEVEL=INFO \
PROMPTS=nanorl/configs/datasets/weird_algebra_train192.jsonl \
EVAL_PROMPTS=nanorl/configs/datasets/weird_algebra_test64.jsonl \
LIMIT_PROMPTS=192 \
EVAL_LIMIT=64 \
EVAL_EVERY=25 \
STEPS=500 \
SYNC_EVERY=2 \
ROUNDS=1000 \
NPROC=8 \
TRAIN_IP=10.102.98.166 \
LOG_DIR="$LOG_DIR" \
SAVE_DIR="$LOG_DIR/checkpoints" \
SAVE_EVERY=100 \
SAVE_FINAL=1 \
bash scripts/m3_fsdp_smoke.sh
```

For a shorter integration check, lower `STEPS`:

```bash
STEPS=20 EVAL_EVERY=5 TRAIN_IP=10.102.98.166 NPROC=8 \
bash scripts/m3_fsdp_smoke.sh
```

TensorBoard for a run:

```bash
tensorboard --logdir "$LOG_DIR/m3_fsdp_tb" --port 6006 --bind_all
```

The smoke script writes TensorBoard events by default through `--tb-dir`. The
JSONL logs remain under `$LOG_DIR` for debugging or post-hoc dashboard
generation.

## Rollout-side logprobs

`sampling.ship_logprobs: true` asks NanoDeploy to return per-response-token logprobs for sampled tokens. NanoRL stores them on each `Trajectory` and the trainer consumes them as `old_logprobs`, so:

- `old_lp=True` in train logs means the inference-side logprobs arrived.
- `logprob_to_old=mean/max` tracks policy drift from the rollout policy to the current train policy.
- `kl_to_old` is the off-policy distance actually used for monitoring. The older `kl`/`kl_mean` field is reference-model KL and remains a diagnostic unless `rl.kl_beta > 0`.

Use `--no-ship-logprobs` on `rollout` to fall back to train-side `current_logprobs.detach()`, which makes ratios equal 1 by construction and is useful for parity debugging.

## Checkpoints

`nanorl train` can save HF-format checkpoints. The trainer is Ray-managed: the
driver can run on a different node, while Ray packs TrainActors onto
`--train-ip`.

```bash
python -m nanorl.cli train ... \
  --save-dir /tmp/nanorl_ckpts/my_run \
  --save-every 50 \
  --save-final
```

Each save writes `step_XXXXXX/model.safetensors`, tokenizer/config files copied from the source HF directory, and `nanorl_checkpoint.json`. In Ray mode the path is local to the train node where the Ray TrainActor runs.

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
  cli.py                    rollout ✅, train ✅, consume-ray ✅
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
  fake_train_consumer.py    pulls from a running rollout over SlimeRPC
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
