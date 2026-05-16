# NanoRL

Ray-orchestrated reinforcement learning for large language models. **GRPO** with [**megatron-core**](https://github.com/NVIDIA/Megatron-LM/) training (DDP single-rank or FSDP/ZeRO-3 multi-rank), [**NanoDeploy**](https://github.com/DeepLink-org/NanoDeploy) rollouts, and [**DLSlime**](https://github.com/DeepLink-org/NanoDeploy) moving both trajectories and weight tensors over RDMA.

## Status

| Milestone                              | What it proves                                                                           | State                              |
| -------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------- |
| **M1** — TrainActor + dataloader       | single-rank megatron-core pulls trajectories over SlimeRPC, runs GRPO step               | ✅ `bash scripts/m1_smoke.sh`      |
| **M2** — RolloutActor + dataloader     | NanoInfra serves rollouts, math verifier scores, samples shipped over RDMA               | ✅ `bash scripts/m2_smoke.sh`      |
| **M3** — Train↔rollout weight sync     | DDP train side gathers, ships 8 GB Qwen3-4B → 4 NanoInfra workers via parallel RDMA pull | ✅ `bash scripts/m3_smoke.sh`      |
| **M3+FSDP** — multi-rank ZeRO-3 + sync | 2-GPU FSDP train, uneven-DTensor gather, then weight sync                                | ✅ `bash scripts/m3_fsdp_smoke.sh` |

The full GRPO loop runs end-to-end on Qwen3-4B-Instruct.

## Quick start

Pre-reqs (already running on this cluster — see `docs/install.md` if any are missing):

- NanoCtrl on `http://10.102.97.179:3000` + Redis on `127.0.0.1:6379`
- Ray cluster reachable at `10.102.97.179:7078`
- RDMA HCAs visible under `/sys/class/infiniband`
- Free GPUs on the configured `master_address` host (default `10.102.97.183` for rollout, local for train)

```bash
pip install -e .
```

**Local rollout smoke** (no SlimeRPC, prints rewards):

```bash
python -m nanorl.cli rollout-only --cfg nanorl/configs/qwen3_4b_grpo.yaml \
  --prompts nanorl/configs/sample_prompts.jsonl --rounds 1 --no-rpc
```

**Full M3 loop** (rollout on `.183`, single-rank DDP train on `.179:GPU-7`, 5 GRPO steps with 2 weight syncs):

```bash
bash scripts/m3_smoke.sh
```

**Multi-rank FSDP** (rollout on `.183`, 2-rank ZeRO-3 train on `.179:GPU-6,7`):

```bash
bash scripts/m3_fsdp_smoke.sh
```

## Operational numbers

Qwen3-4B-Instruct, 1 train rank (DDP) + 4 NanoInfra workers (TP=4 attn, ffn_tp=4):

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
  cli.py                    rollout-only ✅, train-only ✅, train ✅
  config.py                 pydantic schemas; loaded from YAML
  actors/
    train.py                TrainActor: megatron-core (DDP or FSDP), GRPO step, weight gather
    rollout.py              RolloutEngine: NanoInfra LLM + verifier + publisher
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
  train_loop.py             Top-level driver for `nanorl train`
  configs/
    qwen3_4b_grpo.yaml      DDP single-rank baseline
    qwen3_4b_grpo_fsdp.yaml ZeRO-3 multi-rank variant

scripts/
  m1_smoke.sh, m2_smoke.sh, m3_smoke.sh, m3_fsdp_smoke.sh   end-to-end smokes
  fake_train_consumer.py    pulls from a running rollout-only over SlimeRPC
  sanity_apply_weight_update.py  one-shot: NanoInfra patch in/out check
  sanity_qwen3_forward.py   HF↔Megatron logit cross-check (Δ logprob ≈ 4e-4)
  diag_train_vs_ref.py      reproduces the kl-kernel-parity issue (kl_beta=0 cause)
  diag_fsdp_full_tensor.py  reproduces the per-rank uneven-DTensor shape mismatch

NanoInfra patches (in /mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoDeploy):
  nanodeploy/worker/weight_update.py         apply_named_tensors_in_place helper
  nanodeploy/worker/pull_weights.py          worker-direct RDMA pull (the 13× speedup)
  nanodeploy/engine/weight_sync.py           engine fan-out wrapper
  + thin delegating methods on ModelRunner and LLMEngine
```

## Documentation

| Doc                       | Read when                                                       |
| ------------------------- | --------------------------------------------------------------- |
| `docs/install.md`         | Setting up a new host or debugging missing pre-reqs             |
| `docs/architecture.md`    | How Ray, NanoInfra, megatron-core, DLSlime fit together         |
| `docs/cli.md`             | Every CLI flag with examples                                    |
| `docs/rollout.md`         | M2 walkthrough — config, JSONL format, smoke output             |
| `docs/training.md`        | M1/M3 walkthrough — DDP and FSDP recipes, weight sync internals |
| `docs/data_plane.md`      | SlimeRPC trajectory contract, raw-RDMA weight transport         |
| `docs/troubleshooting.md` | Failures we have actually hit and how we fixed them             |

## Known limitations

- **`kl_beta = 0` by default.** Reference KL works mathematically but PyTorch's gradient-mode SDPA picks different attention kernels than no_grad mode, drifting per-token logprobs by ~5 in BF16 on Qwen3. KL term blows up. Pin a deterministic SDPA backend to re-enable. Reproducible via `scripts/diag_train_vs_ref.py`.
- **TP > 1 / PP > 1 train side not supported yet** (only TP=1 PP=1 EP=1; FSDP at world_size > 1 is the multi-rank story today).
- **MoE Qwen3.5-35B-A3B not wired** — the gather walk and shapes assume dense Qwen3.
- **Single math verifier** — no `--verifier` flag yet.
- **Full GRPO group + reward variance not yet exercised** on the bundled prompts (they're trivial arithmetic, all rollouts get reward 1.0 → advantage 0 → loss 0). The pipeline is correct; harder prompts are the path to seeing actual learning.
