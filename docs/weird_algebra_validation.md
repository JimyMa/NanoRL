# Weird Algebra Validation Set

This repository includes a small generated algebra set used to validate the
end-to-end off-policy GRPO path on Qwen3-4B:

| Split    | Path                                                   | Count |
| -------- | ------------------------------------------------------ | ----: |
| Train    | `nanorl/configs/datasets/weird_algebra_train192.jsonl` |   192 |
| Test     | `nanorl/configs/datasets/weird_algebra_test64.jsonl`   |    64 |
| Combined | `nanorl/configs/datasets/weird_algebra_256.jsonl`      |   256 |

Each row follows the same JSONL schema as the other NanoRL math prompts:

```json
{
  "data_source": "nanorl_weird_algebra_v1",
  "prompt": [{"role": "user", "content": "..."}],
  "ability": "MATH",
  "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": "..."},
  "extra_info": {"category": "...", "split": "train|test"}
}
```

The set is intentionally modest. It is not meant to be a public benchmark; it
is a fixed generated smoke set that is useful for checking whether rollout,
verifier rewards, off-policy logprobs, FSDP train, weight sync, and checkpoint
saving work together without relying on a contaminated public math eval.

## Categories

The generated problems are balanced across eight algebra templates:

| Category          | Train | Test |
| ----------------- | ----: | ---: |
| `absolute_shift`  |    24 |    8 |
| `arith_sequence`  |    24 |    8 |
| `exponent_linear` |    24 |    8 |
| `fraction_linear` |    24 |    8 |
| `linear_nested`   |    24 |    8 |
| `poly_remainder`  |    24 |    8 |
| `quadratic_root`  |    24 |    8 |
| `system_combo`    |    24 |    8 |

## Reproduce The Smoke Run

The validated run used rollout-side logprobs as `old_logprobs`, no model
dropout, FSDP/ZeRO-3 train workers on `TRAIN_IP`, and held-out eval every 25
rollout rounds:

```bash
LOG_DIR=/tmp/nanorl_weird_algebra_m3_fsdp_s500_nodropout_save
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

## Observed Result

In the validated 500-step run, a standalone rollout comparison on the corrected
held-out labels improved from 122/256 sampled rollouts correct to 172/256
sampled rollouts correct:

| Model                | Prompts | Samples per prompt | Mean reward | Correct |
| -------------------- | ------: | -----------------: | ----------: | ------: |
| Qwen3-4B-Base        |      64 |                  4 |      0.4766 |     122 |
| NanoRL step-500 ckpt |      64 |                  4 |      0.6719 |     172 |

Earlier raw training logs were emitted before the `linear_nested` label fix and
therefore under-counted correct solutions for that template. The bundled JSONL
files now contain formula-consistent labels, guarded by
`tests/test_weird_algebra_dataset.py`.

Training logs also showed the off-policy path was active:

- `old_logprobs_present = 1`
- `logprob_to_old_mean` remained finite, usually around `0.1`-`0.2`
- `kl_to_old` remained finite and was the useful off-policy drift metric
- checkpoints were written in HF format via `SAVE_DIR` / `SAVE_FINAL`

The result is a sanity check that this branch can learn on a held-out generated
algebra set after the rollout-side logprob and Ray-managed FSDP changes. It is
still a small-N smoke result, so treat it as evidence that the pipeline is
working rather than as a model-quality claim.

## Community Note

This is the result we want readers to take away:

- the off-policy GRPO path was not only wiring-clean, it produced a measurable
  held-out improvement on a fixed generated algebra split;
- rollout-side logprobs were present and used as the old-policy logprobs for
  ratio/clipping;
- the run exercised Ray-managed FSDP train actors, NanoDeploy rollout workers,
  DLSlime trajectory transport, repeated weight sync, held-out eval, and HF
  checkpoint save;
- the evidence is intentionally scoped as a reproducible smoke validation, not
  as a broad benchmark claim.
