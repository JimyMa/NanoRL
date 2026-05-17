"""Filter a DAPO-style prompt set to a 'Goldilocks' subset using rollout reward.

The motivation: GRPO needs reward variance within a sampled group to give a
non-zero advantage. If a prompt is so easy that all 4 samples score 1.0, or
so hard that all 4 score 0.0, the group is saturated and contributes no
gradient — half of our recent training run was wasted that way.

This script reads an existing rollout trajectory JSONL (dumped via
``--save-jsonl``), groups rewards by ``reference``, computes mean pass rate
per problem, and keeps only the prompts whose mean reward sits in
``(low, high)`` — the noisy-but-learnable middle band.

Usage:
  python scripts/filter_prompts.py \\
    --traj /tmp/nanorl_smoke/m3_fsdp_trajectories.jsonl \\
    --src nanorl/configs/sample_prompts.jsonl \\
    --dst nanorl/configs/dapo_goldilocks.jsonl \\
    --low 0.05 --high 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--traj",
        required=True,
        type=Path,
        help="rollout trajectory JSONL (must contain 'reference', 'reward', "
        "'prompt' fields per row).",
    )
    p.add_argument("--dst", required=True, type=Path, help="output filtered JSONL")
    p.add_argument(
        "--low", type=float, default=0.05, help="min mean reward (exclusive)"
    )
    p.add_argument(
        "--high", type=float, default=0.95, help="max mean reward (exclusive)"
    )
    p.add_argument(
        "--min-attempts",
        type=int,
        default=4,
        help="discard refs seen fewer than this many times (noise floor).",
    )
    args = p.parse_args(argv)

    if not args.traj.exists():
        print(f"FATAL: trajectory dump not found: {args.traj}", file=sys.stderr)
        return 2

    # Aggregate (rewards, sample prompt) per reference. We pick the LAST
    # observed prompt as the canonical text for that reference — chat-template
    # padding is identical across attempts of the same problem, so any one
    # sample suffices.
    rewards: dict[str, list[float]] = defaultdict(list)
    prompt_for: dict[str, str] = {}
    raw_n = 0
    for line in args.traj.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ref = str(row.get("reference", ""))
        if not ref:
            continue
        rewards[ref].append(float(row["reward"]))
        if "prompt" in row:
            prompt_for[ref] = row["prompt"]
        raw_n += 1

    refs = []
    for ref, rs in rewards.items():
        if len(rs) < args.min_attempts:
            continue
        mean = sum(rs) / len(rs)
        if args.low < mean < args.high:
            refs.append((ref, mean, len(rs)))
    refs.sort(key=lambda x: x[1])  # easiest-to-hardest within band

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.dst.open("w") as fout:
        for ref, mean, n in refs:
            prompt = prompt_for.get(ref)
            if prompt is None:
                continue
            # Strip chat-template artifacts left by `tokenizer.decode(...,
            # skip_special_tokens=True)`. For Qwen3 the user message
            # decodes to "user\n<problem>\nassistant\n"; we want just the
            # problem. The rollout will re-apply chat templates on next pass
            # so leaving the role markers in would double-wrap.
            for marker in ("\nassistant\n", "<|im_start|>assistant"):
                idx = prompt.rfind(marker)
                if idx > 0:
                    prompt = prompt[:idx].rstrip()
                    break
            for prefix in ("user\n", "<|im_start|>user\n"):
                if prompt.startswith(prefix):
                    prompt = prompt[len(prefix) :]
                    break
            fout.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "reference": ref,
                        "group_id": written,
                    }
                )
                + "\n"
            )
            written += 1

    total_refs = len(rewards)
    print(
        f"read {raw_n} trajectories spanning {total_refs} unique refs; "
        f"kept {written} in goldilocks band ({args.low}, {args.high}) "
        f"with >={args.min_attempts} attempts; wrote {args.dst}"
    )
    if not written:
        print(
            "warning: 0 prompts passed — try widening (--low, --high) or "
            "lowering --min-attempts.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
