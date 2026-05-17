"""Prompt-set loader with auto-detected schema.

The rollout / eval pipelines need ``PromptItem(prompt, reference, group_id)``
rows. We accept any of:

  * NanoRL native::

        {"prompt": "...", "reference": "...", "group_id": 0}

  * DAPO / verl-style chat row::

        {"prompt": [{"role": "user", "content": "..."}],
         "label": "34", "reward_model": {"ground_truth": "34"}}

  * HuggingFace MATH-ish flat::

        {"question": "...", "answer": "..."}

Bundled short names (resolved relative to the dataset roots known at import
time) let callers pass ``--prompts dapo`` instead of a file path. Anything
that doesn't resolve to a file or a known alias raises ``FileNotFoundError``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable

from nanorl.actors.rollout import PromptItem

logger = logging.getLogger(__name__)


# Bundled aliases. Edit when new datasets land on disk.
_BUNDLED: dict[str, str] = {
    "sample": "nanorl/configs/sample_prompts.jsonl",
    "dapo": "/models/dapo_math/dapo-math-17k.jsonl",
    "aime": "/models/dapo_math/aime-2024.jsonl",
    # Goldilocks-zone subset of dapo: prompts with mean reward in (0.05, 0.95)
    # under a baseline run — produced by scripts/filter_prompts.py from a
    # pilot rollout dump. These are the "noisy-but-learnable" problems where
    # GRPO group advantages are non-zero and the policy gradient flows.
    "goldilocks": "nanorl/configs/dapo_goldilocks.jsonl",
}


def _row_to_prompt_item(row: dict, lineno: int) -> PromptItem | None:
    """Map a heterogeneous row to PromptItem; return None if unparsable."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        # Chat-formatted (DAPO/verl): take the user turn(s).
        prompt = "\n".join(
            m.get("content", "") for m in prompt if m.get("role") == "user"
        )
    elif prompt is None:
        # HF MATH/GSM8K-style flat schema.
        prompt = row.get("question")
    if not prompt:
        return None

    reference = (
        row.get("reference")
        or row.get("answer")
        or row.get("label")
        or (row.get("reward_model") or {}).get("ground_truth")
        or ""
    )
    group_id = row.get("group_id", row.get("id", lineno))
    try:
        group_id = int(group_id)
    except (TypeError, ValueError):
        group_id = lineno

    return PromptItem(
        prompt=str(prompt),
        reference=str(reference),
        group_id=group_id,
    )


def load_jsonl(path: str, limit: int | None = None) -> list[PromptItem]:
    """Load a JSONL file. Bad rows are skipped with a warning so a single
    typo doesn't tank the run. ``limit`` caps the result for smoke runs."""
    items: list[PromptItem] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("skipping line %d in %s: %s", lineno, path, exc)
                continue
            item = _row_to_prompt_item(row, lineno)
            if item is None:
                logger.warning(
                    "skipping line %d in %s: no prompt/reference fields", lineno, path
                )
                continue
            items.append(item)
            if limit and len(items) >= limit:
                break
    return items


def resolve(spec: str) -> str:
    """Resolve a path or bundled name to an absolute path."""
    if os.path.isfile(spec):
        return spec
    if spec in _BUNDLED:
        cand = _BUNDLED[spec]
        if os.path.isabs(cand) and os.path.isfile(cand):
            return cand
        # Relative paths are repo-relative.
        repo = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        rel = os.path.join(repo, cand)
        if os.path.isfile(rel):
            return rel
    raise FileNotFoundError(
        f"prompt set not found: {spec!r} (not a file; not in bundled set "
        f"{sorted(_BUNDLED)})"
    )


def load_prompts(spec: str, limit: int | None = None) -> list[PromptItem]:
    """One-stop: accept a JSONL path or a bundled name (``dapo``, ``aime``,
    ``sample``)."""
    path = resolve(spec)
    items = load_jsonl(path, limit=limit)
    logger.info("loaded %d prompts from %s", len(items), path)
    return items
