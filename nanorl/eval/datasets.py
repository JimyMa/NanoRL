"""Eval prompt-set loaders.

Thin wrapper over :mod:`nanorl.data.datasets` that adds the eval-only
bundled directory ``nanorl/configs/eval_prompts/``. Same auto-detected
schema as the rollout side: NanoRL native, DAPO chat, or HF-flat.
"""

from __future__ import annotations

import logging
import os

from nanorl.actors.rollout import PromptItem
from nanorl.data.datasets import load_jsonl, resolve as _resolve_general

logger = logging.getLogger(__name__)


def load_jsonl_eval(path: str) -> list[PromptItem]:
    """Backwards-compatible alias for :func:`nanorl.data.datasets.load_jsonl`."""
    items = load_jsonl(path)
    logger.info("loaded %d eval prompts from %s", len(items), path)
    return items


def load_eval_prompts(spec: str) -> list[PromptItem]:
    """Resolve ``spec`` to a list of PromptItems.

    Accepts:
      - a path to a JSONL file
      - a name in the eval-only set under
        ``nanorl/configs/eval_prompts/<name>.jsonl``
      - a name registered in :data:`nanorl.data.datasets._BUNDLED`
        (``sample``, ``dapo``, ``aime``)
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundled = os.path.join(here, "configs", "eval_prompts", f"{spec}.jsonl")
    if os.path.isfile(bundled):
        return load_jsonl_eval(bundled)
    return load_jsonl_eval(_resolve_general(spec))
