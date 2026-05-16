"""Reward verifiers — pluggable scoring of rollout responses.

A `Verifier` takes a rollout (prompt text + response text + ground-truth) and
returns a scalar reward. Initial implementation: a tiny math verifier that
extracts the last numeric answer (\\boxed{...} or trailing number) and
compares to a reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class Verifier(Protocol):
    def score(self, response: str, reference: str) -> float: ...


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_TRAILING_NUM = re.compile(r"-?\d+(?:\.\d+)?(?!.*\d)")


def _extract_answer(text: str) -> str | None:
    matches = _BOXED.findall(text)
    if matches:
        return matches[-1].strip()
    m = _TRAILING_NUM.search(text)
    return m.group(0) if m else None


def _normalize_number(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


@dataclass
class MathVerifier:
    """0/1 reward by exact-match on the extracted final answer.

    Numeric answers are compared with a small tolerance; otherwise we fall
    back to whitespace-stripped string equality.
    """

    tol: float = 1e-4

    def score(self, response: str, reference: str) -> float:
        pred = _extract_answer(response)
        gold = _extract_answer(reference) or reference.strip()
        if pred is None:
            return 0.0
        p, g = _normalize_number(pred), _normalize_number(gold)
        if p is not None and g is not None:
            return 1.0 if abs(p - g) <= self.tol * max(1.0, abs(g)) else 0.0
        return 1.0 if pred.strip() == gold.strip() else 0.0
