"""Reward verifiers — pluggable scoring of rollout responses.

A `Verifier` takes a rollout (prompt text + response text + ground-truth) and
returns a scalar reward. Initial implementation: a tiny math verifier that
extracts the last numeric answer (\\boxed{...} or trailing number) and
compares to a reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol


class Verifier(Protocol):
    def score(self, response: str, reference: str) -> float: ...


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
# Per-line "Answer: X" — the DAPO/MATH-style instruction tells the model to
# put its final answer on its own line after this prefix. Multi-line aware.
_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*[:\-]\s*(.+?)\s*$")
_BARE_NUMERIC = re.compile(r"\s*-?\d+(?:/\d+|\.\d+)?\s*")
# Trailing number fallback. ``re.DOTALL`` makes ``.`` match newlines so the
# negative-lookahead ``(?!.*\d)`` actually anchors at document-end (without
# DOTALL it picks the first end-of-LINE number, not the last in the whole
# response, and we'd extract "5" from "5²" three lines before the real answer).
_TRAILING_NUM = re.compile(r"-?\d+(?:/\d+|\.\d+)?(?!.*\d)", re.DOTALL)


def _extract_answer(text: str) -> str | None:
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    answer_lines = _ANSWER_LINE.findall(text)
    if answer_lines:
        # Strip a trailing "$" / period / latex wrappers for clean compare.
        cand = answer_lines[-1].strip().strip("$").strip(".").strip()
        if cand:
            return cand
    if _BARE_NUMERIC.fullmatch(text):
        return text.strip()
    m = _TRAILING_NUM.search(text)
    return m.group(0) if m else None


def _normalize_number(s: str) -> float | None:
    s = s.replace(",", "").strip()
    try:
        return float(Fraction(s))
    except (ValueError, AttributeError):
        try:
            return float(s)
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
