import json
import re
from fractions import Fraction
from pathlib import Path


LINEAR_NESTED_RE = re.compile(
    r"Let x be the real number satisfying "
    r"(?P<a>\d+)\((?P<b>\d+)\(x (?P<inner_sign>[+-]) (?P<inner>\d+)\)\) "
    r"(?P<outer_sign>[+-]) (?P<outer>\d+) = (?P<rhs>-?\d+)\. "
    r"Find x (?P<query_sign>[+-]) (?P<query>\d+)\."
)


def _signed(value: str, sign: str) -> int:
    n = int(value)
    return n if sign == "+" else -n


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _solve_linear_nested(problem: str) -> str:
    match = LINEAR_NESTED_RE.fullmatch(problem)
    assert match is not None, problem

    a = int(match["a"])
    b = int(match["b"])
    inner = _signed(match["inner"], match["inner_sign"])
    outer = _signed(match["outer"], match["outer_sign"])
    rhs = int(match["rhs"])
    query = _signed(match["query"], match["query_sign"])

    x = Fraction(rhs - outer, a * b) - inner
    return _format_fraction(x + query)


def test_weird_algebra_linear_nested_labels_are_formula_consistent():
    dataset_dir = Path("nanorl/configs/datasets")
    paths = [
        dataset_dir / "weird_algebra_train192.jsonl",
        dataset_dir / "weird_algebra_test64.jsonl",
        dataset_dir / "weird_algebra_256.jsonl",
    ]

    seen = 0
    for path in paths:
        for raw in path.read_text().splitlines():
            row = json.loads(raw)
            if row.get("extra_info", {}).get("category") != "linear_nested":
                continue

            seen += 1
            problem = row["extra_info"]["raw_problem"]
            assert row["reward_model"]["ground_truth"] == _solve_linear_nested(problem)

    # 24 train + 8 test + the combined 32-record copy.
    assert seen == 64
