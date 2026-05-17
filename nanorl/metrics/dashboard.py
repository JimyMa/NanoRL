"""Static HTML dashboard for NanoRL training runs.

Lives in ``nanorl/metrics/`` because it consumes exactly the JSONL
schema that ``JSONLLogger`` (in this package) produces. The pairing:

    nanorl train --log-jsonl train.jsonl     # writes via JSONLLogger
    nanorl-dashboard --train-jsonl train.jsonl --out run.html  # reads here

No third-party deps. Pure stdlib + inline SVG. Safe to attach to a PR.

Tier-3 readiness checks (auto-graded in ``assess``):
  * loss is finite and trending down
  * reward variance is non-zero (catches saturated-prompts pathology)
  * weight syncs ship a manifest of >0 tensors and every worker loads >0
  * no unknown tensors silently skipped on the rollout side

Tier-2 health checks (added when the corresponding TrainStats fields
are present):
  * importance-weight clip rate (truncated_above + truncated_below)
    must stay < 5% sustained — high clip rate signals stale data
  * policy entropy must not collapse below 50% of step-0 value
  * KL(πθ || πθ_old) must stay < 0.1 sustained — off-policy distance
  * gradient norm must be finite and bounded; vanishing or exploding
    is flagged
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


@dataclass
class StepRecord:
    """One per-step row from ``JSONLLogger.log_step``. Fields default to
    safe values so older logs (pre-Tier-2) still parse."""

    step: int
    loss: float
    kl_mean: float = 0.0
    mean_reward: float = 0.0
    mean_advantage: float = 0.0
    response_tokens: int = 0
    elapsed_s: float = 0.0
    # Tier-2 health signals — added 2026-05 alongside the metrics module.
    # Older logs simply leave these at their defaults; assess() then
    # skips the corresponding checks rather than warning falsely.
    kl_to_old: float = 0.0
    logprob_to_old_mean: float = 0.0
    logprob_to_old_max: float = 0.0
    ratios_mean: float = 1.0
    ratios_max: float = 1.0
    truncated_above_rate: float = 0.0
    truncated_below_rate: float = 0.0
    entropy_mean: float = 0.0
    grad_norm: float = 0.0
    response_length_mean: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncRecord:
    version: int
    n_tensors: int
    pull_s: float = 0.0
    apply_s: float = 0.0
    wall_s: float = 0.0
    loaded: int = 0
    skipped_unknown: int = 0
    used_loader_cb: int = 0
    used_direct_copy: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunData:
    steps: list[StepRecord] = field(default_factory=list)
    syncs: list[SyncRecord] = field(default_factory=list)
    rollout_rounds: list[dict[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def _sum_counts(sync: dict[str, Any], key: str) -> int:
    counts = sync.get("counts")
    if not isinstance(counts, list):
        return 0
    return sum(_as_int(c.get(key, 0)) for c in counts if isinstance(c, dict))


def parse_train_jsonl(path: Path) -> RunData:
    data = RunData()
    for obj in _read_jsonl(path):
        if obj.get("event") == "weight_sync":
            data.syncs.append(
                SyncRecord(
                    version=_as_int(obj.get("version")),
                    n_tensors=_as_int(obj.get("n_tensors")),
                    pull_s=_as_float(obj.get("pull_s")),
                    apply_s=_as_float(obj.get("apply_s")),
                    wall_s=_as_float(obj.get("wall_s")),
                    loaded=_sum_counts(obj, "loaded"),
                    skipped_unknown=_sum_counts(obj, "skipped_unknown"),
                    used_loader_cb=_sum_counts(obj, "used_loader_cb"),
                    used_direct_copy=_sum_counts(obj, "used_direct_copy"),
                    raw=obj,
                )
            )
        elif "loss" in obj:
            data.steps.append(
                StepRecord(
                    step=_as_int(obj.get("step")),
                    loss=_as_float(obj.get("loss"), float("nan")),
                    kl_mean=_as_float(obj.get("kl_mean")),
                    mean_reward=_as_float(obj.get("mean_reward")),
                    mean_advantage=_as_float(obj.get("mean_advantage")),
                    response_tokens=_as_int(obj.get("response_tokens")),
                    elapsed_s=_as_float(obj.get("elapsed_s")),
                    kl_to_old=_as_float(obj.get("kl_to_old")),
                    logprob_to_old_mean=_as_float(
                        obj.get(
                            "logprob_to_old_mean",
                            obj.get("old_logprobs_abs_diff_mean"),
                        )
                    ),
                    logprob_to_old_max=_as_float(
                        obj.get(
                            "logprob_to_old_max",
                            obj.get("old_logprobs_abs_diff_max"),
                        )
                    ),
                    ratios_mean=_as_float(obj.get("ratios_mean"), 1.0),
                    ratios_max=_as_float(obj.get("ratios_max"), 1.0),
                    truncated_above_rate=_as_float(obj.get("truncated_above_rate")),
                    truncated_below_rate=_as_float(obj.get("truncated_below_rate")),
                    entropy_mean=_as_float(obj.get("entropy_mean")),
                    grad_norm=_as_float(obj.get("grad_norm")),
                    response_length_mean=_as_float(obj.get("response_length_mean")),
                    raw=obj,
                )
            )
    return data


_ROLLOUT_RE = re.compile(
    r"rollout: n=(?P<n>\d+) mean=(?P<mean>-?\d+(?:\.\d+)?) "
    r"std=(?P<std>-?\d+(?:\.\d+)?) min=(?P<min>-?\d+(?:\.\d+)?) "
    r"max=(?P<max>-?\d+(?:\.\d+)?) elapsed=(?P<elapsed>-?\d+(?:\.\d+)?)s"
)


def parse_rollout_log(path: Path) -> list[dict[str, float]]:
    rounds: list[dict[str, float]] = []
    with path.open(errors="replace") as f:
        for line in f:
            m = _ROLLOUT_RE.search(line)
            if not m:
                continue
            rounds.append(
                {
                    "n": float(m.group("n")),
                    "mean_reward": float(m.group("mean")),
                    "std_reward": float(m.group("std")),
                    "min_reward": float(m.group("min")),
                    "max_reward": float(m.group("max")),
                    "elapsed_s": float(m.group("elapsed")),
                }
            )
    return rounds


def merge_run_data(base: RunData, other: RunData) -> RunData:
    base.steps.extend(other.steps)
    base.syncs.extend(other.syncs)
    base.rollout_rounds.extend(other.rollout_rounds)
    base.warnings.extend(other.warnings)
    base.steps.sort(key=lambda s: s.step)
    base.syncs.sort(key=lambda s: s.version)
    return base


def _has_tier2_data(data: RunData) -> bool:
    """Did the training run actually emit the new TrainStats fields?
    Older logs leave them at default; we skip the Tier-2 checks then."""
    if not data.steps:
        return False
    return any(
        s.entropy_mean != 0.0
        or s.grad_norm != 0.0
        or s.kl_to_old != 0.0
        or s.ratios_max != 1.0
        for s in data.steps
    )


def assess(data: RunData, *, expect_sync: bool = False) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []

    # ---- Tier-3 correctness ----------------------------------------------
    if not data.steps:
        checks.append(
            (
                "fail",
                "No train steps",
                "No per-step records were found in the train JSONL.",
            )
        )
    else:
        bad = [s for s in data.steps if not math.isfinite(s.loss)]
        if bad:
            checks.append(
                (
                    "fail",
                    "Non-finite loss",
                    f"{len(bad)} step(s) contain NaN or Inf loss.",
                )
            )
        else:
            checks.append(
                (
                    "pass",
                    "Loss is finite",
                    f"All {len(data.steps)} train step(s) have finite loss.",
                )
            )

        if len(data.steps) >= 2:
            first, last = data.steps[0].loss, data.steps[-1].loss
            delta = last - first
            status = "pass" if delta < 0 else "warn"
            checks.append(
                (
                    status,
                    "Loss trend",
                    f"first={first:.6g}, last={last:.6g}, delta={delta:.6g}.",
                )
            )

        rewards = [s.mean_reward for s in data.steps]
        if len(set(round(r, 8) for r in rewards)) <= 1:
            checks.append(
                (
                    "warn",
                    "Reward variance",
                    "Mean reward is constant; trivial prompts may produce zero advantage.",
                )
            )
        else:
            checks.append(
                (
                    "pass",
                    "Reward variance",
                    f"mean reward range {min(rewards):.3g} to {max(rewards):.3g}.",
                )
            )

    if data.syncs:
        empty = [s for s in data.syncs if s.n_tensors <= 0]
        zero_loaded = [s for s in data.syncs if s.loaded <= 0 and s.raw.get("counts")]
        if empty or zero_loaded:
            checks.append(
                (
                    "fail",
                    "Weight sync payload",
                    "At least one weight sync shipped no tensors or loaded none.",
                )
            )
        else:
            avg_tensors = fmean([s.n_tensors for s in data.syncs])
            checks.append(
                (
                    "pass",
                    "Weight sync",
                    f"{len(data.syncs)} sync(s), avg manifest {avg_tensors:.0f} tensors.",
                )
            )

        skipped = sum(s.skipped_unknown for s in data.syncs)
        if skipped:
            checks.append(
                (
                    "warn",
                    "Unknown tensors skipped",
                    f"Rollout skipped {skipped} tensor load(s).",
                )
            )
        else:
            checks.append(
                (
                    "pass",
                    "Unknown tensors skipped",
                    "No skipped tensors reported by rollout workers.",
                )
            )
    elif expect_sync:
        checks.append(
            ("fail", "Weight sync", "Expected weight sync events, but none were found.")
        )
    else:
        checks.append(
            (
                "warn",
                "Weight sync",
                "No weight sync events found; OK for M1/train-only runs.",
            )
        )

    # ---- Tier-2 policy health (only when the new fields are present) -----
    if _has_tier2_data(data):
        # 1. Clip rate. Sustained > 5% means data is stale w.r.t. policy.
        clip_rates = [
            s.truncated_above_rate + s.truncated_below_rate for s in data.steps
        ]
        avg_clip = fmean(clip_rates)
        if avg_clip > 0.05:
            checks.append(
                (
                    "warn",
                    "Clip rate",
                    f"avg total clip = {avg_clip:.1%} (> 5%); rollouts likely stale, lower weight_sync_every.",
                )
            )
        else:
            checks.append(
                ("pass", "Clip rate", f"avg total clip = {avg_clip:.1%} (< 5%).")
            )

        # 2. Entropy collapse. Compare end vs beginning.
        entropies = [s.entropy_mean for s in data.steps if s.entropy_mean != 0.0]
        if len(entropies) >= 4:
            head = fmean(entropies[: max(1, len(entropies) // 4)])
            tail = fmean(entropies[-max(1, len(entropies) // 4) :])
            if head > 0 and tail < 0.5 * head:
                checks.append(
                    (
                        "warn",
                        "Entropy",
                        f"collapsed from {head:.3g} → {tail:.3g} (< 50%); mode-collapse risk.",
                    )
                )
            else:
                checks.append(
                    (
                        "pass",
                        "Entropy",
                        f"head={head:.3g} tail={tail:.3g} (no collapse).",
                    )
                )

        # 3. Distance to the rollout-time policy. ``kl_to_old`` is a stable
        # Schulman-style approx; ``logprob_to_old`` is the direct chosen-token
        # logprob delta and is easier to reason about when ref KL is noisy.
        kl_to_olds = [s.kl_to_old for s in data.steps]
        avg_kl_to_old = fmean(kl_to_olds)
        avg_logprob_to_old = fmean([s.logprob_to_old_mean for s in data.steps])
        if avg_kl_to_old > 0.1:
            checks.append(
                (
                    "warn",
                    "Off-policy distance",
                    f"avg KL(πθ||πθ_old) = {avg_kl_to_old:.3g} (> 0.1), avg |logprob-old| = {avg_logprob_to_old:.3g}; reduce LR or weight_sync_every.",
                )
            )
        else:
            checks.append(
                (
                    "pass",
                    "Off-policy distance",
                    f"avg KL(πθ||πθ_old) = {avg_kl_to_old:.3g}, avg |logprob-old| = {avg_logprob_to_old:.3g}.",
                )
            )

        # 4. Gradient norm sanity.
        gnorms = [s.grad_norm for s in data.steps if s.grad_norm != 0.0]
        if gnorms:
            bad_g = [g for g in gnorms if not math.isfinite(g) or g > 100.0 or g < 1e-6]
            if bad_g:
                checks.append(
                    (
                        "warn",
                        "Gradient norm",
                        f"{len(bad_g)} step(s) with grad_norm out of [1e-6, 100]; numerical instability.",
                    )
                )
            else:
                checks.append(
                    (
                        "pass",
                        "Gradient norm",
                        f"min={min(gnorms):.3g} max={max(gnorms):.3g} avg={fmean(gnorms):.3g}.",
                    )
                )

    if data.rollout_rounds:
        checks.append(
            (
                "pass",
                "Rollout log",
                f"Parsed {len(data.rollout_rounds)} rollout round(s).",
            )
        )

    return checks


def _fmt_num(value: float | int, digits: int = 3) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def _series_svg(
    points: list[tuple[float, float]],
    *,
    width: int = 560,
    height: int = 180,
    stroke: str = "#2563eb",
) -> str:
    if not points:
        return '<div class="empty">No data</div>'
    if len(points) == 1:
        points = [points[0], (points[0][0] + 1.0, points[0][1])]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points if math.isfinite(p[1])]
    if not ys:
        return '<div class="empty">No finite values</div>'
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_y == max_y:
        pad = max(1.0, abs(min_y) * 0.1)
        min_y -= pad
        max_y += pad
    if min_x == max_x:
        max_x += 1
    pad_l, pad_r, pad_t, pad_b = 42, 12, 12, 28
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    def sx(x: float) -> float:
        return pad_l + (x - min_x) / (max_x - min_x) * inner_w

    def sy(y: float) -> float:
        return pad_t + (max_y - y) / (max_y - min_y) * inner_h

    poly = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points if math.isfinite(y))
    circles = "\n".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3"><title>x={x:g}, y={y:g}</title></circle>'
        for x, y in points
        if math.isfinite(y)
    )
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="series chart">
  <line class="axis" x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" />
  <line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" />
  <text class="tick" x="{pad_l}" y="{height-8}">{min_x:g}</text>
  <text class="tick" x="{width-pad_r-24}" y="{height-8}">{max_x:g}</text>
  <text class="tick" x="4" y="{pad_t+5}">{_fmt_num(max_y)}</text>
  <text class="tick" x="4" y="{height-pad_b}">{_fmt_num(min_y)}</text>
  <polyline points="{poly}" fill="none" stroke="{stroke}" stroke-width="2.5" />
  <g fill="{stroke}">{circles}</g>
</svg>"""


def _metric_card(label: str, value: str, sub: str = "") -> str:
    return f"""
<section class="metric">
  <div class="metric-label">{html.escape(label)}</div>
  <div class="metric-value">{html.escape(value)}</div>
  <div class="metric-sub">{html.escape(sub)}</div>
</section>"""


def render_html(
    data: RunData, *, title: str, checks: list[tuple[str, str, str]]
) -> str:
    losses = [(s.step, s.loss) for s in data.steps]
    rewards = [(s.step, s.mean_reward) for s in data.steps]
    sync_wall = [(s.version, s.wall_s) for s in data.syncs]
    sync_loaded = [(s.version, s.loaded) for s in data.syncs]

    # Tier-2 series. Drawn only when at least one step emitted them.
    tier2 = _has_tier2_data(data)
    entropy_pts = [(s.step, s.entropy_mean) for s in data.steps] if tier2 else []
    grad_norm_pts = [(s.step, s.grad_norm) for s in data.steps] if tier2 else []
    kl_to_old_pts = [(s.step, s.kl_to_old) for s in data.steps] if tier2 else []
    clip_pts = (
        [(s.step, s.truncated_above_rate + s.truncated_below_rate) for s in data.steps]
        if tier2
        else []
    )

    finite_losses = [s.loss for s in data.steps if math.isfinite(s.loss)]
    avg_loss = fmean(finite_losses) if finite_losses else float("nan")
    avg_reward = (
        fmean([s.mean_reward for s in data.steps]) if data.steps else float("nan")
    )
    avg_step_s = (
        fmean([s.elapsed_s for s in data.steps]) if data.steps else float("nan")
    )
    avg_sync_s = fmean([s.wall_s for s in data.syncs]) if data.syncs else float("nan")

    check_html = "\n".join(
        f'<li class="{status}"><strong>{html.escape(name)}</strong><span>{html.escape(detail)}</span></li>'
        for status, name, detail in checks
    )
    step_rows = "\n".join(
        "<tr>"
        f"<td>{s.step}</td><td>{_fmt_num(s.loss, 6)}</td><td>{_fmt_num(s.kl_mean)}</td>"
        f"<td>{_fmt_num(s.mean_reward)}</td><td>{_fmt_num(s.entropy_mean)}</td>"
        f"<td>{_fmt_num(s.logprob_to_old_mean)}</td>"
        f"<td>{_fmt_num(s.grad_norm)}</td><td>{s.response_tokens}</td>"
        f"<td>{_fmt_num(s.elapsed_s)}</td>"
        "</tr>"
        for s in data.steps[-20:]
    )
    sync_rows = "\n".join(
        "<tr>"
        f"<td>{s.version}</td><td>{s.n_tensors}</td><td>{s.loaded}</td>"
        f"<td>{s.skipped_unknown}</td><td>{_fmt_num(s.pull_s)}</td>"
        f"<td>{_fmt_num(s.apply_s)}</td><td>{_fmt_num(s.wall_s)}</td>"
        "</tr>"
        for s in data.syncs[-20:]
    )

    tier2_panels = ""
    if tier2:
        tier2_panels = f"""
  <section class="grid">
    <section class="panel"><h2>Entropy</h2>{_series_svg(entropy_pts, stroke="#0f766e")}</section>
    <section class="panel"><h2>Gradient Norm</h2>{_series_svg(grad_norm_pts, stroke="#7c3aed")}</section>
    <section class="panel"><h2>KL(πθ ‖ πθ_old)</h2>{_series_svg(kl_to_old_pts, stroke="#a46000")}</section>
    <section class="panel"><h2>Clip Rate (above + below)</h2>{_series_svg(clip_pts, stroke="#b42318")}</section>
  </section>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f7f4;
  --panel: #ffffff;
  --ink: #1d2528;
  --muted: #627076;
  --line: #d8ddd7;
  --pass: #147a4b;
  --warn: #a46000;
  --fail: #b42318;
  --blue: #2563eb;
  --teal: #0f766e;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--ink); }}
header {{ padding: 28px 32px 10px; }}
h1 {{ margin: 0; font-size: 28px; font-weight: 720; }}
.subtitle {{ color: var(--muted); margin-top: 4px; }}
main {{ padding: 14px 32px 36px; display: grid; gap: 18px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
.metric, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
.metric {{ padding: 14px 16px; }}
.metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .02em; }}
.metric-value {{ font-size: 26px; font-weight: 760; margin-top: 4px; }}
.metric-sub {{ color: var(--muted); min-height: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
.panel {{ padding: 16px; overflow: hidden; }}
h2 {{ margin: 0 0 12px; font-size: 16px; }}
.checks {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }}
.checks li {{ display: flex; gap: 10px; align-items: baseline; border-left: 4px solid var(--line); padding: 8px 10px; background: #fafbf9; }}
.checks li strong {{ min-width: 200px; }}
.checks li span {{ color: var(--muted); }}
.checks .pass {{ border-color: var(--pass); }}
.checks .warn {{ border-color: var(--warn); }}
.checks .fail {{ border-color: var(--fail); }}
svg {{ width: 100%; height: auto; display: block; }}
.axis {{ stroke: var(--line); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 11px; }}
.empty {{ color: var(--muted); padding: 40px 0; text-align: center; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 650; font-size: 12px; }}
@media (max-width: 640px) {{
  header, main {{ padding-left: 16px; padding-right: 16px; }}
  .grid {{ grid-template-columns: 1fr; }}
  .checks li {{ display: block; }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">Generated from NanoRL JSONL/log files. Static, dependency-free, and safe to archive with smoke outputs.</div>
</header>
<main>
  <section class="metrics">
    {_metric_card("Train Steps", str(len(data.steps)), f"avg step {_fmt_num(avg_step_s)} s")}
    {_metric_card("Avg Loss", _fmt_num(avg_loss, 6), "finite train records only")}
    {_metric_card("Avg Reward", _fmt_num(avg_reward), "from TrainStats")}
    {_metric_card("Weight Syncs", str(len(data.syncs)), f"avg wall {_fmt_num(avg_sync_s)} s")}
  </section>
  <section class="panel">
    <h2>Readiness Checks</h2>
    <ul class="checks">{check_html}</ul>
  </section>
  <section class="grid">
    <section class="panel"><h2>Loss</h2>{_series_svg(losses, stroke="#2563eb")}</section>
    <section class="panel"><h2>Mean Reward</h2>{_series_svg(rewards, stroke="#0f766e")}</section>
    <section class="panel"><h2>Weight Sync Wall Time</h2>{_series_svg(sync_wall, stroke="#a46000")}</section>
    <section class="panel"><h2>Loaded Tensor Count</h2>{_series_svg(sync_loaded, stroke="#7c3aed")}</section>
  </section>{tier2_panels}
  <section class="grid">
    <section class="panel">
      <h2>Recent Train Steps</h2>
      <table><thead><tr><th>step</th><th>loss</th><th>kl</th><th>reward</th><th>H</th><th>logprob-old</th><th>gnorm</th><th>tokens</th><th>sec</th></tr></thead>
      <tbody>{step_rows}</tbody></table>
    </section>
    <section class="panel">
      <h2>Recent Weight Syncs</h2>
      <table><thead><tr><th>version</th><th>manifest</th><th>loaded</th><th>skipped</th><th>pull</th><th>apply</th><th>wall</th></tr></thead>
      <tbody>{sync_rows}</tbody></table>
    </section>
  </section>
</main>
</body>
</html>
"""


def build_dashboard(
    train_jsonl: Path,
    output: Path,
    *,
    producer_log: Path | None = None,
    title: str = "NanoRL Run Dashboard",
    expect_sync: bool = False,
) -> RunData:
    data = parse_train_jsonl(train_jsonl)
    if producer_log is not None and producer_log.exists():
        data.rollout_rounds.extend(parse_rollout_log(producer_log))
    checks = assess(data, expect_sync=expect_sync)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data, title=title, checks=checks))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanorl-dashboard")
    parser.add_argument(
        "--train-jsonl",
        required=True,
        type=Path,
        help="JSONL from nanorl train --log-jsonl",
    )
    parser.add_argument(
        "--producer-log", type=Path, default=None, help="Optional rollout producer log"
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/nanorl_dashboard.html"))
    parser.add_argument("--title", default="NanoRL Run Dashboard")
    parser.add_argument(
        "--expect-sync",
        action="store_true",
        help="Fail readiness if no weight_sync events exist",
    )
    args = parser.parse_args(argv)

    data = build_dashboard(
        args.train_jsonl,
        args.out,
        producer_log=args.producer_log,
        title=args.title,
        expect_sync=args.expect_sync,
    )
    print(
        f"wrote {args.out} "
        f"({len(data.steps)} train steps, {len(data.syncs)} weight syncs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
