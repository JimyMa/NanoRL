from __future__ import annotations

import json

from nanorl.metrics import assess, build_dashboard, parse_train_jsonl


def test_parse_train_jsonl_splits_steps_and_syncs(tmp_path):
    path = tmp_path / "train.jsonl"
    records = [
        {
            "step": 0,
            "loss": 1.5,
            "kl_mean": 0.0,
            "mean_reward": 0.25,
            "mean_advantage": 0.0,
            "response_tokens": 12,
            "elapsed_s": 0.3,
        },
        {
            "step": 1,
            "loss": 1.0,
            "kl_mean": 0.0,
            "mean_reward": 0.5,
            "mean_advantage": 0.1,
            "response_tokens": 13,
            "elapsed_s": 0.4,
        },
        {
            "event": "weight_sync",
            "version": 2,
            "n_tensors": 398,
            "pull_s": 1.2,
            "apply_s": 0.8,
            "wall_s": 2.4,
            "counts": [
                {"loaded": 398, "skipped_unknown": 0, "used_loader_cb": 390},
                {"loaded": 398, "skipped_unknown": 0, "used_loader_cb": 390},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    data = parse_train_jsonl(path)

    assert [s.step for s in data.steps] == [0, 1]
    assert [s.loss for s in data.steps] == [1.5, 1.0]
    assert len(data.syncs) == 1
    assert data.syncs[0].loaded == 796
    assert data.syncs[0].n_tensors == 398


def test_assess_marks_basic_ready_run_as_pass(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "loss": 2.0, "mean_reward": 0.0}),
                json.dumps({"step": 1, "loss": 1.0, "mean_reward": 1.0}),
                json.dumps(
                    {
                        "event": "weight_sync",
                        "version": 2,
                        "n_tensors": 10,
                        "counts": [{"loaded": 10, "skipped_unknown": 0}],
                    }
                ),
            ]
        )
        + "\n"
    )

    checks = assess(parse_train_jsonl(path), expect_sync=True)
    statuses = {name: status for status, name, _detail in checks}

    assert statuses["Loss is finite"] == "pass"
    assert statuses["Loss trend"] == "pass"
    assert statuses["Weight sync"] == "pass"


def test_build_dashboard_writes_html(tmp_path):
    train_jsonl = tmp_path / "train.jsonl"
    out = tmp_path / "dashboard.html"
    train_jsonl.write_text(
        json.dumps({"step": 0, "loss": 0.0, "mean_reward": 1.0, "elapsed_s": 0.1})
        + "\n"
    )

    data = build_dashboard(train_jsonl, out, title="Test Dashboard")

    assert len(data.steps) == 1
    html = out.read_text()
    assert "Test Dashboard" in html
    assert "Train Steps" in html
