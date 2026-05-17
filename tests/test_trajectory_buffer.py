"""Unit tests for the trajectory buffer (no RDMA required).

Exercises the producer-side queue logic and pickle round-trip directly.
A full SlimeRPC loopback test is in ``tests/test_slime_rpc_loopback.py`` and
is skipped unless NanoCtrl is reachable.
"""

from __future__ import annotations

import pickle
import threading
import time

from nanorl.data.sample import Trajectory, TrajectoryBatch
from nanorl.data.trajectory_buffer import TrajectoryService


def _traj(group_id=0, length=4):
    return Trajectory(
        prompt_ids=list(range(length)),
        response_ids=list(range(length, length + 3)),
        reward=1.0,
        group_id=group_id,
    )


def test_publish_then_pull_roundtrip():
    svc = TrajectoryService(capacity=128)
    svc.publish([_traj(i) for i in range(5)])
    raw = svc.pull_batch(n=3, timeout_s=0.1)
    assert isinstance(raw, bytes)
    out = pickle.loads(raw)
    assert len(out) == 3
    assert all(isinstance(t, Trajectory) for t in out)
    assert svc.buffered() == 2


def test_pull_blocks_until_publish():
    svc = TrajectoryService(capacity=8)
    out = {}

    def consumer():
        out["raw"] = svc.pull_batch(n=2, timeout_s=2.0)

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    time.sleep(0.1)
    assert t.is_alive()
    svc.publish([_traj(0), _traj(0)])
    t.join(timeout=1.0)
    assert not t.is_alive()
    samples = pickle.loads(out["raw"])
    assert len(samples) == 2


def test_pull_returns_empty_on_timeout():
    svc = TrajectoryService(capacity=8)
    raw = svc.pull_batch(n=4, timeout_s=0.05)
    assert pickle.loads(raw) == []


def test_capacity_drops_oldest():
    svc = TrajectoryService(capacity=3)
    svc.publish([_traj(i) for i in range(5)])
    assert svc.buffered() == 3
    raw = svc.pull_batch(n=10, timeout_s=0.1)
    out = pickle.loads(raw)
    assert [t.group_id for t in out] == [2, 3, 4]


def test_trajectory_batch_pads_correctly():
    trajs = [
        Trajectory(prompt_ids=[1, 2], response_ids=[3, 4, 5], reward=1.0, group_id=0),
        Trajectory(prompt_ids=[6, 7, 8], response_ids=[9], reward=0.0, group_id=0),
    ]
    batch = TrajectoryBatch.from_trajectories(trajs, pad_id=99)
    assert batch.tokens.shape == (2, 5)
    assert batch.tokens[0, :5].tolist() == [1, 2, 3, 4, 5]
    assert batch.tokens[1].tolist() == [6, 7, 8, 9, 99]
    assert batch.response_mask[0].tolist() == [0, 0, 1, 1, 1]
    assert batch.response_mask[1].tolist() == [0, 0, 0, 1, 0]
    assert batch.rewards.tolist() == [1.0, 0.0]
    assert batch.seq_lengths.tolist() == [5, 4]
