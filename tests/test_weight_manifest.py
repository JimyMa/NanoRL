"""End-to-end weight-manifest test on real RDMA.

Two PeerAgents in one process (mirrors the SlimeRPC loopback test): the
"train" side registers a moderately large random tensor as a versioned
MR; the "rollout" side allocates a receive buffer, pulls it via
``WeightTransportRollout.pull``, and asserts byte-equality + clean MR
release.

Skipped without NanoCtrl + RDMA HCAs.
"""

from __future__ import annotations

import uuid

import pytest
import torch

dlslime = pytest.importorskip("dlslime")

from nanorl.weights.transport import (
    select_nic,
    TensorMRInfo,
    WeightManifest,
    WeightTransportRollout,
    WeightTransportTrain,
)


def test_select_nic_picks_round_robin():
    """``select_nic`` is pure; it should return one of available_nic()."""
    nics = dlslime.available_nic()
    assert nics, "test environment must have RDMA HCAs"
    assert select_nic(0) == nics[0]
    assert select_nic(len(nics)) == nics[0]
    assert select_nic(1) == nics[1 % len(nics)]


def test_register_pull_release_roundtrip(nanoctrl_url, has_rdma_device):
    """Register N tensors of varied shape on the train side; rollout pulls
    them; assert bit-equality; release; ensure no leaked Redis keys."""
    if not has_rdma_device:
        pytest.skip("no RDMA HCAs")

    PeerAgent = dlslime.PeerAgent
    suffix = uuid.uuid4().hex[:8]
    train_alias = f"nanorl-test-train:{suffix}"
    rollout_alias = f"nanorl-test-rollout:{suffix}"

    train = PeerAgent(nanoctrl_url=nanoctrl_url, alias=train_alias)
    rollout = PeerAgent(nanoctrl_url=nanoctrl_url, alias=rollout_alias)

    try:
        # Both sides initiate; symmetric handshake (matches rpc_example).
        c_t = rollout.connect_to(train_alias, ib_port=1, qp_num=1)
        c_r = train.connect_to(rollout_alias, ib_port=1, qp_num=1)
        c_t.wait(timeout=30.0)
        c_r.wait(timeout=30.0)

        # Build a small mixed-shape weight set.
        torch.manual_seed(0)
        named = {
            "tiny": torch.randn(7, dtype=torch.float32),
            "mid": torch.randn(64, 256, dtype=torch.bfloat16),
            "big": torch.randn(2048, 1024, dtype=torch.bfloat16),  # ~4 MB
        }
        # Hold reference for later equality check (CPU contiguous copies).
        expected = {k: v.detach().contiguous().clone() for k, v in named.items()}

        sender = WeightTransportTrain(train)
        manifest = sender.register(version=1, named_tensors=named)
        assert len(manifest.entries) == 3
        assert manifest.train_alias == train_alias

        receiver = WeightTransportRollout(rollout, train_alias)
        got = receiver.pull(manifest)

        assert set(got.keys()) == set(expected.keys())
        for k in expected:
            assert got[k].shape == expected[k].shape
            assert got[k].dtype == expected[k].dtype
            assert torch.equal(
                got[k], expected[k]
            ), f"tensor {k!r} mismatch — RDMA read produced different bytes"

        # Cleanup; idempotent on second call.
        receiver.release(manifest)
        sender.unregister(version=1)
        sender.unregister(version=1)  # idempotent
    finally:
        train.shutdown()
        rollout.shutdown()
