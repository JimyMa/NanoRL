"""End-to-end SlimeRPC loopback through ``TrajectoryService``.

Spawns two PeerAgents in this process, connects them over RDMA, runs the
producer-side ``serve()`` loop in a background thread, and pulls a batch
from the consumer-side proxy. Skipped when NanoCtrl or RDMA HCAs are
unavailable.

Mirrors ``dlslime/examples/python/rpc_example.py`` but uses our
``TrajectoryService`` and the consumer-side ``TrajectoryClient`` so any
shape change to the wire format is caught here.
"""

from __future__ import annotations

import pickle
import time
import uuid

import pytest

dlslime = pytest.importorskip("dlslime")

from nanorl.data.sample import Trajectory
from nanorl.data.trajectory_buffer import (
    open_consumer,
    run_rpc_server,
    TrajectoryService,
)


def _traj(g, n=4):
    return Trajectory(
        prompt_ids=list(range(n)),
        response_ids=list(range(n, n + 3)),
        reward=float(g),
        group_id=g,
    )


def test_loopback_publish_then_pull(nanoctrl_url, has_rdma_device):
    if not has_rdma_device:
        pytest.skip("no RDMA HCAs available")

    PeerAgent = dlslime.PeerAgent
    suffix = uuid.uuid4().hex[:8]
    producer_alias = f"nanorl-test-rollout:{suffix}"
    consumer_alias = f"nanorl-test-train:{suffix}"

    producer = PeerAgent(nanoctrl_url=nanoctrl_url, alias=producer_alias)
    consumer = PeerAgent(nanoctrl_url=nanoctrl_url, alias=consumer_alias)

    try:
        c2p = consumer.connect_to(producer_alias, ib_port=1, qp_num=1)
        p2c = producer.connect_to(consumer_alias, ib_port=1, qp_num=1)
        c2p.wait()
        p2c.wait()

        svc = TrajectoryService(capacity=64)
        svc.publish([_traj(i) for i in range(5)])
        run_rpc_server(producer, svc, consumer_alias=consumer_alias)
        # let the server thread bind before the first proxy call
        time.sleep(0.1)

        proxy = open_consumer(consumer, producer_alias)
        fut = proxy.pull_batch(3, 5.0)
        raw = fut.wait(timeout=10.0)
        out = pickle.loads(raw)
        assert len(out) == 3
        assert all(isinstance(t, Trajectory) for t in out)
        assert [t.group_id for t in out] == [0, 1, 2]
        assert svc.buffered() == 2
    finally:
        producer.shutdown()
        consumer.shutdown()
