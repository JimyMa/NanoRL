# Data plane (DLSlime)

NanoRL has two distinct flows on the DLSlime fabric:

1. **Trajectories** (rollout → train) — small-to-medium pickled batches over SlimeRPC.
2. **Weights** (train → rollout, M3) — gathered model parameters via raw `RDMAEndpoint.read/write`.

This document covers (1). For (2), see `nanorl/weights/` (M3, not started).

## SlimeRPC trajectory contract

Defined in `nanorl/data/trajectory_buffer.py:TrajectoryService`. Both producer and consumer must import the *same class*; SlimeRPC dispatches by method name.

```python
class TrajectoryService:
    @method
    def pull_batch(self, n: int, timeout_s: float = 30.0) -> bytes:
        """Block until at least one trajectory is available; return up to n.
        Returns pickled list[Trajectory]. Empty list on timeout."""

    @method
    def stats(self) -> bytes:
        """Returns pickled {"buffered": int, "capacity": int}."""
```

`Trajectory` (`nanorl/data/sample.py`) is a dataclass with:

```python
prompt_ids:   list[int]
response_ids: list[int]
reward:       float
group_id:     int
eos:          bool
meta:         dict
```

The train side reconstructs a padded `TrajectoryBatch` via `TrajectoryBatch.from_trajectories(...)`.

### Backpressure

Producer-side queue is bounded (`capacity` at construction). New samples push out the *oldest* on overflow — train will see a fresh tail rather than a stale snapshot. Consumer-side prefetch (`TrajectoryClient`) keeps `prefetch_depth` batches inflight; if the prefetch queue is full the older one is dropped to make room for the new pull.

### Discovery

Each side constructs a `dlslime.PeerAgent(nanoctrl_url=..., alias=...)`, then both call `connect_to(...)` on the other's alias. NanoCtrl + Redis mediate the RDMA QP handshake.

## Wire size and performance

- Default mode is pickle (`@method` returns `bytes`). Trajectories are tiny (a few hundred ints + a float), so pickle overhead is irrelevant compared to RDMA latency. We have not yet benchmarked.
- For payloads where pickle dominates, `@method(raw=True, inplace=True)` writes the reply directly into the registered send buffer with zero intermediate copy. We'll switch when needed.

## Tests

| Test                               | What it covers                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `tests/test_trajectory_buffer.py`  | Producer-side queue logic (no RDMA): publish/pull, blocking, capacity, batch padding                         |
| `tests/test_slime_rpc_loopback.py` | End-to-end RDMA: 2 PeerAgents in one process, connect, serve, proxy, pull. Skipped without HCAs or NanoCtrl. |

The loopback test is the canonical truth — if it passes against your live NanoCtrl + RDMA, the trajectory plane works. Run it before assuming a NanoRL bug.
