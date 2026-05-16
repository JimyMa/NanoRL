# Data plane (DLSlime)

NanoRL has two flows on the DLSlime fabric:

1. **Trajectories** — small pickled batches over SlimeRPC (rollout → train).
2. **Weights** — large tensors via raw `endpoint.read` (train → each NanoInfra worker, in parallel).

## SlimeRPC trajectory contract

Defined in `nanorl/data/trajectory_buffer.py:TrajectoryService`. Producer and consumer must import the *same class*; SlimeRPC dispatches by method name.

```python
class TrajectoryService:
    @method
    def pull_batch(self, n: int, timeout_s: float = 30.0) -> bytes:
        """Block until ≥1 trajectory; return up to n. Returns pickled list[Trajectory].
        Empty list on timeout."""

    @method
    def stats(self) -> bytes:
        """Returns pickled {"buffered": int, "capacity": int}."""

    @method
    def apply_weight_update(self, manifest_blob: bytes) -> bytes:
        """M3: forward the manifest to LLMComponent.pull_and_apply_weights;
        each NanoInfra worker pulls direct via its own PeerAgent."""
```

`Trajectory` (`nanorl/data/sample.py`):

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

Producer-side queue is bounded (`capacity` at construction). New samples push out the *oldest* on overflow. Consumer-side prefetch (`TrajectoryClient`) keeps `prefetch_depth` batches inflight.

### Discovery

Each side constructs `dlslime.PeerAgent(nanoctrl_url=..., alias=...)`, then both call `connect_to(...)` on the other's alias. NanoCtrl + Redis mediate the RDMA QP handshake.

## Raw RDMA weight transport (M3)

Train side registers each gathered HF tensor as a versioned RDMA memory region. The manifest (small metadata-only blob) is shipped over SlimeRPC. Each NanoInfra worker then issues an RDMA read against the train's MRs **directly** — bypassing the rollout driver entirely.

```
train rank 0           rollout driver         NanoInfra worker[0..3]
   │                        │                       │
   │ register MRs           │                       │
   │ (8GB on CPU)           │                       │
   │                        │                       │
   │── SlimeRPC ──────────► │                       │
   │   apply_weight_update  │── collective_rpc ───► │
   │   (manifest_blob)      │   pull_and_apply_     │
   │                        │   weights             │
   │                        │                       │
   │ ◄── RDMA read ─────────────────────────────────┤  (parallel
   │ ◄── RDMA read ─────────────────────────────────┤   across 4
   │ ◄── RDMA read ─────────────────────────────────┤   workers)
   │ ◄── RDMA read ─────────────────────────────────┤
   │                        │                       │
   │                        │                       │ apply via
   │                        │                       │ weight_loader
   │                        │                       │ (TP slice + copy_)
   │                        │                       │
   │                        │ ◄── counts ───────────┤
   │ ◄── result blob ───────┤                       │
```

Implementation:

| File                                                                         | Role                                                                         |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `nanorl/weights/transport.py:WeightTransportTrain`                           | Register MRs (versioned), publish manifest                                   |
| `nanorl/weights/transport.py:WeightTransportRollout`                         | Pull (used in legacy / fallback path)                                        |
| `NanoDeploy/nanodeploy/worker/pull_weights.py`                               | Per-worker RDMA read + apply (the M3 fast path)                              |
| `NanoDeploy/nanodeploy/worker/weight_update.py:apply_named_tensors_in_place` | In-place copy with `weight_loader` callback (preserves CUDA-graph addresses) |
| `NanoDeploy/nanodeploy/engine/weight_sync.py:update_weights`                 | Engine fan-out wrapper                                                       |

## Performance notes

- **Why each worker pulls direct.** Original design: rollout driver pulls 8 GB to CPU, Ray-RPCs the dict to each worker (cross-host serialization). 65 s per sync. New design: 4 workers RDMA-read in parallel, each from a different NIC. 5 s per sync — 13×.
- **NIC selection.** `nanorl/weights/transport.py:select_nic` picks `available_nic()[local_rank % N]` so multi-rank deployments spread across NICs naturally.
- **Pre-resolved handles.** `WeightTransportRollout.pull` and `pull_and_apply_on_worker` resolve local + remote handles once and build the numeric assign tuple `(local_handle, remote_handle, 0, 0, size)`, then issue ONE batched `endpoint.read(...)` for the whole manifest. No per-call name resolution.

## Tests

| Test                               | What it covers                                                     |
| ---------------------------------- | ------------------------------------------------------------------ |
| `tests/test_trajectory_buffer.py`  | Producer queue logic, no RDMA                                      |
| `tests/test_slime_rpc_loopback.py` | 2 PeerAgents in one process, real RDMA, real NanoCtrl/Redis        |
| `tests/test_weight_manifest.py`    | 2-process register→read on a moderately large tensor, bit-equality |
| `scripts/m2_smoke.sh`              | Cross-process trajectory flow on real RDMA                         |
| `scripts/m3_smoke.sh`              | Cross-host weight sync (DDP)                                       |
| `scripts/m3_fsdp_smoke.sh`         | Cross-host weight sync (2-rank FSDP)                               |
