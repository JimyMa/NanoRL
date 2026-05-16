# Troubleshooting

Most NanoRL failures live at the seams between Ray, NanoInfra, NanoCtrl, and DLSlime — none of which NanoRL controls directly. This page is the playbook for the failures we have actually hit, in the order we have hit them.

## RolloutEngine/NanoInfra startup issues

### `PermissionError: '/tmp/ray/session_*/node_ip_address.json.lock'`

**Cause:** the existing Ray cluster on `10.102.97.179:7078` was started by `root`. Its session directory is world-readable but the lock files are root-owned mode 644. As a non-root user, `ray.init(address=...)` cannot acquire the flock.

**Fix:**

```bash
sudo chmod 666 /tmp/ray/session_*/*.lock
```

Re-apply after every Ray cluster restart (the session dir name embeds a timestamp).

**Long-term:** wrap NanoInfra in a launcher that creates a private Ray cluster per job, or have ops chown the cluster session.

### `FileNotFoundError: model dir does not exist: /models/...`

**Cause:** model path in `cfg.model.hf_path` doesn't exist on the host running rollout. `RolloutEngine.__init__` pre-flights this so we fail in seconds rather than after a 60s NanoInfra startup.

**Fix:** check the path; on this cluster the active models are under `/models/`.

### NanoInfra hangs during startup with `Failed to connect to GCS at address 10.102.97.179:6379`

**Cause:** NanoInfra defaulted `ray_address` to `127.0.0.1:6379`, but `6379` is **Redis** here, not Ray (Ray is on `7078`).

**Fix:** set `infer.ray_address: 10.102.97.179:7078` and `infer.master_address: 10.102.97.183:6006` in the YAML. The reference config already has this.

## SlimeRPC / DLSlime connection issues

### Producer's serve thread crashes with `ValueError: requires a connected endpoint`

**Cause:** `serve()` was called before the QP transitioned to RTS. Producer raced ahead of the consumer.

**Fix already in code:** `nanorl/data/trajectory_buffer.py:run_rpc_server` accepts the `Connection` object and waits on it inside the serve thread before binding the mailbox. Make sure callers pass `connection=` (the CLI already does).

### Consumer's `proxy()` raises `Timeout waiting for MR 'rpc:mailbox:...'`

**Cause:** the producer's `serve()` hasn't run yet (so its mailbox MR isn't registered in NanoCtrl/Redis), or its serve thread silently died.

**Fix:** check the producer log for `trajectory rpc server thread started, exposing to <consumer_alias>`. If absent, look earlier in the log for `connection wait to <consumer_alias> failed` or a Python traceback inside the server thread.

### `RDMA read/write completion failed` / `IBV_WC_RETRY_EXC_ERR` (Vendor Err 129)

**Cause:** the receiver hadn't posted recv WRs by the time the sender's first WR arrived. Even after the QP is up, dlslime needs a beat to arm the mailbox.

**Fix already in code:** producer side has `serve_settle_s=0.2` after `connection.wait()` returns; consumer side has `initial_settle_s=0.2` before the first pull. If you build a new SlimeRPC service, do not skip these.

### `409 Conflict: Peer agent alias 'X' already exists in this scope`

**Cause:** a previous run was killed with `kill -9` (or otherwise crashed) and never called `PeerAgent.shutdown()`. The alias is still registered with NanoCtrl until the heartbeat TTL expires.

**Fixes (cheapest first):**

1. Use unique aliases per run (timestamp suffix). `scripts/m2_smoke.sh` already does this.
2. Wait ~30 seconds for the heartbeat TTL to expire and NanoCtrl to garbage-collect the orphan.
3. Manually evict via NanoCtrl's `/cleanup` endpoint:
   ```bash
   curl -X POST http://10.102.97.179:3000/cleanup \
     -d '{"agent_alias": "rollout:0"}' -H 'content-type: application/json'
   ```

## NanoCtrl / Redis health

### NanoCtrl `/health` returns 404

**Cause:** there is no `/health` endpoint despite what `nanoctrl status` claims. NanoRL's pytest fixture (`tests/conftest.py:_nanoctrl_reachable`) probes `/` instead, which returns `NanoCtrl Server Running`.

**Fix:** if you wrote a new health probe, point it at `/`, not `/health`.

### `connection refused` on `http://10.102.97.179:3000`

**Cause:** NanoCtrl isn't running.

**Fix:** start it (foreground recommended):

```bash
/mnt/nvme1n1/ml_research/majinming/src/DLSlime/NanoCtrl/target/release/nanoctrl server \
  --redis-url redis://127.0.0.1:6379 --host 0.0.0.0 --port 3000
```

The `start` (background) subcommand needs write access to `/tmp/nanoctrl/`, which is currently root-owned.

## Test failures

### `tests/test_grpo_loss.py::test_vendored_matches_upstream_with_inference_logprobs SKIPPED: cannot import name 'override' from 'typing'`

**Cause:** `megatron.rl.rl_utils` imports something that requires `typing.override`, available only in Python ≥ 3.12. The host runs 3.11.

**Fix:** none needed — the *first* upstream-equivalence test passes. The vendored loss is byte-equivalent on the basic case; the inference-logprob branch is locked down by `test_inference_logprob_branch_self_consistent`. Skipping is the correct behavior.

### `tests/test_slime_rpc_loopback.py SKIPPED: NanoCtrl not reachable`

**Cause:** the fixture probes `http://127.0.0.1:3000/` and got `connection refused`.

**Fix:** start NanoCtrl (above) or set `NANORL_NANOCTRL_URL=http://your-host:port`.

## Diagnostics quick reference

```bash
# NanoCtrl alive?
curl -sf http://10.102.97.179:3000/

# Redis alive?
redis-cli -p 6379 PING

# RDMA HCAs visible?
ls /sys/class/infiniband

# Ray cluster reachable?
cat /tmp/ray/ray_current_cluster

# Lock files writable?
ls -la /tmp/ray/session_*/*.lock

# Stuck PeerAgent aliases (look in Redis for entity registrations)?
redis-cli -p 6379 KEYS '*peer*' | head
```
