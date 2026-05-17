# Troubleshooting

Most NanoRL failures live at the seams between Ray, NanoDeploy, NanoCtrl, and DLSlime. This page is the playbook for failures we have actually hit.

## Rollout / NanoDeploy startup

### `PermissionError: '/tmp/ray/session_*/node_ip_address.json.lock'`

**Cause:** Ray cluster session dir lock files are root-owned. As a non-root user, `ray.init(address=...)` cannot acquire the flock.

**Fix:**

```bash
sudo chmod 666 /tmp/ray/session_*/*.lock
```

Re-apply after every Ray cluster restart.

### `FileNotFoundError: model dir does not exist`

**Cause:** model path in `cfg.model.hf_path` doesn't exist. `RolloutEngine.__init__` pre-flights this so we fail in seconds, not after a 60 s startup.

### NanoDeploy hangs with `Failed to connect to GCS at address ...:6379`

**Cause:** NanoDeploy defaulted `ray_address` to `127.0.0.1:6379`, but `6379` is **Redis** here (Ray is on `7078`).

**Fix:** set `infer.ray_address: 10.102.97.179:7078` and `infer.master_address: 10.102.97.183:6006` in YAML.

## SlimeRPC / DLSlime

### Producer's serve thread crashes with `ValueError: requires a connected endpoint`

**Cause:** `serve()` was called before the QP transitioned to RTS.

**Fix already in code:** `nanorl/data/trajectory_buffer.py:run_rpc_server` accepts the `Connection` object and waits on it inside the serve thread before binding the mailbox.

### Consumer's `proxy()` raises `Timeout waiting for MR 'rpc:mailbox:...'`

**Cause:** producer's `serve()` hasn't run yet (mailbox MR not registered), or the serve thread silently died.

**Fix:** check producer log for `trajectory rpc server thread started`. If absent, look earlier for `connection wait to ... failed` or a Python traceback.

### `RDMA read/write completion failed` / `IBV_WC_RETRY_EXC_ERR` (Vendor Err 129)

**Cause:** receiver hadn't posted recv WRs by the time the sender's first WR arrived.

**Fix already in code:** producer adds `serve_settle_s=0.2` after `connection.wait()`; consumer adds `initial_settle_s=0.2` before the first pull.

### `409 Conflict: Peer agent alias 'X' already exists in this scope`

**Cause:** previous run was `kill -9`'d; alias is still registered until heartbeat TTL.

**Fixes (cheapest first):**

1. Use unique aliases per run (timestamp suffix). All `m*_smoke.sh` scripts already do this.
2. Wait ~30 s for TTL.
3. Manually evict: `curl -X POST http://10.102.97.179:3000/cleanup -d '{"agent_alias": "rollout:0"}' -H 'content-type: application/json'`.

## NanoCtrl / Redis

### NanoCtrl `/health` returns 404

**Cause:** there is no `/health` endpoint despite what `nanoctrl status` claims. Probe `/` instead — returns `NanoCtrl Server Running`.

### `connection refused` on `http://10.102.97.179:3000`

**Cause:** NanoCtrl isn't running.

**Fix:**

```bash
/mnt/nvme1n1/ml_research/majinming/src/DLSlime/NanoCtrl/target/release/nanoctrl server \
  --redis-url redis://127.0.0.1:6379 --host 0.0.0.0 --port 3000
```

The `start` (background) subcommand needs write access to `/tmp/nanoctrl/`, which is currently root-owned.

## M3 weight sync

### Rollout side reports `apply_weight_update` worked but generation didn't change

This shouldn't happen with the current `pull_and_apply_weights` path — every NanoDeploy worker logs its `loaded` count. Check:

- All 4 workers report `loaded: 218` (or whatever the per-rank-shard count is for your TP layout). If a worker reports `loaded: 0`, its PeerAgent didn't reach the train side.
- The manifest's `train_alias` matches the train PeerAgent's actual alias (rank-0 only).
- `param.data.copy_` was used (not `param.data = ...`), preserving the CUDA-graph addresses. The path uses the parameter's bound `weight_loader` which always uses `.copy_`.

`scripts/sanity_apply_weight_update.py` reproduces "noop apply preserves greedy decode; zero-out apply changes it" in ~30 s — a known-good baseline.

### M3 sync wall is 60 s, not 5 s

You're on the slow path. The fast path requires the NanoDeploy patches in `NanoDeploy/nanodeploy/worker/pull_weights.py` and the `LLMEngine.pull_and_apply_weights` method. If `LLMComponent.pull_and_apply_weights` is missing, the rollout falls back to the slow `update_weights` (Ray fan-out of full dict).

## Off-policy logprobs

### Train logs show `old_lp=False` or ratios stay exactly `1.000`

**Cause:** rollout did not deliver `Trajectory.response_logprobs`, so TrainActor fell back to `current_logprobs.detach()`.

**Fixes:**

1. Make sure rollout was not launched with `--no-ship-logprobs`.
2. Check `sampling.ship_logprobs: true` in YAML.
3. Check rollout logs for `rollout logprobs: N/N completions carried logprobs`.
4. If it says `0/N`, rebuild NanoDeploy with the `return_completion_logprobs` / sampler logprob patch.

### `logprob_to_old_mean` is around 0.5+ immediately after sync

**Cause:** training forward is stochastic. We hit this when Qwen dropout was nonzero in the Megatron `TransformerConfig`: rollout/HF eval logprobs and Megatron train-mode logprobs disagreed even at identical weights.

**Fix already in code:** `nanorl/weights/hf_to_megatron.py:build_transformer_config` sets both `hidden_dropout=0.0` and `attention_dropout=0.0`. With dropout off, post-sync `logprob_to_old_mean` should usually live around `0.1-0.2` for the current Qwen3-4B runs.

### `kl_mean` is enormous but `kl_to_old` looks reasonable

`kl_mean` is reference-model KL (`rl.kl_beta` path). `kl_to_old` is the rollout-policy distance computed from `old_logprobs` and is the useful off-policy monitor in the current setup. Keep `rl.kl_beta: 0.0` until the SDPA kernel-parity issue below is fixed.

## FSDP

### `ValueError: not enough values to unpack (expected 2, got 1)` from `_import_class_from_path`

**Cause:** `fsdp_unit_modules` got a bare class name. Megatron-FSDP expects a fully-qualified module path.

**Fix:** `["megatron.core.transformer.transformer_layer.TransformerLayer"]`, not `["TransformerLayer"]`.

### Weight gather produces tensors with wrong shape after `full_tensor()`

**Cause:** Megatron-FSDP uses an *uneven DTensor* — `DTensor.full_tensor()` returns each rank's PADDED shard concatenated, not the global view. For `linear_qkv.weight` declared `[6144, 2560]` you get `[11856, 2560]` on rank 0 and `[432, 2560]` on rank 1.

**Fix:** use `megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor.uneven_dtensor_to_full_tensor(dtensor)` instead of `dtensor.full_tensor()`. NanoRL does this in `nanorl/weights/megatron_to_hf.py:_materialize`.

`scripts/diag_fsdp_full_tensor.py` reproduces the per-rank shape mismatch in ~30 s for sanity-checking.

### KL term blows up to 1e8 when `kl_beta > 0`

**Cause:** PyTorch's gradient-mode SDPA picks different attention kernels than no_grad mode in BF16. The trainable model's gradient-mode forward and the reference model's no_grad forward differ by ~5 in logprob space *for the same weights and same input*. `exp(5) - 5 - 1` ≈ 142, scaled across response tokens → tens of thousands.

**Fix (current):** `kl_beta=0` until SDPA kernel parity is solved. Reference model and gather logic are wired correctly; only the value path is broken.

**Future fix:** wrap both forwards in `with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True):` to pin a deterministic backend, then re-enable `kl_beta`.

`scripts/diag_train_vs_ref.py` reproduces the kernel-mode logit drift in ~30 s.

## Tests

### `tests/test_grpo_loss.py::test_vendored_matches_upstream_with_inference_logprobs SKIPPED: cannot import name 'override' from 'typing'`

**Cause:** `megatron.rl.rl_utils` imports something requiring `typing.override`, available only in Python ≥ 3.12. Host runs 3.11.

**Fix:** none needed — the basic-case equivalence test passes. The vendored loss is byte-equal on the unsharded path; the inference-logprob branch is locked down by `test_inference_logprob_branch_self_consistent`.

### `tests/test_slime_rpc_loopback.py SKIPPED: NanoCtrl not reachable`

**Cause:** fixture probed `http://127.0.0.1:3000/` and got connection refused.

**Fix:** start NanoCtrl (above) or set `NANORL_NANOCTRL_URL=http://your-host:port`.

## Diagnostics quick reference

```bash
curl -sf http://10.102.97.179:3000/        # NanoCtrl alive?
redis-cli -p 6379 PING                     # Redis alive?
ls /sys/class/infiniband                   # RDMA HCAs visible?
cat /tmp/ray/ray_current_cluster           # Ray cluster reachable?
ls -la /tmp/ray/session_*/*.lock           # Lock files writable?
redis-cli -p 6379 KEYS '*peer*' | head     # Stuck PeerAgent aliases
```
