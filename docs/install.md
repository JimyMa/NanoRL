# Installation & pre-requisites

NanoRL is the assembly — its components must already be live. This page is the canonical checklist.

## TL;DR pre-flight

```bash
# 1. Editable install
pip install -e .

# 2. Confirm runtime deps are reachable
curl -sf http://10.102.97.179:3000/   | grep -q "NanoCtrl Server Running" && echo NanoCtrl OK
redis-cli -h 127.0.0.1 -p 6379 PING   | grep -q PONG && echo Redis OK
ls /sys/class/infiniband               | head -1 && echo HCAs visible
cat /tmp/ray/ray_current_cluster        # should print Ray address (e.g. 10.102.97.179:7078)

# 3. Sanity-test the math + RPC stack (no GPU)
pytest tests/ -q

# 4. End-to-end smokes
bash scripts/m1_smoke.sh        # single-rank DDP train + rollout (~3 min)
bash scripts/m2_smoke.sh        # rollout-only + consumer (~2 min)
bash scripts/m3_smoke.sh        # DDP train + 2 weight syncs (~3 min)
bash scripts/m3_fsdp_smoke.sh   # 2-rank FSDP train + 2 weight syncs (~5 min)
```

If all four smokes pass, your environment is good for any NanoRL workload at this scale.

## 1. Python package

```bash
cd /mnt/nvme1n1/ml_research/majinming/src/NanoRL
pip install -e .
```

This installs `nanorl` and pulls in `torch`, `ray`, `pydantic`, `PyYAML`, `numpy`, `transformers`. It does **not** install `nanodeploy` (NanoDeploy) or `dlslime` — those are already on `sys.path` on this cluster.

## 2. NanoCtrl + Redis (DLSlime control plane)

Both must be reachable from any host running a NanoRL component.

| What                    | Where                                                                             | Notes                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| NanoCtrl release binary | `/mnt/nvme1n1/ml_research/majinming/src/DLSlime/NanoCtrl/target/release/nanoctrl` | Build with `cargo build --release` if missing                               |
| Redis                   | `127.0.0.1:6379`                                                                  | Already running; `redis-cli -p 6379 PING` returns `PONG`                    |
| NanoCtrl HTTP           | `http://10.102.97.179:3000` (LAN IP, **not** `127.0.0.1`)                         | Health check: `curl http://10.102.97.179:3000/` → `NanoCtrl Server Running` |

**Why LAN IP, not localhost:** any NanoDeploy worker on a remote node must reach NanoCtrl over the network. Mixing `127.0.0.1` and `10.102.97.179` in different configs causes mailbox-MR lookups to silently fail. See `docs/troubleshooting.md`.

To bring up NanoCtrl yourself:

```bash
/mnt/nvme1n1/ml_research/majinming/src/DLSlime/NanoCtrl/target/release/nanoctrl server \
  --redis-url redis://127.0.0.1:6379 --host 0.0.0.0 --port 3000
```

There is **no** `/health` endpoint despite what `nanoctrl status` claims; probe `/`.

## 3. Ray cluster

NanoDeploy calls `ray.init(address=...)`. NanoRL itself does not require Ray.

| Field          | Value                          |
| -------------- | ------------------------------ |
| Address        | `10.102.97.179:7078`           |
| Discovery file | `/tmp/ray/ray_current_cluster` |

**Known gotcha:** the session directory `/tmp/ray/session_*/` lock files may be root-owned. Non-root users hit `PermissionError` from `ray.init`. Workaround:

```bash
sudo chmod 666 /tmp/ray/session_*/*.lock
```

Re-apply after every Ray cluster restart.

## 4. RDMA HCAs

```bash
ls /sys/class/infiniband        # mlx5_0..mlx5_7 expected
```

`tests/test_slime_rpc_loopback.py`, `tests/test_weight_manifest.py`, and the m2/m3/m3_fsdp smokes all skip when no HCAs are visible. There is no NanoRL fallback to TCP.

## 5. Models

Default config (`nanorl/configs/qwen3_4b_grpo.yaml`) points at:

| Path                                          | Use                                                             |
| --------------------------------------------- | --------------------------------------------------------------- |
| `/models/model--Qwen-Qwen3-4B-Instruct-2507/` | DDP + FSDP smoke target                                         |
| `/models/models-Qwen-Qwen3.5-35B-A3B/`        | M3+ MoE target (HF→Megatron MoE conversion not yet implemented) |
| `/models/model--Qwen--Qwen3-30B-A3B-FP8/`     | Different architecture from 35B-A3B; FP8 path deferred          |

Each must be a directory with `config.json`, `tokenizer.json`, and `model*.safetensors`. NanoRL pre-flights this in `RolloutEngine.__init__` so a typo'd path fails in seconds, not after a 60 s NanoDeploy startup.

## 6. GPUs

- Rollout: 4 GPUs on the host pinned by `infer.master_address` (default `10.102.97.183:6006`) for the bundled `attention_tp=4 ffn_tp=4` config.
- Train (DDP single-rank): 1 GPU on the Ray node selected by `--train-ip`.
- Train (FSDP 2-rank): 2 GPUs on the Ray node selected by `--train-ip`.

A 2-rank FSDP run of Qwen3-4B with the bundled config uses ~50 GB per train GPU.
