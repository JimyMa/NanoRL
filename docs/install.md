# Installation & pre-requisites

NanoRL is the assembly. The components it drives must already be live on the cluster — none of them are auto-installed by `pip install -e .`. This page is the canonical checklist.

## TL;DR pre-flight

```bash
# 1. Editable install
pip install -e .

# 2. Confirm runtime deps are reachable
curl -sf http://10.102.97.179:3000/   | grep -q "NanoCtrl Server Running" && echo NanoCtrl OK
redis-cli -h 127.0.0.1 -p 6379 PING   | grep -q PONG && echo Redis OK
ls /sys/class/infiniband               | head -1 && echo HCAs visible
ls /tmp/ray/ray_current_cluster        && echo Ray cluster file present

# 3. Sanity-test the math + RPC stack (no GPU)
pytest tests/ -q

# 4. End-to-end smoke (~2 minutes; uses GPUs on .183)
bash scripts/m2_smoke.sh
```

If all four steps pass, the rollout-only path will work. If any fails, see below.

## 1. Python package

```bash
cd /mnt/nvme1n1/ml_research/majinming/src/NanoRL
pip install -e .
```

This installs `nanorl` and pulls in `torch`, `ray`, `pydantic`, `PyYAML`, `numpy`, `transformers`. It does **not** install `nanodeploy` (NanoInfra) or `dlslime` — those must already be on the system Python path (they are, on this cluster).

Optional extras for tests: `pip install -e '.[test]'` adds `pytest`.

## 2. NanoCtrl + Redis (DLSlime control plane)

DLSlime peers register with NanoCtrl, which uses Redis as the registry. Both must be reachable from any host running a NanoRL component.

| What                    | Where                                                                             | Notes                                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| NanoCtrl release binary | `/mnt/nvme1n1/ml_research/majinming/src/DLSlime/NanoCtrl/target/release/nanoctrl` | Build with `cargo build --release` if missing                                                                               |
| Redis                   | `127.0.0.1:6379`                                                                  | Already running on this cluster; `redis-server` binary not on PATH but the daemon is reachable via `redis-cli -p 6379 PING` |
| NanoCtrl HTTP           | `http://10.102.97.179:3000` (LAN IP, **not** `127.0.0.1`)                         | Health check: `curl http://10.102.97.179:3000/` should return `NanoCtrl Server Running`                                     |

**Why LAN IP, not localhost:** any NanoInfra worker on a remote node must reach NanoCtrl over the network. Using `127.0.0.1` in one config and `10.102.97.179` in another causes mailbox-MR lookup to silently fail in the consumer. See `docs/troubleshooting.md`.

To bring up NanoCtrl yourself (foreground, owned by your user):

```bash
/mnt/nvme1n1/ml_research/majinming/src/DLSlime/NanoCtrl/target/release/nanoctrl server \
  --redis-url redis://127.0.0.1:6379 \
  --host 0.0.0.0 \
  --port 3000
```

The `start` (background) subcommand uses `/tmp/nanoctrl/`, which is currently root-owned on this cluster — prefer the foreground `server` command or `nohup ... &`.

## 3. RDMA HCAs

```bash
ls /sys/class/infiniband        # should show mlx5_0, mlx5_1, ...
```

Tests using `dlslime.PeerAgent` (the loopback test, the M2 smoke) will skip if no HCAs are visible. There is no NanoRL fallback to TCP.

## 4. Ray cluster

NanoInfra's `LLM` calls `ray.init(address=...)`; NanoRL itself does not require Ray. The shared cluster on this host:

| Field          | Value                              |
| -------------- | ---------------------------------- |
| Address        | `10.102.97.179:7078`               |
| Discovery file | `/tmp/ray/ray_current_cluster`     |
| Owned by       | root (started outside our process) |

**Known gotcha:** the session directory `/tmp/ray/session_*/` has root-owned `*.lock` files. As a non-root user, `ray.init(address="10.102.97.179:7078", ...)` raises `PermissionError`. Workaround:

```bash
sudo chmod 666 /tmp/ray/session_*/*.lock
```

Re-apply after every Ray cluster restart (the session dir name changes). See `docs/troubleshooting.md` for alternatives.

## 5. Models

Default config (`nanorl/configs/qwen3_4b_grpo.yaml`) points at:

| Path                                          | Use                                                            |
| --------------------------------------------- | -------------------------------------------------------------- |
| `/models/model--Qwen-Qwen3-4B-Instruct-2507/` | M2 smoke test target                                           |
| `/models/models-Qwen-Qwen3.5-35B-A3B/`        | Long-term M3 target (MoE; HF→Megatron conversion is M3 work)   |
| `/models/model--Qwen--Qwen3-30B-A3B-FP8/`     | Different architecture from 35B-A3B; **not** an FP8 cast of it |

Each must be a directory containing `config.json`, `tokenizer.json`, and `model*.safetensors`. NanoRL pre-flights this in `RolloutEngine.__init__` so a typo'd path fails in seconds, not after a 60s NanoInfra startup.

## 6. GPUs

`nvidia-smi` should show H200s (or equivalent). For the Qwen3-4B smoke run with `attention_tp=4 ffn_tp=4`, NanoInfra needs 4 GPUs on the configured `master_address` host.
