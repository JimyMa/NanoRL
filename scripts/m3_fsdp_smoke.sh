#!/usr/bin/env bash
# Multi-rank FSDP-2 trainer + single-process rollout, end-to-end smoke.
#
# Identical sequence to scripts/m3_smoke.sh but launches the trainer via
# Ray with NPROC TrainActor ranks. Each rank wraps the
# GPTModel in MegatronFSDP (zero_dp_strategy=optim_grads_params), so the
# 4B model + Adam state + grad buffer is sharded across ranks. With NPROC=2
# the per-rank memory drops from ~138 GB (single-rank OOM) to ~70 GB.
#
# Pre-reqs:
#   * cfg.train.fsdp = true (default in qwen3_4b_grpo_fsdp.yaml)
#   * Same NanoCtrl + Redis + RDMA + Ray-cluster expectations as M3 smoke
#   * Each rank uses one GPU allocated by Ray
#
# Usage:
#   bash scripts/m3_fsdp_smoke.sh                       # 2-GPU default
#   NPROC=4 TRAIN_IP=10.102.98.154 bash scripts/m3_fsdp_smoke.sh
set -euo pipefail

CFG="${1:-nanorl/configs/qwen3_4b_grpo_fsdp.yaml}"
PROMPTS="${PROMPTS:-dapo}"
EVAL_PROMPTS="${EVAL_PROMPTS:-aime}"
EVAL_EVERY="${EVAL_EVERY:-20}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
STEPS="${STEPS:-2000}"
SYNC_EVERY="${SYNC_EVERY:-10}"
ROUNDS="${ROUNDS:-1000}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-128}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29600}"
TRAIN_IP="${TRAIN_IP:-10.102.98.154}"
REDIS_URL="${REDIS_URL:-redis://10.102.97.179:6379/0}"
REDIS_KEY="${REDIS_KEY:-nanorl:trajectories}"

SUFFIX="$(date +%s)"
PROD_ALIAS="rollout:m3fsdp-${SUFFIX}"
CONS_ALIAS="train:m3fsdp-${SUFFIX}"
LOG_DIR="${LOG_DIR:-/tmp/nanorl_smoke}"
SAVE_DIR="${SAVE_DIR:-}"
SAVE_EVERY="${SAVE_EVERY:-0}"
SAVE_FINAL="${SAVE_FINAL:-0}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/m3_fsdp_*.log "$LOG_DIR"/m3_fsdp_train.jsonl "$LOG_DIR"/m3_fsdp_trajectories.jsonl "$LOG_DIR"/m3_fsdp_eval.jsonl
rm -rf "$LOG_DIR/m3_fsdp_tb"

echo "[m3-fsdp-smoke] cfg=$CFG steps=$STEPS sync_every=$SYNC_EVERY nproc=$NPROC train_ip=$TRAIN_IP"
echo "[m3-fsdp-smoke] prompts=$PROMPTS eval_prompts=$EVAL_PROMPTS eval_every=$EVAL_EVERY"
echo "[m3-fsdp-smoke] aliases producer=$PROD_ALIAS consumer=$CONS_ALIAS"
if [[ -n "$SAVE_DIR" ]]; then
  echo "[m3-fsdp-smoke] checkpoints save_dir=$SAVE_DIR save_every=$SAVE_EVERY save_final=$SAVE_FINAL"
fi

# Cleanup that runs on EXIT, SIGINT, or SIGTERM. ``setsid`` puts each
# subprocess in its own process group so we can kill local driver processes.
# Ray actors live in the cluster and are cleaned up by the driver/placement
# group path; ``pkill`` is a belt-and-suspenders fallback for orphan drivers.
PROD_PID=""
TRAIN_PID=""
cleanup() {
  local rc=$?
  echo "[m3-fsdp-smoke] cleanup (rc=$rc)..."
  for pid in "$TRAIN_PID" "$PROD_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "$TRAIN_PID" "$PROD_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  pkill -KILL -f "nanorl.cli (rollout-only|train)" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

echo "[m3-fsdp-smoke] starting rollout producer (NanoInfra startup ~90s)..."
# DLSLIME_MR_WAIT_TIMEOUT_S: producer-side wait for the trainer's proxy
# mailbox MR. With 8-rank FSDP + cold HF safetensors load, the trainer
# can take 5-8 minutes to reach _build_trajectory_client; 900s (15 min)
# gives generous headroom. Bump higher (1800, 3600) for slow disks /
# larger models if needed.
DLSLIME_MR_WAIT_TIMEOUT_S="${DLSLIME_MR_WAIT_TIMEOUT_S:-900}" \
DLSLIME_TIMING="${DLSLIME_TIMING:-1}" \
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
setsid python -m nanorl.cli rollout-only \
  --cfg "$CFG" \
  --prompts "$PROMPTS" \
  --rounds "$ROUNDS" \
  --limit-prompts "$LIMIT_PROMPTS" \
  --serve-forever \
  --save-jsonl "$LOG_DIR/m3_fsdp_trajectories.jsonl" \
  --print-traj-every 1 \
  --print-traj-n 2 \
  --eval-prompts "$EVAL_PROMPTS" \
  --eval-every "$EVAL_EVERY" \
  --eval-limit-prompts "$EVAL_LIMIT" \
  --eval-jsonl "$LOG_DIR/m3_fsdp_eval.jsonl" \
  --redis-url "$REDIS_URL" \
  --redis-key "$REDIS_KEY" \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  > "$LOG_DIR/m3_fsdp_producer.log" 2>&1 &
PROD_PID=$!

echo "[m3-fsdp-smoke] producer pid=$PROD_PID; waiting for first round..."
for i in $(seq 1 180); do
  if grep -q "round=0 buffered" "$LOG_DIR/m3_fsdp_producer.log" 2>/dev/null; then
    echo "[m3-fsdp-smoke] producer ready after ${i}s"
    break
  fi
  if ! kill -0 "$PROD_PID" 2>/dev/null; then
    echo "[m3-fsdp-smoke] producer died early; tail:"
    tail -30 "$LOG_DIR/m3_fsdp_producer.log"
    exit 1
  fi
  sleep 1
done
sleep 3

echo "[m3-fsdp-smoke] launching Ray TrainActors on $TRAIN_IP nproc=$NPROC"
TRAIN_RC=0
SAVE_ARGS=()
if [[ -n "$SAVE_DIR" ]]; then
  SAVE_ARGS+=(--save-dir "$SAVE_DIR" --save-every "$SAVE_EVERY")
  if [[ "$SAVE_FINAL" == "1" || "$SAVE_FINAL" == "true" ]]; then
    SAVE_ARGS+=(--save-final)
  fi
fi
PYTHONPATH=/mnt/nvme1n1/ml_research/majinming/src/Megatron-LM \
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
setsid python -m nanorl.cli train-ray \
    --cfg "$CFG" \
    --steps "$STEPS" \
    --weight-sync-every "$SYNC_EVERY" \
    --producer-alias "$PROD_ALIAS" \
    --consumer-alias "$CONS_ALIAS" \
    --master-port "$PORT" \
    --train-ip "$TRAIN_IP" \
    --nproc "$NPROC" \
    --log-jsonl "$LOG_DIR/m3_fsdp_train.jsonl" \
    --tb-dir "$LOG_DIR/m3_fsdp_tb" \
    "${SAVE_ARGS[@]}" \
  > "$LOG_DIR/m3_fsdp_train.log" 2>&1 &
TRAIN_PID=$!
wait "$TRAIN_PID" || TRAIN_RC=$?

echo "[m3-fsdp-smoke] trainer exit=$TRAIN_RC"
echo "[m3-fsdp-smoke] === train log tail ==="
tail -40 "$LOG_DIR/m3_fsdp_train.log"
echo "[m3-fsdp-smoke] === train.jsonl ==="
if [[ -f "$LOG_DIR/m3_fsdp_train.jsonl" ]]; then
  cat "$LOG_DIR/m3_fsdp_train.jsonl"
fi

if [[ -f "$LOG_DIR/m3_fsdp_trajectories.jsonl" ]]; then
  echo "[m3-fsdp-smoke] === trajectory dump summary ==="
  python - <<PY
import json
from collections import Counter
path = "$LOG_DIR/m3_fsdp_trajectories.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
if not rows:
    print("(no trajectories dumped)")
else:
    rewards = [r["reward"] for r in rows]
    lens = [r.get("response_len", 0) for r in rows]
    eos = sum(1 for r in rows if r.get("eos"))
    print(f"trajectories={len(rows)} eos={eos} mean_reward={sum(rewards)/len(rewards):.3f} "
          f"mean_resp_len={sum(lens)/len(lens):.0f} resp_len_max={max(lens)}")
    print(f"reward histogram: {Counter(rewards)}")
PY
fi

if [[ -f "$LOG_DIR/m3_fsdp_eval.jsonl" ]]; then
  echo "[m3-fsdp-smoke] === held-out eval trajectory ==="
  python - <<PY
import json
path = "$LOG_DIR/m3_fsdp_eval.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
if not rows:
    print("(no eval rows)")
else:
    print(f"{'round':>6} {'n':>4} {'mean_reward':>12} {'pos_rate':>10}")
    for r in rows:
        print(f"{r['round']:>6} {r['n']:>4} {r['mean_reward']:>12.4f} "
              f"{r['pos_count']/max(1,r['n']):>10.2%}")
    if len(rows) >= 2:
        d = rows[-1]['mean_reward'] - rows[0]['mean_reward']
        print(f"first→last delta: {d:+.4f}")
PY
fi

if [[ -f "$LOG_DIR/m3_fsdp_train.jsonl" ]]; then
  python - <<PY
import json, math, sys
log = "$LOG_DIR/m3_fsdp_train.jsonl"
losses, syncs = [], []
for line in open(log):
    if not line.strip(): continue
    d = json.loads(line)
    if d.get("event") == "weight_sync":
        syncs.append(d)
    else:
        losses.append(d)
if not losses:
    print("FAIL: no train steps logged"); sys.exit(2)
if not syncs:
    print(f"FAIL: 0 weight syncs across {len(losses)} steps"); sys.exit(2)
for d in losses:
    if not math.isfinite(d['loss']):
        print(f"FAIL: NaN loss at step {d['step']}"); sys.exit(2)
print(f"PASS: {len(losses)} train steps, {len(syncs)} weight syncs, "
      f"avg manifest size={sum(s['n_tensors'] for s in syncs)/len(syncs):.0f} tensors")
PY
  RC=$?
  echo "[m3-fsdp-smoke] verifier exit=$RC"
  exit $RC
else
  echo "[m3-fsdp-smoke] FAIL: no train.jsonl produced"
  exit 2
fi
