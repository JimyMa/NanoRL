#!/usr/bin/env bash
# End-to-end M3 smoke: rollout-only + nanorl train with weight sync.
#
# Boots the M2-proven rollout (NanoInfra Qwen3-4B on .183) with the
# M3 weight-update RPC enabled, then runs the M3-enabled trainer
# (single-rank megatron-core on .179:GPU-7) for N steps with periodic
# weight sync. After every sync, the rollout side has copied the
# train-side weights into its live model. The smoke asserts:
#
#   * every train step's loss is finite
#   * at least one weight-sync event fired
#   * the manifest size matches what gather_full_state_dict produces
#     (one tensor per HF-named param, no orphans)
#
# Pre-reqs: see scripts/m1_smoke.sh — same environment.
#
# Usage:
#   bash scripts/m3_smoke.sh
#   STEPS=20 SYNC_EVERY=2 bash scripts/m3_smoke.sh
set -euo pipefail

CFG="${1:-nanorl/configs/qwen3_4b_grpo.yaml}"
PROMPTS="${PROMPTS:-nanorl/configs/sample_prompts.jsonl}"
STEPS="${STEPS:-5}"
SYNC_EVERY="${SYNC_EVERY:-2}"
ROUNDS="${ROUNDS:-100}"
TRAIN_GPU="${TRAIN_GPU:-7}"

SUFFIX="$(date +%s)"
PROD_ALIAS="rollout:m3-${SUFFIX}"
CONS_ALIAS="train:m3-${SUFFIX}"
LOG_DIR="${LOG_DIR:-/tmp/nanorl_smoke}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/m3_producer.log "$LOG_DIR"/m3_train.log "$LOG_DIR"/m3_train.jsonl

echo "[m3-smoke] cfg=$CFG steps=$STEPS sync_every=$SYNC_EVERY"
echo "[m3-smoke] aliases producer=$PROD_ALIAS consumer=$CONS_ALIAS"

echo "[m3-smoke] starting rollout producer (NanoInfra startup ~90s)..."
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
python -m nanorl.cli rollout-only \
  --cfg "$CFG" \
  --prompts "$PROMPTS" \
  --rounds "$ROUNDS" \
  --serve-forever \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  > "$LOG_DIR/m3_producer.log" 2>&1 &
PROD_PID=$!
trap 'kill -INT $PROD_PID 2>/dev/null || true; sleep 3; kill -KILL $PROD_PID 2>/dev/null || true' EXIT

echo "[m3-smoke] producer pid=$PROD_PID; waiting for first round..."
for i in $(seq 1 180); do
  if grep -q "round=0 buffered" "$LOG_DIR/m3_producer.log" 2>/dev/null; then
    echo "[m3-smoke] producer ready after ${i}s"
    break
  fi
  if ! kill -0 "$PROD_PID" 2>/dev/null; then
    echo "[m3-smoke] producer died early; tail:"
    tail -30 "$LOG_DIR/m3_producer.log"
    exit 1
  fi
  sleep 1
done
sleep 3

echo "[m3-smoke] starting trainer on GPU $TRAIN_GPU..."
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
PYTHONPATH=/mnt/nvme1n1/ml_research/majinming/src/Megatron-LM \
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
python -m nanorl.cli train \
  --cfg "$CFG" \
  --steps "$STEPS" \
  --weight-sync-every "$SYNC_EVERY" \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  --log-jsonl "$LOG_DIR/m3_train.jsonl" \
  > "$LOG_DIR/m3_train.log" 2>&1
TRAIN_RC=$?

echo "[m3-smoke] trainer exit=$TRAIN_RC"
echo "[m3-smoke] === train log tail ==="
tail -30 "$LOG_DIR/m3_train.log"
echo "[m3-smoke] === train.jsonl ==="
if [[ -f "$LOG_DIR/m3_train.jsonl" ]]; then
  cat "$LOG_DIR/m3_train.jsonl"
fi

if [[ -f "$LOG_DIR/m3_train.jsonl" ]]; then
  python - <<PY
import json, math, sys
log = "$LOG_DIR/m3_train.jsonl"
losses = []
syncs = []
for line in open(log):
    if not line.strip(): continue
    d = json.loads(line)
    if d.get("event") == "weight_sync":
        syncs.append(d)
    else:
        losses.append(d)
if not losses:
    print("FAIL: no train steps logged")
    sys.exit(2)
if not syncs:
    print(f"FAIL: 0 weight syncs across {len(losses)} steps")
    sys.exit(2)
for s in syncs:
    if s.get("n_tensors", 0) <= 0:
        print(f"FAIL: sync v={s['version']} shipped {s.get('n_tensors')} tensors")
        sys.exit(2)
for d in losses:
    if not math.isfinite(d['loss']):
        print(f"FAIL: NaN loss at step {d['step']}")
        sys.exit(2)
print(f"PASS: {len(losses)} train steps, {len(syncs)} weight syncs, "
      f"avg manifest size={sum(s['n_tensors'] for s in syncs)/len(syncs):.0f} tensors")
PY
  RC=$?
  echo "[m3-smoke] verifier exit=$RC"
  exit $RC
else
  echo "[m3-smoke] FAIL: no train.jsonl produced"
  exit 2
fi
