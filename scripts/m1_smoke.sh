#!/usr/bin/env bash
# End-to-end M1 smoke test: rollout-only producer + train-only consumer.
#
# Boots a NanoInfra rollout (Qwen3-4B on .183 via M2-proven path) and a
# single-rank megatron-core TrainActor (on local .179 GPU). The train side
# pulls real trajectories over SlimeRPC, runs N GRPO steps with kl_beta=0,
# and asserts every step loss is finite.
#
# Pre-reqs (see docs/install.md):
#   * NanoCtrl on http://10.102.97.179:3000, Redis on 6379
#   * Ray cluster reachable at 10.102.97.179:7078 (locks chmod'd if needed)
#   * RDMA HCAs visible on the host running this script
#   * 4 GPUs on .183 for NanoInfra (TP=4) + 1 GPU on .179 for the trainer
#
# Usage:
#   bash scripts/m1_smoke.sh                          # default 10 steps
#   STEPS=20 bash scripts/m1_smoke.sh                 # more steps
#   bash scripts/m1_smoke.sh path/to/cfg.yaml         # different config
set -euo pipefail

CFG="${1:-nanorl/configs/qwen3_4b_grpo.yaml}"
PROMPTS="${PROMPTS:-nanorl/configs/sample_prompts.jsonl}"
STEPS="${STEPS:-10}"
ROUNDS="${ROUNDS:-100}"
TRAIN_GPU="${TRAIN_GPU:-7}"

SUFFIX="$(date +%s)"
PROD_ALIAS="rollout:m1-${SUFFIX}"
CONS_ALIAS="train:m1-${SUFFIX}"
LOG_DIR="${LOG_DIR:-/tmp/nanorl_smoke}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/m1_producer.log "$LOG_DIR"/m1_train.log "$LOG_DIR"/m1_train.jsonl

echo "[m1-smoke] cfg=$CFG steps=$STEPS"
echo "[m1-smoke] aliases producer=$PROD_ALIAS consumer=$CONS_ALIAS"

echo "[m1-smoke] starting rollout producer (NanoInfra startup ~90s)..."
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
python -m nanorl.cli rollout-only \
  --cfg "$CFG" \
  --prompts "$PROMPTS" \
  --rounds "$ROUNDS" \
  --serve-forever \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  > "$LOG_DIR/m1_producer.log" 2>&1 &
PROD_PID=$!
trap 'kill -INT $PROD_PID 2>/dev/null || true; sleep 3; kill -KILL $PROD_PID 2>/dev/null || true' EXIT

echo "[m1-smoke] producer pid=$PROD_PID; waiting for first round..."
for i in $(seq 1 180); do
  if grep -q "round=0 buffered" "$LOG_DIR/m1_producer.log" 2>/dev/null; then
    echo "[m1-smoke] producer ready after ${i}s"
    break
  fi
  if ! kill -0 "$PROD_PID" 2>/dev/null; then
    echo "[m1-smoke] producer died early; tail:"
    tail -30 "$LOG_DIR/m1_producer.log"
    exit 1
  fi
  sleep 1
done
sleep 3   # producer's serve_settle_s

echo "[m1-smoke] starting trainer on GPU $TRAIN_GPU..."
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
PYTHONPATH=/mnt/nvme1n1/ml_research/majinming/src/Megatron-LM \
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
python -m nanorl.cli train-only \
  --cfg "$CFG" \
  --steps "$STEPS" \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  --log-jsonl "$LOG_DIR/m1_train.jsonl" \
  > "$LOG_DIR/m1_train.log" 2>&1
TRAIN_RC=$?

echo "[m1-smoke] trainer exit=$TRAIN_RC"
echo "[m1-smoke] === train log tail ==="
tail -25 "$LOG_DIR/m1_train.log"
echo "[m1-smoke] === train.jsonl ==="
if [[ -f "$LOG_DIR/m1_train.jsonl" ]]; then
  cat "$LOG_DIR/m1_train.jsonl"
fi

# Sanity checks on the JSONL: every step's loss must be finite.
if [[ -f "$LOG_DIR/m1_train.jsonl" ]]; then
  if python -c "
import json, math, sys
n = 0
for line in open('$LOG_DIR/m1_train.jsonl'):
    if not line.strip(): continue
    d = json.loads(line)
    n += 1
    if not math.isfinite(d['loss']):
        print(f'NOT FINITE at step {d[\"step\"]}: loss={d[\"loss\"]}')
        sys.exit(2)
print(f'all {n} losses finite')
"; then
    echo "[m1-smoke] PASS"
  else
    echo "[m1-smoke] FAIL: non-finite loss"
    exit 2
  fi
else
  echo "[m1-smoke] FAIL: no train.jsonl produced"
  exit 2
fi
