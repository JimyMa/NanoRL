#!/usr/bin/env bash
# End-to-end M2 smoke test: rollout-only producer + Ray-managed consumer.
#
# Spins up a producer (NanoInfra Qwen3-4B), waits for the first round to
# finish, runs the consumer as a Ray actor, and shuts down the producer. Verifies the
# full DLSlimeRPC trajectory dataloader path: NanoInfra → publish →
# SlimeRPC → consumer → padded TrajectoryBatch.
#
# Pre-reqs: NanoCtrl running on http://10.102.97.179:3000, Redis on 6379,
# RDMA HCAs visible, and the shared Ray cluster reachable at .179:7078.
#
# Usage:
#   bash scripts/m2_smoke.sh                      # default config
#   bash scripts/m2_smoke.sh path/to/cfg.yaml     # override config
set -euo pipefail

CFG="${1:-nanorl/configs/qwen3_4b_grpo.yaml}"
PROMPTS="${PROMPTS:-nanorl/configs/sample_prompts.jsonl}"
ROUNDS="${ROUNDS:-3}"
BATCHES="${BATCHES:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONSUMER_IP="${CONSUMER_IP:-10.102.98.154}"

SUFFIX="$(date +%s)"
PROD_ALIAS="rollout:${SUFFIX}"
CONS_ALIAS="train:${SUFFIX}"
LOG_DIR="${LOG_DIR:-/tmp/nanorl_smoke}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/producer.log "$LOG_DIR"/consumer.log

echo "[smoke] cfg=$CFG"
echo "[smoke] consumer_ip=$CONSUMER_IP"
echo "[smoke] aliases producer=$PROD_ALIAS consumer=$CONS_ALIAS"
echo "[smoke] starting producer..."
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
python -m nanorl.cli rollout-only \
  --cfg "$CFG" \
  --prompts "$PROMPTS" \
  --rounds "$ROUNDS" \
  --serve-forever \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  > "$LOG_DIR/producer.log" 2>&1 &
PROD_PID=$!
trap 'kill -INT $PROD_PID 2>/dev/null || true; sleep 3; kill -KILL $PROD_PID 2>/dev/null || true' EXIT

echo "[smoke] producer pid=$PROD_PID; waiting for first round (NanoInfra startup ~90s)..."
for i in $(seq 1 180); do
  if grep -q "round=0 buffered" "$LOG_DIR/producer.log" 2>/dev/null; then
    echo "[smoke] producer ready after ${i}s"
    break
  fi
  if ! kill -0 "$PROD_PID" 2>/dev/null; then
    echo "[smoke] producer died early; tail:"
    tail -30 "$LOG_DIR/producer.log"
    exit 1
  fi
  sleep 1
done
sleep 3  # let serve_settle_s elapse on the producer

echo "[smoke] starting Ray-managed consumer..."
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
python -m nanorl.cli consume-ray \
  --cfg "$CFG" \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  --batches "$BATCHES" \
  --batch-size "$BATCH_SIZE" \
  --consumer-ip "$CONSUMER_IP" \
  > "$LOG_DIR/consumer.log" 2>&1
CONS_RC=$?

echo "[smoke] consumer exit=$CONS_RC"
echo "[smoke] === producer rounds ==="
grep -E "round=|stats|mean=" "$LOG_DIR/producer.log" | head -20
echo "[smoke] === consumer batches ==="
grep -E "Link Established|batch=|pull_batch failed|Error" "$LOG_DIR/consumer.log" | head -20
echo "[smoke] consumer.log: $(ls -lh "$LOG_DIR/consumer.log")"
exit $CONS_RC
