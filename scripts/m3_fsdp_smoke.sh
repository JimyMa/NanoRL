#!/usr/bin/env bash
# Skeleton for launching a multi-rank FSDP TrainActor via torchrun.
#
# Currently UNTESTED — committed as a starting point so the next person
# (or future session) doesn't have to rediscover the env-var contract.
# The TrainActor reads RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR /
# MASTER_PORT from the environment; torchrun sets all of them.
#
# Pre-reqs:
#   * cfg.train.fsdp = true in the YAML
#   * cfg.train.world_size matches NPROC (sanity check)
#   * Same NanoCtrl + Redis + RDMA + Ray-cluster expectations as M3 smoke
#   * Each rank uses one GPU; CUDA_VISIBLE_DEVICES set per-rank by torchrun
#
# Usage:
#   bash scripts/m3_fsdp_smoke.sh                      # 2-GPU default
#   NPROC=4 bash scripts/m3_fsdp_smoke.sh              # 4-GPU
set -euo pipefail

CFG="${1:-nanorl/configs/qwen3_4b_grpo_fsdp.yaml}"
NPROC="${NPROC:-2}"
PORT="${PORT:-29600}"
TRAIN_GPUS="${TRAIN_GPUS:-6,7}"

LOG_DIR="${LOG_DIR:-/tmp/nanorl_smoke}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/m3_fsdp_*.log "$LOG_DIR"/m3_fsdp_train.jsonl

# Boot the rollout-only producer (one process; same as M3 smoke).
SUFFIX="$(date +%s)"
PROD_ALIAS="rollout:m3fsdp-${SUFFIX}"
CONS_ALIAS="train:m3fsdp-${SUFFIX}"

NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
python -m nanorl.cli rollout-only \
  --cfg "$CFG" \
  --prompts nanorl/configs/sample_prompts.jsonl \
  --rounds 100 --serve-forever \
  --producer-alias "$PROD_ALIAS" --consumer-alias "$CONS_ALIAS" \
  > "$LOG_DIR/m3_fsdp_producer.log" 2>&1 &
PROD_PID=$!
trap 'kill -INT $PROD_PID 2>/dev/null || true; sleep 3; kill -KILL $PROD_PID 2>/dev/null || true' EXIT

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

# Launch NPROC ranks of `nanorl train`. Each rank gets one GPU 0..NPROC-1.
# Note: only rank 0 issues the weight-sync RPC; non-zero ranks return
# None from gather_and_publish but participate in the all-gather.
echo "[m3-fsdp-smoke] launching torchrun --nproc_per_node=$NPROC on GPUs $TRAIN_GPUS"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
PYTHONPATH=/mnt/nvme1n1/ml_research/majinming/src/Megatron-LM \
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
  -m nanorl.cli train \
    --cfg "$CFG" --steps 5 --weight-sync-every 2 \
    --producer-alias "$PROD_ALIAS" --consumer-alias "$CONS_ALIAS" \
    --log-jsonl "$LOG_DIR/m3_fsdp_train.jsonl" \
  2>&1 | tee "$LOG_DIR/m3_fsdp_train.log"
TRAIN_RC=${PIPESTATUS[0]}
echo "[m3-fsdp-smoke] trainer exit=$TRAIN_RC"
exit $TRAIN_RC
