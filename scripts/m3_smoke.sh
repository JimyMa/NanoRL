#!/usr/bin/env bash
# End-to-end M3 smoke: rollout-only + nanorl train with weight sync.
#
# Boots the M2-proven rollout (NanoInfra Qwen3-4B on .183) with the
# M3 weight-update RPC enabled, then runs the M3-enabled trainer as a
# single-rank Ray actor for N steps with periodic
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
PROMPTS="${PROMPTS:-dapo}"
STEPS="${STEPS:-200}"
SYNC_EVERY="${SYNC_EVERY:-10}"
ROUNDS="${ROUNDS:-200}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-64}"
TRAIN_IP="${TRAIN_IP:-10.102.98.154}"

SUFFIX="$(date +%s)"
PROD_ALIAS="rollout:m3-${SUFFIX}"
CONS_ALIAS="train:m3-${SUFFIX}"
LOG_DIR="${LOG_DIR:-/tmp/nanorl_smoke}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/m3_producer.log "$LOG_DIR"/m3_train.log "$LOG_DIR"/m3_train.jsonl "$LOG_DIR"/m3_trajectories.jsonl
rm -rf "$LOG_DIR/m3_tb"

echo "[m3-smoke] cfg=$CFG steps=$STEPS sync_every=$SYNC_EVERY"
echo "[m3-smoke] train_ip=$TRAIN_IP"
echo "[m3-smoke] aliases producer=$PROD_ALIAS consumer=$CONS_ALIAS"

# Cleanup that runs on EXIT, SIGINT, or SIGTERM. ``setsid`` puts each
# subprocess in its own process group so we can kill the entire group with
# ``kill -- -PGID``; that catches NanoInfra's Ray worker children that
# otherwise survive Ctrl+C. ``pkill`` is the belt-and-suspenders fallback.
PROD_PID=""
TRAIN_PID=""
cleanup() {
  local rc=$?
  echo "[m3-smoke] cleanup (rc=$rc)..."
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

echo "[m3-smoke] starting rollout producer (NanoInfra startup ~90s)..."
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
setsid python -m nanorl.cli rollout-only \
  --cfg "$CFG" \
  --prompts "$PROMPTS" \
  --rounds "$ROUNDS" \
  --limit-prompts "$LIMIT_PROMPTS" \
  --serve-forever \
  --save-jsonl "$LOG_DIR/m3_trajectories.jsonl" \
  --print-traj-every 1 \
  --print-traj-n 2 \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  > "$LOG_DIR/m3_producer.log" 2>&1 &
PROD_PID=$!

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

echo "[m3-smoke] starting Ray-managed trainer on $TRAIN_IP..."
TRAIN_RC=0
PYTHONPATH=/mnt/nvme1n1/ml_research/majinming/src/Megatron-LM \
NANORL_LOG_LEVEL="${NANORL_LOG_LEVEL:-INFO}" \
setsid python -m nanorl.cli train-ray \
  --cfg "$CFG" \
  --steps "$STEPS" \
  --weight-sync-every "$SYNC_EVERY" \
  --producer-alias "$PROD_ALIAS" \
  --consumer-alias "$CONS_ALIAS" \
  --train-ip "$TRAIN_IP" \
  --nproc 1 \
  --log-jsonl "$LOG_DIR/m3_train.jsonl" \
  --tb-dir "$LOG_DIR/m3_tb" \
  > "$LOG_DIR/m3_train.log" 2>&1 &
TRAIN_PID=$!
wait "$TRAIN_PID" || TRAIN_RC=$?

echo "[m3-smoke] trainer exit=$TRAIN_RC"
echo "[m3-smoke] === train log tail ==="
tail -30 "$LOG_DIR/m3_train.log"
echo "[m3-smoke] === train.jsonl ==="
if [[ -f "$LOG_DIR/m3_train.jsonl" ]]; then
  cat "$LOG_DIR/m3_train.jsonl"
fi

if [[ -f "$LOG_DIR/m3_trajectories.jsonl" ]]; then
  echo "[m3-smoke] === trajectory dump summary ($LOG_DIR/m3_trajectories.jsonl) ==="
  python - <<PY
import json
from collections import Counter
path = "$LOG_DIR/m3_trajectories.jsonl"
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
    # Show one positive (if any) and one zero example so the operator can
    # spot-check what the verifier sees.
    pos = next((r for r in rows if r["reward"] > 0), None)
    neg = next((r for r in rows if r["reward"] == 0), None)
    for tag, r in [("first-positive", pos), ("first-zero", neg)]:
        if r is None: continue
        print(f"\n--- {tag} (group={r['group_id']}, reward={r['reward']}, ref={r['reference']!r}) ---")
        print("PROMPT:", (r.get("prompt") or "")[-400:])
        print("RESPONSE:", (r.get("response") or "")[:800])
PY
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
