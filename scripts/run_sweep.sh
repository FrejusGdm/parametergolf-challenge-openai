#!/usr/bin/env bash
# run_sweep.sh — Run a single training experiment with env var overrides.
# Usage: ./scripts/run_sweep.sh <experiment_name> [ENV_VAR=value ...]
#
# Example:
#   ./scripts/run_sweep.sh 10-lr-sweep MATRIX_LR=0.08
#   ./scripts/run_sweep.sh 14-batch-size TRAIN_BATCH_TOKENS=262144

set -euo pipefail

EXP_NAME="${1:?Usage: run_sweep.sh <experiment_name> [ENV_VAR=value ...]}"
shift

# Defaults for sweep runs
export ITERATIONS="${ITERATIONS:-500}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Apply overrides from command line
for arg in "$@"; do
    export "$arg"
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PG_DIR="$REPO_DIR/parameter-golf"
EXP_DIR="$REPO_DIR/experiments/$EXP_NAME"
RESULTS_FILE="$EXP_DIR/results.json"

mkdir -p "$EXP_DIR"

echo "============================================"
echo "Experiment: $EXP_NAME"
echo "Overrides: $*"
echo "Iterations: $ITERATIONS"
echo "============================================"

# Run training and capture output
cd "$PG_DIR"
RUN_ID="sweep_${EXP_NAME}" \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "/tmp/sweep_${EXP_NAME}.log"

# Extract val_bpb from output
VAL_BPB=$(grep -oP 'val_bpb[=: ]+\K[0-9.]+' "/tmp/sweep_${EXP_NAME}.log" | tail -1 || echo "N/A")
VAL_LOSS=$(grep -oP 'val_loss[=: ]+\K[0-9.]+' "/tmp/sweep_${EXP_NAME}.log" | tail -1 || echo "N/A")

echo ""
echo "============================================"
echo "Result: val_bpb=$VAL_BPB  val_loss=$VAL_LOSS"
echo "============================================"

# Save result to JSON
cat > "$RESULTS_FILE" <<EOF
{
    "experiment": "$EXP_NAME",
    "overrides": "$*",
    "iterations": $ITERATIONS,
    "val_bpb": "$VAL_BPB",
    "val_loss": "$VAL_LOSS",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "Results saved to $RESULTS_FILE"
