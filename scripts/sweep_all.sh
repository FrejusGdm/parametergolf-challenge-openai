#!/usr/bin/env bash
# sweep_all.sh — Run all hyperparameter sweep experiments (10-19) sequentially.
# Usage: ./scripts/sweep_all.sh
#
# Experiments 18 and 19 require manual configuration after reviewing results from 10-17.
# Run those separately once you've identified the winners.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting hyperparameter sweep experiments..."
echo ""

# Experiment 10: Matrix LR sweep
for LR in 0.01 0.02 0.04 0.08; do
    "$SCRIPT_DIR/run_sweep.sh" "10-lr-sweep" "MATRIX_LR=$LR" "RUN_ID=sweep_10_lr_${LR}"
done

# Experiment 11: Embed LR sweep
for LR in 0.02 0.05 0.1 0.15; do
    "$SCRIPT_DIR/run_sweep.sh" "11-embed-lr" "TIED_EMBED_LR=$LR" "RUN_ID=sweep_11_elr_${LR}"
done

# Experiment 12: Warmup steps sweep
for WU in 5 20 50 100; do
    "$SCRIPT_DIR/run_sweep.sh" "12-warmup" "WARMUP_STEPS=$WU" "RUN_ID=sweep_12_wu_${WU}"
done

# Experiment 13: Warmdown iterations sweep
for WD in 600 1200 2400 3600; do
    "$SCRIPT_DIR/run_sweep.sh" "13-warmdown" "WARMDOWN_ITERS=$WD" "RUN_ID=sweep_13_wd_${WD}"
done

# Experiment 14: Batch size sweep
for BS in 262144 524288 1048576; do
    "$SCRIPT_DIR/run_sweep.sh" "14-batch-size" "TRAIN_BATCH_TOKENS=$BS" "RUN_ID=sweep_14_bs_${BS}"
done

# Experiment 15: Sequence length sweep
for SL in 512 1024 2048; do
    "$SCRIPT_DIR/run_sweep.sh" "15-seq-length" "TRAIN_SEQ_LEN=$SL" "RUN_ID=sweep_15_sl_${SL}"
done

# Experiment 16: Muon momentum sweep
for MOM in 0.90 0.95 0.99; do
    "$SCRIPT_DIR/run_sweep.sh" "16-muon-momentum" "MUON_MOMENTUM=$MOM" "RUN_ID=sweep_16_mom_${MOM}"
done

# Experiment 17: Weight decay sweep
# NOTE: Weight decay may require a code change in train_gpt.py.
# Check if WEIGHT_DECAY env var is already supported before running.
for WDEC in 0 0.01 0.04; do
    "$SCRIPT_DIR/run_sweep.sh" "17-weight-decay" "WEIGHT_DECAY=$WDEC" "RUN_ID=sweep_17_wdec_${WDEC}"
done

echo ""
echo "============================================"
echo "Experiments 10-17 complete!"
echo "Review results in experiments/*/results.json"
echo ""
echo "Next steps:"
echo "  1. Identify best values from each experiment"
echo "  2. Configure and run experiment 18 (combined winners)"
echo "  3. Configure and run experiment 19 (combined + MLP 3x)"
echo "============================================"
