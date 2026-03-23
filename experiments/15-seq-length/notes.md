# Experiment 15: Sequence Length Sweep

## Hypothesis
Longer sequences let the model learn longer-range dependencies but cost more compute per step. The leaderboard shows seq_len=2048 and 4096 improving scores. Even at 500 steps, we should see a signal.

## Setup
- Variable: `TRAIN_SEQ_LEN`
- Values: 512, **1024** (baseline), 2048
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU

## Results

| seq_len | val_bpb | Notes |
|---------|---------|-------|
| 512 | — | |
| 1024 | — | baseline |
| 2048 | — | |

## Analysis
TBD
