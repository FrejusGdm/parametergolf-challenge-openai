# Experiment 13: Warmdown Iterations Sweep

## Hypothesis
Warmdown (LR decay at end of training) controls how smoothly the model converges. The baseline uses 1200 steps. With a 10-min wall clock, the warmdown fraction matters — too long wastes peak-LR training time.

## Setup
- Variable: `WARMDOWN_ITERS`
- Values: 600, **1200** (baseline), 2400, 3600
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU

## Results

| warmdown_iters | val_bpb | Notes |
|----------------|---------|-------|
| 600 | — | |
| 1200 | — | baseline |
| 2400 | — | |
| 3600 | — | |

## Analysis
TBD
