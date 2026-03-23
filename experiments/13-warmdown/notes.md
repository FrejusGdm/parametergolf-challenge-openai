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
| 600 | 1.6317 | winner |
| 1200 | 1.6331 | baseline |
| 2400 | 1.7014 | |
| 3600 | 1.7914 | |

## Analysis
Shorter warmdown is better: 600 iters edges out the 1200 baseline by 0.0014 bpb. Longer warmdown values (2400, 3600) degrade significantly because with only 500 training steps, a large warmdown fraction means the LR decays for most of training. This parameter must be tuned relative to total training steps.
