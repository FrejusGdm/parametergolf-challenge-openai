# Experiment 10: Matrix Learning Rate Sweep

## Hypothesis
The baseline matrix_lr=0.04 may not be optimal for this model size and training budget. With only ~500 steps, a higher LR might converge faster; too high will diverge.

## Setup
- Variable: `MATRIX_LR` (controls learning rate for all weight matrices)
- Values: 0.01, 0.02, **0.04** (baseline), 0.08
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU

## Results

| matrix_lr | val_bpb | Notes |
|-----------|---------|-------|
| 0.01 | — | |
| 0.02 | — | |
| 0.04 | — | baseline |
| 0.08 | — | |

## Analysis
TBD
