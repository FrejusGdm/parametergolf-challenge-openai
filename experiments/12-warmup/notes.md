# Experiment 12: Warmup Steps Sweep

## Hypothesis
With only ~500 steps in our test runs (and ~10min on H100 for real runs), warmup is a significant fraction of training. Too much warmup wastes steps; too little may cause instability.

## Setup
- Variable: `WARMUP_STEPS`
- Values: 5, **20** (baseline), 50, 100
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU

## Results

| warmup_steps | val_bpb | Notes |
|--------------|---------|-------|
| 5 | 1.6328 | winner |
| 20 | 1.6329 | baseline |
| 50 | 1.6331 | |
| 100 | 1.6330 | |

## Analysis
Warmup steps barely matter: the entire range from 5 to 100 spans only 0.0003 bpb, which is within noise. 5 steps is technically best but the difference is negligible. Safe to use any value in this range; fewer warmup steps means more steps at peak LR.
