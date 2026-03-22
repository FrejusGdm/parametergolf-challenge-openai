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
| 5 | — | |
| 20 | — | baseline |
| 50 | — | |
| 100 | — | |

## Analysis
TBD
