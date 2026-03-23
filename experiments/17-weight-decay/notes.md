# Experiment 17: Weight Decay Sweep

## Hypothesis
The baseline has no weight decay. Multiple leaderboard entries use WD=0.04 with the Muon optimizer ("Muon WD"). Weight decay acts as regularization and may help the model generalize better, especially important since we're quantizing weights post-training.

## Setup
- Variable: Weight decay (requires code modification to optimizer)
- Values: 0 (baseline), 0.01, 0.04
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU

## Results

| weight_decay | val_bpb | Notes |
|-------------|---------|-------|
| 0 | — | baseline |
| 0.01 | — | |
| 0.04 | — | |

## Analysis
TBD
