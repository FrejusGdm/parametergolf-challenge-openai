# Experiment 16: Muon Momentum Sweep

## Hypothesis
The Muon optimizer uses a momentum parameter (default 0.95). Higher momentum (0.99) is used by several leaderboard entries. This may help with smoother convergence, especially important for short training runs.

## Setup
- Variable: `MUON_MOMENTUM`
- Values: 0.90, **0.95** (baseline), 0.99
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU

## Results

| muon_momentum | val_bpb | Notes |
|---------------|---------|-------|
| 0.90 | — | |
| 0.95 | — | baseline |
| 0.99 | — | |

## Analysis
TBD
