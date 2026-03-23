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
| 0.90 | 1.6397 | |
| 0.95 | 1.6330 | baseline |
| 0.99 | 1.6306 | winner |

## Analysis
Higher momentum is slightly better: 0.99 beats baseline 0.95 by 0.0024 bpb, while 0.90 is worse by 0.0067. The effect is modest but consistent with leaderboard entries using 0.99. Higher momentum provides smoother gradient estimates, which helps with short training runs.
