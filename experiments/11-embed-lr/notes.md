# Experiment 11: Embedding Learning Rate Sweep

## Hypothesis
The tied embedding LR (0.05) is separate from matrix LR. Since embeddings are tied (input=output), the optimal LR may differ significantly from weight matrices.

## Setup
- Variable: `TIED_EMBED_LR`
- Values: 0.02, **0.05** (baseline), 0.1, 0.15
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU

## Results

| embed_lr | val_bpb | Notes |
|----------|---------|-------|
| 0.02 | 1.6161 | winner |
| 0.05 | 1.6325 | baseline |
| 0.1 | 1.6363 | |
| 0.15 | 1.6793 | |

## Analysis
Lower embed LR is better: 0.02 beats the 0.05 baseline by 0.016 bpb, while higher values degrade monotonically. The tied embeddings prefer a more conservative learning rate than the weight matrices, likely because they serve double duty as both input and output projections.
