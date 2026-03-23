# Experiment 14: Batch Size (Tokens/Step) Sweep

## Hypothesis
Larger batches give more stable gradients but fewer update steps in the same wall-clock time. Smaller batches give noisier gradients but more updates. The optimal trade-off depends on the model size and training budget.

## Setup
- Variable: `TRAIN_BATCH_TOKENS`
- Values: 262144 (~256K), **524288** (~524K, baseline), 1048576 (~1M)
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU

## Results

| batch_tokens | val_bpb | Notes |
|-------------|---------|-------|
| 262144 | — | |
| 524288 | — | baseline |
| 1048576 | — | |

## Analysis
TBD
