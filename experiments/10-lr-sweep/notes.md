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
| 0.01 | 1.8766 | |
| 0.02 | 1.7129 | |
| 0.04 | 1.6332 | baseline |
| 0.08 | 1.6051 | winner |

## How to run

All experiments run as HF Jobs on L4 GPUs via `scripts/submit_sweeps.py`. Each job:
1. Clones `openai/parameter-golf`
2. Downloads 1 training shard (~100M tokens)
3. Runs `train_gpt.py` for 500 iterations with the specified override
4. Pushes results JSON to `JosueG/parameter-golf-sweeps` on HF Hub

```bash
# Submit all sweep experiments
python scripts/submit_sweeps.py

# Submit just this experiment
python scripts/submit_sweeps.py 10

# Check results
python scripts/monitor_sweeps.py --results
```

## Analysis
Higher LR wins decisively: 0.08 beats the 0.04 baseline by 0.028 bpb, and lower LRs degrade sharply (0.01 is +0.27 worse). With only 500 steps, the model benefits from aggressive learning rates that converge faster. Worth testing 0.10-0.12 to see if we can push further before divergence.
