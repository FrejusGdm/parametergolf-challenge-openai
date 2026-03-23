# Experiment 06 — 10 Layers

## Hypothesis
Adding a 10th transformer layer (up from the baseline 9) will improve val_bpb
by increasing model depth, at a manageable artifact size increase.

## Background
Deeper models can learn more complex hierarchical representations. The baseline
uses 9 transformer layers with U-Net skip connections. Leaderboard winners use
10-11 layers, suggesting the depth sweet spot is slightly above the baseline.
The `train_gpt.py` script supports this via the `NUM_LAYERS` environment variable.

Adding one layer increases:
- Parameter count (one extra attention + MLP block)
- Artifact size (roughly +10% for one layer on a 9-layer model)
- Training time per step (more forward/backward compute)

The question is whether the bpb improvement justifies these costs within the
10-minute training window and 16MB artifact budget.

## Configuration
- **Script:** `scripts/hf_exp06_layers_job.py`
- **Hardware:** L4 x1 via HF Jobs
- **Training:** 2000 steps for both baseline (9L) and variant (10L)
- **Key env var:** `NUM_LAYERS=10`
- **Comparison:** Same seed, same data, same steps — only layer count differs

## What This Tests
- Impact of one additional layer on val_bpb at fixed step count
- Artifact size increase from the extra layer
- Per-step time increase (fewer total steps possible in 10-min window)

## Expected Outcome
- val_bpb improvement of 0.005-0.02
- Artifact size increase of ~0.5-1.5MB
- Slightly slower per-step time, meaning fewer steps in 10-min budget
- Net positive if bpb gain exceeds the step-count penalty

## Results
*Pending — run with `scripts/hf_exp06_layers_job.py`*
