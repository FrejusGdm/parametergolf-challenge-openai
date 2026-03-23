# Experiment 04 — 3x MLP Expansion

## Hypothesis
Widening the MLP from the default expansion ratio to 3x will improve model
capacity and lower val_bpb, at the cost of increased artifact size.

## Background
The feed-forward (MLP) block in each transformer layer maps from d_model to
an expanded intermediate dimension, then back. A wider MLP gives the model
more capacity to learn nonlinear transformations per layer. The leaderboard
winning entries use 3x MLP expansion, suggesting it is a net win even after
accounting for the larger artifact.

The baseline `train_gpt.py` supports this via the `MLP_MULT` environment
variable. No code changes required.

## Configuration
- **Script:** `scripts/hf_exp04_mlp3x_job.py`
- **Hardware:** L4 x1 via HF Jobs
- **Training:** 2000 steps for both baseline and 3x variant
- **Key env var:** `MLP_MULT=3`
- **Comparison:** Same seed, same data, same steps — only MLP width differs

## What This Tests
- Impact of MLP width on val_bpb at fixed step count
- Artifact size increase from wider MLP
- Whether the bpb improvement justifies the artifact cost within 16MB budget

## Expected Outcome
- val_bpb improvement of 0.01-0.03 (based on leaderboard trends)
- Artifact size increase of ~1-3MB (more parameters to store)
- Net positive: bpb gain should outweigh artifact size cost

## Results
*Pending — run with `scripts/hf_exp04_mlp3x_job.py`*
