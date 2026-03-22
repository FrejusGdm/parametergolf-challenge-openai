# Experiment 00 — Baseline

## Hypothesis
Run the unmodified baseline to establish our starting point. Understanding every metric before changing anything.

## Configuration
- **Script:** `train_gpt_mlx.py` (unmodified MLX baseline)
- **Hardware:** Apple M1, 16GB unified memory
- **Training:** 200 steps (short run for local testing)
- **Batch:** 65,536 tokens/step (reduced from 524,288 for M1 memory)
- **Grad accum:** 8 steps
- **Sequence length:** 1024
- **Model:** 9 layers, 512 dim, 8 heads (4 KV), 2x MLP

## Results
- val_loss: TBD
- val_bpb: TBD
- Artifact size: TBD
- Training time: TBD
- Tokens/sec: TBD

## What I Learned
(To be filled after run completes)

## Notes
- Running with fewer iterations (200 vs 20,000) and smaller batch to fit M1
- This means our val_bpb will be much worse than the H100 baseline
- The goal isn't to match baseline score — it's to verify everything works and understand the pipeline

## Next Steps
- Run the architecture walkthrough notebook
- Try the quantization notebook
- Move to Experiment 01: sliding window eval
