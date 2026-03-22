# 2026-03-22 — Project Kickoff

## What Happened

Set up the project structure for the OpenAI Parameter Golf challenge. Spent time reading through the entire challenge repo, understanding the baseline, and studying top submissions on the leaderboard.

## What I Learned

### The Challenge
- You get 16MB total (code + model weights) and 10 minutes on 8x H100 GPUs
- The metric is bits-per-byte (bpb) — a tokenizer-agnostic measure of how well your model compresses text
- The dataset is FineWeb 10B, tokenized with a tiny 1024-token SentencePiece vocabulary

### The Baseline Architecture
- 9-layer transformer with some cool tricks:
  - **GQA (Grouped Query Attention):** 8 attention heads but only 4 key-value heads — saves memory without losing much quality
  - **RoPE:** Rotary positional embeddings baked into the attention computation
  - **U-Net skip connections:** The model has an encoder-decoder structure where earlier layers feed into later layers
  - **ReLU² activation:** Squaring the ReLU output in the MLP — sharper, sparser activations
  - **Muon optimizer:** A custom optimizer that does orthogonal updates for matrix parameters, different from standard Adam

### What the Winners Did
The top submissions (1.1428 bpb vs 1.2244 baseline) used a combination of:
1. **Aggressive quantization** — int6 or even int5 instead of int8 (saves ~25-37% space)
2. **Quantization-aware training (QAT)** — train with fake quantization so the model learns to be robust to it
3. **Bigger MLPs** — 3x expansion instead of 2x (more capacity, funded by compression savings)
4. **More layers** — 10-11 instead of 9
5. **BigramHash / SmearGate** — cheap bigram modeling that helps with local context
6. **Sliding window eval** — evaluate with overlapping context windows (~0.034 bpb free!)
7. **SWA (Stochastic Weight Averaging)** — average checkpoints for smoother, more quantization-friendly weights
8. **zstd compression** — ~5% better than zlib for the model artifact

### Key Numbers
- Baseline: 1.2244 bpb (post-quantization)
- Current SOTA: 1.1428 bpb
- That's a ~6.7% improvement — sounds small but it's significant in this domain

## Assumptions & Open Questions
- How fast will training be on my MacBook with MLX? Might be much slower than H100s but good enough for iteration
- Which improvements give the biggest bang-for-buck? My guess: sliding window eval + int6 quant + 3x MLP
- Can I find novel approaches that the leaderboard hasn't tried yet?

## Next Steps
1. Get the baseline running locally with MLX
2. Create the architecture walkthrough notebook
3. Understand the quantization pipeline deeply
4. Start with Experiment 01 (sliding window eval — the easiest win)
