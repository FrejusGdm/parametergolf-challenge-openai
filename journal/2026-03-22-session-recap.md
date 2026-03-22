# 2026-03-22 — Full Session Recap

## What We Did Today

### Set Up the Whole Project
- Created the project structure: experiments, notebooks, journal, papers, scripts
- Initialized git, wrote README telling my story (CS senior wanting to learn ML)
- Installed everything: PyTorch, MLX, Jupyter, SentencePiece, zstandard
- Downloaded 5 training shards of the FineWeb dataset to my Mac

### Learned the Architecture
Created 6 Jupyter notebooks to understand every piece of the baseline:

1. **Architecture Walkthrough** — traced through the GPT model: token embedding → RMSNorm → encoder/decoder with U-Net skips → attention (GQA with RoPE) → MLP (ReLU²) → logit softcap. The model is surprisingly small (~3.3M params) but packs clever tricks.

2. **Quantization Deep Dive** — this is THE key technique. Int8 quantization squishes float32 weights into 1 byte each. The winners go further with int6 (6 bits!) and even int5. Going int8→int6 frees up ~25% more space for parameters. Quantization-Aware Training (QAT) uses the Straight-Through Estimator to train models that are robust to this compression.

3. **Dataset Exploration** — the FineWeb data is tokenized with a tiny 1024-token BPE vocabulary. Each token is roughly 2 bytes of text. The distribution is very skewed — a few common tokens (spaces, 'e', 'the') dominate.

4. **Curriculum Learning** — my idea! Analyzing whether shard ordering matters for training.

5. **Muon Optimizer** — the custom optimizer that does orthogonal updates via Newton-Schulz iteration. It's like SGD but every update is forced to be a rotation, not a stretch. Combined with separate Adam for embeddings and scalars.

6. **Sliding Window Eval** — the easiest win: evaluate with overlapping context windows instead of non-overlapping chunks. Each token gets ~960 tokens of context instead of 0-1023 on average. Free ~0.034 BPB improvement!

### Built the Curriculum Learning Experiment
This was my main original idea — testing if the order you present training data matters.

- **Wrote a shard analysis script** that measures entropy, bigram entropy, vocabulary coverage, and repetition ratio for each training shard
- **Built CurriculumTokenStream** — a drop-in replacement for the baseline's data loader that reorders shards based on a strategy (easy→hard, hard→easy, interleaved, random, quality-first)
- **Created an experiment runner** that trains with each strategy and compares loss curves
- **Ran analysis on 5 local shards** — found they're very similar (entropy std=0.002) but have some variation in repetition ratio

### Set Up Cloud Compute
- GCP has GPU quota issues (need to request increase)
- Discovered HF Jobs works great — submitted the curriculum experiment to run on an NVIDIA L4 GPU through Hugging Face
- The job downloads all 80 training shards (fast on HF's network), analyzes them all, and runs 6 strategies × 500 training steps
- Results get pushed to `JosueG/parameter-golf-curriculum` on the Hub

## Key Insights

1. **The 16MB budget is everything** — quantization determines how much model fits. Int6 vs int8 is the difference between 9 and 11 layers.

2. **Evaluation tricks matter** — sliding window eval is a free 0.034 BPB win with zero model changes. That's bigger than many architectural improvements.

3. **The shards are very homogeneous** (at least the 5 we tested locally). Curriculum learning might not produce dramatic results, but the full 80-shard analysis on HF might tell a different story.

4. **Short runs (500 steps) are a great feedback loop** — can't measure final BPB, but directional signal for comparing approaches is there.

5. **HF Jobs is great for this** — instant GPU access, no quota hassles, results pushed to Hub automatically.

## What's Running Right Now
- HF Jobs: curriculum experiment on L4 GPU (job `69c0788125abd6f920b4e08f`)
- Local: baseline training still grinding on M1 (very slow)

## Next Steps
- Check curriculum results when the HF job finishes
- Implement Experiment 01 (sliding window eval) — the easiest BPB win
- Start on Experiment 02 (zstd compression) — another easy win
- Request GCP GPU quota increase for longer experiments
- Apply for RunPod credits through the challenge form
