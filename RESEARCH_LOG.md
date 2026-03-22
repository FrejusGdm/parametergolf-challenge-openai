# Research Log

A chronological record of everything I try, learn, and discover during this challenge.

---

## 2026-03-22 — Project Kickoff

- Set up project structure and repository
- Explored the [Parameter Golf](https://github.com/openai/parameter-golf) challenge repo
- Studied the baseline architecture: 9-layer GPT with GQA, RoPE, U-Net skip connections
- Studied the top submissions to understand what wins look like
- Key insight: the gap between baseline (1.2244 bpb) and SOTA (1.1428 bpb) comes from aggressive quantization, bigger MLPs, and clever evaluation
- [Detailed notes](journal/2026-03-22-project-kickoff.md)

**Next:** Get the baseline running locally on my Mac, start the architecture walkthrough notebook.

---

## 2026-03-22 — Baseline Running + Notebooks Created

- Got baseline training running on M1 MacBook (16GB)
- Key learning: M1 validation is *slow* — 947 batches for the full val set. Need to skip step-0 val for faster iteration.
- Created 3 notebooks:
  1. [Architecture Walkthrough](notebooks/01-architecture-walkthrough.ipynb) — traces every component
  2. [Quantization Deep Dive](notebooks/02-quantization-deep-dive.ipynb) — int8 vs int6, compression, QAT
  3. [Dataset Exploration](notebooks/03-dataset-exploration.ipynb) — token distributions, shard analysis, BPB math
- Training config for M1: 65K tokens/step (vs 524K on H100), 8 grad accum steps, 4K microbatch
- **Idea:** Use short runs (200-500 steps) on M1 as a fast feedback loop to test hypotheses
- **Idea:** Explore data-related angles — tokenizer optimization, data ordering (allowed by rules)
- **Insight:** The rules allow custom tokenizers! With only 1024 tokens, there might be room to optimize

**Baseline run results:** (updating when training finishes)
