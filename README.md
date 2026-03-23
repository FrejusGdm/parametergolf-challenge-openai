# Parameter Golf Challenge — My Journey

Hi! I'm Josue, a senior at Dartmouth College studying Computer Science. This repo is my open-source adventure into the [OpenAI Parameter Golf](https://openai.com/index/parameter-golf/) challenge — a research competition where you try to build the most efficient tiny language model under some pretty wild constraints.

## What's the Challenge?

The goal is simple (but hard!): **minimize language model loss while keeping everything under 16MB and training in just 10 minutes on 8 GPUs.**

Think of it like golf — lowest score wins, and you have very little room to work with.

- **Dataset:** FineWeb 10B (a big pile of web text)
- **Budget:** 16MB total (your model weights + training code, combined!)
- **Training time:** 10 minutes on 8x H100 GPUs
- **Metric:** Bits-per-byte (bpb) — how well your model compresses text

## Why Am I Doing This?

I want to learn more about machine learning — not just from textbooks, but by getting my hands dirty with real research constraints. This challenge is the perfect playground: small enough to wrap your head around, deep enough to learn a ton. I spent too much time on the Applied side, building apps, webapps finetuning models and etc. I wanted to dive deeper into the layer up so here I am :)

Plus, it sounded really fun.

## How This Repo is Organized

```
experiments/       Each experiment in its own folder, with notes and results
notebooks/         Jupyter notebooks where I explore and learn concepts
journal/           Detailed write-ups of what I tried and what I learned
papers/            Reading list — papers I found useful along the way
scripts/           Utility scripts for evaluation, comparison, etc.
parameter-golf/    The original challenge repo (upstream code)
```

## The Experiment Roadmap

| # | Experiment | Status | BPB | Notes |
|---|-----------|--------|-----|-------|
| 00 | Baseline | Pending | — | Understand the starting point |
| 01 | Sliding Window Eval | Pending | — | Free win from smarter evaluation |
| 02 | Better Compression (zstd) | Pending | — | Fit more model in 16MB |
| 03 | Int6 Quantization | Pending | — | Aggressive weight compression |
| 04 | Bigger MLP (3x) | Pending | — | More capacity per layer |
| 05 | Bigram Features | Pending | — | Lightweight local context |
| 06 | More Layers | Pending | — | Deeper is (usually) better |
| 07 | Weight Averaging (SWA) | Pending | — | Smoother weights for quantization |
| 08 | Curriculum Learning | Pending | — | Does shard order matter? |
| XX | Novel Ideas | Pending | — | The fun part! |

## Research Log

I'm keeping a running [Research Log](RESEARCH_LOG.md) of everything — what I tried, what worked, what surprised me, and what papers I read along the way. If you're curious about the journey, that's the place to start.

## Current Best Result

**TBD** — just getting started!

(The current leaderboard SOTA is 1.1428 bpb. The baseline is 1.2244 bpb. Let's see how close I can get.)

## Want to Follow Along?

Feel free to browse the journal entries, check out the notebooks, or just read the research log. If you're also participating in the challenge or learning about ML, I'd love to hear from you!

## Acknowledgments

- [OpenAI](https://openai.com) for hosting this awesome challenge
- The Parameter Golf community for sharing their approaches openly
- Dartmouth CS for giving me the foundation to even attempt this

---

*Built with curiosity, coffee, and a MacBook.*
