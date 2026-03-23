# Experiments

Each experiment builds on the previous one. The idea is to start simple, understand what works, and gradually add complexity.

## Results Summary

| # | Experiment | val_bpb | Delta | Artifact Size | Date |
|---|-----------|---------|-------|---------------|------|
| 00 | Baseline | — | — | — | — |
| 01 | Sliding Window Eval | — | — | — | — |
| 02 | zstd Compression | — | — | — | — |
| 03 | Int6 Quantization + QAT | — | — | — | — |
| 04 | 3x MLP Expansion | — | — | — | — |
| 05 | BigramHash / SmearGate | — | — | — | — |
| 06 | More Layers (10-11) | — | — | — | — |
| 07 | SWA | — | — | — | — |
| 08 | Curriculum Learning | — | — | — | — |
| 09 | ChunkGate-Lite | — | — | — | — |
| 21 | Winner Reproduction (Proxy) | — | — | — | — |
| XX | Novel Ideas | — | — | — | — |

### Hyperparameter Sweep (10-19)

Short runs (500 steps) on 1x L4 GPU to understand training dynamics. Run with `scripts/sweep_all.sh`.

| # | Experiment | Best val_bpb | Baseline | Best Value | Date |
|---|-----------|-------------|----------|------------|------|
| 10 | Matrix LR Sweep | **1.6051** | 0.04 | **0.08** | 2026-03-23 |
| 11 | Embed LR Sweep | **1.6161** | 0.05 | **0.02** | 2026-03-23 |
| 12 | Warmup Steps Sweep | 1.6328 | 20 | 5 (negligible) | 2026-03-23 |
| 13 | Warmdown Iters Sweep | 1.6317 | 1200 | 600 | 2026-03-23 |
| 14 | Batch Size Sweep | OOM on L4 | 524K | — | — |
| 15 | Sequence Length Sweep | **1.6154** | 1024 | **2048** | 2026-03-23 |
| 16 | Muon Momentum Sweep | **1.6306** | 0.95 | **0.99** | 2026-03-23 |
| 17 | Weight Decay Sweep | 1.6327 | 0 | 0 (no effect) | 2026-03-23 |
| 18 | Combined Winners | — | — | — | — |
| 19 | Combined + 3x MLP | — | — | — | — |

### Activation Function Sweep (20)

Testing 13 activation functions (10 non-gated + 3 gated). Run with `scripts/submit_activation_sweep.py`.

| # | Activation | val_bpb | Gated? | Paper |
|---|-----------|---------|--------|-------|
| 20 | **geglu** | **1.6223** | Yes | Shazeer, 2020 |
| 20 | **swiglu** | **1.6230** | Yes | Shazeer, 2020 |
| 20 | **reglu** | **1.6250** | Yes | Shazeer, 2020 |
| 20 | relu_sq (baseline) | 1.6325 | No | So et al., 2021 |
| 20 | relu | 1.6449 | No | Nair & Hinton, 2010 |
| 20 | gelu | 1.6456 | No | Hendrycks & Gimpel, 2016 |
| 20 | mish | 1.6607 | No | Misra, 2019 |
| 20 | relu_cubed | 1.6621 | No | So et al., 2021 |
| 20 | silu/swish | 1.6644 | No | Ramachandran et al., 2017 |
| 20 | tanh | 1.6942 | No | LeCun et al., 1998 |
| 20 | sigmoid | 1.7161 | No | Classical |
| 20 | softmax | 1.7295 | No | Bridle, 1990 |
| 20 | softplus | — | No | Dugas et al., 2001 |

## How Each Experiment Works

Every experiment folder contains:
- `train_gpt.py` or `train_gpt_mlx.py` — the modified training script
- `notes.md` — what was changed, why, and what I learned
- `results.json` — raw metrics

Experiments are meant to be run independently, but each one incorporates the wins from previous experiments.
