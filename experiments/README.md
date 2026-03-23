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
| XX | Novel Ideas | — | — | — | — |

### Hyperparameter Sweep (10-19)

Short runs (500 steps) on 1x L4 GPU to understand training dynamics. Run with `scripts/sweep_all.sh`.

| # | Experiment | val_bpb | Baseline | Best Value | Date |
|---|-----------|---------|----------|------------|------|
| 10 | Matrix LR Sweep | — | 0.04 | — | — |
| 11 | Embed LR Sweep | — | 0.05 | — | — |
| 12 | Warmup Steps Sweep | — | 20 | — | — |
| 13 | Warmdown Iters Sweep | — | 1200 | — | — |
| 14 | Batch Size Sweep | — | 524K | — | — |
| 15 | Sequence Length Sweep | — | 1024 | — | — |
| 16 | Muon Momentum Sweep | — | 0.95 | — | — |
| 17 | Weight Decay Sweep | — | 0 | — | — |
| 18 | Combined Winners | — | — | — | — |
| 19 | Combined + 3x MLP | — | — | — | — |

### Activation Function Sweep (20)

Testing 13 activation functions (10 non-gated + 3 gated). Run with `scripts/submit_activation_sweep.py`.

| # | Activation | val_bpb | Gated? | Paper |
|---|-----------|---------|--------|-------|
| 20 | relu_sq (baseline) | — | No | So et al., 2021 |
| 20 | gelu | — | No | Hendrycks & Gimpel, 2016 |
| 20 | silu/swish | — | No | Ramachandran et al., 2017 |
| 20 | relu | — | No | Nair & Hinton, 2010 |
| 20 | softmax | — | No | Bridle, 1990 |
| 20 | relu_cubed | — | No | So et al., 2021 |
| 20 | sigmoid | — | No | Classical |
| 20 | tanh | — | No | LeCun et al., 1998 |
| 20 | softplus | — | No | Dugas et al., 2001 |
| 20 | mish | — | No | Misra, 2019 |
| 20 | swiglu | — | Yes | Shazeer, 2020 |
| 20 | geglu | — | Yes | Shazeer, 2020 |
| 20 | reglu | — | Yes | Shazeer, 2020 |

## How Each Experiment Works

Every experiment folder contains:
- `train_gpt.py` or `train_gpt_mlx.py` — the modified training script
- `notes.md` — what was changed, why, and what I learned
- `results.json` — raw metrics

Experiments are meant to be run independently, but each one incorporates the wins from previous experiments.
