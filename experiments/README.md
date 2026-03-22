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

## How Each Experiment Works

Every experiment folder contains:
- `train_gpt.py` or `train_gpt_mlx.py` — the modified training script
- `notes.md` — what was changed, why, and what I learned
- `results.json` — raw metrics

Experiments are meant to be run independently, but each one incorporates the wins from previous experiments.
