# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a competition entry for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) challenge. The goal is to minimize bits-per-byte (bpb) on FineWeb validation under strict constraints:
- **16MB artifact limit** (model weights + training code combined, decimal 16,000,000 bytes)
- **10 minutes training** on 8×H100 GPUs
- **Metric:** val_bpb (lower is better). Baseline: 1.2244, current SOTA: 1.1428

## Architecture

- `parameter-golf/` — upstream challenge repo (submodule/clone). Contains `train_gpt.py` (CUDA) and `train_gpt_mlx.py` (Apple Silicon). Do not modify directly.
- `experiments/` — numbered experiment folders (00-baseline through XX-novel). Each contains a modified `train_gpt.py`, `notes.md`, and `results.json`. Experiments are cumulative — each incorporates wins from previous ones.
- `notebooks/` — Jupyter notebooks for understanding architecture, quantization, and dataset properties.
- `journal/` — detailed write-ups of findings.
- `RESEARCH_LOG.md` — chronological log of all attempts and results.

## Common Commands

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

### Download Dataset
```bash
# Full dataset (80 shards, ~8B tokens)
python3 parameter-golf/data/cached_challenge_fineweb.py --variant sp1024

# Minimal for local testing (1 shard)
python3 parameter-golf/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```
Data lands in `parameter-golf/data/datasets/fineweb10B_sp1024/` and `parameter-golf/data/tokenizers/`.

### Local Training (Apple Silicon / MLX)
```bash
cd parameter-golf
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```
Set `VAL_LOSS_EVERY=0` to skip periodic validation (M1 validation is very slow — 947 batches).

### GPU Training (CUDA)
```bash
cd parameter-golf
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
Use `nproc_per_node=8` for 8×H100 leaderboard runs. Override time cap with `MAX_WALLCLOCK_SECONDS=0` for longer runs.

### Running Experiment Scripts
```bash
cd experiments/00-baseline
python3 train_gpt_mlx.py  # or torchrun for CUDA version
```

### Hyperparameter Sweep (experiments 10-19)
```bash
# Run a single experiment with overrides
./scripts/run_sweep.sh 10-lr-sweep MATRIX_LR=0.08

# Run all sweep experiments sequentially
./scripts/sweep_all.sh
```

## GCP Compute

- **Project:** `testingout-423013` (Dartmouth edu account)
- **GPU quota:** L4 (1x available in us-central1), no A100/H100 quota
- **VM:** `g2-standard-8` with 1x L4 in `us-central1-a`

## Key Technical Details

- **Baseline model:** 9-layer GPT, 512 dim, 1024 vocab, GQA (4 KV heads), RoPE, U-Net skip connections, tied embeddings
- **Winning techniques from leaderboard:** sliding window eval, zstd compression, int6/int5 quantization + QAT, 3× MLP expansion, BigramHash/SmearGate, more layers (10-11), SWA, Muon optimizer with weight decay
- **Artifact size** = code bytes (`train_gpt.py`) + compressed model bytes. Must be under 16MB.
- **Evaluation:** val_bpb computed on fixed first-50k-document FineWeb val split. Sliding window eval (stride=64) is a free improvement. Eval time limit is 10 min separate from training.
- **Custom tokenizers are allowed** but heavily scrutinized. The default is a 1024-token BPE vocabulary.
- **Local M1 config:** ~65K tokens/step (vs 524K on H100), 8 grad accumulation steps, 4K microbatch. Use short runs (200-500 steps) as a fast feedback loop.

## HF Jobs (Remote GPU)

Use Hugging Face Jobs for GPU experiments instead of local M1. See `docs/hf-jobs-workflow.md` for full guide.

```python
from huggingface_hub import HfApi, get_token
api = HfApi()
job = api.run_uv_job(
    "scripts/your_experiment.py",  # Self-contained script with PEP 723 deps
    flavor="l4x1",                 # L4 GPU
    timeout="3h",
    secrets={"HF_TOKEN": get_token()},
)
print(f"Job ID: {job.id}")
```

- **Account:** `JosueG` (Pro), results go to `JosueG/parameter-golf-*` dataset repos
- **Preferred flavor:** `l4x1` (24GB VRAM, ~$0.50/hr)
- **Scripts must be self-contained** — no local imports, clone repos at runtime
- **Always push results to Hub** — environment is ephemeral

## Rules

- Do NOT mention Anthropic, Claude, or Claude Code in commits, README, or project files
- Commit often as work progresses
- Update RESEARCH_LOG.md and journal entries with findings
