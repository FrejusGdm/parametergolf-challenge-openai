# Experiment 09 — ChunkGate-Lite

## Hypothesis
A tiny fixed-ratio compressed branch can recover some of the upside of dynamic chunking ideas without the full complexity of H-Net.

## Configuration
- **Scripts:** `train_gpt.py` (CUDA/GCP), `train_gpt_mlx.py` (local M1)
- **Default hardware target:** 1 GPU smoke run on GCP (cost-controlled)
- **Run style:** short hypothesis test, not leaderboard attempt
- **Key objective:** measure quality/throughput tradeoff of compression branch

## New Environment Knobs (Both Scripts)
- `CHUNKGATE_ENABLE` (default `0`)
- `CHUNKGATE_STRIDE` (default `4`)
- `CHUNKGATE_INNER_LAYERS` (default `2`)
- `CHUNKGATE_GATE_TEMP` (default `1.0`)
- `CHUNKGATE_FUSION_INIT` (default `0.10`)

## What Changed
- Added an optional `ChunkGate-Lite` module in both training paths:
  - `train_gpt.py` (CUDA)
  - `train_gpt_mlx.py` (MLX)
- `ChunkGate-Lite` does:
  - token-level gate prediction (`sigmoid(linear(rms_norm(x)))`)
  - fixed-stride downsampling (`x[:, ::stride, :]`)
  - a small transformer stack on the compressed stream
  - cheap upsampling via repeat and gated residual fusion back to full sequence
- Baseline path is unchanged when disabled (`CHUNKGATE_ENABLE=0`).

## Suggested Smoke Run (Local MLX)
```bash
cd experiments/09-chunkgate-lite
RUN_ID=chunkgate_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=65536 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=65536 \
CHUNKGATE_ENABLE=1 \
CHUNKGATE_STRIDE=4 \
CHUNKGATE_INNER_LAYERS=2 \
python3 train_gpt_mlx.py
```

## Suggested Smoke Run (CUDA, 1 GPU)
```bash
cd experiments/09-chunkgate-lite
RUN_ID=chunkgate_cuda_smoke \
MAX_WALLCLOCK_SECONDS=900 \
ITERATIONS=600 \
TRAIN_BATCH_TOKENS=262144 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
CHUNKGATE_ENABLE=1 \
CHUNKGATE_STRIDE=4 \
CHUNKGATE_INNER_LAYERS=2 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Results
- val_loss: TBD
- val_bpb: TBD
- Artifact size: TBD
- Training time: TBD
- Tokens/sec: TBD

## What I Learned
- TBD (fill after first CUDA smoke run)

## Success Criteria (Hypothesis Gate)
- Throughput drop vs baseline ≤ 10%
- Pre-quant `val_bpb` improvement in short runs (~200-500 iters)
- Artifact size still comfortably under 16MB cap after quantized export

## Cost Controls (GCP)
- Use 1 GPU only
- Use Spot VM
- Set `MAX_WALLCLOCK_SECONDS` (default smoke target: 15 minutes)
- Schedule auto-shutdown at VM boot
- Delete VM immediately after run and copy logs back

## Attribution / Citations
- **Baseline code base:** OpenAI Parameter Golf reference scripts (`train_gpt.py`, `train_gpt_mlx.py`)  
  Source: https://github.com/openai/parameter-golf
- **Research inspiration:** H-Net / Dynamic Chunking  
  Paper: https://arxiv.org/abs/2507.07955  
  Code: https://github.com/goombalab/hnet
