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

## Suggested GCP Smoke Run (Auto Shutdown + Cleanup)
```bash
cd /Users/josuegodeme/Downloads/projects/open-ai-challenge
PROJECT=testingout-423013 \
ZONE=us-central1-a \
AUTO_SHUTDOWN_MINUTES=75 \
MAX_WALLCLOCK_SECONDS=900 \
ITERATIONS=600 \
TRAIN_BATCH_TOKENS=262144 \
DELETE_INSTANCE_AT_END=1 \
bash scripts/gcp_exp09_l4_smoke.sh
```

## Suggested HF Jobs Smoke Run (Preferred)
```bash
cd /Users/josuegodeme/Downloads/projects/open-ai-challenge
RUN_ID=exp09_hf_smoke \
HF_FLAVOR=l4x1 \
HF_TIMEOUT=2h \
TRAIN_SHARDS=1 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=300 \
TRAIN_BATCH_TOKENS=131072 \
ENABLE_TORCH_COMPILE=0 \
WARMUP_STEPS=0 \
python3 scripts/hf_submit_exp09_job.py
```

Monitor:
```bash
python3 - << 'PY'
from huggingface_hub import HfApi
job_id = "PASTE_JOB_ID"
j = HfApi().inspect_job(job_id=job_id, namespace="JosueG")
print("status:", j.status.stage)
for line in HfApi().fetch_job_logs(job_id=job_id, namespace="JosueG"):
    print(line)
PY
```

## Results
- **Run ID:** `exp09_hf_smoke_retry2`
- **Hardware:** Hugging Face Jobs `l4x1`
- **Train time cap:** `600s`
- **Step time avg:** `1349.57 ms`
- **Throughput:** `97,117 tok/s`
- **Pre-quant:** `val_loss=2.9230`, `val_bpb=1.7312`
- **Post-quant roundtrip:** `val_loss=2.93721661`, `val_bpb=1.73958512`
- **Artifact size:** `11,407,582 bytes` (under 16MB cap)
- **HF Job ID:** `69c078da71691dc46f163eb1`

## What I Learned
- HF Jobs is a viable iteration loop for this challenge once compile is disabled (`ENABLE_TORCH_COMPILE=0`) for stability on `l4x1`.
- Cost controls worked: short single-GPU run completed and uploaded logs/artifacts automatically.
- This run is a functional smoke test, not a quality win. Need controlled A/B vs equal-budget baseline to isolate ChunkGate impact.

## Reporting Template (Fill This After Each Run)
- **Run ID:** `...`
- **Hardware:** `...`
- **Train time cap:** `...`
- **Step time avg:** `... ms`
- **Throughput:** `... tok/s`
- **Pre-quant:** `val_loss=...`, `val_bpb=...`
- **Post-quant roundtrip:** `val_loss=...`, `val_bpb=...`
- **Artifact size:** `... bytes`
- **Hypothesis verdict:** `support / reject / inconclusive`
- **What I learned:** `1-3 concrete bullets`

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
