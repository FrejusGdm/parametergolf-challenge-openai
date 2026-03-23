# Experiment 21 — Winner Reproduction (Proxy)

## Hypothesis
Reproducing the current winning recipe exactly (architecture + quantization path) gives us a strong baseline to iterate from. On single-GPU L4 we should not expect leaderboard-level bpb, but we should reproduce the same behavior trends and artifact constraints.

## Reference Being Reproduced
- **Source repo:** https://github.com/openai/parameter-golf
- **Pinned commit:** `9f9d53343aa44fe1fbd94ae32650ca2e83602a10`
- **Winning folder:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- **Reported score (3 seeds, 8xH100):** `1.14276` bpb

## Why "Proxy" Reproduction?
The winning score was on 8xH100 with much higher throughput. HF `l4x1` is useful for fast/cheap iteration, but is not compute-equivalent.

## What We Reproduce Exactly
- 10 layers, 512 dim, GQA (KV=4), MLP 3x
- BigramHash (`10240`) + SmearGate path
- SWA defaults (`start_frac=0.4`, `every=50`)
- Mixed int5/int6 quantization logic from winner script
- Sliding-window eval (`stride=64`) behavior in that script

## What We Scale for L4 Safety
- `TRAIN_BATCH_TOKENS=131072` (vs winner default 786432)
- `VAL_BATCH_SIZE=131072` (to avoid OOM risk)
- Keep wallclock at 600s to match challenge time budget shape

## Run on HF Jobs
```bash
cd /Users/josuegodeme/Downloads/projects/open-ai-challenge
RUN_ID=exp21_winner_repro_l4 \
HF_FLAVOR=l4x1 \
HF_TIMEOUT=2h \
TRAIN_SHARDS=1 \
MAX_WALLCLOCK_SECONDS=600 \
python3 scripts/hf_submit_exp21_job.py
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

## Local Artifacts Location
- HF remote artifacts: `JosueG/parameter-golf-curriculum` under `jobs/exp21/...`
- Download and store local copies in:
  - `experiments/21-winner-repro/hf-logs/`

## Results
- **Run ID:** `exp21_winner_repro_l4_v1`
- **HF Job ID:** `69c0a2c771691dc46f1640b3` (running)
- **Hardware:** `...`
- **Pre-quant:** `val_loss=...`, `val_bpb=...`
- **Post-quant:** `val_loss=...`, `val_bpb=...`
- **Artifact size:** `... bytes`
- **Hypothesis verdict:** `support / reject / inconclusive`

## Next Improvements After Repro
1. Keep winner stack fixed and only sweep one lever at a time (A/B): `SWA_START_FRAC`, `BIGRAM_VOCAB_SIZE`, `WEIGHT_DECAY`.
2. Add controlled curriculum variants on top of winner stack (default vs easy-first vs 2-stage schedule).
3. Test small MoE-lite branch only if it beats dense by step-2000 proxy checkpoints; otherwise kill quickly.

## Citation / Attribution
- OpenAI Parameter Golf reference and winning record scripts:
  - https://github.com/openai/parameter-golf
  - https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50
