# Experiment 07 — Stochastic Weight Averaging (SWA)

## Hypothesis
Averaging model checkpoints from the last 50% of training will produce a
smoother model that generalizes better, reducing val_bpb compared to the
final single checkpoint, with essentially zero training cost.

## Why It Matters
SWA is a well-established technique for improving generalization in deep learning.
It works by averaging multiple checkpoints along the SGD/optimizer trajectory,
which tends to find wider optima in the loss landscape. Wider optima correlate
with better generalization. For the Parameter Golf challenge, this is attractive
because:

1. **Free quality improvement:** No extra training compute; just save and average.
2. **Better quantization robustness:** Smoother weight distributions from averaging
   tend to quantize better (fewer outliers).
3. **Used by the winner:** The top leaderboard submission uses SWA (50% of training),
   confirming its effectiveness in this setting.

The main cost is memory: storing N checkpoints in CPU RAM. For the 5-6M parameter
baseline model, each checkpoint is ~20-24MB in fp32, so 20 checkpoints would use
~480MB — well within typical GPU node RAM.

## Technique Details
- **Collection start:** After `swa_start_frac` of total iterations (default: 50%)
- **Collection frequency:** Every `swa_every` steps (default: 50)
- **Storage:** Full state_dict cloned to CPU at each collection point
- **Averaging:** Arithmetic mean of all collected checkpoints in fp32
- **With 2000 iterations:** Collects from step 1000 onward, every 50 steps = ~20 checkpoints

### What Gets Averaged
All model parameters: embedding weights, attention matrices, MLP weights,
layer norms, skip weights, etc. The averaged model is then used for both
final pre-quant evaluation and quantized export.

## Expected Results
- Pre-quant val_bpb: slightly better (smoother weights generalize better)
- Post-quant val_bpb: noticeably better (smoother weights quantize better)
- Training time: essentially unchanged (checkpoint cloning is fast)
- Artifact size: similar to baseline (same architecture, same quantization)

## Configuration
- **HF Job script:** `scripts/hf_exp07_swa_job.py`
- **Hardware:** HF Jobs `l4x1`
- **Default:** 2000 iterations, 20 train shards
- **SWA config:** start at 50%, collect every 50 steps
- **Two runs:** baseline (A) vs SWA (B)

## How to Run
```bash
python scripts/submit_all_exp02_07.py --exp 07
# or with custom config:
# SWA_START_FRAC=0.75 SWA_EVERY=25 HF_TOKEN=... uv run scripts/hf_exp07_swa_job.py
```

## Results

### Baseline
- **Pre-quant:** `val_loss=...`, `val_bpb=...`
- **Post-quant:** `val_loss=...`, `val_bpb=...`
- **Artifact size:** `... bytes`

### SWA (last 50%)
- **Pre-quant (final step):** `val_loss=...`, `val_bpb=...`
- **Pre-quant (SWA averaged):** `val_loss=...`, `val_bpb=...`
- **Post-quant:** `val_loss=...`, `val_bpb=...`
- **Artifact size:** `... bytes`
- **Checkpoints collected:** `...`

### A/B Verdict
- Post-quant val_bpb delta: `...`
- SWA pre-quant vs baseline pre-quant delta: `...`
- Hypothesis verdict: `pending`

## Follow-up Ideas
- Sweep SWA start fraction: 25%, 50%, 75%
- Sweep collection frequency: every 25, 50, 100 steps
- Try exponential moving average (EMA) instead of uniform averaging
- Combine with learning rate cycling (SWAG or cyclical SWA)
- Test interaction with warmdown schedule (does SWA help more without warmdown?)

## References
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization" (2018)
- SWA used in top Parameter Golf leaderboard submissions
- Baseline code: [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)
