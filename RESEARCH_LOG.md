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

---

## 2026-03-22 — Experiment 09 Setup (ChunkGate-Lite + GCP Smoke Workflow)

- Added `Experiment 09` implementation with `ChunkGate-Lite` in:
  - `experiments/09-chunkgate-lite/train_gpt_mlx.py` (local MLX path)
  - `experiments/09-chunkgate-lite/train_gpt.py` (CUDA path for cloud GPUs)
- Added a cost-controlled GCP smoke runner:
  - `scripts/gcp_exp09_l4_smoke.sh`
  - Uses 1xL4 (`g2-standard-8`), Spot VM, timed auto-shutdown, and optional auto-delete.
- Added explicit attribution/citations in code and notes:
  - OpenAI Parameter Golf baseline code
  - H-Net paper and reference implementation
- Added a reporting template in `experiments/09-chunkgate-lite/notes.md` and expanded `results.json` fields for consistent run reporting.

**Current GCP quota status (project `testingout-423013`):**
- `NVIDIA_A100_GPUS`: `0`
- `NVIDIA_A100_80GB_GPUS`: `0`
- `NVIDIA_L4_GPUS`: `1`

**Next:** Run one short L4 smoke experiment (`~15 min`) and fill the Exp 09 report template.

---

## 2026-03-22 — Experiment 09 HF Jobs Smoke Run (Completed)

- Switched from GCP to Hugging Face Jobs due to instant GPU availability.
- Submitted 3 jobs:
  - `69c0780025abd6f920b4e083` — failed (`pip` not available in UV job env)
  - `69c0784671691dc46f163ea1` — failed (`torch.compile`/Inductor crash on L4)
  - `69c078da71691dc46f163eb1` — completed successfully after setting `ENABLE_TORCH_COMPILE=0`, `WARMUP_STEPS=0`
- Final successful run:
  - **Run ID:** `exp09_hf_smoke_retry2`
  - **Hardware:** HF Jobs `l4x1`
  - **Pre-quant:** `val_loss=2.9230`, `val_bpb=1.7312`
  - **Post-quant:** `val_loss=2.93721661`, `val_bpb=1.73958512`
  - **Artifact:** `11,407,582 bytes` (under 16MB cap)
  - **Train time:** `404,872 ms`

**Takeaway:** infrastructure and reporting pipeline now work end-to-end on HF Jobs; model quality impact from ChunkGate-Lite remains inconclusive and needs baseline A/B under identical short-run settings.

---

## 2026-03-22 — Experiment 08: Curriculum Learning Results

**THE BIG FINDING: Easy-first curriculum learning works!**

Ran 6 shard ordering strategies × 500 steps on all 80 FineWeb shards using HF Jobs (NVIDIA L4).

| Strategy | Final Loss | vs Default |
|----------|-----------|------------|
| **easy_first** | **3.7846** | **-0.108 (best!)** |
| interleaved | 3.8358 | -0.057 |
| default | 3.8926 | baseline |
| hard_first | 3.9419 | +0.049 |
| random | 3.9580 | +0.065 |
| quality_first | 3.9600 | +0.067 |

- Delta between best and worst: **0.1754** — surprisingly large for data that looks nearly identical
- Easy→hard ordering gives ~0.11 improvement over default sequential
- Hard-first actively hurts convergence — model wastes high-LR early steps on hard data
- Shard "difficulty" measured by entropy — easy shards have more predictable patterns
- [Detailed notes](experiments/08-curriculum-learning/notes.md)
- [Full results on HF Hub](https://huggingface.co/datasets/JosueG/parameter-golf-curriculum)

**Caveats:** Used simplified model (Adam, no Muon/U-Net), 500 steps, single seed.

**Next:** Validate at longer runs, test with full baseline model, explore adaptive curriculum.

---

## 2026-03-22 — Experiment 09 Full A/B on HF Jobs (Completed)

Completed a clean full-run A/B where only `CHUNKGATE_ENABLE` changes.

### Files for this experiment
- [Experiment notes](experiments/09-chunkgate-lite/notes.md)
- [Experiment results JSON](experiments/09-chunkgate-lite/results.json)
- [CUDA trainer (Exp 09)](experiments/09-chunkgate-lite/train_gpt.py)
- [HF submit script](scripts/hf_submit_exp09_job.py)
- [HF self-contained job script](scripts/hf_jobs/exp09_hf_job.py)
- [Local raw HF logs folder](experiments/09-chunkgate-lite/hf-logs)

### Job pair (matched settings)
- **Baseline control (`CHUNKGATE_ENABLE=0`)**  
  Job ID: `69c0837471691dc46f16400b`  
  URL: https://huggingface.co/jobs/JosueG/69c0837471691dc46f16400b  
  Status: `COMPLETED`
  Run ID: `baseline_hf_full_l4_v1`
  Pre-quant: `val_loss=5.1764`, `val_bpb=3.0658`
  Post-quant: `val_loss=5.19749717`, `val_bpb=3.07825059`
  Step avg: `6106.34 ms`
  Artifact size: `5,965,306 bytes`

- **Exp 09 ChunkGate (`CHUNKGATE_ENABLE=1`)**  
  Job ID: `69c083d425abd6f920b4e172`  
  URL: https://huggingface.co/jobs/JosueG/69c083d425abd6f920b4e172  
  Status: `COMPLETED`
  Run ID: `exp09_hf_full_l4_v2`
  Pre-quant: `val_loss=5.3055`, `val_bpb=3.1422`
  Post-quant: `val_loss=5.32222805`, `val_bpb=3.15212324`
  Step avg: `6426.60 ms`
  Artifact size: `7,050,647 bytes`

### Important fix applied before ChunkGate full rerun
- Fixed Rotary cache backward crash after step-0 validation (`inference_mode` tensor reuse issue) in:
  - [experiments/09-chunkgate-lite/train_gpt.py](experiments/09-chunkgate-lite/train_gpt.py)

### A/B outcome
- Lower bpb is better; ChunkGate was worse.
- Post-quant delta (ChunkGate - Baseline): `+0.07387265 bpb`
- Pre-quant delta (ChunkGate - Baseline): `+0.0764 bpb`
- Speed delta (step avg): `+5.25%` slower
- **Hypothesis verdict:** `reject` for this configuration.

### Notes
- Earlier full ChunkGate attempt (`run_id=exp09_hf_full_l4_v1`) failed with the Rotary cache issue and was superseded by the fixed rerun above.
- HF artifact folders used for this summary:
  - `jobs/exp09/baseline_hf_full_l4_v1_20260323_002956/`
  - `jobs/exp09/exp09_hf_full_l4_v2_20260323_003203/`

---

## 2026-03-23 — Experiment 08: Baseline Validation (Negative Result)

Ran the **full baseline model** (Muon optimizer, U-Net skips, GQA) with easy_first vs default shard ordering for 2000 steps on HF Jobs L4.

| Strategy | val_loss | val_bpb | final_bpb (post-quant) |
|----------|---------|---------|------------------------|
| default | 2.4010 | 1.4220 | 1.42316 |
| easy_first | 2.4015 | 1.4223 | 1.42346 |
| **Delta** | **0.0005** | **0.0003** | **0.0003** |

**Verdict: No effect.** The 0.0003 bpb difference is noise.

**Why the simplified model showed an effect but the real model doesn't:**
1. Muon optimizer is more robust to data ordering than Adam
2. U-Net skip connections help learn from both easy and hard patterns simultaneously
3. 2000 steps smooths out early ordering effects
4. The full model architecture is simply better conditioned

**This is still valuable** — we know data ordering is NOT a lever for this challenge, saving us from pursuing it further. The real model is robust to shard ordering.

[Full results on HF Hub](https://huggingface.co/datasets/JosueG/parameter-golf-curriculum)

**Moving on to:** Experiment 01 (Sliding Window Eval) — guaranteed ~0.034 free bpb improvement.

---

## 2026-03-23 — Hyperparameter Sweep (Experiments 10-17) + Activation Sweep (Experiment 20)

Ran 41 short experiments (500 steps each) on HF Jobs L4 GPUs to sweep hyperparameters and activation functions.

### HP Sweep Winners (experiments 10-17)

| Param | Baseline | Best | val_bpb | Delta |
|-------|----------|------|---------|-------|
| **Matrix LR** | 0.04 | **0.08** | 1.6051 | -0.028 (biggest win!) |
| **Embed LR** | 0.05 | **0.02** | 1.6161 | -0.016 |
| **Seq Length** | 1024 | **2048** | 1.6154 | -0.017 |
| **Muon Momentum** | 0.95 | **0.99** | 1.6306 | -0.002 |
| Warmdown | 1200 | 600 | 1.6317 | -0.001 |
| Warmup | 20 | 5 | 1.6328 | negligible |
| Weight Decay | 0 | 0 | 1.6327 | no effect at 500 steps |
| Batch Size | — | — | OOM on L4 | untested |

### Activation Function Sweep (experiment 20)

| Rank | Activation | val_bpb | vs baseline (relu²) |
|------|-----------|---------|---------------------|
| 1 | **geglu** | **1.6223** | -0.010 |
| 2 | **swiglu** | **1.6230** | -0.010 |
| 3 | **reglu** | **1.6250** | -0.008 |
| 4 | relu_sq (baseline) | 1.6325 | — |
| 5 | relu | 1.6449 | +0.012 |
| 6 | gelu | 1.6456 | +0.013 |
| ... | sigmoid, softmax | 1.71+ | much worse |

### Key Insights
1. **Higher matrix LR (0.08)** is the single biggest knob — 2× baseline LR helps converge faster in short runs
2. **Lower embed LR (0.02)** — tied embeddings prefer conservative updates
3. **Gated activations dominate** — GEGLU/SwiGLU beat ReLU² despite having fewer effective hidden units (2/3 hidden dim to compensate for gate matrix)
4. **Seq length 2048** confirms leaderboard trend — longer context helps even at 500 steps
5. **Weight decay doesn't help at 500 steps** — may need longer runs to see regularization benefit (leaderboard uses WD=0.04)
6. **Warmup barely matters** — all values within 0.0003 bpb

### Next Steps
- Run experiment 18: combine winners (LR=0.08, embed_lr=0.02, seq_len=2048, momentum=0.99, geglu)
- Run experiment 19: combined + MLP 3x
- Validate top configs at longer runs (2000+ steps)
- Test batch size sweep on H100 when available

---

## 2026-03-22 — Experiment 21 Setup: Winner Reproduction (Proxy)

Created a new experiment to reproduce the current winning recipe from the official records, pinned to the exact commit that introduced the winner update.

### Goal
- Reproduce the winner architecture/quantization path faithfully.
- Run a cost-controlled single-GPU proxy on HF Jobs (`l4x1`) for fast iteration.
- Use this as the base for targeted improvements (SWA schedule, bigram size, WD, curriculum on top).

### Source pinned for repro
- Repo: https://github.com/openai/parameter-golf
- Commit: `9f9d53343aa44fe1fbd94ae32650ca2e83602a10`
- Script: `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`

### Files added
- [Experiment notes](experiments/21-winner-repro/notes.md)
- [Experiment results JSON](experiments/21-winner-repro/results.json)
- [HF self-contained job script](scripts/hf_jobs/exp21_hf_job.py)
- [HF submit script](scripts/hf_submit_exp21_job.py)

### Notes
- This is a **proxy reproduction** on `l4x1`, not a compute-matched 8xH100 run.
- L4-safe defaults are applied (`TRAIN_BATCH_TOKENS=131072`, `VAL_BATCH_SIZE=131072`) to reduce OOM risk while preserving the winning model path.
- Submitted first run:
  - Job ID: `69c0a2c771691dc46f1640b3`
  - Run ID: `exp21_winner_repro_l4_v1`
  - Status at log time: `RUNNING`
