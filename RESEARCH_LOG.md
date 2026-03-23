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
