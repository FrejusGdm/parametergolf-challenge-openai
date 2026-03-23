# Hyperparameter Sweep (Experiments 10-19)

## Overview
A systematic sweep of basic training hyperparameters to understand what each knob does and find better defaults. Each experiment varies one parameter while keeping everything else at baseline.

## How to run

All experiments run remotely on HF Jobs (L4 GPUs, ~$0.50/hr each).

### Submit experiments
```bash
# Activate venv
source .venv/bin/activate

# Submit all 28 jobs (experiments 10-17, 3-4 values each)
python scripts/submit_sweeps.py

# Submit specific experiments only
python scripts/submit_sweeps.py 10 11 12

# Preview without submitting
python scripts/submit_sweeps.py --dry-run
```

### Monitor and collect results
```bash
# Fetch results from HF Hub and display summary
python scripts/monitor_sweeps.py --results
```

Results are pushed to: https://huggingface.co/datasets/JosueG/parameter-golf-sweeps

### What each job does
1. Clones `openai/parameter-golf` on the remote GPU
2. Downloads 1 training shard (~100M tokens) via `cached_challenge_fineweb.py`
3. Runs `torchrun --nproc_per_node=1 train_gpt.py` for 500 iterations
4. Parses `val_bpb` from output
5. Pushes result JSON to HF Hub

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/hf_sweep_job.py` | Self-contained job script (runs on HF Jobs) |
| `scripts/submit_sweeps.py` | Submits jobs from local machine |
| `scripts/monitor_sweeps.py` | Fetches and displays results from Hub |

## Experiments 18-19 (manual)
After reviewing results from 10-17, manually configure the combined experiments:
- **18**: Best values from all sweeps combined
- **19**: Best values + `MLP_MULT=3`

These require editing `EXPERIMENTS` in `submit_sweeps.py` or running `hf_sweep_job.py` directly with multiple overrides.
