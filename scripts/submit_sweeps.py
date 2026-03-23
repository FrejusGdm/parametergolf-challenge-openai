"""
Submit all hyperparameter sweep experiments to HF Jobs.

Usage:
    python scripts/submit_sweeps.py           # Submit all experiments
    python scripts/submit_sweeps.py 10 11 12  # Submit specific experiments
    python scripts/submit_sweeps.py --dry-run # Preview without submitting
"""

import sys
import time

from huggingface_hub import HfApi, get_token

# ── Define all sweep experiments ───────────────────────────────────────────

EXPERIMENTS = {
    # Exp 10: Matrix LR sweep
    "10-lr-sweep-0.01": {"EXP_NAME": "10-lr-sweep-lr0.01", "SWEEP_OVERRIDES": "MATRIX_LR=0.01"},
    "10-lr-sweep-0.02": {"EXP_NAME": "10-lr-sweep-lr0.02", "SWEEP_OVERRIDES": "MATRIX_LR=0.02"},
    "10-lr-sweep-0.04": {"EXP_NAME": "10-lr-sweep-lr0.04", "SWEEP_OVERRIDES": "MATRIX_LR=0.04"},
    "10-lr-sweep-0.08": {"EXP_NAME": "10-lr-sweep-lr0.08", "SWEEP_OVERRIDES": "MATRIX_LR=0.08"},

    # Exp 11: Embed LR sweep
    "11-embed-lr-0.02": {"EXP_NAME": "11-embed-lr-0.02", "SWEEP_OVERRIDES": "TIED_EMBED_LR=0.02"},
    "11-embed-lr-0.05": {"EXP_NAME": "11-embed-lr-0.05", "SWEEP_OVERRIDES": "TIED_EMBED_LR=0.05"},
    "11-embed-lr-0.10": {"EXP_NAME": "11-embed-lr-0.10", "SWEEP_OVERRIDES": "TIED_EMBED_LR=0.1"},
    "11-embed-lr-0.15": {"EXP_NAME": "11-embed-lr-0.15", "SWEEP_OVERRIDES": "TIED_EMBED_LR=0.15"},

    # Exp 12: Warmup steps sweep
    "12-warmup-5":   {"EXP_NAME": "12-warmup-5",   "SWEEP_OVERRIDES": "WARMUP_STEPS=5"},
    "12-warmup-20":  {"EXP_NAME": "12-warmup-20",  "SWEEP_OVERRIDES": "WARMUP_STEPS=20"},
    "12-warmup-50":  {"EXP_NAME": "12-warmup-50",  "SWEEP_OVERRIDES": "WARMUP_STEPS=50"},
    "12-warmup-100": {"EXP_NAME": "12-warmup-100", "SWEEP_OVERRIDES": "WARMUP_STEPS=100"},

    # Exp 13: Warmdown iterations sweep
    "13-warmdown-600":  {"EXP_NAME": "13-warmdown-600",  "SWEEP_OVERRIDES": "WARMDOWN_ITERS=600"},
    "13-warmdown-1200": {"EXP_NAME": "13-warmdown-1200", "SWEEP_OVERRIDES": "WARMDOWN_ITERS=1200"},
    "13-warmdown-2400": {"EXP_NAME": "13-warmdown-2400", "SWEEP_OVERRIDES": "WARMDOWN_ITERS=2400"},
    "13-warmdown-3600": {"EXP_NAME": "13-warmdown-3600", "SWEEP_OVERRIDES": "WARMDOWN_ITERS=3600"},

    # Exp 14: Batch size sweep
    "14-batch-262k": {"EXP_NAME": "14-batch-262k", "SWEEP_OVERRIDES": "TRAIN_BATCH_TOKENS=262144"},
    "14-batch-524k": {"EXP_NAME": "14-batch-524k", "SWEEP_OVERRIDES": "TRAIN_BATCH_TOKENS=524288"},
    "14-batch-1M":   {"EXP_NAME": "14-batch-1M",   "SWEEP_OVERRIDES": "TRAIN_BATCH_TOKENS=1048576"},

    # Exp 15: Sequence length sweep
    "15-seqlen-512":  {"EXP_NAME": "15-seqlen-512",  "SWEEP_OVERRIDES": "TRAIN_SEQ_LEN=512"},
    "15-seqlen-1024": {"EXP_NAME": "15-seqlen-1024", "SWEEP_OVERRIDES": "TRAIN_SEQ_LEN=1024"},
    "15-seqlen-2048": {"EXP_NAME": "15-seqlen-2048", "SWEEP_OVERRIDES": "TRAIN_SEQ_LEN=2048"},

    # Exp 16: Muon momentum sweep
    "16-momentum-0.90": {"EXP_NAME": "16-momentum-0.90", "SWEEP_OVERRIDES": "MUON_MOMENTUM=0.90"},
    "16-momentum-0.95": {"EXP_NAME": "16-momentum-0.95", "SWEEP_OVERRIDES": "MUON_MOMENTUM=0.95"},
    "16-momentum-0.99": {"EXP_NAME": "16-momentum-0.99", "SWEEP_OVERRIDES": "MUON_MOMENTUM=0.99"},

    # Exp 17: Weight decay sweep (may need code change — test first)
    "17-wd-0":    {"EXP_NAME": "17-wd-0",    "SWEEP_OVERRIDES": "WEIGHT_DECAY=0"},
    "17-wd-0.01": {"EXP_NAME": "17-wd-0.01", "SWEEP_OVERRIDES": "WEIGHT_DECAY=0.01"},
    "17-wd-0.04": {"EXP_NAME": "17-wd-0.04", "SWEEP_OVERRIDES": "WEIGHT_DECAY=0.04"},
}

FLAVOR = "l4x1"
TIMEOUT = "3h"
SCRIPT_PATH = "scripts/hf_sweep_job.py"


def submit_experiment(api, name, env_vars, dry_run=False, max_retries=3):
    """Submit a single sweep experiment to HF Jobs with retry on rate limit."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Submitting: {name}")
    print(f"  Overrides: {env_vars.get('SWEEP_OVERRIDES', '')}")

    if dry_run:
        return None

    token = get_token()
    for attempt in range(max_retries):
        try:
            job = api.run_uv_job(
                SCRIPT_PATH,
                flavor=FLAVOR,
                timeout=TIMEOUT,
                env=env_vars,
                secrets={"HF_TOKEN": token},
            )
            print(f"  Job ID: {job.id}")
            print(f"  Status: {job.status.stage}")
            return job
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    return None


def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # Filter experiments if specific numbers given
    if args:
        prefixes = [f"{a}-" for a in args]
        filtered = {k: v for k, v in EXPERIMENTS.items() if any(k.startswith(p) for p in prefixes)}
    else:
        filtered = EXPERIMENTS

    if not filtered:
        print("No matching experiments found.")
        sys.exit(1)

    print(f"{'[DRY RUN] ' if dry_run else ''}Submitting {len(filtered)} sweep jobs to HF Jobs ({FLAVOR})")

    api = HfApi()
    jobs = []

    for name, env_vars in filtered.items():
        job = submit_experiment(api, name, env_vars, dry_run=dry_run)
        if job:
            jobs.append((name, job))
        # Delay between submissions to avoid HF rate limiting on /whoami
        if not dry_run:
            time.sleep(15)

    if jobs:
        print(f"\n{'='*50}")
        print(f"Submitted {len(jobs)} jobs. Monitor with:")
        print(f"  python scripts/monitor_sweeps.py")
        print(f"\nJob IDs:")
        for name, job in jobs:
            print(f"  {name}: {job.id}")


if __name__ == "__main__":
    main()
