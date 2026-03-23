"""
Submit activation function sweep experiments to HF Jobs.

Usage:
    python scripts/submit_activation_sweep.py                          # Submit all 13
    python scripts/submit_activation_sweep.py --activations swiglu gelu silu  # Specific ones
    python scripts/submit_activation_sweep.py --dry-run                # Preview only
"""

import sys
import time

from huggingface_hub import HfApi, get_token

ALL_ACTIVATIONS = [
    # Non-gated (same param count as baseline)
    "relu_sq",      # baseline
    "gelu",
    "silu",
    "relu",
    "softmax",
    "relu_cubed",
    "sigmoid",
    "tanh",
    "softplus",
    "mish",
    # Gated (extra gate matrix, reduced hidden dim)
    "swiglu",
    "geglu",
    "reglu",
]

FLAVOR = "l4x1"
TIMEOUT = "3h"
SCRIPT_PATH = "scripts/hf_activation_sweep.py"


def submit(api, activation, dry_run=False, max_retries=3):
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Submitting: activation={activation}")

    if dry_run:
        return None

    token = get_token()
    env_vars = {"ACTIVATION": activation, "ITERATIONS": "500"}

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

    # Parse --activations flag
    activations = ALL_ACTIVATIONS
    if "--activations" in sys.argv:
        idx = sys.argv.index("--activations")
        activations = []
        for a in sys.argv[idx + 1:]:
            if a.startswith("--"):
                break
            activations.append(a)

    if not activations:
        print("No activations specified.")
        sys.exit(1)

    print(f"{'[DRY RUN] ' if dry_run else ''}Submitting {len(activations)} activation sweep jobs ({FLAVOR})")
    print(f"Activations: {', '.join(activations)}")

    api = HfApi()
    jobs = []

    for act in activations:
        job = submit(api, act, dry_run=dry_run)
        if job:
            jobs.append((act, job))
        if not dry_run:
            time.sleep(20)

    if jobs:
        print(f"\n{'='*50}")
        print(f"Submitted {len(jobs)} jobs.")
        print(f"\nJob IDs:")
        for act, job in jobs:
            print(f"  {act}: {job.id}")
        print(f"\nMonitor: python scripts/monitor_sweeps.py --results")


if __name__ == "__main__":
    main()
