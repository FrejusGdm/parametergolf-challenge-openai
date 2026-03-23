"""
Submit experiment 18 variants to HF Jobs.

Usage:
    python scripts/submit_exp18.py              # Submit both variants
    python scripts/submit_exp18.py 18a          # Just GEGLU
    python scripts/submit_exp18.py 18b          # Just HP tweaks
    python scripts/submit_exp18.py --dry-run    # Preview
"""

import sys
import time

from huggingface_hub import HfApi, get_token

VARIANTS = {
    "18a": {"VARIANT": "18a_geglu", "ITERATIONS": "2000"},
    "18b": {"VARIANT": "18b_hp_tweaks", "ITERATIONS": "2000"},
}

FLAVOR = "l4x1"
TIMEOUT = "4h"  # 2000 steps takes longer
SCRIPT_PATH = "scripts/hf_exp18_job.py"


def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        selected = {k: v for k, v in VARIANTS.items() if k in args}
    else:
        selected = VARIANTS

    api = HfApi()
    token = get_token()

    print(f"{'[DRY RUN] ' if dry_run else ''}Submitting {len(selected)} exp 18 jobs ({FLAVOR})")

    for name, env_vars in selected.items():
        print(f"\nSubmitting: {name} ({env_vars['VARIANT']})")
        if not dry_run:
            job = api.run_uv_job(
                SCRIPT_PATH,
                flavor=FLAVOR,
                timeout=TIMEOUT,
                env=env_vars,
                secrets={"HF_TOKEN": token},
            )
            print(f"  Job ID: {job.id}")
            print(f"  Status: {job.status.stage}")
            time.sleep(20)

    print("\nMonitor: python scripts/monitor_sweeps.py --results")


if __name__ == "__main__":
    main()
