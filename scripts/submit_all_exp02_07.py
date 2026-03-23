#!/usr/bin/env python3
"""
Submit experiments 02-07 to HF Jobs.

Usage:
    python scripts/submit_all_exp02_07.py --all          # Submit all 6
    python scripts/submit_all_exp02_07.py --exp 04       # Submit one
    python scripts/submit_all_exp02_07.py --exp 02,04,06 # Submit specific ones
    python scripts/submit_all_exp02_07.py --list         # List available experiments
"""

import argparse
import sys

EXPERIMENTS = {
    "02": {"script": "scripts/hf_exp02_zstd_job.py", "name": "zstd compression"},
    "03": {"script": "scripts/hf_exp03_int6_job.py", "name": "Int6 + QAT"},
    "04": {"script": "scripts/hf_exp04_mlp3x_job.py", "name": "3x MLP"},
    "05": {"script": "scripts/hf_exp05_bigram_job.py", "name": "BigramHash + SmearGate"},
    "06": {"script": "scripts/hf_exp06_layers_job.py", "name": "10 Layers"},
    "07": {"script": "scripts/hf_exp07_swa_job.py", "name": "SWA"},
}


def main():
    parser = argparse.ArgumentParser(description="Submit exp 02-07 to HF Jobs")
    parser.add_argument("--all", action="store_true", help="Submit all 6 experiments")
    parser.add_argument("--exp", type=str, help="Experiment number(s), comma-separated (e.g., '04' or '02,04,06')")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--flavor", type=str, default="l4x1", help="HF Jobs GPU flavor")
    parser.add_argument("--timeout", type=str, default="3h", help="Job timeout")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for num, info in EXPERIMENTS.items():
            print(f"  {num}: {info['name']} ({info['script']})")
        return

    if not args.all and not args.exp:
        parser.print_help()
        return

    # Determine which experiments to run
    if args.all:
        to_run = list(EXPERIMENTS.keys())
    else:
        to_run = [e.strip() for e in args.exp.split(",")]
        for e in to_run:
            if e not in EXPERIMENTS:
                print(f"Unknown experiment: {e}. Use --list to see available.")
                sys.exit(1)

    from huggingface_hub import HfApi, get_token
    api = HfApi()
    token = get_token()

    print(f"Submitting {len(to_run)} experiment(s) to HF Jobs ({args.flavor})...\n")

    jobs = []
    for exp_num in to_run:
        info = EXPERIMENTS[exp_num]
        print(f"  Exp {exp_num} ({info['name']})...", end=" ", flush=True)
        try:
            job = api.run_uv_job(
                info["script"],
                flavor=args.flavor,
                timeout=args.timeout,
                secrets={"HF_TOKEN": token},
            )
            print(f"Job ID: {job.id}")
            jobs.append({"exp": exp_num, "name": info["name"], "job_id": job.id})
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"Submitted {len(jobs)}/{len(to_run)} jobs")
    for j in jobs:
        print(f"  Exp {j['exp']} ({j['name']}): {j['job_id']}")
    print(f"\nMonitor with:")
    print(f"  python -c \"from huggingface_hub import HfApi; api=HfApi(); "
          f"[print(f'{{j.id}}: {{j.status.stage}}') for j in api.list_jobs()]\"")


if __name__ == "__main__":
    main()
