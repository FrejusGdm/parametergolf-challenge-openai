"""
Monitor and collect results from HF Jobs sweep experiments.

Usage:
    python scripts/monitor_sweeps.py              # Check all recent jobs
    python scripts/monitor_sweeps.py --results     # Fetch and display results from Hub
"""

import json
import sys

from huggingface_hub import HfApi, get_token, hf_hub_download, list_repo_files

RESULTS_REPO = "JosueG/parameter-golf-sweeps"


def fetch_results():
    """Download and display all sweep results from Hub."""
    api = HfApi()

    try:
        files = list_repo_files(RESULTS_REPO, repo_type="dataset")
    except Exception as e:
        print(f"Could not access {RESULTS_REPO}: {e}")
        return

    sweep_files = sorted([f for f in files if f.startswith("sweeps/") and f.endswith(".json")])

    if not sweep_files:
        print("No results found yet.")
        return

    print(f"{'Experiment':<30} {'val_bpb':>10} {'val_loss':>10} {'Time (s)':>10}")
    print("-" * 65)

    results = []
    for f in sweep_files:
        try:
            path = hf_hub_download(RESULTS_REPO, f, repo_type="dataset")
            with open(path) as fh:
                data = json.load(fh)
            results.append(data)
            print(
                f"{data.get('experiment', '?'):<30} "
                f"{data.get('val_bpb', 'N/A'):>10} "
                f"{data.get('val_loss', 'N/A'):>10} "
                f"{data.get('elapsed_seconds', 'N/A'):>10}"
            )
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    # Group by experiment prefix and find winners
    if results:
        print(f"\n{'='*65}")
        print("Winners by experiment group:")
        print("-" * 65)

        groups = {}
        for r in results:
            prefix = r["experiment"].rsplit("-", 1)[0] if "-" in r["experiment"] else r["experiment"]
            # Use the experiment number as group key
            group = prefix.split("-")[0]
            if group not in groups:
                groups[group] = []
            groups[group].append(r)

        for group in sorted(groups.keys()):
            valid = [r for r in groups[group] if r.get("val_bpb", "N/A") != "N/A"]
            if valid:
                best = min(valid, key=lambda r: float(r["val_bpb"]))
                print(f"  Exp {group}: best val_bpb={best['val_bpb']} ({best['experiment']})")


def main():
    if "--results" in sys.argv:
        fetch_results()
    else:
        # Show job status
        api = HfApi()
        print("Use --results to fetch and display results from Hub.")
        print(f"Results repo: https://huggingface.co/datasets/{RESULTS_REPO}")


if __name__ == "__main__":
    main()
