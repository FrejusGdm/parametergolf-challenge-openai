# /// script
# dependencies = [
#     "numpy",
#     "tqdm",
#     "torch",
#     "huggingface-hub",
#     "setuptools",
#     "typing-extensions==4.15.0",
#     "datasets",
#     "sentencepiece",
#     "zstandard",
#     "kernels",
# ]
# ///
"""
Curriculum Learning Validation — Full Baseline Model

Runs the ACTUAL train_gpt.py baseline with easy_first vs default shard ordering.
2000 steps each on L4 GPU, single process (no DDP).

This validates whether curriculum learning transfers from our simplified model
to the real competition model (Muon optimizer, U-Net skips, GQA, etc.)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi


# ============================================================================
# SETUP
# ============================================================================

def setup():
    print("=" * 60)
    print("  Curriculum Learning — Full Baseline Validation")
    print("=" * 60)

    # Clone repo
    if not Path("parameter-golf").exists():
        print("Cloning parameter-golf...")
        subprocess.run(["git", "clone", "https://github.com/openai/parameter-golf.git"], check=True)

    # Download all 80 shards
    print("Downloading FineWeb (80 shards)...")
    subprocess.run([
        sys.executable, "parameter-golf/data/cached_challenge_fineweb.py",
        "--variant", "sp1024", "--train-shards", "80"
    ], check=True)

    import torch
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# SHARD ANALYSIS (quick — just entropy for ordering)
# ============================================================================

def load_data_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)


def compute_entropy(shard_path: Path, vocab_size: int = 1024) -> float:
    tokens = load_data_shard(shard_path)
    counts = np.bincount(tokens, minlength=vocab_size)
    freq = counts / counts.sum()
    nonzero = freq[freq > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def analyze_and_order_shards(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return default-ordered and easy-first-ordered shard lists."""
    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    print(f"\nAnalyzing {len(train_files)} shards for entropy...")

    entropies = {}
    for i, f in enumerate(train_files):
        ent = compute_entropy(f)
        entropies[f] = ent
        if (i + 1) % 20 == 0 or i == 0 or i == len(train_files) - 1:
            print(f"  [{i+1}/{len(train_files)}] {f.name}: entropy={ent:.4f}")

    easy_first = sorted(train_files, key=lambda f: entropies[f])
    print(f"\nEasy-first order: {easy_first[0].name} (ent={entropies[easy_first[0]]:.4f}) → "
          f"{easy_first[-1].name} (ent={entropies[easy_first[-1]]:.4f})")

    return train_files, easy_first


# ============================================================================
# CREATE REORDERED DATA DIRECTORY
# ============================================================================

def create_ordered_data_dir(original_dir: Path, ordered_files: list[Path], output_dir: Path):
    """Create a new data directory with shards renamed to force the desired order.

    Since TokenStream reads sorted(glob(pattern)), we rename shards so that
    the desired order matches lexicographic sorting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy validation files as-is
    for val_file in original_dir.glob("fineweb_val_*.bin"):
        dst = output_dir / val_file.name
        if not dst.exists():
            os.link(val_file, dst)

    # Create hardlinks with new names that sort in our desired order
    for i, src in enumerate(ordered_files):
        dst = output_dir / f"fineweb_train_{i:06d}.bin"
        if not dst.exists():
            os.link(src, dst)

    print(f"Created ordered data dir: {output_dir} ({len(ordered_files)} shards)")


# ============================================================================
# RUN TRAINING
# ============================================================================

def run_training(strategy_name: str, data_path: str, n_steps: int, run_id: str) -> dict:
    """Run the actual baseline train_gpt.py with torchrun."""
    print(f"\n{'='*60}")
    print(f"  Training: {strategy_name} — {n_steps} steps")
    print(f"  Data: {data_path}")
    print(f"  Run ID: {run_id}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env.update({
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": "./parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
        "RUN_ID": run_id,
        "ITERATIONS": str(n_steps),
        "TRAIN_BATCH_TOKENS": "65536",  # Smaller batch for L4
        "VAL_LOSS_EVERY": str(n_steps),  # Only validate at the end
        "VAL_BATCH_SIZE": "65536",
        "MAX_WALLCLOCK_SECONDS": "0",  # No wallclock cap
        "WARMUP_STEPS": "10",
        "TRAIN_LOG_EVERY": "100",
        "SEED": "1337",
    })

    t0 = time.perf_counter()
    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=1", "parameter-golf/train_gpt.py"],
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout
    )
    elapsed = time.perf_counter() - t0

    # Parse output for metrics
    stdout = result.stdout
    stderr = result.stderr
    output = stdout + "\n" + stderr

    print(f"\n--- stdout (last 30 lines) ---")
    for line in stdout.strip().split("\n")[-30:]:
        print(line)

    if result.returncode != 0:
        print(f"\n--- stderr (last 20 lines) ---")
        for line in stderr.strip().split("\n")[-20:]:
            print(line)
        print(f"\nWARNING: Training exited with code {result.returncode}")

    # Extract metrics from logs
    metrics = {
        "strategy": strategy_name,
        "run_id": run_id,
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "train_losses": [],
        "val_loss": None,
        "val_bpb": None,
    }

    for line in output.split("\n"):
        if "train_loss:" in line and "step:" in line:
            try:
                parts = line.strip().split()
                step_part = [p for p in parts if p.startswith("step:")][0]
                loss_part = [p for p in parts if p.startswith("train_loss:")][0]
                step = int(step_part.split(":")[1].split("/")[0])
                loss = float(loss_part.split(":")[1])
                metrics["train_losses"].append({"step": step, "loss": loss})
            except (IndexError, ValueError):
                pass

        if "val_loss:" in line and "val_bpb:" in line and "final" not in line.lower():
            try:
                parts = line.strip().split()
                vl = [p for p in parts if p.startswith("val_loss:")][0]
                vb = [p for p in parts if p.startswith("val_bpb:")][0]
                metrics["val_loss"] = float(vl.split(":")[1])
                metrics["val_bpb"] = float(vb.split(":")[1])
            except (IndexError, ValueError):
                pass

        if "final_int8" in line and "val_bpb:" in line:
            try:
                parts = line.strip().split()
                vl = [p for p in parts if p.startswith("val_loss:")][0]
                vb = [p for p in parts if p.startswith("val_bpb:")][0]
                metrics["final_val_loss"] = float(vl.split(":")[1])
                metrics["final_val_bpb"] = float(vb.split(":")[1])
            except (IndexError, ValueError):
                pass

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    setup()

    data_dir = Path("parameter-golf/data/datasets/fineweb10B_sp1024")
    default_files, easy_first_files = analyze_and_order_shards(data_dir)

    n_steps = 2000

    # Create easy-first data directory with reordered shards
    easy_dir = Path("data_easy_first")
    create_ordered_data_dir(data_dir, easy_first_files, easy_dir)

    # Run both strategies
    results = []

    # 1. Default ordering
    r1 = run_training(
        "default",
        str(data_dir),
        n_steps,
        "curriculum_default_2k",
    )
    results.append(r1)
    print(f"\nDefault result: val_loss={r1.get('val_loss')}, val_bpb={r1.get('val_bpb')}")

    # 2. Easy-first ordering
    r2 = run_training(
        "easy_first",
        str(easy_dir),
        n_steps,
        "curriculum_easyfirst_2k",
    )
    results.append(r2)
    print(f"\nEasy-first result: val_loss={r2.get('val_loss')}, val_bpb={r2.get('val_bpb')}")

    # Summary
    print("\n" + "=" * 80)
    print("CURRICULUM LEARNING — FULL BASELINE VALIDATION")
    print("=" * 80)
    for r in results:
        train_final = r["train_losses"][-1]["loss"] if r["train_losses"] else "N/A"
        print(f"  {r['strategy']:15s}  train_loss={train_final}  "
              f"val_loss={r.get('val_loss', 'N/A')}  val_bpb={r.get('val_bpb', 'N/A')}  "
              f"final_bpb={r.get('final_val_bpb', 'N/A')}  time={r['elapsed_s']:.0f}s")

    if results[0].get("val_bpb") and results[1].get("val_bpb"):
        delta = results[0]["val_bpb"] - results[1]["val_bpb"]
        print(f"\n  Delta (default - easy_first): {delta:.4f} bpb")
        if delta > 0:
            print(f"  >>> easy_first is BETTER by {delta:.4f} bpb!")
        else:
            print(f"  >>> default is better by {-delta:.4f} bpb")

    # Push results
    token = os.environ.get("HF_TOKEN")
    if token:
        api = HfApi(token=token)
        repo_id = "JosueG/parameter-golf-curriculum"
        output = {
            "experiment": "curriculum_baseline_validation",
            "model": "full_baseline_train_gpt.py",
            "optimizer": "Muon + Adam (real baseline)",
            "n_steps": n_steps,
            "strategies": ["default", "easy_first"],
            "results": results,
        }
        api.upload_file(
            path_or_fileobj=json.dumps(output, indent=2).encode(),
            path_in_repo="baseline_validation_results.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"\nResults pushed to https://huggingface.co/datasets/{repo_id}")

    print("\nDone!")


if __name__ == "__main__":
    main()
