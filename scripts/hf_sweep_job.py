# /// script
# dependencies = [
#     "numpy",
#     "torch",
#     "huggingface-hub",
#     "sentencepiece",
#     "datasets",
#     "tqdm",
#     "setuptools",
#     "typing-extensions==4.15.0",
#     "kernels",
#     "tiktoken",
# ]
# ///
"""
Self-contained hyperparameter sweep experiment for HF Jobs.

Expects these environment variables:
  HF_TOKEN         — for pushing results to Hub
  EXP_NAME         — experiment name (e.g., "10-lr-sweep")
  SWEEP_OVERRIDES  — comma-separated env overrides (e.g., "MATRIX_LR=0.08,ITERATIONS=500")
"""

import json
import os
import re
import subprocess
import sys
import time

# ── Configuration from environment ──────────────────────────────────────────
EXP_NAME = os.environ.get("EXP_NAME", "sweep-test")
SWEEP_OVERRIDES = os.environ.get("SWEEP_OVERRIDES", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-sweeps")
ITERATIONS = os.environ.get("ITERATIONS", "500")

# ── Clone the parameter-golf repo ──────────────────────────────────────────
print(f"=== HF Jobs Sweep: {EXP_NAME} ===")
print(f"Overrides: {SWEEP_OVERRIDES}")

subprocess.run(
    ["git", "clone", "https://github.com/openai/parameter-golf.git"],
    check=True,
)
os.chdir("parameter-golf")

# ── Download dataset (1 shard for fast sweeps) ─────────────────────────────
print("\n=== Downloading dataset (1 shard) ===")
subprocess.run(
    [sys.executable, "data/cached_challenge_fineweb.py", "--variant", "sp1024", "--train-shards", "1"],
    check=True,
)

# ── Build environment for training ─────────────────────────────────────────
train_env = os.environ.copy()
train_env.update({
    "ITERATIONS": ITERATIONS,
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_LOSS_EVERY": "0",
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "RUN_ID": f"sweep_{EXP_NAME}",
    # L4 has 24GB VRAM — grad_accum_steps is hardcoded to 8 in train_gpt.py,
    # so reduce TRAIN_BATCH_TOKENS so each microbatch (tokens/8) fits in 24GB.
    # 65536/8 = 8192 tokens per microbatch. Effective batch is smaller than H100
    # but sufficient for relative comparisons.
    "TRAIN_BATCH_TOKENS": "65536",
    "VAL_BATCH_SIZE": "65536",
})

# Apply sweep overrides
if SWEEP_OVERRIDES:
    for pair in SWEEP_OVERRIDES.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            train_env[key] = value
            print(f"  Override: {key}={value}")

# ── Run training ───────────────────────────────────────────────────────────
print(f"\n=== Training ({ITERATIONS} iterations) ===")
start_time = time.time()

result = subprocess.run(
    ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
    env=train_env,
    capture_output=True,
    text=True,
)

elapsed = time.time() - start_time
output = result.stdout + "\n" + result.stderr

# Print full output for HF Jobs logs
print(output)

if result.returncode != 0:
    print(f"\n=== Training FAILED (exit code {result.returncode}) ===")
    sys.exit(1)

# ── Parse results ──────────────────────────────────────────────────────────
val_bpb_match = re.findall(r"val_bpb[=: ]+([0-9.]+)", output)
val_loss_match = re.findall(r"val_loss[=: ]+([0-9.]+)", output)

val_bpb = val_bpb_match[-1] if val_bpb_match else "N/A"
val_loss = val_loss_match[-1] if val_loss_match else "N/A"

print(f"\n=== Result: val_bpb={val_bpb}  val_loss={val_loss} ===")
print(f"=== Elapsed: {elapsed:.1f}s ===")

# ── Push results to Hub ────────────────────────────────────────────────────
results = {
    "experiment": EXP_NAME,
    "overrides": SWEEP_OVERRIDES,
    "iterations": int(ITERATIONS),
    "val_bpb": val_bpb,
    "val_loss": val_loss,
    "elapsed_seconds": round(elapsed, 1),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

results_json = json.dumps(results, indent=2)
print(f"\n{results_json}")

if HF_TOKEN:
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # Ensure repo exists
    try:
        api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Upload results
    filename = f"{EXP_NAME}.json"
    api.upload_file(
        path_or_fileobj=results_json.encode(),
        path_in_repo=f"sweeps/{filename}",
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"\nResults pushed to https://huggingface.co/datasets/{RESULTS_REPO}/blob/main/sweeps/{filename}")
else:
    print("\nNo HF_TOKEN — skipping Hub push")
