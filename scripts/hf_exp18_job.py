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
Experiment 18: Test novel tweaks on top of the winner recipe at 2000 steps.

Variants:
  18a_geglu    — Winner + GEGLU activation (swap relu² for GEGLU)
  18b_hp_tweaks — Winner + matrix_lr=0.04, tied_embed_lr=0.02 (testing different LRs)

Winner baseline (exp 21):
  matrix_lr=0.02, tied_embed_lr=0.03, embed_lr=0.6, muon_momentum=0.99, WD=0.04
  10 layers, MLP 3x, BigramHash(10240), SWA, int5/int6 quant

Environment variables:
  VARIANT     — 18a_geglu or 18b_hp_tweaks
  HF_TOKEN    — for pushing results
  ITERATIONS  — training steps (default 2000)
"""

import json
import os
import re
import subprocess
import sys
import time

VARIANT = os.environ.get("VARIANT", "18a_geglu")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-sweeps")
ITERATIONS = os.environ.get("ITERATIONS", "2000")
PINNED_COMMIT = "9f9d53343aa44fe1fbd94ae32650ca2e83602a10"
WINNER_SCRIPT = "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"

print(f"=== Experiment 18: {VARIANT} ===")
print(f"Iterations: {ITERATIONS}")

# ── Clone repo at pinned commit ────────────────────────────────────────────
subprocess.run(["git", "clone", "https://github.com/openai/parameter-golf.git"], check=True)
os.chdir("parameter-golf")
subprocess.run(["git", "checkout", PINNED_COMMIT], check=True)

# ── Copy winner script to root ─────────────────────────────────────────────
import shutil
shutil.copy(WINNER_SCRIPT, "train_gpt.py")
print(f"Using winner script from {WINNER_SCRIPT}")

# ── Download dataset ───────────────────────────────────────────────────────
print("\n=== Downloading dataset (1 shard) ===")
subprocess.run(
    [sys.executable, "data/cached_challenge_fineweb.py", "--variant", "sp1024", "--train-shards", "1"],
    check=True,
)

# ── Apply variant-specific patches ─────────────────────────────────────────
if VARIANT == "18a_geglu":
    # Patch MLP to use GEGLU
    with open("train_gpt.py", "r") as f:
        code = f.read()

    old_mlp = """class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())"""

    new_mlp = """class MLP(nn.Module):
    # GEGLU gated MLP (Shazeer, 2020 arXiv:2002.05202)
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        # Reduce hidden dim by 2/3 to compensate for extra gate matrix
        hidden = int(2 * mlp_mult * dim / 3)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.nn.functional.gelu(self.gate(x)) * self.fc(x))"""

    if old_mlp not in code:
        print("WARNING: Could not find MLP class to patch! Trying with float mlp_mult...")
        # The winner may use float mlp_mult
        old_mlp2 = old_mlp.replace("mlp_mult: int", "mlp_mult: float")
        if old_mlp2 in code:
            code = code.replace(old_mlp2, new_mlp.replace("mlp_mult: int", "mlp_mult: float"))
            print("Patched MLP with GEGLU (float mlp_mult)")
        else:
            print("ERROR: Could not patch MLP class")
            sys.exit(1)
    else:
        code = code.replace(old_mlp, new_mlp)
        print("Patched MLP with GEGLU")

    with open("train_gpt.py", "w") as f:
        f.write(code)

# ── Build training env ─────────────────────────────────────────────────────
train_env = os.environ.copy()
train_env.update({
    "ITERATIONS": ITERATIONS,
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_LOSS_EVERY": "0",
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "RUN_ID": f"exp18_{VARIANT}",
    # L4-safe batch settings (matching exp 21)
    "TRAIN_BATCH_TOKENS": "131072",
    "VAL_BATCH_SIZE": "131072",
})

if VARIANT == "18b_hp_tweaks":
    # Override LR params — testing whether our sweep findings help the winner
    train_env["MATRIX_LR"] = "0.04"       # Winner uses 0.02, baseline 0.04, sweep said 0.08
    train_env["TIED_EMBED_LR"] = "0.02"   # Winner uses 0.03, sweep said 0.02
    print("HP tweaks: MATRIX_LR=0.04, TIED_EMBED_LR=0.02")

# ── Train ──────────────────────────────────────────────────────────────────
print(f"\n=== Training ({ITERATIONS} iters, variant={VARIANT}) ===")
start_time = time.time()

result = subprocess.run(
    ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
    env=train_env,
    capture_output=True,
    text=True,
)

elapsed = time.time() - start_time
output = result.stdout + "\n" + result.stderr
print(output)

if result.returncode != 0:
    print(f"\n=== Training FAILED (exit code {result.returncode}) ===")
    sys.exit(1)

# ── Parse results ──────────────────────────────────────────────────────────
val_bpb_match = re.findall(r"val_bpb[=: ]+([0-9.]+)", output)
val_loss_match = re.findall(r"val_loss[=: ]+([0-9.]+)", output)

val_bpb = val_bpb_match[-1] if val_bpb_match else "N/A"
val_loss = val_loss_match[-1] if val_loss_match else "N/A"

print(f"\n=== Result: variant={VARIANT} val_bpb={val_bpb} val_loss={val_loss} ===")
print(f"=== Elapsed: {elapsed:.1f}s ===")

# ── Push to Hub ────────────────────────────────────────────────────────────
results = {
    "experiment": f"18-{VARIANT}",
    "variant": VARIANT,
    "base": "winner_10L_Int5MLP_BigramHash_SWA_WD04",
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
    try:
        api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception:
        pass
    filename = f"18-{VARIANT}.json"
    api.upload_file(
        path_or_fileobj=results_json.encode(),
        path_in_repo=f"sweeps/{filename}",
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"\nResults pushed to https://huggingface.co/datasets/{RESULTS_REPO}/blob/main/sweeps/{filename}")
