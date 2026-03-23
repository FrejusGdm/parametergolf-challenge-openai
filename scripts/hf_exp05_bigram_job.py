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
Experiment 05: BigramHash + SmearGate

Adds two cheap modules to capture local bigram statistics and blend adjacent
token representations, inspired by top leaderboard submissions.

BigramHash: hashes (prev_token, cur_token) into a fixed bucket embedding,
then projects into model_dim and adds to the token embedding. This gives the
model explicit bigram frequency information without extra vocabulary.

SmearGate: a learned per-dimension gate that blends each token's embedding
with its predecessor, providing a cheap local context signal before the
first attention layer.

Two runs:
  1. Baseline (unmodified) for A/B comparison
  2. Baseline + BigramHash + SmearGate

Environment variables:
  HF_TOKEN        — for pushing results (required)
  ITERATIONS      — training steps (default 2000)
  BIGRAM_BUCKETS  — number of hash buckets (default 4096)
  BIGRAM_DIM      — intermediate embedding dim (default 128)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

ITERATIONS = os.environ.get("ITERATIONS", "2000")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RESULTS_REPO = "JosueG/parameter-golf-experiments"
TRAIN_SHARDS = os.environ.get("TRAIN_SHARDS", "20")
BIGRAM_BUCKETS = os.environ.get("BIGRAM_BUCKETS", "4096")
BIGRAM_DIM = os.environ.get("BIGRAM_DIM", "128")
PINNED_COMMIT = "9f9d53343aa44fe1fbd94ae32650ca2e83602a10"


def run_cmd(cmd, cwd=None, env=None):
    print(f"[cmd] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    rc = proc.wait()
    output = "".join(output_lines)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")
    return output


def parse_results(output):
    """Extract val_bpb, val_loss, artifact size from training output."""
    results = {}

    step_rows = re.findall(
        r"step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+) train_time:([0-9]+)ms step_avg:([0-9.]+)ms",
        output,
    )
    if step_rows:
        last = step_rows[-1]
        results["pre_quant_val_loss"] = float(last[2])
        results["pre_quant_val_bpb"] = float(last[3])
        results["train_time_ms"] = int(last[4])
        results["step_avg_ms"] = float(last[5])

    final_q = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)", output
    )
    if final_q:
        results["post_quant_val_loss"] = float(final_q.group(1))
        results["post_quant_val_bpb"] = float(final_q.group(2))

    artifact = re.search(r"Total submission size int8\+zlib: ([0-9]+) bytes", output)
    if artifact:
        results["artifact_bytes"] = int(artifact.group(1))

    return results


def get_train_env(run_id):
    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id,
        "ITERATIONS": ITERATIONS,
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_LOSS_EVERY": ITERATIONS,
        "VAL_BATCH_SIZE": "65536",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMUP_STEPS": "10",
        "TRAIN_LOG_EVERY": "200",
        "SEED": "1337",
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
    })
    return env


def apply_bigram_smeargate_patches(script_path):
    """Patch train_gpt.py to add BigramHash and SmearGate modules."""
    with open(script_path, "r") as f:
        code = f.read()

    n_buckets = int(BIGRAM_BUCKETS)
    bigram_dim = int(BIGRAM_DIM)

    # ── Patch 1: Add BigramHash class before the GPT class ──
    bigram_class = f'''
class BigramHash(nn.Module):
    """Hash-based bigram feature extractor.
    Maps (prev_token, cur_token) pairs to learned embeddings via modular hashing."""
    def __init__(self, n_buckets={n_buckets}, bigram_dim={bigram_dim}, model_dim=512):
        super().__init__()
        self.n_buckets = n_buckets
        self.embed = nn.Embedding(n_buckets, bigram_dim)
        self.proj = nn.Linear(bigram_dim, model_dim, bias=False)
        nn.init.normal_(self.embed.weight, std=0.01)
        nn.init.zeros_(self.proj.weight)

    def forward(self, input_ids):
        prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
        hashes = (prev * 31 + input_ids) % self.n_buckets
        return self.proj(self.embed(hashes))

'''

    gpt_anchor = "class GPT(nn.Module):"
    if gpt_anchor not in code:
        raise RuntimeError(f"Could not find '{gpt_anchor}' in train_gpt.py")
    code = code.replace(gpt_anchor, bigram_class + gpt_anchor)
    print("Patch 1: Added BigramHash class")

    # ── Patch 2: Add BigramHash and SmearGate to GPT.__init__ ──
    # Find the line where self.final_norm is created and add our modules before it
    old_init_line = "        self.final_norm = RMSNorm()"
    new_init_lines = """        self.bigram_hash = BigramHash(n_buckets={n_buckets}, bigram_dim={bigram_dim}, model_dim=model_dim)
        self.smear_gate = nn.Parameter(torch.zeros(model_dim))
        self.final_norm = RMSNorm()""".format(
        n_buckets=n_buckets, bigram_dim=bigram_dim
    )

    if old_init_line not in code:
        raise RuntimeError("Could not find 'self.final_norm = RMSNorm()' in GPT.__init__")
    code = code.replace(old_init_line, new_init_lines)
    print("Patch 2: Added BigramHash and smear_gate to GPT.__init__")

    # ── Patch 3: Modify GPT.forward to use BigramHash and SmearGate ──
    # The forward starts with embedding, then RMS norm, then x0 = x
    old_forward_start = """    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x"""

    new_forward_start = """    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        # Add bigram hash features
        x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        # SmearGate: blend with previous token embedding
        prev_emb = F.pad(x[:, :-1], (0, 0, 1, 0))
        gate = torch.sigmoid(self.smear_gate)
        x = (1 - gate) * x + gate * prev_emb
        x0 = x"""

    if old_forward_start not in code:
        raise RuntimeError("Could not find GPT.forward start to patch")
    code = code.replace(old_forward_start, new_forward_start)
    print("Patch 3: Modified GPT.forward with BigramHash + SmearGate")

    # ── Patch 4: Add bigram_hash params to optimizer (scalar/Adam group) ──
    # The bigram_hash projection is a 2D matrix, but it's small enough to go with Adam.
    # The embedding is also best with Adam. Add them to the scalar_params list.
    old_scalar_append = """    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)"""
    new_scalar_append = """    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    # Add BigramHash and SmearGate params to scalar optimizer
    for p in base_model.bigram_hash.parameters():
        scalar_params.append(p)
    scalar_params.append(base_model.smear_gate)"""

    if old_scalar_append not in code:
        raise RuntimeError("Could not find skip_weights optimizer setup to patch")
    code = code.replace(old_scalar_append, new_scalar_append)
    print("Patch 4: Added bigram_hash + smear_gate params to optimizer")

    with open(script_path, "w") as f:
        f.write(code)
    print("All BigramHash + SmearGate patches applied successfully")


def main():
    print("=" * 80)
    print("Experiment 05: BigramHash + SmearGate")
    print(f"Iterations: {ITERATIONS}, Train shards: {TRAIN_SHARDS}")
    print(f"BigramHash config: buckets={BIGRAM_BUCKETS}, dim={BIGRAM_DIM}")
    print("=" * 80)

    # ── Clone and setup ──
    run_cmd(["git", "clone", "https://github.com/openai/parameter-golf.git"])
    os.chdir("parameter-golf")
    run_cmd(["git", "checkout", PINNED_COMMIT])

    print("\n=== Downloading dataset ===")
    run_cmd([
        sys.executable, "data/cached_challenge_fineweb.py",
        "--variant", "sp1024",
        "--train-shards", TRAIN_SHARDS,
    ])

    # ── Run 1: Baseline ──
    print("\n" + "=" * 80)
    print("Run 1: BASELINE (unmodified)")
    print("=" * 80)

    shutil.copy("train_gpt.py", "train_gpt_baseline.py")

    t_start = time.time()
    baseline_output = run_cmd(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
        env=get_train_env("exp05_baseline"),
    )
    baseline_elapsed = time.time() - t_start
    baseline_results = parse_results(baseline_output)
    baseline_results["elapsed_seconds"] = round(baseline_elapsed, 1)
    print(f"\nBaseline results: {json.dumps(baseline_results, indent=2)}")

    # ── Run 2: BigramHash + SmearGate ──
    print("\n" + "=" * 80)
    print("Run 2: BigramHash + SmearGate")
    print("=" * 80)

    shutil.copy("train_gpt_baseline.py", "train_gpt.py")
    apply_bigram_smeargate_patches("train_gpt.py")

    t_start = time.time()
    bigram_output = run_cmd(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
        env=get_train_env("exp05_bigram_smeargate"),
    )
    bigram_elapsed = time.time() - t_start
    bigram_results = parse_results(bigram_output)
    bigram_results["elapsed_seconds"] = round(bigram_elapsed, 1)
    print(f"\nBigram+SmearGate results: {json.dumps(bigram_results, indent=2)}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("EXPERIMENT 05 SUMMARY")
    print("=" * 80)

    summary = {
        "experiment": "05-bigram-features",
        "description": "BigramHash + SmearGate local context features",
        "config": {
            "bigram_buckets": int(BIGRAM_BUCKETS),
            "bigram_dim": int(BIGRAM_DIM),
        },
        "iterations": int(ITERATIONS),
        "train_shards": int(TRAIN_SHARDS),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline_results,
        "bigram_smeargate": bigram_results,
    }

    if "post_quant_val_bpb" in baseline_results and "post_quant_val_bpb" in bigram_results:
        delta = bigram_results["post_quant_val_bpb"] - baseline_results["post_quant_val_bpb"]
        summary["delta_post_quant_val_bpb"] = round(delta, 6)
        print(f"Post-quant val_bpb delta: {delta:+.6f} ({'better' if delta < 0 else 'worse'})")

    if "pre_quant_val_bpb" in baseline_results and "pre_quant_val_bpb" in bigram_results:
        delta = bigram_results["pre_quant_val_bpb"] - baseline_results["pre_quant_val_bpb"]
        summary["delta_pre_quant_val_bpb"] = round(delta, 6)
        print(f"Pre-quant val_bpb delta: {delta:+.6f} ({'better' if delta < 0 else 'worse'})")

    if "artifact_bytes" in baseline_results and "artifact_bytes" in bigram_results:
        ratio = bigram_results["artifact_bytes"] / baseline_results["artifact_bytes"]
        summary["artifact_size_ratio"] = round(ratio, 4)
        print(f"Artifact size ratio: {ratio:.4f}x ({bigram_results['artifact_bytes']} vs {baseline_results['artifact_bytes']} bytes)")

    if "step_avg_ms" in baseline_results and "step_avg_ms" in bigram_results:
        slowdown = (bigram_results["step_avg_ms"] / baseline_results["step_avg_ms"]) - 1.0
        summary["throughput_delta_pct"] = round(slowdown * 100, 2)
        print(f"Throughput impact: {slowdown*100:+.2f}% step time")

    results_json = json.dumps(summary, indent=2)
    print(f"\n{results_json}")

    # ── Push to Hub ──
    if HF_TOKEN:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        try:
            api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
        except Exception:
            pass

        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        prefix = f"jobs/exp05/{stamp}"
        api.upload_file(
            path_or_fileobj=results_json.encode(),
            path_in_repo=f"{prefix}/results.json",
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print(f"\nResults pushed to https://huggingface.co/datasets/{RESULTS_REPO}/blob/main/{prefix}/results.json")
    else:
        print("\nWARNING: HF_TOKEN not set, results not pushed to Hub")


if __name__ == "__main__":
    main()
