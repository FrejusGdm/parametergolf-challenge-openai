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
Experiment 03: Int6 Quantization + Quantization-Aware Training (QAT)

This experiment tests whether training with simulated int6 quantization noise
(via Straight-Through Estimator) produces a model that degrades less when
actually quantized to int6 at export time.

Two runs:
  1. Baseline (unmodified train_gpt.py) — for A/B comparison
  2. Int6 + QAT — fake-quantize during training, export with int6 range [-32, 31]

Environment variables:
  HF_TOKEN    — for pushing results (required)
  ITERATIONS  — training steps (default 2000)
"""

import io
import json
import os
import re
import subprocess
import sys
import time
import zlib

ITERATIONS = os.environ.get("ITERATIONS", "2000")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RESULTS_REPO = "JosueG/parameter-golf-experiments"
TRAIN_SHARDS = os.environ.get("TRAIN_SHARDS", "20")
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

    # Pre-quant metrics (last eval step)
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

    # Post-quant metrics
    final_q = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)", output
    )
    if final_q:
        results["post_quant_val_loss"] = float(final_q.group(1))
        results["post_quant_val_bpb"] = float(final_q.group(2))

    # Artifact size
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


def apply_int6_qat_patches(script_path):
    """Patch train_gpt.py with int6 fake-quantize (STE) during training and int6 export."""
    with open(script_path, "r") as f:
        code = f.read()

    # ── Patch 1: Add fake-quantize function after the imports ──
    # Insert the STE fake-quantize function right before the CastedLinear class
    fake_quant_fn = '''
def fake_quantize_int6(w):
    """Straight-Through Estimator for int6 quantization simulation during training."""
    with torch.no_grad():
        abs_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / 31.0
    q = (w / scale).round().clamp(-32, 31)
    return w + (q * scale - w).detach()  # STE: forward uses quantized, backward uses original

'''
    # Insert before CastedLinear
    anchor = "class CastedLinear(nn.Linear):"
    if anchor not in code:
        raise RuntimeError(f"Could not find '{anchor}' in train_gpt.py")
    code = code.replace(anchor, fake_quant_fn + anchor)
    print("Patch 1: Added fake_quantize_int6 function")

    # ── Patch 2: Modify CastedLinear forward to use fake-quantize ──
    old_forward = """    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)"""

    new_forward = """    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        w = self.weight
        if self.training and w.ndim == 2 and w.shape[0] > 1:
            w_fq = fake_quantize_int6(w)
            return F.linear(x, w_fq.to(x.dtype), bias)
        return F.linear(x, w.to(x.dtype), bias)"""

    if old_forward not in code:
        raise RuntimeError("Could not find CastedLinear.forward to patch")
    code = code.replace(old_forward, new_forward)
    print("Patch 2: Modified CastedLinear.forward with fake-quantize")

    # ── Patch 3: Change quantization export from int8 [-127,127] to int6 [-32,31] ──
    # Change scale denominator from 127.0 to 31.0 and clamp range from [-127, 127] to [-32, 31]

    # Per-row quantization (2D tensors)
    old_row_scale = """scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()"""
    new_row_scale = """scale = (clip_abs / 31.0).clamp_min(1.0 / 31.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -32, 31).to(torch.int8).contiguous()"""

    if old_row_scale not in code:
        raise RuntimeError("Could not find per-row quantization code to patch")
    code = code.replace(old_row_scale, new_row_scale)
    print("Patch 3a: Changed per-row quantization to int6 range [-32, 31]")

    # Per-tensor quantization (vectors/scalars)
    old_tensor_scale = """scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()"""
    new_tensor_scale = """scale = torch.tensor(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -32, 31).to(torch.int8).contiguous()"""

    if old_tensor_scale not in code:
        raise RuntimeError("Could not find per-tensor quantization code to patch")
    code = code.replace(old_tensor_scale, new_tensor_scale)
    print("Patch 3b: Changed per-tensor quantization to int6 range [-32, 31]")

    with open(script_path, "w") as f:
        f.write(code)
    print("All int6+QAT patches applied successfully")


def main():
    print("=" * 80)
    print(f"Experiment 03: Int6 Quantization + QAT")
    print(f"Iterations: {ITERATIONS}, Train shards: {TRAIN_SHARDS}")
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

    import shutil
    shutil.copy("train_gpt.py", "train_gpt_baseline.py")

    t_start = time.time()
    baseline_output = run_cmd(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
        env=get_train_env("exp03_baseline"),
    )
    baseline_elapsed = time.time() - t_start
    baseline_results = parse_results(baseline_output)
    baseline_results["elapsed_seconds"] = round(baseline_elapsed, 1)
    print(f"\nBaseline results: {json.dumps(baseline_results, indent=2)}")

    # ── Run 2: Int6 + QAT ──
    print("\n" + "=" * 80)
    print("Run 2: INT6 + QAT")
    print("=" * 80)

    # Restore clean baseline script, then apply patches
    shutil.copy("train_gpt_baseline.py", "train_gpt.py")
    apply_int6_qat_patches("train_gpt.py")

    t_start = time.time()
    int6_output = run_cmd(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
        env=get_train_env("exp03_int6_qat"),
    )
    int6_elapsed = time.time() - t_start
    int6_results = parse_results(int6_output)
    int6_results["elapsed_seconds"] = round(int6_elapsed, 1)
    print(f"\nInt6+QAT results: {json.dumps(int6_results, indent=2)}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("EXPERIMENT 03 SUMMARY")
    print("=" * 80)

    summary = {
        "experiment": "03-int6-quant",
        "description": "Int6 quantization with QAT (STE fake-quantize during training)",
        "iterations": int(ITERATIONS),
        "train_shards": int(TRAIN_SHARDS),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline_results,
        "int6_qat": int6_results,
    }

    # Compute deltas
    if "post_quant_val_bpb" in baseline_results and "post_quant_val_bpb" in int6_results:
        delta = int6_results["post_quant_val_bpb"] - baseline_results["post_quant_val_bpb"]
        summary["delta_post_quant_val_bpb"] = round(delta, 6)
        print(f"Post-quant val_bpb delta: {delta:+.6f} ({'better' if delta < 0 else 'worse'})")

    if "pre_quant_val_bpb" in baseline_results and "pre_quant_val_bpb" in int6_results:
        delta = int6_results["pre_quant_val_bpb"] - baseline_results["pre_quant_val_bpb"]
        summary["delta_pre_quant_val_bpb"] = round(delta, 6)
        print(f"Pre-quant val_bpb delta: {delta:+.6f} ({'better' if delta < 0 else 'worse'})")

    if "artifact_bytes" in baseline_results and "artifact_bytes" in int6_results:
        ratio = int6_results["artifact_bytes"] / baseline_results["artifact_bytes"]
        summary["artifact_size_ratio"] = round(ratio, 4)
        print(f"Artifact size ratio: {ratio:.4f}x")

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
        prefix = f"jobs/exp03/{stamp}"
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
