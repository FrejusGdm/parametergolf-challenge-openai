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
Sliding Window Eval — Experiment 01

Trains the baseline model for 2000 steps, then evaluates with:
  1. Standard non-overlapping eval (baseline)
  2. Sliding window eval with stride=64

This demonstrates the free BPB improvement from better evaluation.
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor, nn
from huggingface_hub import HfApi


# ============================================================================
# SETUP
# ============================================================================

def setup():
    print("=" * 60)
    print("  Experiment 01: Sliding Window Eval")
    print("=" * 60)

    if not Path("parameter-golf").exists():
        subprocess.run(["git", "clone", "https://github.com/openai/parameter-golf.git"], check=True)

    # Need val data + a few train shards
    subprocess.run([
        sys.executable, "parameter-golf/data/cached_challenge_fineweb.py",
        "--variant", "sp1024", "--train-shards", "20"
    ], check=True)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================================
# STEP 1: TRAIN BASELINE (2000 steps)
# ============================================================================

def train_baseline(n_steps=2000):
    """Train the actual baseline model."""
    print(f"\n{'='*60}")
    print(f"  Training baseline for {n_steps} steps")
    print(f"{'='*60}")

    env = os.environ.copy()
    env.update({
        "DATA_PATH": "./parameter-golf/data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
        "RUN_ID": "exp01_sliding_window",
        "ITERATIONS": str(n_steps),
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_LOSS_EVERY": str(n_steps),  # Validate at end
        "VAL_BATCH_SIZE": "65536",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMUP_STEPS": "10",
        "TRAIN_LOG_EVERY": "200",
        "SEED": "1337",
    })

    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=1", "parameter-golf/train_gpt.py"],
        env=env, capture_output=True, text=True, timeout=7200,
    )

    for line in result.stdout.strip().split("\n")[-15:]:
        print(line)

    if result.returncode != 0:
        print("STDERR (last 10):")
        for line in result.stderr.strip().split("\n")[-10:]:
            print(line)

    # Find the saved model
    import glob
    model_files = sorted(glob.glob("logs/*_model.pt"))
    quant_files = sorted(glob.glob("logs/*.int8.ptz"))
    print(f"Model files: {model_files}")
    print(f"Quant files: {quant_files}")

    # Parse baseline eval metrics
    baseline_metrics = {}
    for line in result.stdout.split("\n"):
        if f"step:{n_steps}/{n_steps}" in line and "val_bpb:" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("val_loss:"):
                    baseline_metrics["val_loss"] = float(p.split(":")[1])
                if p.startswith("val_bpb:"):
                    baseline_metrics["val_bpb"] = float(p.split(":")[1])
        if "final_int8_zlib_roundtrip " in line and "val_bpb:" in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith("val_loss:"):
                    baseline_metrics["quant_val_loss"] = float(p.split(":")[1])
                if p.startswith("val_bpb:"):
                    baseline_metrics["quant_val_bpb"] = float(p.split(":")[1])

    print(f"\nBaseline metrics: {baseline_metrics}")
    return model_files[-1] if model_files else None, quant_files[-1] if quant_files else None, baseline_metrics


# ============================================================================
# STEP 2: LOAD MODEL AND RUN SLIDING WINDOW EVAL
# ============================================================================

def sliding_window_eval(model_path, stride=64):
    """Load the trained model and run sliding window evaluation."""
    print(f"\n{'='*60}")
    print(f"  Sliding Window Eval (stride={stride})")
    print(f"{'='*60}")

    # We need to use the train_gpt.py code to load the model properly.
    # Easiest approach: import from the repo and reconstruct.
    sys.path.insert(0, "parameter-golf")

    # Import everything we need from the baseline
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", "parameter-golf/train_gpt.py")
    tgpt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgpt)

    args = tgpt.Hyperparameters()
    device = torch.device("cuda")

    # Load tokenizer for BPB computation
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tgpt.build_sentencepiece_luts(sp, args.vocab_size)

    base_bytes_lut = torch.tensor(base_bytes_lut, dtype=torch.int16, device=device)
    has_leading_space_lut = torch.tensor(has_leading_space_lut, dtype=torch.bool, device=device)
    is_boundary_token_lut = torch.tensor(is_boundary_token_lut, dtype=torch.bool, device=device)

    # Load validation tokens
    val_tokens = tgpt.load_validation_tokens(args.val_files, args.train_seq_len)
    val_tokens = torch.tensor(val_tokens, dtype=torch.int64, device=device)
    print(f"Val tokens: {val_tokens.numel():,}")

    # Load the model
    print(f"Loading model from {model_path}...")
    saved = torch.load(model_path, map_location=device, weights_only=False)

    model = tgpt.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    ).to(device)

    model.load_state_dict(saved, strict=False)

    # Run sliding window eval
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    batch_seqs = 16  # L4 has 24GB, be conservative

    window_starts = list(range(0, total_tokens, stride))
    # Filter windows that are too short
    window_starts = [ws for ws in window_starts if min(ws + seq_len, total_tokens) - ws >= 1]
    print(f"Total windows: {len(window_starts)} (stride={stride})")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    t0 = time.perf_counter()

    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            # Forward pass — get logits
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Use the model's forward to get hidden states, then project to logits
                hidden = model(x_batch)
                # Project to vocab (tied embeddings)
                logits = hidden @ model.tok_emb.weight.to(hidden.dtype).T
                logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                # First window: score all tokens; others: score last `stride` tokens
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)

                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            # Progress
            done = min(bi + batch_seqs, len(window_starts))
            if done % (batch_seqs * 50) < batch_seqs or done == len(window_starts):
                pct = done / len(window_starts) * 100
                elapsed = time.perf_counter() - t0
                running_loss = (loss_sum / token_count).item() if token_count.item() > 0 else 0
                running_bpb = running_loss / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
                print(f"  [{pct:5.1f}%] {done}/{len(window_starts)} windows  "
                      f"bpb={running_bpb:.4f}  elapsed={elapsed:.0f}s")

    eval_time = time.perf_counter() - t0
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    val_bpb = bits_per_token * tokens_per_byte

    print(f"\nSliding window eval complete!")
    print(f"  val_loss: {val_loss:.4f}")
    print(f"  val_bpb:  {val_bpb:.4f}")
    print(f"  eval_time: {eval_time:.0f}s")
    print(f"  tokens scored: {int(token_count.item()):,}")

    return {"val_loss": val_loss, "val_bpb": round(val_bpb, 6), "eval_time_s": round(eval_time, 1)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    setup()

    # Step 1: Train baseline
    model_path, quant_path, baseline_metrics = train_baseline(n_steps=2000)

    if not model_path:
        print("ERROR: No model file found after training!")
        return

    # Step 2: Sliding window eval on the raw (pre-quant) model
    sw_metrics = sliding_window_eval(model_path, stride=64)

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 01: SLIDING WINDOW EVAL RESULTS")
    print("=" * 80)
    print(f"  {'Method':<25s}  {'val_loss':>10s}  {'val_bpb':>10s}")
    print(f"  {'-'*50}")

    bl_bpb = baseline_metrics.get("val_bpb", "N/A")
    sw_bpb = sw_metrics.get("val_bpb", "N/A")
    bl_loss = baseline_metrics.get("val_loss", "N/A")
    sw_loss = sw_metrics.get("val_loss", "N/A")

    print(f"  {'Non-overlapping (base)':<25s}  {bl_loss:>10}  {bl_bpb:>10}")
    print(f"  {'Sliding window (s=64)':<25s}  {sw_loss:>10}  {sw_bpb:>10}")

    if isinstance(bl_bpb, float) and isinstance(sw_bpb, float):
        delta = bl_bpb - sw_bpb
        print(f"\n  BPB improvement: {delta:.4f} (expected ~0.032-0.034)")
        if delta > 0.02:
            print(f"  >>> Sliding window wins by {delta:.4f} bpb! Free improvement confirmed.")
        elif delta > 0:
            print(f"  >>> Small improvement of {delta:.4f} bpb.")
        else:
            print(f"  >>> Unexpected: sliding window didn't help ({delta:.4f}).")

    # Also show post-quant baseline for reference
    if baseline_metrics.get("quant_val_bpb"):
        print(f"\n  Post-quant baseline bpb: {baseline_metrics['quant_val_bpb']}")
        print(f"  (Sliding window on post-quant model would improve this too)")

    # Push results
    token = os.environ.get("HF_TOKEN")
    if token:
        api = HfApi(token=token)
        repo_id = "JosueG/parameter-golf-experiments"
        try:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        except Exception:
            pass

        output = {
            "experiment": "01_sliding_window_eval",
            "n_steps": 2000,
            "baseline_metrics": baseline_metrics,
            "sliding_window_metrics": sw_metrics,
            "stride": 64,
        }
        api.upload_file(
            path_or_fileobj=json.dumps(output, indent=2).encode(),
            path_in_repo="exp01_sliding_window_results.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"\nResults pushed to https://huggingface.co/datasets/{repo_id}")

    print("\nDone!")


if __name__ == "__main__":
    main()
