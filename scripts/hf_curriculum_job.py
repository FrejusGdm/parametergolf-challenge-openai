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
# ]
# ///
"""
Curriculum Learning Experiment — HF Jobs Version

Downloads all 80 FineWeb shards, analyzes shard difficulty,
runs 6 curriculum strategies × 500 steps each, and pushes results to HF Hub.

This is a self-contained script designed to run on HF Jobs with a GPU.
"""

import json
import math
import os
import random
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi


# ============================================================================
# STEP 0: Download the Parameter Golf repo and data
# ============================================================================

def setup_environment():
    """Clone repo and download data."""
    print("=" * 60)
    print("  STEP 0: Environment Setup")
    print("=" * 60)

    if not Path("parameter-golf").exists():
        print("Cloning parameter-golf repo...")
        subprocess.run(["git", "clone", "https://github.com/openai/parameter-golf.git"], check=True)

    # Download ALL 80 training shards
    print("Downloading FineWeb dataset (80 shards)...")
    subprocess.run([
        sys.executable, "parameter-golf/data/cached_challenge_fineweb.py",
        "--variant", "sp1024", "--train-shards", "80"
    ], check=True)

    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ============================================================================
# STEP 1: Shard Analysis
# ============================================================================

def load_data_shard(path: Path) -> np.ndarray:
    """Load tokenized shard (from baseline code)."""
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)


def analyze_shard(shard_path: Path, vocab_size: int = 1024) -> dict:
    """Compute metrics for a single shard."""
    tokens = load_data_shard(shard_path)
    n_tokens = len(tokens)

    counts = np.bincount(tokens, minlength=vocab_size)
    freq = counts / counts.sum()

    # Shannon entropy
    nonzero = freq[freq > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    # Vocab coverage
    unique_tokens = int(np.sum(counts > 0))
    vocab_coverage = unique_tokens / vocab_size

    # Bigram entropy (sample for speed)
    sample = tokens[:min(1_000_000, n_tokens)]
    bigrams = sample[:-1].astype(np.int64) * vocab_size + sample[1:]
    bigram_counts = np.bincount(bigrams)
    bigram_freq = bigram_counts[bigram_counts > 0] / bigram_counts.sum()
    bigram_entropy = float(-np.sum(bigram_freq * np.log2(bigram_freq)))

    # Repetition ratio
    rep_sample = tokens[:min(500_000, n_tokens)]
    ngrams = set()
    repeated = 0
    for i in range(len(rep_sample) - 3):
        ng = tuple(rep_sample[i:i+4].tolist())
        if ng in ngrams:
            repeated += 1
        else:
            ngrams.add(ng)
    repetition_ratio = repeated / max(len(rep_sample) - 3, 1)

    return {
        "file": shard_path.name,
        "n_tokens": n_tokens,
        "entropy": round(entropy, 6),
        "bigram_entropy": round(bigram_entropy, 6),
        "unique_tokens": unique_tokens,
        "vocab_coverage": round(vocab_coverage, 4),
        "repetition_ratio": round(repetition_ratio, 6),
    }


def analyze_all_shards(data_dir: Path) -> dict:
    """Analyze all training shards."""
    print("\n" + "=" * 60)
    print("  STEP 1: Shard Analysis")
    print("=" * 60)

    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    print(f"Found {len(train_files)} training shards")

    results = []
    for i, f in enumerate(train_files):
        print(f"  [{i+1}/{len(train_files)}] {f.name}...", end=" ", flush=True)
        metrics = analyze_shard(f)
        results.append(metrics)
        print(f"entropy={metrics['entropy']:.4f}  rep={metrics['repetition_ratio']:.4f}")

    entropies = [m["entropy"] for m in results]
    print(f"\nEntropy: mean={np.mean(entropies):.4f}, std={np.std(entropies):.4f}, "
          f"range=[{np.min(entropies):.4f}, {np.max(entropies):.4f}]")

    return {
        "shards": results,
        "summary": {
            "n_shards": len(results),
            "entropy_mean": round(float(np.mean(entropies)), 6),
            "entropy_std": round(float(np.std(entropies)), 6),
            "entropy_min": round(float(np.min(entropies)), 6),
            "entropy_max": round(float(np.max(entropies)), 6),
        }
    }


# ============================================================================
# STEP 2: Curriculum Ordering Strategies
# ============================================================================

STRATEGIES = ["default", "easy_first", "hard_first", "interleaved", "random", "quality_first"]


def reorder_files(files: list[Path], shard_metrics: list[dict], strategy: str) -> list[Path]:
    """Reorder shard files based on curriculum strategy."""
    metrics_map = {m["file"]: m for m in shard_metrics}

    def get_metric(f, key, default=0.0):
        m = metrics_map.get(f.name)
        return m[key] if m else default

    if strategy == "default":
        return files
    elif strategy == "easy_first":
        return sorted(files, key=lambda f: get_metric(f, "entropy", 999.0))
    elif strategy == "hard_first":
        return sorted(files, key=lambda f: get_metric(f, "entropy", 0.0), reverse=True)
    elif strategy == "interleaved":
        by_entropy = sorted(files, key=lambda f: get_metric(f, "entropy", 999.0))
        easy = by_entropy[:len(by_entropy)//2]
        hard = by_entropy[len(by_entropy)//2:]
        result = []
        for e, h in zip(easy, hard):
            result.extend([e, h])
        result.extend(easy[len(hard):] if len(easy) > len(hard) else hard[len(easy):])
        return result
    elif strategy == "random":
        shuffled = list(files)
        random.Random(42).shuffle(shuffled)
        return shuffled
    elif strategy == "quality_first":
        return sorted(files, key=lambda f: get_metric(f, "vocab_coverage", 0.0), reverse=True)
    return files


# ============================================================================
# STEP 3: Minimal GPT Model (from baseline, PyTorch)
# ============================================================================

class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class CastedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.fc = CastedLinear(dim, dim * mult)
        self.proj = CastedLinear(dim * mult, dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.proj(x * x)  # ReLU²


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = Attention(dim, num_heads, num_kv_heads)
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=1024, dim=512, num_layers=9, num_heads=8, num_kv_heads=4, mlp_mult=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads, num_kv_heads, mlp_mult) for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self.vocab_size = vocab_size
        nn.init.normal_(self.tok_emb.weight, std=0.005)
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)
            nn.init.zeros_(b.mlp.proj.weight)

    def forward(self, x):
        x = self.tok_emb(x)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_emb.weight.T
        return 30.0 * torch.tanh(logits / 30.0)


# ============================================================================
# STEP 4: Training Loop
# ============================================================================

class TokenStream:
    def __init__(self, files: list[Path]):
        self.files = files
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            if self.pos >= len(self.tokens):
                self.next_file()
            k = min(left, len(self.tokens) - self.pos)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def train_strategy(strategy, files, shard_metrics, n_steps=500, log_every=50, device="cuda"):
    """Run training for one curriculum strategy."""
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy}")
    print(f"{'='*60}")

    ordered_files = reorder_files(files, shard_metrics, strategy)
    order_str = " → ".join(f.name[-10:-4] for f in ordered_files[:10])
    if len(ordered_files) > 10:
        order_str += f" ... ({len(ordered_files)} total)"
    print(f"  Order: {order_str}")

    # Reset seed for fair comparison
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    model = MiniGPT().to(device).bfloat16()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.95))

    stream = TokenStream(ordered_files)
    seq_len = 1024
    batch_tokens = 65_536
    batch_seqs = batch_tokens // seq_len

    loss_curve = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        chunk = stream.take(batch_tokens + 1)
        x = torch.tensor(chunk[:-1].reshape(batch_seqs, seq_len), dtype=torch.long, device=device)
        y = torch.tensor(chunk[1:].reshape(batch_seqs, seq_len), dtype=torch.long, device=device)

        logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, 1024), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_curve.append({"step": step + 1, "loss": round(loss_val, 6)})

        if (step + 1) % log_every == 0 or step == 0 or step == n_steps - 1:
            elapsed = time.perf_counter() - t0
            print(f"  step:{step+1:>4d}/{n_steps}  loss:{loss_val:.4f}  elapsed:{elapsed:.0f}s")

    total_time = time.perf_counter() - t0
    return {
        "strategy": strategy,
        "n_steps": n_steps,
        "total_time_s": round(total_time, 1),
        "final_loss": loss_curve[-1]["loss"],
        "loss_curve": loss_curve,
        "shard_order": [f.name for f in ordered_files],
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  CURRICULUM LEARNING EXPERIMENT")
    print("  OpenAI Parameter Golf Challenge")
    print("=" * 60)

    setup_environment()

    data_dir = Path("parameter-golf/data/datasets/fineweb10B_sp1024")
    analysis = analyze_all_shards(data_dir)

    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    n_steps = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Run all strategies
    all_results = []
    for strategy in STRATEGIES:
        result = train_strategy(strategy, train_files, analysis["shards"], n_steps=n_steps, device=device)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    checkpoints = [50, 100, 250, 500]
    header = f"{'Strategy':<20s}"
    for cp in checkpoints:
        header += f" {'Step '+str(cp):>10s}"
    header += f" {'Time(s)':>10s}"
    print(header)
    print("-" * 80)

    for r in all_results:
        row = f"{r['strategy']:<20s}"
        for cp in checkpoints:
            losses = [p["loss"] for p in r["loss_curve"] if p["step"] == cp]
            row += f" {losses[0]:>10.4f}" if losses else f" {'N/A':>10s}"
        row += f" {r['total_time_s']:>10.1f}"
        print(row)

    final_losses = [(r["strategy"], r["final_loss"]) for r in all_results]
    best = min(final_losses, key=lambda x: x[1])
    worst = max(final_losses, key=lambda x: x[1])
    print("-" * 80)
    print(f"Best:  {best[0]} ({best[1]:.4f})")
    print(f"Worst: {worst[0]} ({worst[1]:.4f})")
    print(f"Delta: {worst[1] - best[1]:.4f}")

    # Save results
    output = {
        "config": {"n_steps": n_steps, "n_shards": len(train_files), "device": device},
        "shard_analysis": analysis,
        "results": all_results,
    }

    # Push to HF Hub
    token = os.environ.get("HF_TOKEN")
    if token:
        api = HfApi(token=token)
        repo_id = "JosueG/parameter-golf-curriculum"
        try:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"Repo creation note: {e}")

        results_json = json.dumps(output, indent=2)
        api.upload_file(
            path_or_fileobj=results_json.encode(),
            path_in_repo="results.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"\nResults pushed to https://huggingface.co/datasets/{repo_id}")
    else:
        print("\nNo HF_TOKEN — results printed but not saved to Hub")
        print(json.dumps(output, indent=2)[:2000])

    print("\nDone!")


if __name__ == "__main__":
    main()
