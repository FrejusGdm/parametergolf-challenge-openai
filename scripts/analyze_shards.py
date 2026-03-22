#!/usr/bin/env python3
"""
Shard Analysis Script for Curriculum Learning

Measures difficulty/quality metrics for each training shard in the FineWeb dataset.
Used to inform curriculum learning strategies (shard ordering).

Usage:
    python scripts/analyze_shards.py [--data-dir PATH] [--tokenizer PATH] [--output PATH]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

# Reuse the shard loader from the baseline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "parameter-golf"))
from train_gpt_mlx import load_data_shard, build_sentencepiece_luts


def compute_shard_metrics(shard_path: Path, sp: spm.SentencePieceProcessor, vocab_size: int) -> dict:
    """Compute difficulty/quality metrics for a single shard."""
    tokens = load_data_shard(shard_path)
    n_tokens = len(tokens)

    # Token frequency distribution
    counts = np.bincount(tokens, minlength=vocab_size)
    freq = counts / counts.sum()

    # Shannon entropy (bits) — higher = more diverse/harder
    nonzero = freq[freq > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    # Vocabulary coverage — how many unique tokens used
    unique_tokens = int(np.sum(counts > 0))
    vocab_coverage = unique_tokens / vocab_size

    # Top token concentration — what % of tokens are in the top 10
    top10_ids = np.argsort(-counts)[:10]
    top10_pct = float(counts[top10_ids].sum() / counts.sum())

    # Bigram entropy — measure of local predictability
    # Sample first 1M tokens for speed
    sample = tokens[:min(1_000_000, n_tokens)]
    bigrams = sample[:-1].astype(np.int64) * vocab_size + sample[1:]
    bigram_counts = np.bincount(bigrams)
    bigram_freq = bigram_counts[bigram_counts > 0] / bigram_counts.sum()
    bigram_entropy = float(-np.sum(bigram_freq * np.log2(bigram_freq)))

    # Repetition ratio — fraction of 4-grams that appear more than once
    # Sample first 500K tokens for speed
    rep_sample = tokens[:min(500_000, n_tokens)]
    if len(rep_sample) >= 4:
        ngrams = set()
        repeated = 0
        total = 0
        for i in range(len(rep_sample) - 3):
            ng = tuple(rep_sample[i:i+4].tolist())
            if ng in ngrams:
                repeated += 1
            else:
                ngrams.add(ng)
            total += 1
        repetition_ratio = repeated / max(total, 1)
    else:
        repetition_ratio = 0.0

    # Average bytes per token (affects BPB)
    base_bytes_lut, _, _ = build_sentencepiece_luts(sp, vocab_size)
    avg_bytes_per_token = float(np.sum(base_bytes_lut[:vocab_size].astype(np.float64) * freq))

    # Decode first 500 chars as sample
    sample_text = sp.decode(tokens[:256].tolist())[:500]

    return {
        "file": shard_path.name,
        "path": str(shard_path),
        "n_tokens": n_tokens,
        "entropy": round(entropy, 6),
        "bigram_entropy": round(bigram_entropy, 6),
        "unique_tokens": unique_tokens,
        "vocab_coverage": round(vocab_coverage, 4),
        "top10_concentration": round(top10_pct, 4),
        "repetition_ratio": round(repetition_ratio, 6),
        "avg_bytes_per_token": round(avg_bytes_per_token, 4),
        "sample_text": sample_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze FineWeb training shards")
    parser.add_argument("--data-dir", type=str,
                        default="parameter-golf/data/datasets/fineweb10B_sp1024",
                        help="Path to dataset directory")
    parser.add_argument("--tokenizer", type=str,
                        default="parameter-golf/data/tokenizers/fineweb_1024_bpe.model",
                        help="Path to SentencePiece model")
    parser.add_argument("--output", type=str,
                        default="scripts/shard_analysis.json",
                        help="Output JSON path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    if not train_files:
        print(f"No training shards found in {data_dir}")
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.vocab_size()

    print(f"Analyzing {len(train_files)} training shards...")
    print(f"Tokenizer: {args.tokenizer} (vocab={vocab_size})")
    print()

    results = []
    for i, f in enumerate(train_files):
        print(f"  [{i+1}/{len(train_files)}] {f.name}...", end=" ", flush=True)
        metrics = compute_shard_metrics(f, sp, vocab_size)
        results.append(metrics)
        print(f"entropy={metrics['entropy']:.4f}  bigram_ent={metrics['bigram_entropy']:.4f}  "
              f"vocab_cov={metrics['vocab_coverage']:.4f}  rep={metrics['repetition_ratio']:.6f}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Shard':<30s} {'Tokens':>12s} {'Entropy':>10s} {'BigramEnt':>10s} "
          f"{'VocabCov':>10s} {'Rep%':>8s} {'Byt/Tok':>8s}")
    print("-" * 90)

    entropies = []
    for m in results:
        entropies.append(m["entropy"])
        print(f"{m['file']:<30s} {m['n_tokens']:>12,} {m['entropy']:>10.4f} "
              f"{m['bigram_entropy']:>10.4f} {m['vocab_coverage']:>10.4f} "
              f"{m['repetition_ratio']*100:>7.4f}% {m['avg_bytes_per_token']:>8.4f}")

    print("-" * 90)
    print(f"{'MEAN':<30s} {'':>12s} {np.mean(entropies):>10.4f}")
    print(f"{'STD':<30s} {'':>12s} {np.std(entropies):>10.4f}")
    print(f"{'MIN':<30s} {'':>12s} {np.min(entropies):>10.4f}")
    print(f"{'MAX':<30s} {'':>12s} {np.max(entropies):>10.4f}")

    # Suggested orderings
    by_entropy = sorted(results, key=lambda x: x["entropy"])
    print("\n=== Suggested Orderings ===")
    print("Easy → Hard (entropy): " + " → ".join(m["file"].replace("fineweb_train_", "").replace(".bin", "") for m in by_entropy))
    print("Hard → Easy (entropy): " + " → ".join(m["file"].replace("fineweb_train_", "").replace(".bin", "") for m in reversed(by_entropy)))

    by_vocab = sorted(results, key=lambda x: x["vocab_coverage"], reverse=True)
    print("Quality first (vocab): " + " → ".join(m["file"].replace("fineweb_train_", "").replace(".bin", "") for m in by_vocab))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove sample_text from JSON (too long), keep separately
    for m in results:
        m.pop("sample_text", None)

    output = {
        "shards": results,
        "summary": {
            "n_shards": len(results),
            "entropy_mean": round(float(np.mean(entropies)), 6),
            "entropy_std": round(float(np.std(entropies)), 6),
            "entropy_min": round(float(np.min(entropies)), 6),
            "entropy_max": round(float(np.max(entropies)), 6),
        },
        "orderings": {
            "easy_first": [m["file"] for m in by_entropy],
            "hard_first": [m["file"] for m in reversed(by_entropy)],
            "quality_first": [m["file"] for m in by_vocab],
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
