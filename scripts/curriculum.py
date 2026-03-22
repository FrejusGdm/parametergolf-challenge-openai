#!/usr/bin/env python3
"""
Curriculum Learning Module for Parameter Golf

Provides CurriculumTokenStream — a drop-in replacement for TokenStream
that reorders training shards based on curriculum strategies.

Usage:
    from curriculum import CurriculumTokenStream, CurriculumTokenLoader

    stream = CurriculumTokenStream(
        pattern="data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
        strategy="easy_first",
        shard_metrics_path="scripts/shard_analysis.json",
    )
    loader = CurriculumTokenLoader(stream)
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sys

# Import the base classes from the baseline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "parameter-golf"))
from train_gpt_mlx import TokenStream, TokenLoader, load_data_shard
import mlx.core as mx


STRATEGIES = [
    "default",        # sorted glob (baseline behavior)
    "easy_first",     # lowest entropy → highest
    "hard_first",     # highest entropy → lowest
    "interleaved",    # alternating easy/hard
    "random",         # shuffled with seed
    "quality_first",  # highest vocab coverage first
]


class CurriculumTokenStream(TokenStream):
    """TokenStream that reorders shards based on a curriculum strategy.

    Drop-in compatible with TokenStream — same take() API.
    The only difference is the order in which shards are consumed.
    """

    def __init__(
        self,
        pattern: str,
        strategy: str = "default",
        shard_metrics_path: str = "scripts/shard_analysis.json",
        random_seed: int = 42,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        # Initialize the base TokenStream (loads files in sorted order)
        super().__init__(pattern, log_fn=log_fn, dataset_name=dataset_name)

        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {STRATEGIES}")

        if strategy == "default":
            # Keep sorted glob order (baseline behavior)
            return

        # Load shard metrics
        metrics_path = Path(shard_metrics_path)
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Shard metrics not found at {metrics_path}. "
                f"Run 'python scripts/analyze_shards.py' first."
            )
        with open(metrics_path) as f:
            analysis = json.load(f)

        # Build a mapping from filename to metrics
        shard_metrics = {m["file"]: m for m in analysis["shards"]}

        # Reorder self.files based on strategy
        self.files = self._reorder(self.files, shard_metrics, strategy, random_seed)

        # Reset to start of reordered files
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

        if log_fn:
            order_str = " → ".join(f.name.replace("fineweb_train_", "").replace(".bin", "") for f in self.files)
            log_fn(f"curriculum:{strategy} order={order_str}")

    @staticmethod
    def _reorder(
        files: list[Path],
        shard_metrics: dict,
        strategy: str,
        random_seed: int,
    ) -> list[Path]:
        """Reorder file list based on curriculum strategy."""

        def get_metric(f: Path, key: str, default: float = 0.0) -> float:
            m = shard_metrics.get(f.name)
            return m[key] if m else default

        if strategy == "easy_first":
            return sorted(files, key=lambda f: get_metric(f, "entropy", 999.0))

        elif strategy == "hard_first":
            return sorted(files, key=lambda f: get_metric(f, "entropy", 0.0), reverse=True)

        elif strategy == "interleaved":
            by_entropy = sorted(files, key=lambda f: get_metric(f, "entropy", 999.0))
            easy = by_entropy[: len(by_entropy) // 2]
            hard = by_entropy[len(by_entropy) // 2 :]
            result = []
            for e, h in zip(easy, hard):
                result.extend([e, h])
            # Add any remaining if odd count
            if len(easy) > len(hard):
                result.extend(easy[len(hard):])
            elif len(hard) > len(easy):
                result.extend(hard[len(easy):])
            return result

        elif strategy == "random":
            shuffled = list(files)
            rng = random.Random(random_seed)
            rng.shuffle(shuffled)
            return shuffled

        elif strategy == "quality_first":
            return sorted(files, key=lambda f: get_metric(f, "vocab_coverage", 0.0), reverse=True)

        else:
            return files


class CurriculumTokenLoader:
    """TokenLoader that uses CurriculumTokenStream."""

    def __init__(self, stream: CurriculumTokenStream):
        self.stream = stream

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def list_strategies() -> list[str]:
    """Return all available curriculum strategies."""
    return list(STRATEGIES)


if __name__ == "__main__":
    # Quick test: show ordering for each strategy
    import glob

    pattern = "parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"
    files = sorted(glob.glob(pattern))
    if not files:
        print("No training shards found. Run data download first.")
        sys.exit(1)

    print(f"Found {len(files)} shards")
    for strategy in STRATEGIES:
        try:
            stream = CurriculumTokenStream(
                pattern=pattern,
                strategy=strategy,
                shard_metrics_path="scripts/shard_analysis.json",
            )
            order = [f.name.replace("fineweb_train_", "").replace(".bin", "") for f in stream.files]
            print(f"  {strategy:20s}: {' → '.join(order)}")
        except FileNotFoundError:
            print(f"  {strategy:20s}: (needs shard_analysis.json — run analyze_shards.py first)")
            if strategy == "default":
                order = [Path(f).name.replace("fineweb_train_", "").replace(".bin", "") for f in files]
                print(f"  {'':20s}  default order: {' → '.join(order)}")
