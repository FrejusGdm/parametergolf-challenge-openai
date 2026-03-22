#!/usr/bin/env python3
"""
Curriculum Learning Experiment Runner

Runs the baseline training with different shard ordering strategies
and compares training loss curves.

Usage:
    python experiments/08-curriculum-learning/run_experiment.py [--steps 500] [--strategies all]

Prerequisites:
    python scripts/analyze_shards.py  (generates shard_analysis.json)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "parameter-golf"))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from train_gpt_mlx import (
    Hyperparameters, GPT, SplitOptimizers, TokenLoader,
    accumulate_flat_grads, loss_and_grad_chunked, clip_grad_tree,
    token_chunks,
)
from curriculum import CurriculumTokenStream, CurriculumTokenLoader, STRATEGIES


def run_single_strategy(
    strategy: str,
    n_steps: int,
    log_every: int,
    shard_metrics_path: str,
    data_pattern: str,
    tokenizer_path: str,
) -> dict:
    """Train for n_steps with a given curriculum strategy, return loss curve."""

    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy}")
    print(f"  Steps: {n_steps}, logging every {log_every}")
    print(f"{'='*60}")

    # Reset random seed for fair comparison
    mx.random.seed(1337)

    # Build model fresh each time
    args = Hyperparameters()
    # Override for fast M1 runs
    args.train_batch_tokens = 65_536
    args.grad_accum_steps = 8
    args.mlx_max_microbatch_tokens = 4096
    args.iterations = n_steps
    args.val_loss_every = 0  # No validation — just training loss
    args.max_wallclock_seconds = 0  # No wallclock cap
    args.warmup_steps = 5  # Minimal warmup
    args.train_log_every = log_every

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    opt = SplitOptimizers(model, args)

    # Compiled train function
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # Create data loader with curriculum strategy
    stream = CurriculumTokenStream(
        pattern=data_pattern,
        strategy=strategy,
        shard_metrics_path=shard_metrics_path,
    )
    train_loader = CurriculumTokenLoader(stream)

    # Quick warmup (just compile paths, no param updates)
    for _ in range(args.warmup_steps):
        warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
        mx.eval(warmup_loss)
        mx.synchronize()

    # Reset loader to start from the beginning of the curriculum
    stream = CurriculumTokenStream(
        pattern=data_pattern,
        strategy=strategy,
        shard_metrics_path=shard_metrics_path,
    )
    train_loader = CurriculumTokenLoader(stream)

    # Training loop
    loss_curve = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        step_t0 = time.perf_counter()

        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps

        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)

        grads_tree = tree_unflatten(list(accum.items()))
        grads_tree = clip_grad_tree(grads_tree, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())

        lr_mul = args.lr_mul(step, 1000.0 * (time.perf_counter() - t0))
        opt.step(model, grads_tree, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        actual_step = step + 1

        loss_curve.append({"step": actual_step, "loss": round(train_loss_value, 6)})

        if actual_step % log_every == 0 or actual_step == 1 or actual_step == n_steps:
            elapsed = time.perf_counter() - t0
            tok_s = args.train_batch_tokens / (step_ms / 1000.0)
            print(f"  step:{actual_step:>4d}/{n_steps}  loss:{train_loss_value:.4f}  "
                  f"step_ms:{step_ms:.0f}  tok/s:{tok_s:.0f}  elapsed:{elapsed:.0f}s")

    total_time = time.perf_counter() - t0

    return {
        "strategy": strategy,
        "n_steps": n_steps,
        "total_time_s": round(total_time, 1),
        "final_loss": loss_curve[-1]["loss"],
        "loss_curve": loss_curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Curriculum Learning Experiment")
    parser.add_argument("--steps", type=int, default=500, help="Training steps per strategy")
    parser.add_argument("--log-every", type=int, default=50, help="Log training loss every N steps")
    parser.add_argument("--strategies", type=str, default="all",
                        help="Comma-separated strategies or 'all'")
    parser.add_argument("--shard-metrics", type=str, default="scripts/shard_analysis.json",
                        help="Path to shard analysis JSON")
    parser.add_argument("--data-dir", type=str,
                        default="parameter-golf/data/datasets/fineweb10B_sp1024",
                        help="Path to dataset directory")
    parser.add_argument("--output", type=str,
                        default="experiments/08-curriculum-learning/results.json",
                        help="Output results JSON")
    args = parser.parse_args()

    data_pattern = f"{args.data_dir}/fineweb_train_*.bin"
    tokenizer_path = "parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

    # Check prerequisites
    if not Path(args.shard_metrics).exists():
        print(f"ERROR: {args.shard_metrics} not found.")
        print(f"Run: python scripts/analyze_shards.py")
        sys.exit(1)

    strategies = STRATEGIES if args.strategies == "all" else args.strategies.split(",")
    for s in strategies:
        if s not in STRATEGIES:
            print(f"ERROR: Unknown strategy '{s}'. Choose from: {STRATEGIES}")
            sys.exit(1)

    print(f"Curriculum Learning Experiment")
    print(f"  Steps per strategy: {args.steps}")
    print(f"  Strategies: {strategies}")
    print(f"  Data: {data_pattern}")
    print(f"  Shard metrics: {args.shard_metrics}")

    # Run experiments
    all_results = []
    for strategy in strategies:
        result = run_single_strategy(
            strategy=strategy,
            n_steps=args.steps,
            log_every=args.log_every,
            shard_metrics_path=args.shard_metrics,
            data_pattern=data_pattern,
            tokenizer_path=tokenizer_path,
        )
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Collect loss at key checkpoints
    checkpoints = [50, 100, 250, args.steps]
    checkpoints = [c for c in checkpoints if c <= args.steps]

    header = f"{'Strategy':<20s}"
    for cp in checkpoints:
        header += f" {'Step '+str(cp):>10s}"
    header += f" {'Time(s)':>10s}"
    print(header)
    print("-" * 80)

    for r in all_results:
        row = f"{r['strategy']:<20s}"
        for cp in checkpoints:
            # Find closest step
            losses_at = [p["loss"] for p in r["loss_curve"] if p["step"] == cp]
            if losses_at:
                row += f" {losses_at[0]:>10.4f}"
            else:
                # Find nearest
                nearest = min(r["loss_curve"], key=lambda p: abs(p["step"] - cp))
                row += f" {nearest['loss']:>10.4f}"
        row += f" {r['total_time_s']:>10.1f}"
        print(row)

    # Best and worst
    final_losses = [(r["strategy"], r["final_loss"]) for r in all_results]
    best = min(final_losses, key=lambda x: x[1])
    worst = max(final_losses, key=lambda x: x[1])
    print("-" * 80)
    print(f"Best:  {best[0]} ({best[1]:.4f})")
    print(f"Worst: {worst[0]} ({worst[1]:.4f})")
    print(f"Delta: {worst[1] - best[1]:.4f}")

    if worst[1] - best[1] > 0.01:
        print("\n*** Shard ordering appears to matter! ***")
    elif worst[1] - best[1] > 0.005:
        print("\n*** Small but possibly meaningful difference. Need more steps to confirm. ***")
    else:
        print("\n*** Ordering doesn't seem to matter much at this step count. ***")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "steps": args.steps,
                "strategies": strategies,
                "data_pattern": data_pattern,
                "shard_metrics": args.shard_metrics,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
