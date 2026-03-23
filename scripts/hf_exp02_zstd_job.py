# /// script
# dependencies = [
#     "numpy",
#     "tqdm",
#     "torch",
#     "huggingface-hub>=0.36.0",
#     "setuptools",
#     "typing-extensions==4.15.0",
#     "datasets",
#     "sentencepiece",
#     "zstandard",
#     "kernels",
# ]
# ///

"""
Experiment 02 — zstd Compression comparison.

Trains a baseline model for 2000 steps, then compares zlib-9 vs zstd-22
compression on the int8 quantized artifact. The model is identical in both
cases — this measures purely the artifact size savings from switching
compression backends.

Workflow:
1) Clone parameter-golf repo, download 20 train shards
2) Train baseline for 2000 steps with torchrun
3) Load the int8 quantized model artifact
4) Decompress with zlib, recompress with zstd-22
5) Compare artifact sizes and report savings
6) Push results to HF Hub
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import time
import zlib
from pathlib import Path

import zstandard as zstd
from huggingface_hub import HfApi


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_file=None,
) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if log_file is not None:
            log_file.write(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")


def parse_metrics(log_text: str) -> dict[str, float | int] | None:
    out: dict[str, float | int] = {}

    final_q = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)",
        log_text,
    )
    if final_q:
        out["post_quant_val_loss"] = float(final_q.group(1))
        out["post_quant_val_bpb"] = float(final_q.group(2))

    final_sw = re.search(
        r"final_sliding_window_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)",
        log_text,
    )
    if final_sw:
        out["post_quant_sliding_val_loss"] = float(final_sw.group(1))
        out["post_quant_sliding_val_bpb"] = float(final_sw.group(2))

    step_rows = re.findall(
        r"step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+) train_time:([0-9]+)ms step_avg:([0-9.]+)ms",
        log_text,
    )
    if step_rows:
        last = step_rows[-1]
        out["last_eval_step"] = int(last[0])
        out["iterations"] = int(last[1])
        out["pre_quant_val_loss"] = float(last[2])
        out["pre_quant_val_bpb"] = float(last[3])
        out["train_time_ms"] = int(last[4])
        out["step_avg_ms"] = float(last[5])

    artifact = re.search(r"Total submission size int8\+zlib: ([0-9]+) bytes", log_text)
    if artifact:
        out["artifact_bytes_zlib"] = int(artifact.group(1))

    return out or None


def compare_compression(repo_dir: Path) -> dict[str, int | float]:
    """Load the int8+zlib artifact, decompress, and recompress with zstd-22."""
    # The baseline saves to logs/<run_id>/model.ptz or final_model.int8.ptz
    # Search for the artifact
    candidates = list(repo_dir.glob("logs/*/model.ptz")) + list(
        repo_dir.glob("final_model.int8.ptz")
    )
    if not candidates:
        raise FileNotFoundError("No int8 model artifact found in repo_dir")

    artifact_path = candidates[0]
    print(f"[compress] Loading artifact: {artifact_path}")

    raw_bytes = artifact_path.read_bytes()
    zlib_size = len(raw_bytes)
    print(f"[compress] zlib artifact size: {zlib_size:,} bytes")

    # Decompress with zlib (the baseline uses zlib level 9)
    decompressed = zlib.decompress(raw_bytes)
    raw_size = len(decompressed)
    print(f"[compress] Raw (decompressed) size: {raw_size:,} bytes")

    # Recompress with zstd level 22
    compressor = zstd.ZstdCompressor(level=22)
    zstd_bytes = compressor.compress(decompressed)
    zstd_size = len(zstd_bytes)
    print(f"[compress] zstd-22 artifact size: {zstd_size:,} bytes")

    savings_bytes = zlib_size - zstd_size
    savings_pct = (savings_bytes / zlib_size) * 100 if zlib_size > 0 else 0.0

    print(f"[compress] Savings: {savings_bytes:,} bytes ({savings_pct:.2f}%)")

    return {
        "raw_model_bytes": raw_size,
        "zlib9_artifact_bytes": zlib_size,
        "zstd22_artifact_bytes": zstd_size,
        "savings_bytes": savings_bytes,
        "savings_pct": round(savings_pct, 4),
    }


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is required")

    results_repo = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-experiments")
    run_id = os.environ.get("RUN_ID", f"exp02_zstd_{int(time.time())}")
    train_shards = os.environ.get("TRAIN_SHARDS", "20")

    # Common training config
    train_env = {
        "RUN_ID": run_id,
        "SEED": "1337",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_LOSS_EVERY": "2000",
        "VAL_BATCH_SIZE": "65536",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMUP_STEPS": "10",
        "TRAIN_LOG_EVERY": "200",
    }

    api = HfApi(token=token)
    api.create_repo(
        repo_id=results_repo, repo_type="dataset", private=False, exist_ok=True
    )

    workspace = Path("/tmp/exp02_zstd")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = workspace / "parameter-golf"
    train_log_path = workspace / "train.log"

    status = "success"
    error_message = None
    compression_results = None
    started_at = time.time()

    try:
        # Clone and setup
        run_cmd(
            [
                "git",
                "clone",
                "https://github.com/openai/parameter-golf.git",
                str(repo_dir),
            ]
        )
        run_cmd(
            [
                "python3",
                "data/cached_challenge_fineweb.py",
                "--variant",
                "sp1024",
                "--train-shards",
                train_shards,
            ],
            cwd=repo_dir,
        )

        # Train baseline
        env = os.environ.copy()
        env.update(train_env)
        with train_log_path.open("w", encoding="utf-8") as logf:
            run_cmd(
                [
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node=1",
                    "train_gpt.py",
                ],
                cwd=repo_dir,
                env=env,
                log_file=logf,
            )

        # Compare compression
        compression_results = compare_compression(repo_dir)

    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        print(f"[error] {error_message}")

    finished_at = time.time()
    log_text = (
        train_log_path.read_text(encoding="utf-8") if train_log_path.exists() else ""
    )
    metrics = parse_metrics(log_text)

    result_payload = {
        "status": status,
        "error": error_message,
        "experiment": "02-compression",
        "description": "zstd-22 vs zlib-9 compression comparison on int8 model artifact",
        "run_id": run_id,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "duration_seconds": finished_at - started_at,
        "config": {
            "train_shards": int(train_shards),
            "seed": int(train_env["SEED"]),
            "iterations": int(train_env["ITERATIONS"]),
            "train_batch_tokens": int(train_env["TRAIN_BATCH_TOKENS"]),
            "val_batch_size": int(train_env["VAL_BATCH_SIZE"]),
            "val_loss_every": int(train_env["VAL_LOSS_EVERY"]),
            "warmup_steps": int(train_env["WARMUP_STEPS"]),
        },
        "metrics": metrics,
        "compression": compression_results,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_prefix = f"jobs/exp02/{run_id}_{stamp}"
    results_json_path = workspace / "results.json"
    results_json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    api.upload_file(
        path_or_fileobj=str(results_json_path),
        path_in_repo=f"{out_prefix}/results.json",
        repo_id=results_repo,
        repo_type="dataset",
        token=token,
    )
    if train_log_path.exists():
        api.upload_file(
            path_or_fileobj=str(train_log_path),
            path_in_repo=f"{out_prefix}/train.log",
            repo_id=results_repo,
            repo_type="dataset",
            token=token,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 02 — COMPRESSION COMPARISON RESULTS")
    print("=" * 60)
    if metrics:
        print(f"  val_bpb (post-quant): {metrics.get('post_quant_val_bpb', 'N/A')}")
    if compression_results:
        print(
            f"  zlib-9 artifact:  {compression_results['zlib9_artifact_bytes']:>10,} bytes"
        )
        print(
            f"  zstd-22 artifact: {compression_results['zstd22_artifact_bytes']:>10,} bytes"
        )
        print(
            f"  Savings:          {compression_results['savings_bytes']:>10,} bytes ({compression_results['savings_pct']:.2f}%)"
        )
    print(f"  Status: {status}")
    print(f"  Results uploaded to: {results_repo}/{out_prefix}")
    print("=" * 60)

    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
