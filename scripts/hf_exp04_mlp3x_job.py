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
Experiment 04 — 3x MLP Expansion.

Compares MLP_MULT=3 against the default MLP expansion ratio. A wider MLP
should improve model capacity (lower bpb) at the cost of a larger artifact.
Both baseline and 3x runs use 2000 steps for a fair comparison.

Workflow:
1) Clone parameter-golf repo, download 20 train shards
2) Train baseline (default MLP) for 2000 steps
3) Train with MLP_MULT=3 for 2000 steps
4) Compare val_bpb and artifact sizes
5) Push results to HF Hub
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

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
        out["artifact_bytes"] = int(artifact.group(1))

    return out or None


def train_run(
    repo_dir: Path,
    log_path: Path,
    run_id: str,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Run a single training job and return the log text."""
    base_env = {
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
    env = os.environ.copy()
    env.update(base_env)
    if extra_env:
        env.update(extra_env)

    with log_path.open("w", encoding="utf-8") as logf:
        run_cmd(
            ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
            cwd=repo_dir,
            env=env,
            log_file=logf,
        )

    return log_path.read_text(encoding="utf-8")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is required")

    results_repo = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-experiments")
    train_shards = os.environ.get("TRAIN_SHARDS", "20")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=results_repo, repo_type="dataset", private=False, exist_ok=True
    )

    workspace = Path("/tmp/exp04_mlp3x")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = workspace / "parameter-golf"

    status = "success"
    error_message = None
    baseline_metrics = None
    mlp3x_metrics = None
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

        # Run 1: Baseline
        print("\n" + "=" * 60)
        print("RUNNING BASELINE (default MLP)")
        print("=" * 60)
        baseline_log = train_run(
            repo_dir,
            workspace / "baseline.log",
            f"exp04_baseline_{int(time.time())}",
        )
        baseline_metrics = parse_metrics(baseline_log)

        # Run 2: MLP_MULT=3
        print("\n" + "=" * 60)
        print("RUNNING MLP_MULT=3")
        print("=" * 60)
        mlp3x_log = train_run(
            repo_dir,
            workspace / "mlp3x.log",
            f"exp04_mlp3x_{int(time.time())}",
            extra_env={"MLP_MULT": "3"},
        )
        mlp3x_metrics = parse_metrics(mlp3x_log)

    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        print(f"[error] {error_message}")

    finished_at = time.time()

    result_payload = {
        "status": status,
        "error": error_message,
        "experiment": "04-bigger-mlp",
        "description": "3x MLP expansion vs baseline — comparing val_bpb and artifact size",
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "duration_seconds": finished_at - started_at,
        "config": {
            "train_shards": int(train_shards),
            "seed": 1337,
            "iterations": 2000,
            "train_batch_tokens": 65536,
            "val_batch_size": 65536,
            "warmup_steps": 10,
        },
        "baseline": baseline_metrics,
        "mlp3x": mlp3x_metrics,
    }

    # Compute delta if both succeeded
    if baseline_metrics and mlp3x_metrics:
        b_bpb = baseline_metrics.get("post_quant_val_bpb")
        m_bpb = mlp3x_metrics.get("post_quant_val_bpb")
        if b_bpb and m_bpb:
            result_payload["delta_bpb"] = round(m_bpb - b_bpb, 6)

        b_art = baseline_metrics.get("artifact_bytes")
        m_art = mlp3x_metrics.get("artifact_bytes")
        if b_art and m_art:
            result_payload["delta_artifact_bytes"] = m_art - b_art

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_id = f"exp04_mlp3x_{stamp}"
    out_prefix = f"jobs/exp04/{run_id}"
    results_json_path = workspace / "results.json"
    results_json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    api.upload_file(
        path_or_fileobj=str(results_json_path),
        path_in_repo=f"{out_prefix}/results.json",
        repo_id=results_repo,
        repo_type="dataset",
        token=token,
    )
    for logname in ["baseline.log", "mlp3x.log"]:
        logpath = workspace / logname
        if logpath.exists():
            api.upload_file(
                path_or_fileobj=str(logpath),
                path_in_repo=f"{out_prefix}/{logname}",
                repo_id=results_repo,
                repo_type="dataset",
                token=token,
            )

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 04 — 3x MLP EXPANSION RESULTS")
    print("=" * 60)
    if baseline_metrics:
        print(
            f"  Baseline val_bpb:  {baseline_metrics.get('post_quant_val_bpb', 'N/A')}"
        )
        print(
            f"  Baseline artifact: {baseline_metrics.get('artifact_bytes', 'N/A')} bytes"
        )
    if mlp3x_metrics:
        print(
            f"  MLP 3x val_bpb:   {mlp3x_metrics.get('post_quant_val_bpb', 'N/A')}"
        )
        print(
            f"  MLP 3x artifact:  {mlp3x_metrics.get('artifact_bytes', 'N/A')} bytes"
        )
    if "delta_bpb" in result_payload:
        print(f"  Delta bpb:         {result_payload['delta_bpb']:+.6f}")
    if "delta_artifact_bytes" in result_payload:
        print(
            f"  Delta artifact:    {result_payload['delta_artifact_bytes']:+,} bytes"
        )
    print(f"  Status: {status}")
    print(f"  Results uploaded to: {results_repo}/{out_prefix}")
    print("=" * 60)

    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
