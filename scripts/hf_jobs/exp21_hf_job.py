# /// script
# dependencies = ["huggingface-hub>=0.36.0", "datasets", "sentencepiece", "numpy", "torch", "tqdm", "zstandard"]
# ///

"""
Self-contained HF Job script for Experiment 21 (winner repro proxy).

Workflow:
1) Clone OpenAI Parameter Golf repo at a pinned commit
2) Download challenge data shards
3) Run the winning record train_gpt.py with single-GPU-safe overrides
4) Upload logs + parsed metrics + optional artifact to HF dataset repo
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


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, log_file=None) -> None:
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

    final_q = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)", log_text)
    if final_q:
        out["post_quant_val_loss"] = float(final_q.group(1))
        out["post_quant_val_bpb"] = float(final_q.group(2))

    final_sw = re.search(r"final_sliding_window_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)", log_text)
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


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is required")

    results_repo = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-curriculum")
    run_id = os.environ.get("RUN_ID", f"exp21_winner_repro_{int(time.time())}")
    train_shards = os.environ.get("TRAIN_SHARDS", "1")

    pg_ref = os.environ.get("PG_REF", "9f9d53343aa44fe1fbd94ae32650ca2e83602a10")
    winner_script = os.environ.get(
        "WINNER_SCRIPT",
        "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py",
    )

    train_env = {
        "RUN_ID": run_id,
        "SEED": os.environ.get("SEED", "42"),
        "MAX_WALLCLOCK_SECONDS": os.environ.get("MAX_WALLCLOCK_SECONDS", "600"),
        "ITERATIONS": os.environ.get("ITERATIONS", "20000"),
        "TRAIN_BATCH_TOKENS": os.environ.get("TRAIN_BATCH_TOKENS", "131072"),
        "TRAIN_SEQ_LEN": os.environ.get("TRAIN_SEQ_LEN", "2048"),
        "VAL_BATCH_SIZE": os.environ.get("VAL_BATCH_SIZE", "131072"),
        "VAL_LOSS_EVERY": os.environ.get("VAL_LOSS_EVERY", "500"),
        "TRAIN_LOG_EVERY": os.environ.get("TRAIN_LOG_EVERY", "100"),
        "EVAL_STRIDE": os.environ.get("EVAL_STRIDE", "64"),
        "EVAL_BATCH_SEQS": os.environ.get("EVAL_BATCH_SEQS", "8"),
    }

    api = HfApi(token=token)
    api.create_repo(repo_id=results_repo, repo_type="dataset", private=False, exist_ok=True)

    workspace = Path("/tmp/exp21_hf_job")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = workspace / "parameter-golf"
    train_log_path = workspace / "train.log"

    status = "success"
    error_message = None
    started_at = time.time()
    try:
        run_cmd(["git", "clone", "https://github.com/openai/parameter-golf.git", str(repo_dir)])
        run_cmd(["git", "checkout", pg_ref], cwd=repo_dir)
        run_cmd(
            ["python3", "data/cached_challenge_fineweb.py", "--variant", "sp1024", "--train-shards", train_shards],
            cwd=repo_dir,
        )

        env = os.environ.copy()
        env.update(train_env)
        with train_log_path.open("w", encoding="utf-8") as logf:
            run_cmd(
                ["torchrun", "--standalone", "--nproc_per_node=1", winner_script],
                cwd=repo_dir,
                env=env,
                log_file=logf,
            )
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error_message = str(exc)
        print(f"[error] {error_message}")

    finished_at = time.time()
    log_text = train_log_path.read_text(encoding="utf-8") if train_log_path.exists() else ""
    metrics = parse_metrics(log_text)

    result_payload = {
        "status": status,
        "error": error_message,
        "experiment": "21-winner-repro",
        "run_id": run_id,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "duration_seconds": finished_at - started_at,
        "source": {
            "repo": "https://github.com/openai/parameter-golf",
            "commit": pg_ref,
            "winner_script": winner_script,
            "winner_submission": "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/submission.json",
        },
        "config": {
            "train_shards": int(train_shards),
            "seed": int(train_env["SEED"]),
            "max_wallclock_seconds": float(train_env["MAX_WALLCLOCK_SECONDS"]),
            "iterations": int(train_env["ITERATIONS"]),
            "train_batch_tokens": int(train_env["TRAIN_BATCH_TOKENS"]),
            "train_seq_len": int(train_env["TRAIN_SEQ_LEN"]),
            "val_batch_size": int(train_env["VAL_BATCH_SIZE"]),
            "val_loss_every": int(train_env["VAL_LOSS_EVERY"]),
            "train_log_every": int(train_env["TRAIN_LOG_EVERY"]),
            "eval_stride": int(train_env["EVAL_STRIDE"]),
            "eval_batch_seqs": int(train_env["EVAL_BATCH_SEQS"]),
        },
        "metrics": metrics,
        "citations": [
            {
                "name": "OpenAI Parameter Golf repository",
                "url": "https://github.com/openai/parameter-golf",
            },
            {
                "name": "Winning reference snapshot (Mar 20, 2026)",
                "url": "https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50",
            },
        ],
    }

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_prefix = f"jobs/exp21/{run_id}_{stamp}"
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
    model_artifact = repo_dir / "final_model.int8.ptz"
    if model_artifact.exists():
        api.upload_file(
            path_or_fileobj=str(model_artifact),
            path_in_repo=f"{out_prefix}/final_model.int8.ptz",
            repo_id=results_repo,
            repo_type="dataset",
            token=token,
        )

    print(f"[done] uploaded results to dataset repo: {results_repo}/{out_prefix}")
    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
