#!/usr/bin/env python3
"""
Submit Experiment 09 (ChunkGate-Lite) to Hugging Face Jobs.

This script:
1) Uploads local CUDA trainer to dataset artifact path
2) Submits a UV job (default: l4x1) with short smoke-run config
3) Prints job id and quick monitor snippet
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main() -> None:
    token = get_token()
    if not token:
        raise RuntimeError("No HF token found. Run `huggingface-cli login` first.")

    api = HfApi(token=token)

    results_repo = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-curriculum")
    script_artifact_path = os.environ.get("EXP09_SCRIPT_PATH", "artifacts/exp09/train_gpt_exp09.py")
    script_file = Path(os.environ.get("EXP09_LOCAL_SCRIPT", "experiments/09-chunkgate-lite/train_gpt.py")).resolve()
    job_script = Path(os.environ.get("EXP09_JOB_SCRIPT", "scripts/hf_jobs/exp09_hf_job.py")).resolve()

    flavor = os.environ.get("HF_FLAVOR", "l4x1")
    timeout = os.environ.get("HF_TIMEOUT", "2h")

    run_id = os.environ.get("RUN_ID", "exp09_hf_smoke")
    train_shards = os.environ.get("TRAIN_SHARDS", "1")
    max_wallclock_seconds = os.environ.get("MAX_WALLCLOCK_SECONDS", "600")
    iterations = os.environ.get("ITERATIONS", "300")
    train_batch_tokens = os.environ.get("TRAIN_BATCH_TOKENS", "262144")
    train_seq_len = os.environ.get("TRAIN_SEQ_LEN", "1024")
    val_loss_every = os.environ.get("VAL_LOSS_EVERY", "0")
    train_log_every = os.environ.get("TRAIN_LOG_EVERY", "50")

    if not script_file.exists():
        raise FileNotFoundError(f"Missing local trainer script: {script_file}")
    if not job_script.exists():
        raise FileNotFoundError(f"Missing HF job script: {job_script}")

    api.create_repo(repo_id=results_repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(script_file),
        path_in_repo=script_artifact_path,
        repo_id=results_repo,
        repo_type="dataset",
        token=token,
    )
    print(f"Uploaded trainer: {results_repo}/{script_artifact_path}")

    job = api.run_uv_job(
        str(job_script),
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": token},
        env={
            "RESULTS_REPO": results_repo,
            "EXP09_SCRIPT_REPO": results_repo,
            "EXP09_SCRIPT_PATH": script_artifact_path,
            "RUN_ID": run_id,
            "TRAIN_SHARDS": train_shards,
            "MAX_WALLCLOCK_SECONDS": max_wallclock_seconds,
            "ITERATIONS": iterations,
            "TRAIN_BATCH_TOKENS": train_batch_tokens,
            "TRAIN_SEQ_LEN": train_seq_len,
            "VAL_LOSS_EVERY": val_loss_every,
            "TRAIN_LOG_EVERY": train_log_every,
            "WARMUP_STEPS": os.environ.get("WARMUP_STEPS", "0"),
            "ENABLE_TORCH_COMPILE": os.environ.get("ENABLE_TORCH_COMPILE", "0"),
            "CHUNKGATE_ENABLE": os.environ.get("CHUNKGATE_ENABLE", "1"),
            "CHUNKGATE_STRIDE": os.environ.get("CHUNKGATE_STRIDE", "4"),
            "CHUNKGATE_INNER_LAYERS": os.environ.get("CHUNKGATE_INNER_LAYERS", "2"),
            "CHUNKGATE_GATE_TEMP": os.environ.get("CHUNKGATE_GATE_TEMP", "1.0"),
            "CHUNKGATE_FUSION_INIT": os.environ.get("CHUNKGATE_FUSION_INIT", "0.10"),
        },
        token=token,
    )

    print(f"Job ID: {job.id}")
    print(f"Status: {job.status.stage}")
    print("Monitor:")
    print(
        "python3 - << 'PY'\n"
        "from huggingface_hub import HfApi\n"
        f"j=HfApi().inspect_job(job_id='{job.id}')\n"
        "print(j.status.stage)\n"
        "for line in HfApi().fetch_job_logs(job_id=j.id):\n"
        "    print(line)\n"
        "PY"
    )


if __name__ == "__main__":
    main()
