#!/usr/bin/env python3
"""
Submit Experiment 21 (winner repro proxy) to Hugging Face Jobs.

This script runs the current winning reference train_gpt snapshot with
single-GPU-safe overrides so we can reproduce behavior on L4 while keeping
the exact architecture and quantization path.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main() -> None:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("No HF token found. Run `huggingface-cli login` first.")

    api = HfApi(token=token)

    results_repo = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-curriculum")
    job_script = Path(os.environ.get("EXP21_JOB_SCRIPT", "scripts/hf_jobs/exp21_hf_job.py")).resolve()

    if not job_script.exists():
        raise FileNotFoundError(f"Missing HF job script: {job_script}")

    flavor = os.environ.get("HF_FLAVOR", "l4x1")
    timeout = os.environ.get("HF_TIMEOUT", "2h")

    job = api.run_uv_job(
        str(job_script),
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": token},
        env={
            "RESULTS_REPO": results_repo,
            "RUN_ID": os.environ.get("RUN_ID", "exp21_winner_repro_l4"),
            "TRAIN_SHARDS": os.environ.get("TRAIN_SHARDS", "1"),
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
            "PG_REF": os.environ.get("PG_REF", "9f9d53343aa44fe1fbd94ae32650ca2e83602a10"),
            "WINNER_SCRIPT": os.environ.get(
                "WINNER_SCRIPT",
                "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py",
            ),
        },
        token=token,
    )

    print(f"Job ID: {job.id}")
    print(f"Status: {job.status.stage}")
    print("Monitor:")
    print(
        "python3 - << 'PY'\n"
        "from huggingface_hub import HfApi\n"
        f"j=HfApi().inspect_job(job_id='{job.id}', namespace='JosueG')\n"
        "print(j.status.stage)\n"
        "for line in HfApi().fetch_job_logs(job_id=j.id, namespace='JosueG'):\n"
        "    print(line)\n"
        "PY"
    )


if __name__ == "__main__":
    main()
