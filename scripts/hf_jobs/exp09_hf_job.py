# /// script
# dependencies = ["huggingface-hub>=0.36.0", "datasets", "sentencepiece", "numpy", "torch", "tqdm"]
# ///

"""
Self-contained HF Job script for Parameter Golf Experiment 09 (ChunkGate-Lite).

Workflow:
1) Clone OpenAI Parameter Golf baseline repo
2) Download Experiment 09 CUDA trainer from HF dataset artifact storage
3) Run a short single-GPU smoke training
4) Upload logs + metrics + optional model artifact back to HF dataset repo
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


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


def parse_final_metrics(log_text: str) -> dict[str, float] | None:
    m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)", log_text)
    if not m:
        return None
    return {"post_quant_val_loss": float(m.group(1)), "post_quant_val_bpb": float(m.group(2))}


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is required")

    results_repo = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-curriculum")
    script_repo = os.environ.get("EXP09_SCRIPT_REPO", results_repo)
    script_path = os.environ.get("EXP09_SCRIPT_PATH", "artifacts/exp09/train_gpt_exp09.py")
    run_id = os.environ.get("RUN_ID", f"exp09_hf_{int(time.time())}")
    train_shards = os.environ.get("TRAIN_SHARDS", "1")
    timeout_s = os.environ.get("MAX_WALLCLOCK_SECONDS", "600")
    iterations = os.environ.get("ITERATIONS", "300")
    train_batch_tokens = os.environ.get("TRAIN_BATCH_TOKENS", "262144")
    train_seq_len = os.environ.get("TRAIN_SEQ_LEN", "1024")
    val_loss_every = os.environ.get("VAL_LOSS_EVERY", "0")
    train_log_every = os.environ.get("TRAIN_LOG_EVERY", "50")
    chunkgate_enable = os.environ.get("CHUNKGATE_ENABLE", "1")

    api = HfApi(token=token)
    api.create_repo(repo_id=results_repo, repo_type="dataset", private=False, exist_ok=True)

    workspace = Path("/tmp/exp09_hf_job")
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
        run_cmd(["python3", "data/cached_challenge_fineweb.py", "--variant", "sp1024", "--train-shards", train_shards], cwd=repo_dir)

        exp09_script_local = hf_hub_download(
            repo_id=script_repo,
            repo_type="dataset",
            filename=script_path,
            token=token,
        )
        shutil.copy(exp09_script_local, repo_dir / "train_gpt.py")

        train_env = os.environ.copy()
        train_env.update(
            {
                "RUN_ID": run_id,
                "MAX_WALLCLOCK_SECONDS": timeout_s,
                "ITERATIONS": iterations,
                "TRAIN_BATCH_TOKENS": train_batch_tokens,
                "TRAIN_SEQ_LEN": train_seq_len,
                "VAL_LOSS_EVERY": val_loss_every,
                "TRAIN_LOG_EVERY": train_log_every,
                "WARMUP_STEPS": os.environ.get("WARMUP_STEPS", "0"),
                "ENABLE_TORCH_COMPILE": os.environ.get("ENABLE_TORCH_COMPILE", "0"),
                "CHUNKGATE_ENABLE": chunkgate_enable,
                "CHUNKGATE_STRIDE": os.environ.get("CHUNKGATE_STRIDE", "4"),
                "CHUNKGATE_INNER_LAYERS": os.environ.get("CHUNKGATE_INNER_LAYERS", "2"),
                "CHUNKGATE_GATE_TEMP": os.environ.get("CHUNKGATE_GATE_TEMP", "1.0"),
                "CHUNKGATE_FUSION_INIT": os.environ.get("CHUNKGATE_FUSION_INIT", "0.10"),
            }
        )
        with train_log_path.open("w", encoding="utf-8") as logf:
            run_cmd(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], cwd=repo_dir, env=train_env, log_file=logf)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error_message = str(exc)
        print(f"[error] {error_message}")

    finished_at = time.time()
    log_text = train_log_path.read_text(encoding="utf-8") if train_log_path.exists() else ""
    metrics = parse_final_metrics(log_text)

    result_payload = {
        "status": status,
        "error": error_message,
        "run_id": run_id,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "duration_seconds": finished_at - started_at,
        "config": {
            "train_shards": int(train_shards),
            "max_wallclock_seconds": int(timeout_s),
            "iterations": int(iterations),
            "train_batch_tokens": int(train_batch_tokens),
            "train_seq_len": int(train_seq_len),
            "val_loss_every": int(val_loss_every),
            "train_log_every": int(train_log_every),
            "warmup_steps": int(os.environ.get("WARMUP_STEPS", "0")),
            "enable_torch_compile": bool(int(os.environ.get("ENABLE_TORCH_COMPILE", "0"))),
            "chunkgate_enable": bool(int(chunkgate_enable)),
            "chunkgate_stride": int(os.environ.get("CHUNKGATE_STRIDE", "4")),
            "chunkgate_inner_layers": int(os.environ.get("CHUNKGATE_INNER_LAYERS", "2")),
            "chunkgate_gate_temp": float(os.environ.get("CHUNKGATE_GATE_TEMP", "1.0")),
            "chunkgate_fusion_init": float(os.environ.get("CHUNKGATE_FUSION_INIT", "0.10")),
        },
        "metrics": metrics,
        "citations": [
            {"name": "OpenAI Parameter Golf baseline", "url": "https://github.com/openai/parameter-golf"},
            {"name": "H-Net paper", "url": "https://arxiv.org/abs/2507.07955"},
            {"name": "H-Net code", "url": "https://github.com/goombalab/hnet"},
        ],
    }

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_prefix = f"jobs/exp09/{run_id}_{stamp}"
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
