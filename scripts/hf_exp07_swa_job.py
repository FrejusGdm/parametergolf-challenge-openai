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
#     "kernels",
# ]
# ///
"""
Experiment 07: Stochastic Weight Averaging (SWA)

Collects model checkpoints during the last 50% of training and averages them
to produce a smoother, better-generalizing model. SWA is known to flatten the
loss landscape and improve generalization with minimal overhead.

Implementation: checkpoints are collected every N steps after the halfway point.
After training, all checkpoints are averaged and used for final eval + quantization.

Two runs:
  1. Baseline (unmodified) for A/B comparison
  2. Baseline + SWA (average last 50% of checkpoints)

Environment variables:
  HF_TOKEN       — for pushing results (required)
  ITERATIONS     — training steps (default 2000)
  SWA_START_FRAC — fraction of training after which to start collecting (default 0.5)
  SWA_EVERY      — collect a checkpoint every N steps (default 50)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

ITERATIONS = os.environ.get("ITERATIONS", "2000")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RESULTS_REPO = "JosueG/parameter-golf-experiments"
TRAIN_SHARDS = os.environ.get("TRAIN_SHARDS", "20")
SWA_START_FRAC = os.environ.get("SWA_START_FRAC", "0.5")
SWA_EVERY = os.environ.get("SWA_EVERY", "50")
PINNED_COMMIT = "9f9d53343aa44fe1fbd94ae32650ca2e83602a10"


def run_cmd(cmd, cwd=None, env=None):
    print(f"[cmd] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    rc = proc.wait()
    output = "".join(output_lines)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")
    return output


def parse_results(output):
    """Extract val_bpb, val_loss, artifact size from training output."""
    results = {}

    step_rows = re.findall(
        r"step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+) train_time:([0-9]+)ms step_avg:([0-9.]+)ms",
        output,
    )
    if step_rows:
        last = step_rows[-1]
        results["pre_quant_val_loss"] = float(last[2])
        results["pre_quant_val_bpb"] = float(last[3])
        results["train_time_ms"] = int(last[4])
        results["step_avg_ms"] = float(last[5])

    final_q = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)", output
    )
    if final_q:
        results["post_quant_val_loss"] = float(final_q.group(1))
        results["post_quant_val_bpb"] = float(final_q.group(2))

    artifact = re.search(r"Total submission size int8\+zlib: ([0-9]+) bytes", output)
    if artifact:
        results["artifact_bytes"] = int(artifact.group(1))

    return results


def get_train_env(run_id):
    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id,
        "ITERATIONS": ITERATIONS,
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_LOSS_EVERY": ITERATIONS,
        "VAL_BATCH_SIZE": "65536",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMUP_STEPS": "10",
        "TRAIN_LOG_EVERY": "200",
        "SEED": "1337",
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
    })
    return env


def apply_swa_patches(script_path):
    """Patch train_gpt.py to collect checkpoints and average them (SWA)."""
    with open(script_path, "r") as f:
        code = f.read()

    swa_start_frac = float(SWA_START_FRAC)
    swa_every = int(SWA_EVERY)

    # ── Patch 1: Add SWA checkpoint list initialization before training loop ──
    # Insert right before "training_time_ms = 0.0"
    old_init = "    training_time_ms = 0.0"
    new_init = f"""    # SWA: collect checkpoints during last {swa_start_frac*100:.0f}% of training
    swa_checkpoints = []
    swa_start_step = int(args.iterations * {swa_start_frac})
    swa_every = {swa_every}
    training_time_ms = 0.0"""

    if old_init not in code:
        raise RuntimeError("Could not find 'training_time_ms = 0.0' to patch")
    code = code.replace(old_init, new_init, 1)
    print("Patch 1: Added SWA checkpoint list initialization")

    # ── Patch 2: Add checkpoint collection after optimizer step ──
    # Insert right after "zero_grad_all()" at the end of the optimizer step block
    # The pattern is: "for opt in optimizers:\n            opt.step()\n        zero_grad_all()"
    old_step_end = """        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1"""

    new_step_end = """        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1

        # SWA: collect checkpoint if past start point
        if step >= swa_start_step and step % swa_every == 0:
            swa_checkpoints.append({{k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}})
            if master_process:
                log0(f"SWA: collected checkpoint at step {{step}} (total: {{len(swa_checkpoints)}})")"""

    if old_step_end not in code:
        raise RuntimeError("Could not find optimizer step end pattern to patch")
    code = code.replace(old_step_end, new_step_end, 1)
    print("Patch 2: Added SWA checkpoint collection in training loop")

    # ── Patch 3: Average checkpoints after training loop, before serialization ──
    # Insert right after the peak memory log line
    old_peak = '''    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )'''

    new_peak = '''    log0(
        f"peak memory allocated: {{torch.cuda.max_memory_allocated() // 1024 // 1024}} MiB "
        f"reserved: {{torch.cuda.max_memory_reserved() // 1024 // 1024}} MiB"
    )

    # SWA: average collected checkpoints and load into model
    if swa_checkpoints:
        log0(f"SWA: averaging {{len(swa_checkpoints)}} checkpoints (steps {{swa_start_step}}..{{step}})")
        avg_state = {{}}
        for key in swa_checkpoints[0]:
            stacked = torch.stack([ckpt[key].float() for ckpt in swa_checkpoints])
            avg_state[key] = stacked.mean(0).to(dtype=swa_checkpoints[0][key].dtype)
        base_model.load_state_dict(avg_state, strict=True)
        del swa_checkpoints  # free memory
        log0("SWA: averaged model loaded")

        # Re-evaluate with SWA-averaged model
        torch.cuda.synchronize()
        swa_val_loss, swa_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"SWA pre-quant: val_loss={{swa_val_loss:.4f}} val_bpb={{swa_val_bpb:.4f}}")
    else:
        log0("SWA: no checkpoints collected, using final model as-is")'''

    if old_peak not in code:
        # The f-string uses curly braces which need careful handling.
        # Try a simpler anchor approach.
        raise RuntimeError("Could not find peak memory log to patch")
    code = code.replace(old_peak, new_peak, 1)
    print("Patch 3: Added SWA averaging after training loop")

    with open(script_path, "w") as f:
        f.write(code)
    print("All SWA patches applied successfully")


def apply_swa_patches_safe(script_path):
    """
    Alternative patching strategy that works line-by-line to avoid f-string
    matching issues. Reads lines and inserts SWA code at the right positions.
    """
    with open(script_path, "r") as f:
        lines = f.readlines()

    code = "".join(lines)
    swa_start_frac = float(SWA_START_FRAC)
    swa_every = int(SWA_EVERY)

    # ── Patch 1: Add SWA init before training_time_ms ──
    swa_init = f'''    # SWA: collect checkpoints during last {swa_start_frac*100:.0f}% of training
    swa_checkpoints = []
    swa_start_step = int(args.iterations * {swa_start_frac})
    swa_every = {swa_every}
'''
    target1 = "    training_time_ms = 0.0\n"
    idx1 = code.find(target1)
    if idx1 < 0:
        raise RuntimeError("Could not find 'training_time_ms = 0.0'")
    code = code[:idx1] + swa_init + code[idx1:]
    print("Patch 1: Added SWA init")

    # ── Patch 2: Add checkpoint collection after step increment ──
    # Find "        step += 1\n" and add SWA collection after it
    target2 = "        step += 1\n"
    # Find the FIRST occurrence after "zero_grad_all()" near optimizer
    search_start = code.find("for opt in optimizers:\n            opt.step()")
    if search_start < 0:
        raise RuntimeError("Could not find optimizer step pattern")
    idx2 = code.find(target2, search_start)
    if idx2 < 0:
        raise RuntimeError("Could not find 'step += 1'")
    insert_after = idx2 + len(target2)

    swa_collect = '''
        # SWA: collect checkpoint if past start point
        if step >= swa_start_step and step % swa_every == 0:
            swa_checkpoints.append({k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()})
            if master_process:
                log0(f"SWA: collected checkpoint at step {step} (total: {len(swa_checkpoints)})")
'''
    code = code[:insert_after] + swa_collect + code[insert_after:]
    print("Patch 2: Added SWA checkpoint collection")

    # ── Patch 3: Add SWA averaging after peak memory log, before serialization ──
    # Find the serialization section marker
    target3 = "    # SERIALIZATION + ROUNDTRIP VALIDATION"
    idx3 = code.find(target3)
    if idx3 < 0:
        # Try alternate
        target3 = "    if master_process:\n        torch.save(base_model.state_dict()"
        idx3 = code.find(target3)
        if idx3 < 0:
            raise RuntimeError("Could not find serialization section")

    swa_average = '''
    # SWA: average collected checkpoints and load into model
    if swa_checkpoints:
        log0(f"SWA: averaging {len(swa_checkpoints)} checkpoints (steps {swa_start_step}..{step})")
        avg_state = {}
        for key in swa_checkpoints[0]:
            stacked = torch.stack([ckpt[key].float() for ckpt in swa_checkpoints])
            avg_state[key] = stacked.mean(0).to(dtype=swa_checkpoints[0][key].dtype)
        base_model.load_state_dict(avg_state, strict=True)
        del swa_checkpoints  # free memory
        log0("SWA: averaged model loaded")

        # Re-evaluate with SWA-averaged model
        torch.cuda.synchronize()
        swa_val_loss, swa_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"SWA pre-quant: val_loss={swa_val_loss:.4f} val_bpb={swa_val_bpb:.4f}")
    else:
        log0("SWA: no checkpoints collected, using final model as-is")

'''
    code = code[:idx3] + swa_average + code[idx3:]
    print("Patch 3: Added SWA averaging before serialization")

    with open(script_path, "w") as f:
        f.write(code)
    print("All SWA patches applied successfully")


def main():
    print("=" * 80)
    print("Experiment 07: Stochastic Weight Averaging (SWA)")
    print(f"Iterations: {ITERATIONS}, Train shards: {TRAIN_SHARDS}")
    print(f"SWA config: start_frac={SWA_START_FRAC}, collect_every={SWA_EVERY} steps")
    print("=" * 80)

    # ── Clone and setup ──
    run_cmd(["git", "clone", "https://github.com/openai/parameter-golf.git"])
    os.chdir("parameter-golf")
    run_cmd(["git", "checkout", PINNED_COMMIT])

    print("\n=== Downloading dataset ===")
    run_cmd([
        sys.executable, "data/cached_challenge_fineweb.py",
        "--variant", "sp1024",
        "--train-shards", TRAIN_SHARDS,
    ])

    # ── Run 1: Baseline ──
    print("\n" + "=" * 80)
    print("Run 1: BASELINE (unmodified)")
    print("=" * 80)

    shutil.copy("train_gpt.py", "train_gpt_baseline.py")

    t_start = time.time()
    baseline_output = run_cmd(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
        env=get_train_env("exp07_baseline"),
    )
    baseline_elapsed = time.time() - t_start
    baseline_results = parse_results(baseline_output)
    baseline_results["elapsed_seconds"] = round(baseline_elapsed, 1)
    print(f"\nBaseline results: {json.dumps(baseline_results, indent=2)}")

    # ── Run 2: SWA ──
    print("\n" + "=" * 80)
    print("Run 2: Stochastic Weight Averaging")
    print("=" * 80)

    shutil.copy("train_gpt_baseline.py", "train_gpt.py")
    apply_swa_patches_safe("train_gpt.py")

    t_start = time.time()
    swa_output = run_cmd(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
        env=get_train_env("exp07_swa"),
    )
    swa_elapsed = time.time() - t_start
    swa_results = parse_results(swa_output)
    swa_results["elapsed_seconds"] = round(swa_elapsed, 1)

    # Also parse the SWA-specific pre-quant line
    swa_prequant = re.search(r"SWA pre-quant: val_loss=([0-9.]+) val_bpb=([0-9.]+)", swa_output)
    if swa_prequant:
        swa_results["swa_pre_quant_val_loss"] = float(swa_prequant.group(1))
        swa_results["swa_pre_quant_val_bpb"] = float(swa_prequant.group(2))

    num_ckpts = re.findall(r"SWA: collected checkpoint at step \d+ \(total: (\d+)\)", swa_output)
    if num_ckpts:
        swa_results["swa_checkpoints_collected"] = int(num_ckpts[-1])

    print(f"\nSWA results: {json.dumps(swa_results, indent=2)}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("EXPERIMENT 07 SUMMARY")
    print("=" * 80)

    summary = {
        "experiment": "07-swa",
        "description": "Stochastic Weight Averaging (last 50% of training)",
        "config": {
            "swa_start_frac": float(SWA_START_FRAC),
            "swa_every": int(SWA_EVERY),
        },
        "iterations": int(ITERATIONS),
        "train_shards": int(TRAIN_SHARDS),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline_results,
        "swa": swa_results,
    }

    if "post_quant_val_bpb" in baseline_results and "post_quant_val_bpb" in swa_results:
        delta = swa_results["post_quant_val_bpb"] - baseline_results["post_quant_val_bpb"]
        summary["delta_post_quant_val_bpb"] = round(delta, 6)
        print(f"Post-quant val_bpb delta: {delta:+.6f} ({'better' if delta < 0 else 'worse'})")

    if "pre_quant_val_bpb" in baseline_results and "pre_quant_val_bpb" in swa_results:
        delta = swa_results["pre_quant_val_bpb"] - baseline_results["pre_quant_val_bpb"]
        summary["delta_pre_quant_val_bpb"] = round(delta, 6)
        print(f"Pre-quant val_bpb delta (last step vs baseline): {delta:+.6f}")

    if "swa_pre_quant_val_bpb" in swa_results and "pre_quant_val_bpb" in baseline_results:
        delta = swa_results["swa_pre_quant_val_bpb"] - baseline_results["pre_quant_val_bpb"]
        summary["delta_swa_pre_quant_val_bpb"] = round(delta, 6)
        print(f"SWA-averaged pre-quant val_bpb delta: {delta:+.6f} ({'better' if delta < 0 else 'worse'})")

    results_json = json.dumps(summary, indent=2)
    print(f"\n{results_json}")

    # ── Push to Hub ──
    if HF_TOKEN:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        try:
            api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
        except Exception:
            pass

        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        prefix = f"jobs/exp07/{stamp}"
        api.upload_file(
            path_or_fileobj=results_json.encode(),
            path_in_repo=f"{prefix}/results.json",
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print(f"\nResults pushed to https://huggingface.co/datasets/{RESULTS_REPO}/blob/main/{prefix}/results.json")
    else:
        print("\nWARNING: HF_TOKEN not set, results not pushed to Hub")


if __name__ == "__main__":
    main()
