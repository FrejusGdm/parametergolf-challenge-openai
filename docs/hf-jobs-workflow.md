# Running Experiments on Hugging Face Jobs

## Why HF Jobs?

Local M1 MacBook is too slow for real experiments. GCP requires GPU quota approval. HF Jobs gives instant access to GPUs (L4, A10G, A100) with a Pro account.

## How It Works

1. Write a **self-contained Python script** with PEP 723 inline dependencies
2. Submit via the `huggingface_hub` Python API
3. Job runs on HF cloud GPU, results pushed to HF Hub dataset repo
4. Monitor logs and fetch results from the Hub

## Quick Start

### 1. Script Format

Your script must be self-contained — no local imports. Use PEP 723 for dependencies:

```python
# /// script
# dependencies = ["numpy", "torch", "huggingface-hub", "sentencepiece"]
# ///

import os
import torch
from huggingface_hub import HfApi

# Your code here...

# Push results to Hub
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=results_json.encode(),
    path_in_repo="results.json",
    repo_id="YourUsername/your-repo",
    repo_type="dataset",
    token=token,
)
```

### 2. Submit Job

```python
from huggingface_hub import HfApi, get_token

api = HfApi()
job = api.run_uv_job(
    "path/to/your_script.py",   # Local file path — gets uploaded
    flavor="l4x1",              # GPU type (see flavors below)
    timeout="3h",               # Max runtime
    secrets={"HF_TOKEN": get_token()},  # For Hub push
)
print(f"Job ID: {job.id}")
print(f"Status: {job.status.stage}")
```

### 3. Monitor

```python
# Check status
j = api.inspect_job(job_id="YOUR_JOB_ID")
print(f"Status: {j.status.stage}")

# View logs
for line in api.fetch_job_logs(job_id="YOUR_JOB_ID"):
    print(line)
```

### 4. Fetch Results

Results are at `https://huggingface.co/datasets/YourUsername/your-repo`

## GPU Flavors

| Flavor | GPU | VRAM | Cost/hr | Good for |
|--------|-----|------|---------|----------|
| `l4x1` | L4 | 24GB | ~$0.50 | Our experiments |
| `a10g-small` | A10G | 24GB | ~$1.50 | Larger models |
| `a10g-large` | A10G | 24GB | ~$3.00 | More CPU/RAM |
| `a100-large` | A100 | 80GB | ~$5.00 | Big models |
| `t4-small` | T4 | 16GB | ~$0.30 | Quick tests |

## Key Gotchas

1. **Scripts must be self-contained** — no imports from your local project
2. **Environment is ephemeral** — push results to Hub or they're lost
3. **Pass file path, not inline code** — `api.run_uv_job("script.py", ...)` works, inline string doesn't
4. **Use `get_token()` not `"$HF_TOKEN"`** — the string placeholder only works with MCP tools
5. **Set timeout** — default is 30min, which may be too short
6. **Clone repos at runtime** — if you need external code, `git clone` inside the script

## Our Setup

- HF Account: `JosueG` (Pro)
- Results repo: `JosueG/parameter-golf-curriculum`
- Preferred flavor: `l4x1` (good balance of speed and cost)
