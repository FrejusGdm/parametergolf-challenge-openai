# /// script
# dependencies = [
#     "numpy",
#     "torch",
#     "huggingface-hub",
#     "sentencepiece",
#     "datasets",
#     "tqdm",
#     "setuptools",
#     "typing-extensions==4.15.0",
#     "kernels",
#     "tiktoken",
# ]
# ///
"""
Activation function sweep for Parameter Golf.
Clones parameter-golf, patches the MLP activation, trains 500 steps, pushes results to Hub.

Environment variables:
  ACTIVATION  — which activation to use (relu_sq, gelu, silu, relu, softmax, relu_cubed,
                sigmoid, tanh, softplus, mish, swiglu, geglu, reglu)
  HF_TOKEN    — for pushing results
  ITERATIONS  — training steps (default 500)
"""

import json
import os
import re
import subprocess
import sys
import time

ACTIVATION = os.environ.get("ACTIVATION", "relu_sq")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "JosueG/parameter-golf-sweeps")
ITERATIONS = os.environ.get("ITERATIONS", "500")

# ── MLP patches ────────────────────────────────────────────────────────────

# Non-gated: replace just the forward method
NON_GATED_FORWARDS = {
    "relu_sq": """    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())""",

    "gelu": """    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.nn.functional.gelu(self.fc(x)))""",

    "silu": """    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.nn.functional.silu(self.fc(x)))""",

    "relu": """    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)))""",

    "softmax": """    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.softmax(self.fc(x), dim=-1))""",

    "relu_cubed": """    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x * x * x)""",

    "sigmoid": """    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.sigmoid(self.fc(x)))""",

    "tanh": """    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.tanh(self.fc(x)))""",

    "softplus": """    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.nn.functional.softplus(self.fc(x)))""",

    "mish": """    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        return self.proj(x * torch.tanh(torch.nn.functional.softplus(x)))""",
}

# Gated: replace both __init__ and forward
GATED_ACTIVATIONS = {
    "swiglu": {
        "gate_act": "torch.nn.functional.silu",
    },
    "geglu": {
        "gate_act": "torch.nn.functional.gelu",
    },
    "reglu": {
        "gate_act": "torch.relu",
    },
}

GATED_MLP_TEMPLATE = """class MLP(nn.Module):
    # {name} gated MLP (Shazeer, 2020 arXiv:2002.05202)
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        # Reduce hidden dim by 2/3 to compensate for extra gate matrix
        hidden = int(2 * mlp_mult * dim / 3)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj({gate_act}(self.gate(x)) * self.fc(x))"""


def patch_train_gpt(filepath, activation):
    """Patch the MLP class in train_gpt.py with the specified activation."""
    with open(filepath, "r") as f:
        code = f.read()

    if activation in NON_GATED_FORWARDS:
        # Replace just the forward method
        old_forward = """    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())"""
        new_forward = NON_GATED_FORWARDS[activation]
        code = code.replace(old_forward, new_forward)

    elif activation in GATED_ACTIVATIONS:
        # Replace the entire MLP class
        old_mlp = """class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())"""

        info = GATED_ACTIVATIONS[activation]
        new_mlp = GATED_MLP_TEMPLATE.format(
            name=activation.upper(),
            gate_act=info["gate_act"],
        )
        code = code.replace(old_mlp, new_mlp)
    else:
        print(f"ERROR: Unknown activation '{activation}'")
        sys.exit(1)

    with open(filepath, "w") as f:
        f.write(code)

    print(f"Patched MLP with activation: {activation}")


# ── Main ───────────────────────────────────────────────────────────────────

print(f"=== Activation Sweep: {ACTIVATION} ===")

# Clone repo
subprocess.run(
    ["git", "clone", "https://github.com/openai/parameter-golf.git"],
    check=True,
)
os.chdir("parameter-golf")

# Download dataset (1 shard)
print("\n=== Downloading dataset (1 shard) ===")
subprocess.run(
    [sys.executable, "data/cached_challenge_fineweb.py", "--variant", "sp1024", "--train-shards", "1"],
    check=True,
)

# Patch the MLP
patch_train_gpt("train_gpt.py", ACTIVATION)

# Build training env
train_env = os.environ.copy()
train_env.update({
    "ITERATIONS": ITERATIONS,
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_LOSS_EVERY": "0",
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "RUN_ID": f"activation_{ACTIVATION}",
    # L4 has 24GB VRAM — grad_accum_steps is hardcoded to 8 in train_gpt.py,
    # so reduce TRAIN_BATCH_TOKENS so each microbatch (tokens/8) fits in 24GB.
    "TRAIN_BATCH_TOKENS": "65536",
    "VAL_BATCH_SIZE": "65536",
})

# Train
print(f"\n=== Training ({ITERATIONS} iters, activation={ACTIVATION}) ===")
start_time = time.time()

result = subprocess.run(
    ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
    env=train_env,
    capture_output=True,
    text=True,
)

elapsed = time.time() - start_time
output = result.stdout + "\n" + result.stderr
print(output)

if result.returncode != 0:
    print(f"\n=== Training FAILED (exit code {result.returncode}) ===")
    sys.exit(1)

# Parse results
val_bpb_match = re.findall(r"val_bpb[=: ]+([0-9.]+)", output)
val_loss_match = re.findall(r"val_loss[=: ]+([0-9.]+)", output)

val_bpb = val_bpb_match[-1] if val_bpb_match else "N/A"
val_loss = val_loss_match[-1] if val_loss_match else "N/A"

print(f"\n=== Result: activation={ACTIVATION} val_bpb={val_bpb} val_loss={val_loss} ===")
print(f"=== Elapsed: {elapsed:.1f}s ===")

# Push to Hub
results = {
    "experiment": f"20-activation-{ACTIVATION}",
    "activation": ACTIVATION,
    "gated": ACTIVATION in GATED_ACTIVATIONS,
    "iterations": int(ITERATIONS),
    "val_bpb": val_bpb,
    "val_loss": val_loss,
    "elapsed_seconds": round(elapsed, 1),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

results_json = json.dumps(results, indent=2)
print(f"\n{results_json}")

if HF_TOKEN:
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    try:
        api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Repo creation note: {e}")

    filename = f"20-activation-{ACTIVATION}.json"
    api.upload_file(
        path_or_fileobj=results_json.encode(),
        path_in_repo=f"sweeps/{filename}",
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"\nResults pushed to https://huggingface.co/datasets/{RESULTS_REPO}/blob/main/sweeps/{filename}")
else:
    print("\nNo HF_TOKEN — skipping Hub push")
