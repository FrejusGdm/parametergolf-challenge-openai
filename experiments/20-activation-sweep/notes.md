# Experiment 20: Activation Function Sweep

## Hypothesis
The baseline uses ReLU² (`relu(x).square()`), inherited from modded-nanogpt. While ReLU² promotes sparsity, other activations may achieve better val_bpb under the 16MB constraint. Gated variants (SwiGLU, GEGLU) are SOTA in modern LLMs but add parameters — we test with adjusted hidden sizes to keep param count comparable.

## Setup
- Variable: MLP activation function
- 13 variants (10 non-gated, 3 gated)
- All other hyperparameters at baseline defaults
- 500 iterations on 1x L4 GPU via HF Jobs
- Gated variants use `hidden = int(2/3 * mlp_mult * dim)` to compensate for extra gate matrix

## Non-gated variants (same param count)

| Activation | Formula | val_bpb | Paper |
|-----------|---------|---------|-------|
| relu_sq (baseline) | `proj(relu(fc(x))²)` | — | So et al., 2021 (Primer) arXiv:2109.08668 |
| gelu | `proj(gelu(fc(x)))` | — | Hendrycks & Gimpel, 2016 arXiv:1606.08415 |
| silu (swish) | `proj(silu(fc(x)))` | — | Ramachandran et al., 2017 arXiv:1710.05941 |
| relu | `proj(relu(fc(x)))` | — | Nair & Hinton, 2010 |
| softmax | `proj(softmax(fc(x)))` | — | Bridle, 1990 |
| relu_cubed | `proj(relu(fc(x))³)` | — | So et al., 2021 |
| sigmoid | `proj(sigmoid(fc(x)))` | — | Classical |
| tanh | `proj(tanh(fc(x)))` | — | LeCun et al., 1998 |
| softplus | `proj(softplus(fc(x)))` | — | Dugas et al., 2001 |
| mish | `proj(x·tanh(softplus(fc(x))))` | — | Misra, 2019 arXiv:1908.08681 |

## Gated variants (extra gate matrix, reduced hidden dim)

| Activation | Formula | val_bpb | Paper |
|-----------|---------|---------|-------|
| swiglu | `proj(silu(W1·x) * W2·x)` | — | Shazeer, 2020 arXiv:2002.05202 |
| geglu | `proj(gelu(W1·x) * W2·x)` | — | Shazeer, 2020 |
| reglu | `proj(relu(W1·x) * W2·x)` | — | Shazeer, 2020 |

## How to run

```bash
# Submit all 13 activation experiments
python scripts/submit_activation_sweep.py

# Submit specific ones
python scripts/submit_activation_sweep.py --activations swiglu gelu silu

# Dry run
python scripts/submit_activation_sweep.py --dry-run

# Check results
python scripts/monitor_sweeps.py --results
```

## Analysis
TBD
