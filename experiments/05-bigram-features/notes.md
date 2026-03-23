# Experiment 05 — BigramHash + SmearGate

## Hypothesis
Adding explicit bigram features and a simple token-blending gate will improve
val_bpb by giving the model cheap access to local context information that would
otherwise require attention layers to learn from scratch.

## Why It Matters
Small language models with limited depth struggle to capture local token
co-occurrence patterns. Top leaderboard submissions use BigramHash and SmearGate
to inject this information directly:

- **BigramHash** provides explicit bigram frequency features via hash-based embeddings.
  Instead of expanding the vocabulary (which is costly in embedding parameters),
  it hashes (prev_token, cur_token) pairs into a fixed number of buckets and learns
  embeddings for each bucket. This is especially powerful with the small 1024-token
  BPE vocabulary where many bigram patterns are predictable.

- **SmearGate** blends each token's representation with its predecessor using a
  learned per-dimension gate. This provides a cheap "look-back-one" signal before
  the first attention layer, helping the model start with richer local context.

Both modules add minimal parameters and compute overhead while potentially
providing significant quality gains.

## Technique Details

### BigramHash
- Hash function: `(prev_token * 31 + cur_token) % n_buckets`
- Default: 4096 buckets, 128-dim intermediate embeddings
- Projected to model_dim (512) via a linear layer initialized to zeros
- Added to token embeddings before RMS norm
- Added parameters: `4096 * 128 + 128 * 512 = 524,288 + 65,536 = 589,824`

### SmearGate
- Learned per-dimension gate: `sigmoid(gate_param)` where gate_param is initialized to 0
- Blends: `x = (1 - gate) * x + gate * prev_x`
- Applied after RMS norm of embeddings, before transformer blocks
- Added parameters: 512 (one per model dimension)

### Total Parameter Overhead
~590K additional parameters (~10% of baseline model size)

## Expected Results
- Pre-quant val_bpb: improved (bigram features help convergence)
- Post-quant val_bpb: improved (extra parameters are small, quantize well)
- Throughput: minimal slowdown (< 5%), since both modules are cheap
- Artifact size: slightly larger due to extra parameters, but still well under 16MB

## Configuration
- **HF Job script:** `scripts/hf_exp05_bigram_job.py`
- **Hardware:** HF Jobs `l4x1`
- **Default:** 2000 iterations, 20 train shards
- **Bigram config:** 4096 buckets, 128-dim (configurable via env vars)
- **Two runs:** baseline (A) vs bigram+smeargate (B)

## How to Run
```bash
python scripts/submit_all_exp02_07.py --exp 05
# or with custom config:
# BIGRAM_BUCKETS=8192 BIGRAM_DIM=64 HF_TOKEN=... uv run scripts/hf_exp05_bigram_job.py
```

## Results

### Baseline
- **Pre-quant:** `val_loss=...`, `val_bpb=...`
- **Post-quant:** `val_loss=...`, `val_bpb=...`
- **Artifact size:** `... bytes`
- **Step time:** `... ms`

### BigramHash + SmearGate
- **Pre-quant:** `val_loss=...`, `val_bpb=...`
- **Post-quant:** `val_loss=...`, `val_bpb=...`
- **Artifact size:** `... bytes`
- **Step time:** `... ms`

### A/B Verdict
- Post-quant val_bpb delta: `...`
- Pre-quant val_bpb delta: `...`
- Throughput impact: `...`
- Artifact size ratio: `...`
- Hypothesis verdict: `pending`

## Follow-up Ideas
- Sweep n_buckets: 2048, 4096, 8192, 10240 (the winner uses 10240)
- Sweep bigram_dim: 64, 128, 256
- Try trigram hashing (prev2 * 31^2 + prev1 * 31 + cur)
- Try separate gate per layer instead of a single pre-attention gate

## References
- BigramHash used in top leaderboard submissions
- SmearGate: simple token-blending technique from competitive small LM training
- Baseline code: [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)
