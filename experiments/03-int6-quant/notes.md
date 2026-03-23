# Experiment 03 — Int6 Quantization + QAT

## Hypothesis
Training with simulated int6 quantization noise (Straight-Through Estimator) will
produce a model that degrades less when quantized to int6 at export time, yielding
a smaller artifact with comparable or better post-quantization val_bpb than the
standard int8 export.

## Why It Matters
The 16MB artifact limit is a hard constraint. The baseline uses int8 quantization
(range [-127, 127]) for export. Switching to int6 (range [-32, 31]) reduces the
effective bits per weight from 8 to 6, shrinking the compressed artifact. However,
naive int6 quantization after training at full precision causes larger degradation
than int8. QAT bridges this gap: by injecting quantization noise during training,
the model learns weight distributions that are more robust to the reduced precision.

The STE trick is standard in quantization-aware training: the forward pass uses
the quantized weights, but gradients flow through the original (unquantized) weights
so the optimizer can still make smooth updates.

## Technique Details
- **Fake-quantize function:** Per-row abs-max scaling to [-32, 31], with STE
- **Applied in:** `CastedLinear.forward()` during training only (2D weight matrices)
- **Export change:** `quantize_float_tensor()` uses int6 range [-32, 31] instead of [-127, 127]
- **Everything else:** Unchanged from baseline

## Expected Results
- Pre-quant val_bpb: slightly worse than baseline (noise hurts convergence a bit)
- Post-quant val_bpb: better than baseline int8 (model is robust to quantization)
- Artifact size: ~25% smaller (6 bits vs 8 bits per weight, plus better zlib compression)
- The key metric is the gap between pre-quant and post-quant val_bpb

## Configuration
- **HF Job script:** `scripts/hf_exp03_int6_job.py`
- **Hardware:** HF Jobs `l4x1`
- **Default:** 2000 iterations, 20 train shards
- **Two runs:** baseline (A) vs int6+QAT (B)

## How to Run
```bash
python scripts/submit_all_exp02_07.py --exp 03
# or directly:
# HF_TOKEN=... uv run scripts/hf_exp03_int6_job.py
```

## Results

### Baseline
- **Pre-quant:** `val_loss=...`, `val_bpb=...`
- **Post-quant (int8):** `val_loss=...`, `val_bpb=...`
- **Artifact size:** `... bytes`

### Int6 + QAT
- **Pre-quant:** `val_loss=...`, `val_bpb=...`
- **Post-quant (int6):** `val_loss=...`, `val_bpb=...`
- **Artifact size:** `... bytes`

### A/B Verdict
- Post-quant delta: `...`
- Artifact size ratio: `...`
- Hypothesis verdict: `pending`

## References
- STE for quantization: Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons" (2013)
- QAT overview: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
- Baseline code: [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)
