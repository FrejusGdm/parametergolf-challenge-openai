# Experiment 02 — zstd Compression

## Hypothesis
Switching from zlib-9 to zstd-22 for model artifact compression will reduce
artifact size without any change to model quality, freeing up bytes within
the 16MB artifact budget for a larger model.

## Background
The baseline serialization pipeline quantizes the model to int8 and compresses
with zlib level 9. Zstandard (zstd) at high compression levels (e.g., 22) is
known to achieve better compression ratios than zlib on structured data like
neural network weights, at the cost of slower compression (decompression remains
fast). Since compression happens once at training time and decompression happens
once at eval time, this tradeoff is free in practice.

## Configuration
- **Script:** `scripts/hf_exp02_zstd_job.py`
- **Hardware:** L4 x1 via HF Jobs
- **Training:** 2000 steps, baseline config (no model changes)
- **Comparison:** After training, the int8 artifact is decompressed from zlib
  and recompressed with zstd-22. Sizes are compared directly.

## What This Tests
- Pure serialization change: same model, same weights, different compressor
- val_bpb should be identical between zlib and zstd (same model)
- Only artifact_bytes should differ

## Expected Outcome
- zstd-22 should save 5-15% on artifact size vs zlib-9
- This translates to ~500KB-1.5MB freed from the 16MB budget
- No impact on training time or model quality

## Results
*Pending — run with `scripts/hf_exp02_zstd_job.py`*
