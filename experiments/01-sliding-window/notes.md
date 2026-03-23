# Experiment 01 — Sliding Window Evaluation

## Hypothesis
Evaluating with overlapping sliding windows instead of non-overlapping chunks will improve val_bpb by ~0.032-0.034, for free (no model changes, no artifact cost).

## How It Works

**The problem:** Baseline eval splits text into non-overlapping 1024-token chunks. The first token of each chunk has zero context — the model is guessing blind. On average, each token gets ~512 tokens of context.

**The fix:** Slide a 1024-token window forward by just 64 tokens at a time. Only "grade" the model on the last 64 tokens of each window, where every token has 960+ tokens of context.

```
Baseline:   [........GRADE ALL........] [........GRADE ALL........]
             ^ 0 context                 ^ 0 context

Sliding:    [..............GRADE last 64]
               [..............GRADE last 64]
                  [..............GRADE last 64]
                     ^ always 960+ context
```

**Cost:** ~16x more forward passes (eval takes ~70-90s instead of ~16s)
**Gain:** ~0.034 bpb improvement
**Budget impact:** Zero — same model, same artifact size

## Reference Implementations
Two submissions in `parameter-golf/records/` already use this:
- `2026-03-19_SlidingWindowEval/train_gpt.py` (lines 837-931) — pure eval improvement
- `2026-03-19_MixedQuant_Int6Int8_SlidingWindow/train_gpt.py` (lines 291-383)

## What We Changed
- Added `eval_val_sliding()` function: overlapping windows with configurable stride
- Model needs to return logits (not just loss) for per-token scoring
- `EVAL_STRIDE` env var controls stride (0 = disabled, 64 = recommended)

## Results

*(Running on HF Jobs — will update)*

| Method | val_loss | val_bpb | Eval Time |
|--------|---------|---------|-----------|
| Non-overlapping (baseline) | — | — | — |
| Sliding window (stride=64) | — | — | — |
| **Delta** | — | — | — |

## Key Takeaways
- This is the easiest win in the challenge — change eval, not training
- Every top submission uses it
- Should be the first thing added to any competitive submission
