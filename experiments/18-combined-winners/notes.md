# Experiment 18: Winner + Novel Tweaks (2000-step validation)

## Hypothesis
Our 500-step sweeps found GEGLU activation beats relu² by 0.010 bpb, and different LR settings look promising. But 500-step results on simplified configs don't always transfer (curriculum learning vanished at 2000 steps). We need to validate on top of the actual winner recipe at 2000 steps.

## Key insight from checking the winner
The winner already uses many of our "sweep winners":
- muon_momentum=0.99 (matches our finding)
- WD=0.04 (we saw no effect at 500 steps, but winner uses it)
- MLP 3x (already there)

But the winner uses **matrix_lr=0.02** (lower than baseline 0.04), while our sweep said 0.08 was best. Short-run LR preferences don't transfer to longer training.

## Variants

### 18a: Winner + GEGLU activation
- Base: winner recipe (10L, MLP 3x, BigramHash, SWA, int5/int6, WD=0.04)
- Change: swap relu² MLP for GEGLU (hidden reduced to 2/3 to match param count)
- Rationale: GEGLU was our strongest novel finding (-0.010 bpb at 500 steps)

### 18b: Winner + LR tweaks
- Base: winner recipe
- Changes: matrix_lr=0.04 (2x winner's 0.02), tied_embed_lr=0.02 (vs winner's 0.03)
- Rationale: test if the winner's LR choices are optimal

## Setup
- 2000 iterations on 1x L4 GPU via HF Jobs
- TRAIN_BATCH_TOKENS=131072 (matching exp 21's L4-safe setting)
- Winner script from pinned commit `9f9d53343`

## How to run
```bash
python scripts/submit_exp18.py         # Submit both
python scripts/submit_exp18.py 18a     # Just GEGLU
python scripts/submit_exp18.py 18b     # Just HP tweaks
```

## Results

| Variant | val_bpb | vs exp 21 baseline | Notes |
|---------|---------|-------------------|-------|
| 18a (GEGLU) | — | — | |
| 18b (HP tweaks) | — | — | |
| exp 21 (winner repro) | — | — | reference |

## Analysis
TBD
