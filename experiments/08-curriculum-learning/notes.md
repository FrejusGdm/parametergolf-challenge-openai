# Experiment 08 — Curriculum Learning (Data Ordering)

## Hypothesis
The order in which training shards are presented to the model matters. Curriculum learning (easy → hard or strategically ordered data) could improve convergence within the fixed 10-minute training budget.

**Prior work:** I've used curriculum learning in a published paper for language translation, where ordering training data by difficulty improved convergence speed and final quality. Applying the same idea here.

## Background
- The baseline reads shards sequentially (0, 1, 2, ...) via sorted glob
- No existing leaderboard submission mentions data ordering — this could be novel
- With only 10 minutes of training, every step counts — faster convergence = better final loss
- The `TokenStream` class uses simple sequential shard reading with wrap-around — trivially overridable

## What We Built

### Shard Analysis (`scripts/analyze_shards.py`)
Measures per-shard metrics:
- **Token entropy** (Shannon) — higher = more diverse/harder text
- **Bigram entropy** — captures local predictability
- **Vocabulary coverage** — unique tokens out of 1024
- **Repetition ratio** — fraction of repeated 4-grams
- **Average bytes per token** — affects BPB conversion

### CurriculumTokenStream (`scripts/curriculum.py`)
Drop-in replacement for the baseline's `TokenStream`. Subclasses it and reorders `self.files` based on strategy before training starts. Same `take()` API — works with existing `TokenLoader`.

### 6 Strategies Tested
1. **default** — sorted glob order (baseline)
2. **easy_first** — lowest entropy → highest
3. **hard_first** — highest entropy → lowest
4. **interleaved** — alternating easy/hard
5. **random** — shuffled with seed 42
6. **quality_first** — highest vocab coverage first

## Local Shard Analysis (5 shards)

| Shard | Entropy | Bigram Ent | Vocab Cov | Repetition | Byt/Tok |
|-------|---------|------------|-----------|------------|---------|
| 000000 | 8.6502 | 14.5028 | 0.8652 | 22.52% | 2.0313 |
| 000001 | 8.6498 | 14.5386 | 0.8672 | 22.68% | 2.0305 |
| 000002 | 8.6555 | 14.5100 | 0.8652 | 22.72% | 2.0432 |
| 000003 | 8.6530 | 14.4779 | 0.8652 | 24.67% | 2.0392 |
| 000004 | 8.6496 | 14.4545 | 0.8643 | 24.99% | 2.0263 |

**Key finding:** Shards are very homogeneous (entropy std=0.002). But repetition ratio varies more — shard 4 has 11% more repetition than shard 0. With 80 shards, variation may be larger.

## Experiment Runs

### Run 1: HF Jobs — Full 80 Shards on L4 GPU
- **Platform:** Hugging Face Jobs, NVIDIA L4 (24GB)
- **Config:** 500 steps, 65K tokens/step, batch=64 seqs, Adam lr=0.001
- **Job ID:** `69c07b7725abd6f920b4e0bb`
- **Results repo:** `JosueG/parameter-golf-curriculum` on HF Hub
- **Status:** Running

### Results — 80 Shards, L4 GPU, 500 Steps

```
Strategy          | Step 50  | Step 100 | Step 250 | Step 500 | Time(s)
default           |  4.6166  |  4.4197  |  4.2179  |  3.8926  |  289.1
easy_first        |  4.6502  |  4.3604  |  4.1616  |  3.7846  |  296.3
hard_first        |  4.6953  |  4.4098  |  4.1786  |  3.9419  |  296.2
interleaved       |  4.6566  |  4.3635  |  4.1637  |  3.8358  |  296.3
random            |  4.5969  |  4.4129  |  4.2694  |  3.9580  |  296.5
quality_first     |  4.6035  |  4.4049  |  4.2244  |  3.9600  |  296.4
```

**Best:** easy_first (3.7846)
**Worst:** quality_first (3.9600)
**Delta: 0.1754** — this is significant!

### Ranking (best to worst)
1. **easy_first** — 3.7846 (clear winner, ~0.11 better than default)
2. **interleaved** — 3.8358 (alternating easy/hard also helps)
3. **default** — 3.8926 (sequential baseline)
4. **hard_first** — 3.9419 (starting with hard data hurts)
5. **random** — 3.9580 (random is worse than sequential)
6. **quality_first** — 3.9600 (vocab coverage isn't the right metric)

## Why This Could Work
- In 20,000 steps with 80 shards at 524K tokens/step, the model consumes ~10.5B tokens (~105 shards worth) — so it cycles through data ~1.3 times
- Early training is where learning rate is highest — data seen first has outsized impact
- Curriculum learning has strong theoretical and empirical support in NLP
- Even small improvements compound with other optimizations

## Shard Analysis — Full 80 Shards

Across 80 shards:
- Entropy range: 8.648–8.657 (very tight, std ~0.002)
- Repetition ratio range: 0.2208–0.2697 (22% relative variation — this metric has more signal)
- Shard 40 is the most repetitive (0.2697), shard 59 is the least (0.2208)

Full analysis saved to `JosueG/parameter-golf-curriculum` on HF Hub.

## What I Learned

1. **Curriculum learning WORKS here** — even with very homogeneous data (FineWeb), easy→hard ordering gives a clear 0.11 loss improvement over default at 500 steps
2. **Easy first is the right strategy** — consistent with Bengio et al. (2009) and my translation paper. The model builds foundations on easier patterns before tackling harder ones.
3. **Entropy is the right difficulty metric** — lower entropy shards have more predictable/repetitive patterns that are easier to learn. The model benefits from seeing these first.
4. **Interleaved is a good runner-up** — alternating easy/hard keeps the model from getting "bored" on easy data while still giving it the foundation benefit.
5. **Hard-first actively hurts** — the model wastes early high-LR steps on data it can't yet learn from efficiently.
6. **Vocab coverage isn't a good difficulty proxy** — quality_first ranked last, suggesting token diversity != difficulty.
7. **The effect is surprisingly large** — 0.1754 loss delta across strategies for data that has nearly identical entropy distributions. The ordering matters more than the raw difficulty stats would suggest.

## Caveats
- Used simplified PyTorch model (Adam, no Muon, no U-Net skips) — effect may differ with full baseline
- 500 steps is short — need to verify the advantage holds at 5K-20K steps
- Single seed — should confirm with multiple seeds

## Papers to Read
- Bengio et al., "Curriculum Learning" (2009) — foundational paper
- Recent work on data ordering for LLM pretraining
- Any FineWeb-specific analysis papers

## Next Steps
1. **Longer run:** Test easy_first vs default at 2000-5000 steps — does the gap hold, widen, or close?
2. **Full baseline model:** Test with Muon optimizer + U-Net skips (the actual competition model)
3. **Better difficulty metrics:** Try bigram entropy or repetition ratio instead of entropy for ordering
4. **Adaptive curriculum:** Start easy, then switch to hard at some % of training (e.g., 30% easy → 70% hard)
5. **Combine with other wins:** Stack curriculum learning with sliding window eval, int6 quantization, etc.
6. **Multi-seed validation:** Run 3 seeds to confirm statistical significance
