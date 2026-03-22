# Experiment 08 — Curriculum Learning (Data Ordering)

## Hypothesis
The order in which training shards are presented to the model matters. Curriculum learning (easy → hard or strategically ordered data) could improve convergence within the fixed 10-minute training budget.

**Prior work:** Josue has published work on curriculum learning for language translation, where ordering training data by difficulty improved convergence speed and final quality.

## Background
- The baseline reads shards sequentially (0, 1, 2, ...)
- No existing leaderboard submission mentions data ordering — this could be novel
- With only 10 minutes of training, every step counts — faster convergence = better final loss

## Approach

### Phase 1: Shard Characterization
Score each shard on:
- **Token entropy** — higher entropy = more diverse/harder text
- **Perplexity** (using trained model) — direct measure of difficulty
- **Vocabulary coverage** — how many unique tokens used
- **Text quality signals** — average token length, repetition ratio

### Phase 2: Ordering Experiments
Test orderings with short runs (500 steps each):
1. Default (sequential: 0, 1, 2, ...)
2. Easy → Hard (lowest entropy/perplexity first)
3. Hard → Easy (highest entropy first)
4. Interleaved (alternating easy/hard)
5. Random shuffle
6. Quality-first (highest quality score first)

### Phase 3: Analysis
- Compare training loss curves across orderings
- Statistical significance test on final loss
- If ordering matters, design an optimal curriculum for the full 80-shard run

## Why This Could Work
- In 20,000 steps with 80 shards, the model sees each shard ~2.5 times
- Early training is where learning rate is highest — data seen first has outsized impact
- Curriculum learning has strong theoretical and empirical support in NLP

## Potential Challenges
- Effect might be too small to measure in 500-step runs
- Shard ordering might not matter much when all shards are from the same distribution (FineWeb)
- The sequential order might already be accidentally good/bad

## Papers to Read
- Bengio et al., "Curriculum Learning" (2009) — foundational paper
- Recent work on data ordering for LLM pretraining
- Any FineWeb-specific analysis papers

## Results
TBD

## Status: Pending (after baseline experiments)
