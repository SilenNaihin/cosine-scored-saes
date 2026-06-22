# Experiment 63: Does the cosine advantage survive other selectors?

Tests whether the cosine-scoring advantage is specific to BatchTopK's batch-wide
competition or reflects inner-product scoring more generally. The cosine score
modifies only the encoder pre-activation, not the sparsity mechanism, so it
composes with any selector. We re-run the headline comparison under two
additional per-token selectors at the headline setting (Qwen3-8B L18, 50M
FineWeb tokens, d_sae=65536, matched L0=80, saprmarks recipe + aux-k).

Selectors:
- **BatchTopK** (baseline; reused from exp43d / exp59): single batch-wide budget.
- **per-token TopK** (Gao et al. 2024): k largest activations per token.
- **AbsTopK** (Zhu et al. 2025): k largest by absolute value per token, sign preserved.

Encoders: inner-product (standard) vs cosine (global-a and per-feature+delta).

## Results

Sparse-probing top-1 at matched L0=80 (SAEBench, 8 datasets):

| Selector | Encoder | FVE | Dead% | Probe top-1 |
|---|---|---|---|---|
| BatchTopK | inner-product | 0.723 | 0.0 | 0.530 |
| BatchTopK | cosine (per-feat) | 0.726 | 0.0 | **0.648** |
| per-token TopK | inner-product | 0.641 | 91.7 | 0.731 |
| per-token TopK | + unit-enc (||w||-free) | 0.680 | 80.5 | 0.645 |
| per-token TopK | cosine (global) | 0.711 | 6.1 | **0.802** |
| per-token TopK | cosine (per-feat) | 0.713 | 6.6 | 0.780 |
| AbsTopK | inner-product | 0.636 | 92.4 | 0.690 |
| AbsTopK | + unit-enc (||w||-free) | 0.670 | 0.0 | 0.668 |
| AbsTopK | cosine (global) | 0.691 | 0.0 | **0.812** |
| AbsTopK | cosine (per-feat) | 0.692 | 6.2 | **0.827** |

(BatchTopK rows: FVE/dead from exp43d, probing top-1 from exp59.)

## Key findings

1. **The advantage is not BatchTopK-specific.** The cosine encoder beats
   inner-product on sparse-probing top-1 under all three selectors: +11.8 pts
   (BatchTopK), +5 to +7 (per-token TopK), +12 to +14 (AbsTopK). The effect is
   about inner-product scoring generally, not batch-wide competition.

2. **Batch-wide competition explains part, not all, of the gap.** The mechanism
   account predicts per-token selection should *shrink* the advantage (the
   per-token norm scalar cancels from the within-token ranking). It does shrink
   under per-token TopK (+11.8 → +5-7) but does not vanish, and under AbsTopK it
   is as large as BatchTopK — so a selector-independent component remains.

3. **The advantage decomposes into two magnitude channels** (the `unit-enc`
   arms = inner-product with L2-normalized encoder rows, removing ||w_i|| but
   keeping ||x||; under per-token selection ||x|| cancels from the ranking):
   - **Weight norm ||w_i|| drives feature death.** Unit-normalizing encoder rows
     rescues features from the dead state — decisively under AbsTopK (92.4% →
     0.0% dead), partially under per-token TopK (91.7% → 80.5%). A few large-norm
     rows otherwise win slots on most tokens and starve the rest.
   - **Input norm ||x|| drives probing quality.** Rescuing the features does NOT
     make them interpretable: unit-enc probing stays at the inner-product level
     (0.645, 0.668), far below cosine (0.802, 0.827). Only the cosine score,
     which also strips ||x||, recovers probing.
   Cosine is the only encoder that removes both, so it alone wins on survival
   *and* probing.

## JumpReLU / Gated (not completed — penalty selectors don't reach matched L0)

`exp63b_jumprelu_gated.py` implements JumpReLU (straight-through threshold) and
Gated selectors with the same pluggable cosine scorer. These are penalty-trained
(reach target L0 via a sparsity coefficient) rather than fixed-k. At the 50M-token
mechanism-sweep budget, the learned threshold initializes correctly at L0≈80 but
reconstruction pressure pushes it back up within a few hundred steps and the
straight-through penalty cannot hold it: across lambda spanning >30x AND at two
dictionary sizes (16384 and 65536), the converged L0 stayed in the thousands, not
near 80. Reaching matched sparsity needs the much longer schedules of the original
work; left to future work. Code kept for reproducibility / future runs. Set
`EXP60B_DSAE=16384` to run the smaller-dictionary sweep.

## Files
- `exp63_selectors.py` — TopK + AbsTopK x {inner, cos_global, cos_perfeature}; train + SAEBench probing.
- `exp63b_jumprelu_gated.py` — JumpReLU + Gated (penalty-trained; not converged at this budget).
- `results.json` — full metrics.
