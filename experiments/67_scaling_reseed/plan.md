# Experiment 67 — Scaling-matrix reseed (ground the d_model claim)

## Motivation
Camera-ready B6/A3. The paper claims (§4.5, Appendix `app:cross-model`) that the cosine
advantage **grows with model dimension and is flat across expansion ratio**. Verification
against the public repo found the supporting exp57 matrix is **single-seed (SEED=42), has no
raw results file for 4 of 9 cells, and the 8B/16x cell is hand-eyeballed (`~+14`)**. The
"flat across expansion" half is robust within rows; the "grows with d_model" half rests on
n=1 point estimates and one fake cell. exp67 grounds it on one free A100 (a100-backup-1).

## Design
Reuses exp57's exact recipe (Saprmarks, 50M tokens, k=80, BatchTopK, aux-on) by importing
the module; only the seed and the on-disk paths change. Two SAEs per cell (standard +
adaptive_l2 cosine). Hook layers match exp57 (1.7B=L14, 4B=L18, 8B=L18) so results stay
comparable to the existing matrix and the L18 headline.

| Block | Cells (model/expansion) | Seeds | Why |
|-------|-------------------------|-------|-----|
| A. Complete 8B row (single seed) | 8b/4x, 8b/8x, 8b/16x | 42 | 8B row had no real data; replaces the eyeballed 8b/16x `~+14` |
| B. d_model trend error bars | 1.7b/8x, 4b/8x, 8b/8x | 123, 456 | seeds on the 8x column; combine with existing seed-42 (1.7b/4b) + block-A 8b/8x for n=3 |

New training cells: 3 (block A) + 6 (block B) = **9 cells x 2 SAEs = 18 SAEs**. Ordered
cheap->expensive (1.7B, 4B, then 8B) so the trend signal lands before the expensive tail.
8B dominates wall-clock; if A100 time runs short, blocks A + the 1.7B/4B half of B alone
already kill the two worst caveats (no error bars at the cheap end + the eyeballed cell).

## Predictions
- **H1 (claim holds):** seeded row means reproduce the trend (1.7B < 4B, with 8B near 4B),
  seed SD small (exp34 precedent: SD < 0.001 FVE; probing SD a few tenths of a pp). The real
  8b/16x lands near the eyeballed +14. -> promote `fig:scaling-matrix` with error bars.
- **H2 (trend weaker than claimed):** real 8B cells come in lower, or seed SD on the probing
  gap is large enough (> ~2pp) that 1.7B vs 4B/8B is not separable -> soften to "advantage
  present at all sizes; magnitude not cleanly ordered by d_model"; do NOT promote a clean
  monotone story.
- **H3 (eyeballed cell wrong):** real 8b/16x differs materially from +14 -> correct the matrix
  number regardless of promotion.

## Pass criteria
Trend is "grounded" if: (a) all 9 cells have real SAEBench numbers (no eyeballed values), and
(b) the seeded 8x column gives mean +/- SD where the across-model ordering is either confirmed
(SD small enough to separate sizes) or explicitly reported as not separable. Either outcome is
publishable; the criterion is *real numbers with error bars*, not a particular direction.

## Context
Owner: Chat 3 (transferred from Chat 2/A3). Feeds the §4.5 wording (Option A already landed)
and the deferred `fig:scaling-matrix` promotion (B6). Does not block B1/B2/B8 (those wait on
A1/exp61). Hook-layer note: 8B sits at a shallower *relative* depth than 1.7B/4B (L18 of more
layers); exp67 keeps exp57's layers for comparability and flags depth as an uncontrolled
covariate in analysis rather than reseeding at matched relative depth (out of minimum scope).
