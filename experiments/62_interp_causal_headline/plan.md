# Experiment 62 — Headline interpretability (multi-seed) + causal-cleanliness eval

Camera-ready item B5 (Tier 1) + A4. **Blocked on exp61's saved 500M L18 checkpoints (ETA ~2026-06-19).**

## Motivation
The paper's headline auto-interp story currently fuses two mismatched runs under one citation:
exp40 (100M, aux-off, arm `no_C` = Magnitude-Bypass) supplies the "80.1/82.1%, 16,332 vs
3,529 alive, ~4.5x" numbers, while the `p=0.88` is from exp33 (50M, L27, 200 features,
rates 40.0/37.0). Neither is our recommended variant at the headline recipe. exp53 already
evaluated the correct variants at 500M (standard 25.0% / global-a 19.0% / per-feature 24.0%),
but only n=200 features, single seed, so it lacks CIs to *own* the claim. This experiment
produces one principled, multi-seed interpretability number on the recommended variants, and
the no-compute-blocked half of A4: a causal-cleanliness eval (spec here; Chat 1 runs).

## Design
**Tier 1 (62a, auto-interp re-run).** Reuse exp53's describe-then-predict harness verbatim;
override only: arms = {standard, adaptive_l2 (global-a), perfeature_l2 (per-feature+delta)};
N_FEATURES_PER_SAE 200 -> 1000; loop seeds {42, 123, 456} (matched to exp61 SAE-training
seeds); checkpoints from exp61 (`EXP62_CKPT_DIR`, one subdir per seed). Same model/layer/
budget (Qwen3-8B L18 500M d_sae=65,536 k=80), same judge (Sonnet-4-6), same threshold (>=50%).
Report mean +/- SD interp rate per arm, plus the frequency-stratified (low/med/high) breakdown
that exp53 surfaced a crossover on. Replaces exp33 and exp40 as the headline interp source.

**A4 (62b, causal-cleanliness, SPEC ONLY — Chat 1 runs on box-8).** See `a4_causal_spec.md`.
Steering + ablation on discovered (cosine-unique vs standard-matched) features; metric =
target-effect / collateral (extends exp55d steering parity + exp57b TPP collateral 0.305 vs
0.017). On the same exp61 checkpoints.

## Predictions
- **H1 (most likely):** per-feature interp rate matched to standard within +/- ~3% (CI overlap);
  3-seed SD ~1-3%. Confirms "per-feature quality matched at scale; advantage is discovery/volume."
- **H2:** frequency crossover replicates (per-feature wins low-freq, loses high-freq). Strengthens
  the "discovers rare-concept features" story.
- **H3 (would force a rewrite):** per-feature significantly *beats* standard (non-overlapping CIs).
  Then we can make a per-feature *quality* claim, not just discovery. Lower prior.
- **A4:** cosine features give cleaner causal effects (higher target/collateral) at high N,
  matching exp57b's TPP divergence; per-feature steering power stays matched (exp55d).

## Pass criteria
- 62a: completes 1000 feat x 3 seeds x 3 arms; produces mean+/-SD + freq breakdown; SD reported,
  not hidden. PASS regardless of direction (H1/H2/H3 all publishable; the point is calibrated CIs).
- A4: spec is runnable by Chat 1 without further design; metric + baselines pre-registered here.

## Context
Feeds B5 (headline interp table), B9 (causal down-payment), B8 (Chat 3 abstract number),
and the calibration recalibration (B1). Shares exp61 checkpoints with A1 — do not collide.
