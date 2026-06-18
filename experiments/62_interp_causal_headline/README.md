# Experiment 62: Headline interpretability (multi-seed) + causal-cleanliness eval

Camera-ready items **B5 (Tier 1)** and **A4**. **STATUS: planned, BLOCKED on exp61's saved
500M L18 checkpoints (A1 multi-seed, box-8, ETA ~2026-06-19).**

Two parts:
- **62a (auto-interp, Chat 4 runs):** re-run exp53's describe-then-predict at 1000 features/arm
  across SAE-training seeds {42, 123, 456} on the recommended variants only (standard,
  adaptive_l2 = global-a, perfeature_l2 = per-feature + delta) at the headline recipe
  (Qwen3-8B L18, 500M, d_sae=65,536, k=80). Replaces exp33 (50M/L27/200) and exp40
  (100M/no_C/>=4) as the headline interpretability source, with mean +/- SD CIs.
  Thin wrapper over exp53 (`exp62a_interp_multiseed.py`); no logic duplicated.
- **62b / A4 (causal-cleanliness, SPEC only — Chat 1 runs):** see `a4_causal_spec.md`.
  Steering + ablation on discovered cosine vs standard features; metric = intended/collateral,
  swept over N. Builds on exp55d (steering parity) + exp57b (TPP collateral 0.305 vs 0.017).

See `plan.md` for motivation, predictions, and pass criteria.

## Why this exists
The paper's headline auto-interp numbers fuse two mismatched runs: exp40 (100M, aux-off, arm
`no_C` = Magnitude-Bypass) supplies "80.1/82.1%, 16,332 vs 3,529 alive, ~4.5x", while `p=0.88`
is exp33's (50M/L27/200, rates 40.0/37.0). Neither is our recommended variant at the headline
recipe. exp53 has the right variants at 500M but only n=200, single seed. This produces one
principled, multi-seed number to own the claim.

## Results
_Pending checkpoints (exp61). Outputs: `exp62a_results_seed{42,123,456}.json`,
aggregate `exp62a_results.json`; A4 results per `a4_causal_spec.md`._
