# Experiment 62: Headline interpretability (multi-seed) + causal-cleanliness eval

Camera-ready items **B5 (Tier 1)** and **A4**. **STATUS: ready to run (2026-06-22).** The 500M
L18 headline checkpoints are published at
[Silen/cosine-scored-saes-qwen3-8b](https://huggingface.co/Silen/cosine-scored-saes-qwen3-8b)
and verified to load into exp53's SAE classes (0 missing/unexpected keys, step=244140).
`exp62a` defaults to `--source hf` (single published seed); `--source box8` runs the 3-seed
version off exp61's per-seed checkpoints.

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
_62a RUNNING (2026-06-22): launched on h100-dev-box-4 GPU1, `--source hf` single-seed,
tmux `exp62a_collect`, work area `/mnt/work/cosine-scored-saes` (root disk full; everything
on /mnt). Phase 1 (collect) in progress: Qwen3-8B loaded, streaming FineWeb (skip 200k docs).
Then Phase 2 `--score` (needs AWS Bedrock creds) and Phase 3 `--aggregate`._

_Outputs: `exp62a_results_{hf,box8_seed*}.json`, aggregate `exp62a_results.json`;
A4 results per `a4_causal_spec.md`._
