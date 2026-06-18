# A4 — Causal-cleanliness eval (SPEC; Chat 1 runs on box-8)

Camera-ready item A4. **Chat 4 specs (this doc); Chat 1 builds the harness + runs on box-8
against exp61 checkpoints.** Division agreed with Silen 2026-06-16: no GPU collision, Chat 4
writes it up into the interpretability/B9 narrative once results land.

## Reviewer ask (A4)
> 762k: "Report a causal downstream task, not only sparse probing ... intervene on [discovered
> features] and measure whether cosine features give cleaner causal effects or cleaner feature
> ablations than standard features."
> NTVF / EZEE: validate the claimed downstream-interp benefit, not just probing proxy.

## What already exists (the down-payment, surfaced now in B9, no new compute)
- **exp55d** feature steering: per-feature behavioral power is MATCHED across arms
  (KL ratio 0.94-1.00x). So any A4 advantage must be about *which features exist / collateral*,
  NOT per-feature intervention strength.
- **exp57b** TPP collateral: at high N, standard unintended (collateral) damage explodes while
  cosine stays clean. Precision (intended/unintended) at N=500: standard 1.4x vs cosine 21.8x;
  unintended at N=500: standard 0.3046 vs cosine 0.0170. This IS the causal-cleanliness signal.

## A4 design (new, pre-registered)
Build on exp57b's TPP decomposition but tie it to *named, discovered* concept features so the
result is a concept-level causal claim, not an aggregate-N curve.

1. **Feature selection.** From the exp62a auto-interp run, take the top discovered features per
   arm that map to a labeled SAEBench concept category (language, code, etc.) at interp >=
   threshold. Match counts across arms. Cosine-unique (discovered) vs standard-matched is the
   key contrast (mirrors exp58b discovery decomposition).
2. **Intervention.** For each selected feature, ablate (x0.0) and amplify (x2.0, x5.0) its
   activation at L18 during a forward pass on held-out FineWeb text (reuse exp55d hooks).
3. **Metric = target-effect / collateral.** Decompose the output-distribution shift (KL) into:
   - *intended*: shift on tokens/contexts where the concept is present (probe-positive),
   - *unintended (collateral)*: shift on concept-absent contexts.
   Report precision ratio (intended/unintended) per arm, swept over number of features ablated
   (N = 2..500, matching exp57b), AND per-concept for a handful of named features (the
   qualitative case studies for the paper).
4. **Baselines / controls.** standard vs adaptive_l2 vs perfeature_l2 on the SAME exp61
   checkpoints (n=3 seeds if budget allows, else seed 42). Random-feature ablation as a floor.

## Predictions
- **A4-H1:** cosine precision ratio > standard, growing with N (replicates exp57b: standard
  collateral grows, cosine stays clean). This is the headline causal-cleanliness result.
- **A4-H2:** per-feature *single*-feature intended effect matched across arms (consistent with
  exp55d) — so the win is collateral/cleanliness, not raw steering power. Keeps the story honest.
- **A4-H3 (null / would temper the claim):** precision ratios overlap once features are
  concept-matched -> then A4 only confirms exp55d/57b, no new claim. Report honestly.

## Pass criteria
- Produces precision-ratio-vs-N curves per arm + >=3 named per-concept case studies.
- Cosine advantage (if any) reported with seed variance, not single-seed.
- Explicitly states whether it goes beyond exp57b (concept-level) or merely reproduces it.

## Inputs needed from Chat 1
- exp61 L18 500M checkpoints retained (standard, adaptive_l2, perfeature_l2; per seed).
- exp55d steering-hook code + exp57b TPP harness (both in public repo) as the starting point.

## Handoff
Results -> Chat 4 writes into the main-text causal paragraph (currently `04-experiments` l.92
steering sentence + B9) and the limitations "future work: circuit-level validation" hedge.
