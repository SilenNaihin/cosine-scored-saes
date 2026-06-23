# Experiment 62a — Headline auto-interpretability (1000 features, single seed)

## TL;DR
Describe-then-predict auto-interp at **1000 features/arm** on the published 500M L18 headline
checkpoints (Qwen3-8B, d_sae=65,536, k=80), Sonnet-4-6 judge, interpretable iff >=50%
held-out prediction accuracy. Single seed (HF publishes one checkpoint per variant).

| Arm | Interp rate | Low-freq | Med-freq | High-freq | Alive | Mean acc |
|-----|:-:|:-:|:-:|:-:|:-:|:-:|
| Standard | **20.1%** | 22.3% | 21.6% | 14.8% | 17,320 | 0.270 |
| Global `a` | **21.3%** | 24.3% | 21.4% | 18.0% | 17,814 | 0.288 |
| Per-feature | **19.2%** | 23.5% | 19.5% | 14.3% | 18,655 | 0.273 |

All three arms fall in a **2.1-point band (19.2–21.3%)**. Alive counts are matched (17.3k–18.7k).

## Key findings

1. **Per-feature interpretability is matched at scale — confirmed at 5x the feature count.**
   The single-seed n=200 exp53 result (standard 25.0 / global 19.0 / per-feature 24.0) is
   reproduced in shape at n=1000: no arm has a per-feature *quality* advantage; the spread is
   within noise. This is the honest headline: **the cosine advantage is discovery/volume
   (more interpretable features alive at matched count), not higher per-feature
   interpretability.** Consistent with exp58b (87% of the probing gap is feature discovery).

2. **The exp53 frequency "crossover" does NOT replicate at n=1000 (negative result).**
   exp53 (n=200) reported per-feature winning low-frequency features (38.0% vs 25.5%) but
   losing high-frequency (14.0% vs 24.0%). At n=1000 that pattern dissolves:
   - Low-freq: per-feature 23.5% vs standard 22.3% (tied, not +12.5).
   - High-freq: per-feature 14.3% vs standard 14.8% (tied, not -10).
   The n=200 crossover was almost certainly small-sample noise (50 features/band).
   **Action: the frequency-crossover claim in the paper (06c `tab:interp-500m` discussion)
   must be softened or dropped — it does not survive the larger sample.**

3. **Global `a` is (weakly) the top arm here (21.3%).** Within noise of standard, but it
   undercuts any "per-feature is best on interpretability too" framing. Per-feature remains
   the best on the *headline* metric (sparse probing top-1, +13.7% multi-seed), not on
   auto-interp. Keep the two metrics distinct in the writeup.

## Verdict vs plan predictions
- **H1 (per-feature rate matched within ~3%): CONFIRMED.** 19.2 vs 20.1 (standard), well inside.
- **H2 (frequency crossover replicates): REFUTED.** Crossover vanishes at n=1000. Honest
  negative result; the paper sub-claim built on it needs revision.
- **H3 (per-feature significantly beats standard): REFUTED.** No quality win for any arm.

## What this confirms (and what it doesn't)
- **Confirms:** at the true headline recipe, per-feature auto-interp quality is matched; the
  cosine story is correctly framed as discovery/volume + probing, not per-feature legibility.
  Replaces the mislabeled exp40 (100M/no_C/>=4) and exp33 (50M/L27/200) numbers in the paper
  with one principled run at the right recipe and 5x the features.
- **Does NOT establish:** (a) error bars — single seed, since HF publishes one checkpoint per
  variant. A 3-seed version (`--source box8` on exp61's per-seed checkpoints) would add CIs;
  worth it only if a reviewer presses on interp variance. (b) The frequency-stratified story,
  which is now a null.

## Provenance
- Checkpoints: huggingface.co/Silen/cosine-scored-saes-qwen3-8b (standard/global-a/perfeature).
- Run: h100-dev-box-4 GPU1 (collect) + Bedrock Sonnet-4-6 (score), 2026-06-22.
- Data: `exp62a_results_hf.json` (per-feature judgments), `exp62a_results.json` (aggregate).
- Protocol identical to exp53 (describe 10 / predict 10 / >=50% threshold); only N=200->1000
  and arms restricted to the 3 recommended variants.
