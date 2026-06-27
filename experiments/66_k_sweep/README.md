# Experiment 66: Sparsity-budget (k) sweep

Reviewer item (pYoQ): "it would be interesting to see whether the autointerp
score gains persist at low sparsity, for example [low k]." We vary the BatchTopK
sparsity budget k (active features per token) and measure both sparse probing
and auto-interp.

## Setup

- Qwen3-8B, layer 18, 50M FineWeb tokens, d_sae = 65,536, exp43d recipe
  (Adam lr 5e-5, AuxK dead-feature loss on, decoder unit-norm, geometric-median
  b_dec init, W_enc = W_dec init). Headline uses k=80.
- Arms: Standard BatchTopK vs Per-Feature Adaptive Cosine.
- k in {10, 20, 40, 80, 160}.
- `exp66_k_sweep.py` trains (imports exp43d building blocks, sets k per arm).
  `exp66_probing.py` runs SAEBench sparse probing (per-(k,arm) sae_name +
  output_dir + force_rerun, so no cross-run result contamination).
  `exp66_autointerp.py` runs the exp53 describe-then-predict protocol
  (collect contexts on GPU, then score with Bedrock Sonnet, 200 features/arm).

## Results

Sparse-probing top-1 and auto-interp interpretable rate vs k:

| k | Top-1 Std | Top-1 PF | gap | Interp Std | Interp PF |
|---|-----------|----------|-----|------------|-----------|
| 10  | 0.653 | 0.733 | +8.0%  | 0.530 | 0.495 |
| 20  | 0.695 | 0.782 | +8.7%  | 0.395 | 0.440 |
| 40  | 0.678 | 0.792 | +11.4% | 0.350 | 0.325 |
| 80  | 0.731 | 0.816 | +8.5%  | 0.350 | 0.300 |
| 160 | 0.693 | 0.788 | +9.5%  | 0.180 | 0.215 |

FVE vs k (Std / PF): k10 0.454/0.432, k20 0.579/0.557, k40 0.665/0.667,
k80 0.723/0.726, k160 0.767/0.768.

## Findings

1. **Probing advantage is robust across the sparsity range.** The per-feature
   top-1 gain over Standard is +8 to +11% at every k, with no collapse at low k
   (largest, +11.4%, at k=40). At k=10 (both arms ~95% dead) it is still +8%.
2. **Auto-interp parity holds at every k.** The describe-then-predict rate tracks
   between the two architectures within 200-feature sampling noise (the arms
   trade the lead); neither dominates. This extends the headline matched-interp
   result (p=0.88 at 50M; 2.1-pt band at 500M) across the whole sparsity range.
3. **Both arms' interpretability falls as k rises** (0.53 -> 0.18 for Standard):
   more simultaneously-active features are individually harder to characterize.

The sweep reinforces the central decomposition: the cosine advantage is in
feature **discovery** (probing), not per-feature interpretability, and this holds
across sparsity budgets.

Note: the SAE sparsity budget k here is distinct from the probe top-k (top-1/2/5)
reported elsewhere; both are labeled "k" but refer to different quantities.

## Files

- `exp66_k_sweep.py` — training (k-parameterized, imports exp43d blocks).
- `exp66_probing.py` — SAEBench sparse-probing eval per (k, arm).
- `exp66_autointerp.py` — describe-then-predict auto-interp (collect + score).
- `exp66_{standard,perfeature}_results.json` — FVE/dead per (k, arm).
- `exp66_probing_{standard,perfeature}.json` — top-1/2/5 per (k, arm).
- `autointerp_summary.json` — interpretable rate per (k, arm).
