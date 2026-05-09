# Experiment 19: Norm-stratified analysis of cosine advantage

This experiment stratifies reconstruction and causal metrics by activation norm quartile to determine whether the cosine SAE advantage concentrates on specific regions of the norm distribution. It evaluates exp17 checkpoints (50M tokens, Qwen3-8B layers 9/18/27, d_sae=16384, k=80) on 1M eval tokens per layer, splitting results into four norm quartiles. No new training is performed.

## Results

Norm distributions:

| Layer | Mean norm | Std | Q1 mean | Q2 mean | Q3 mean | Q4 mean | Q4/Q1 ratio |
|---|---|---|---|---|---|---|---|
| 9 | 58.1 | 6.6 | 50.1 | 56.0 | 60.0 | 66.4 | 1.33x |
| 18 | 99.9 | 9.7 | 88.3 | 96.2 | 102.5 | 112.8 | 1.28x |
| 27 | 407.4 | 34.9 | 360.8 | 400.4 | 421.4 | 447.1 | 1.24x |

FVE by norm quartile, Layer 9:

| Variant | Q1 (50.1) | Q2 (56.0) | Q3 (60.0) | Q4 (66.4) | Range |
|---|---|---|---|---|---|
| standard | 0.642 | 0.680 | 0.689 | 0.671 | 0.047 |
| adaptive_l2 | 0.663 | 0.703 | 0.714 | 0.695 | 0.051 |
| perfeature_l2 | 0.668 | 0.711 | 0.724 | 0.712 | 0.056 |
| Gap (adaptive-std) | +0.021 | +0.023 | +0.025 | +0.024 | |

FVE by norm quartile, Layer 18:

| Variant | Q1 (88.3) | Q2 (96.2) | Q3 (102.5) | Q4 (112.8) | Range |
|---|---|---|---|---|---|
| standard | 0.606 | 0.625 | 0.637 | 0.645 | 0.039 |
| adaptive_l2 | 0.602 | 0.623 | 0.636 | 0.644 | 0.042 |
| perfeature_l2 | 0.618 | 0.640 | 0.654 | 0.666 | 0.048 |
| Gap (perfeature-std) | +0.012 | +0.015 | +0.017 | +0.021 | |

FVE by norm quartile, Layer 27:

| Variant | Q1 (360.8) | Q2 (400.4) | Q3 (421.4) | Q4 (447.1) | Range |
|---|---|---|---|---|---|
| standard | 0.617 | 0.648 | 0.661 | 0.674 | 0.057 |
| adaptive_l2 | 0.699 | 0.727 | 0.740 | 0.757 | 0.058 |
| perfeature_l2 | 0.696 | 0.723 | 0.735 | 0.752 | 0.056 |
| Gap (adaptive-std) | +0.082 | +0.079 | +0.079 | +0.083 | |

cos>inner win rate by norm quartile, Layer 9:

| Variant | Q1 | Q2 | Q3 | Q4 | Overall |
|---|---|---|---|---|---|
| standard | 15/30 (50%) | 19/30 (63%) | 18/30 (60%) | 22/30 (73%) | 24/30 (80%) |
| adaptive_l2 | 14/27 (52%) | 13/30 (43%) | 17/30 (57%) | 27/30 (90%) | 27/30 (90%) |
| perfeature_l2 | 14/23 (61%) | 18/30 (60%) | 17/30 (57%) | 23/30 (77%) | 30/30 (100%) |

cos>inner win rate by norm quartile, Layer 18:

| Variant | Q1 | Q2 | Q3 | Q4 | Overall |
|---|---|---|---|---|---|
| standard | 21/30 (70%) | 20/30 (67%) | 12/30 (40%) | 19/30 (63%) | 25/30 (83%) |
| adaptive_l2 | 14/30 (47%) | 16/30 (53%) | 15/30 (50%) | 21/30 (70%) | 28/30 (93%) |
| perfeature_l2 | 16/30 (53%) | 15/30 (50%) | 20/30 (67%) | 18/30 (60%) | 29/30 (97%) |

cos>inner win rate by norm quartile, Layer 27:

| Variant | Q1 | Q2 | Q3 | Q4 | Overall |
|---|---|---|---|---|---|
| standard | 21/30 (70%) | 15/30 (50%) | 13/30 (43%) | 16/30 (53%) | 26/30 (87%) |
| adaptive_l2 | 20/30 (67%) | 16/30 (53%) | 17/30 (57%) | 21/30 (70%) | 20/30 (67%) |
| perfeature_l2 | 19/30 (63%) | 17/30 (57%) | 20/30 (67%) | 19/30 (63%) | 24/30 (80%) |

cos->KL and inner->KL correlations by quartile, Layer 27:

| Variant | Metric | Q1 | Q2 | Q3 | Q4 | Overall |
|---|---|---|---|---|---|---|
| standard | cos->KL | 0.289 | 0.353 | 0.363 | 0.452 | 0.350 |
| standard | inner->KL | 0.263 | 0.350 | 0.363 | 0.451 | 0.322 |
| adaptive_l2 | cos->KL | 0.325 | 0.314 | 0.369 | 0.389 | 0.353 |
| adaptive_l2 | inner->KL | 0.307 | 0.314 | 0.368 | 0.383 | 0.338 |
| perfeature_l2 | cos->KL | 0.326 | 0.375 | 0.435 | 0.462 | 0.378 |
| perfeature_l2 | inner->KL | 0.299 | 0.373 | 0.432 | 0.483 | 0.364 |

Cosine reconstruction by quartile, Layer 27:

| Variant | Q1 cos | Q2 cos | Q3 cos | Q4 cos |
|---|---|---|---|---|
| standard | 0.880 | 0.897 | 0.904 | 0.910 |
| adaptive_l2 | 0.907 | 0.921 | 0.928 | 0.934 |
| perfeature_l2 | 0.906 | 0.920 | 0.926 | 0.932 |

L2 reconstruction loss by quartile, Layer 27:

| Variant | Q1 L2 | Q2 L2 | Q3 L2 | Q4 L2 |
|---|---|---|---|---|
| standard | 28987 | 31116 | 32183 | 34227 |
| adaptive_l2 | 22761 | 24133 | 24662 | 25447 |
| perfeature_l2 | 22982 | 24511 | 25143 | 26049 |
