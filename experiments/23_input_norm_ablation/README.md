# Experiment 23: Input normalization vs geometry ablation

This experiment ablates the cosine SAE into components to determine whether the advantage comes from input normalization alone (gradient stabilization hypothesis) or from full cosine geometry (both input and weight normalization). Four variants are trained on Qwen3-8B layer 27 with d_sae=16384, k=80, 5M tokens.

## Results

Reconstruction quality:

| Variant | FVE | cos_recon | L2 loss | Dead% | Alive |
|---|---|---|---|---|---|
| standard | 0.564 | 0.867 | 40234 | 72.7% | 4480 |
| inputnorm_standard | 0.483 | 0.841 | 47741 | 81.7% | 2995 |
| cosine | 0.446 | 0.830 | 51082 | 74.4% | 4194 |
| adaptive_l2 | 0.530 | 0.858 | 43306 | 51.2% | 7993 |

Norm invariance:

| Variant | 0.5x ratio | 2x ratio | 5x ratio | 5x cos_sim |
|---|---|---|---|---|
| standard | 0.507 | 1.994 | 4.974 | 0.847 |
| inputnorm_standard | 1.016 | 0.992 | 0.987 | 0.998 |
| cosine | 1.000 | 1.000 | 1.000 | 1.000 |
| adaptive_l2 | 0.932 | 1.073 | 1.178 | 0.979 |

Ablation (causal importance):

| Variant | cos->KL | inner->KL | SAE->KL | norm->KL | cos>inner |
|---|---|---|---|---|---|
| standard | 0.394 | 0.360 | 0.249 | -0.048 | 21/30 |
| inputnorm_standard | 0.329 | 0.295 | 0.223 | -0.066 | 18/30 |
| cosine | 0.243 | 0.180 | 0.064 | -0.125 | 25/30 |
| adaptive_l2 | 0.311 | 0.276 | 0.138 | -0.078 | 22/30 |

Learned parameters:

| Variant | scale_b (exp) | scale_a |
|---|---|---|
| inputnorm_standard | 70.44 | - |
| cosine | 70.50 | - |
| adaptive_l2 | 70.25 | 0.1017 |
