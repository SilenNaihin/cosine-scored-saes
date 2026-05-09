# Experiment 21: FVE-causality composite loss exploration

This experiment tests whether a weighted combination of post-RMSNorm loss and L2 loss can achieve both good FVE (from L2) and good causal feature quality (from post-norm). The composite loss is `alpha * postnorm_loss + (1-alpha) * L2_loss`, swept over alpha in {0.50, 0.80, 0.90, 0.95, 0.99}. Trained on Qwen3-8B layer 27, d_sae=16384, k=80, 5M tokens.

## Results

Learned adaptive parameter `a` by alpha:

| Variant | alpha | scale_a |
|---|---|---|
| adaptive_l2 | 0 (pure L2) | 0.1035 |
| adaptive_postnorm | 1 (pure postnorm) | -0.0148 |
| composite_a0.50 | 0.50 | 0.1024 |
| composite_a0.80 | 0.80 | 0.1069 |
| composite_a0.90 | 0.90 | 0.1142 |
| composite_a0.95 | 0.95 | 0.1253 |
| composite_a0.99 | 0.99 | 0.1606 |

Reconstruction metrics:

| Variant | alpha | FVE | pnFVE | Dead fraction |
|---|---|---|---|---|
| standard | - | 0.5802 | 0.6377 | 0.803 |
| adaptive_l2 | - | 0.5304 | 0.5866 | 0.758 |
| adaptive_postnorm | - | 0.0749 | 0.7230 | 0.820 |
| composite_a0.50 | 0.50 | 0.5202 | 0.5967 | 0.764 |
| composite_a0.80 | 0.80 | 0.4923 | 0.5974 | 0.803 |
| composite_a0.90 | 0.90 | 0.4733 | 0.6004 | 0.841 |
| composite_a0.95 | 0.95 | 0.4629 | 0.6104 | 0.863 |
| composite_a0.99 | 0.99 | 0.5332 | 0.6878 | 0.797 |

Ablation metrics:

| Variant | alpha | cos->KL | SAE->KL | cos>inner |
|---|---|---|---|---|
| standard | - | 0.3141 | 0.2557 | 14/30 (47%) |
| adaptive_l2 | - | 0.2558 | 0.0889 | 26/30 (87%) |
| adaptive_postnorm | - | 0.3696 | 0.2517 | 22/30 (73%) |
| composite_a0.50 | 0.50 | 0.2616 | 0.0898 | 24/30 (80%) |
| composite_a0.80 | 0.80 | 0.3101 | 0.0831 | 22/30 (73%) |
| composite_a0.90 | 0.90 | 0.2793 | 0.1431 | 23/30 (77%) |
| composite_a0.95 | 0.95 | 0.2511 | 0.0930 | 24/30 (80%) |
| composite_a0.99 | 0.99 | 0.2074 | 0.1021 | 21/30 (70%) |
