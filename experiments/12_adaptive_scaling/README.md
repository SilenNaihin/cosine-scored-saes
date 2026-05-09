# Experiment 12: Learned norm-dependence parameter optimization

A learnable norm-dependence parameter `a` is introduced into the cosine SAE encoder via `scale(x) = exp(a * log(||x||) + b)`, where a=0 gives pure cosine and a=1 gives inner-product behavior. The optimizer freely chooses how much magnitude the encoder should use for reconstruction. A cosine-loss control variant confirms zero gradient on scale parameters under scale-invariant loss.

## Results

Model: Qwen3-8B, d_sae=16384, k=80, 5M tokens/layer, layers 9/18/27.

### Learned scale_a values

| Layer | Activation norm | adaptive_l2 a | adaptive_cosloss a |
|-------|-----------------|---------------|---------------------|
| 9 | ~58 | 0.044 | -0.019 (frozen) |
| 18 | ~165 | 0.103 | -0.008 (frozen) |
| 27 | ~407 | 0.103 | -0.006 (frozen) |

### Reconstruction (FVE)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 0.601 | 0.534 | 0.564 |
| cosine_l2 | 0.623 | 0.551 | 0.449 |
| cosine_cosloss | 0.534 | 0.327 | 0.124 |
| adaptive_l2 | 0.623 | 0.561 | 0.534 |
| adaptive_cosloss | 0.517 | 0.319 | 0.120 |

### Reconstruction (cosine)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 0.852 | 0.853 | 0.867 |
| cosine_l2 | 0.861 | 0.858 | 0.830 |
| cosine_cosloss | 0.851 | 0.856 | 0.873 |
| adaptive_l2 | 0.861 | 0.862 | 0.859 |
| adaptive_cosloss | 0.851 | 0.857 | 0.874 |

### Norm invariance (2x input scaling, activation ratio)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 1.984 | 1.984 | 1.994 |
| cosine_l2 | 1.001 | 1.000 | 1.000 |
| adaptive_l2 | 1.032 | 1.075 | 1.074 |
| adaptive_cosloss | 0.985 | 0.994 | 0.996 |

### Ablation (cos to KL correlation)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 0.238 | 0.390 | 0.368 |
| cosine_l2 | 0.232 | 0.357 | 0.232 |
| cosine_cosloss | 0.267 | 0.397 | 0.308 |
| adaptive_l2 | 0.226 | 0.368 | 0.285 |
| adaptive_cosloss | 0.229 | 0.367 | 0.301 |

### cos>inner win rate

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 23/30 (77%) | 21/30 (70%) | 23/30 (77%) |
| cosine_l2 | 25/30 (83%) | 27/30 (90%) | 24/30 (80%) |
| adaptive_l2 | 24/30 (80%) | 25/30 (83%) | 21/30 (70%) |
| adaptive_cosloss | 26/30 (87%) | 24/30 (80%) | 20/30 (67%) |

### Dead features

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 71.5% | 91.9% | 83.6% |
| cosine_l2 | 65.9% | 88.0% | 85.6% |
| cosine_cosloss | 77.7% | 92.8% | 76.3% |
| adaptive_l2 | 66.9% | 88.6% | 75.0% |
| adaptive_cosloss | 77.6% | 92.8% | 76.2% |
