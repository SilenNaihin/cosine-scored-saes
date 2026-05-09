# Experiment 16: Per-feature learned norm-dependence

Each of the 16,384 SAE features receives its own learnable norm-dependence parameter `a_i` via `scale_i(x) = exp(a_i * log(||x||) + b_i)`, initialized at a_i=0 (pure cosine). The optimizer independently chooses how much magnitude each feature should use. A cosine-loss control variant (perfeature_cosloss) confirms zero gradient on all 16,384 scale parameters under scale-invariant loss.

## Results

Model: Qwen3-8B, d_sae=16384, k=80, 5M tokens/layer, layers 9/18/27.

### Per-feature scale_a distribution (perfeature_l2)

| Layer | Mean | Std | Median | 5th-95th pctile | Near-zero (|a|<0.05) | Low (0.05-0.2) | Medium (0.2-0.5) | High (>0.5) |
|-------|------|------|--------|-----------------|----------------------|----------------|-------------------|-------------|
| 9 | -0.000 | 0.011 | -0.003 | [-0.008, 0.018] | 99% | 1% | 0% | 0% |
| 18 | 0.007 | 0.026 | -0.001 | [-0.004, 0.079] | 92% | 8% | 0% | 0% |
| 27 | 0.016 | 0.040 | 0.001 | [-0.000, 0.124] | 88% | 12% | 0.1% | 0% |

### Depth gradient summary

| Layer | Norm | Features wanting a~0 | Features wanting a>0.05 | Max a_i |
|-------|------|----------------------|-------------------------|---------|
| 9 | ~58 | 99% | 1% | 0.13 |
| 18 | ~165 | 92% | 8% | 0.19 |
| 27 | ~407 | 88% | 12% | 0.23 |

### Reconstruction (FVE)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 0.601 | 0.534 | 0.564 |
| cosine_l2 | 0.623 | 0.551 | 0.449 |
| adaptive_l2 (exp12) | 0.623 | 0.561 | 0.534 |
| perfeature_l2 | 0.626 | 0.558 | 0.530 |

### Reconstruction (cosine)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 0.852 | 0.853 | 0.867 |
| cosine_l2 | 0.861 | 0.858 | 0.830 |
| adaptive_l2 | 0.861 | 0.862 | 0.859 |
| perfeature_l2 | 0.862 | 0.861 | 0.857 |

### Ablation (cos to KL correlation)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 0.238 | 0.390 | 0.368 |
| cosine_cosloss | 0.267 | 0.397 | 0.308 |
| adaptive_l2 | 0.226 | 0.368 | 0.285 |
| perfeature_l2 | 0.228 | 0.358 | 0.307 |
| perfeature_cosloss | 0.246 | 0.397 | 0.344 |

### Dead features

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 71.5% | 91.9% | 83.6% |
| adaptive_l2 | 66.9% | 88.6% | 75.0% |
| perfeature_l2 | 65.2% | 88.2% | 79.6% |
