# Experiment 14: Post-RMSNorm reconstruction loss

A cosine SAE is trained with post-RMSNorm loss: `||RMSNorm(x) - RMSNorm(x_hat)||^2`, using the model's actual next-layer RMSNorm module (including its learned per-dimension gain parameter). This optimizes for reconstruction quality as seen by the downstream layer, weighting dimensions proportionally to the gain. The gain parameter's coefficient of variation (25-41% across layers) ensures this loss differs meaningfully from unweighted cosine loss.

## Results

Model: Qwen3-8B, d_sae=16384, k=80, 5M tokens/layer, layers 9/18/27.

### RMSNorm gain analysis

| Layer | Next norm | Gain mean | Gain std | Gain min | Gain max | CV |
|-------|-----------|-----------|----------|----------|----------|------|
| 9 | L10 input_layernorm | 0.2573 | 0.0645 | -0.0063 | 1.3047 | 25.1% |
| 18 | L19 input_layernorm | 0.4854 | 0.1965 | 0.1006 | 2.3750 | 40.5% |
| 27 | L28 input_layernorm | 1.1006 | 0.3383 | 0.5117 | 7.7812 | 30.7% |

### Reconstruction (raw FVE and cosine)

| Variant | L9 FVE | L9 cos | L18 FVE | L18 cos | L27 FVE | L27 cos |
|---------|--------|--------|---------|---------|---------|---------|
| standard | 0.601 | 0.852 | 0.534 | 0.853 | 0.564 | 0.867 |
| cosine_l2 | 0.623 | 0.861 | 0.551 | 0.858 | 0.449 | 0.830 |
| cosine_cosloss | 0.534 | 0.851 | 0.327 | 0.856 | 0.124 | 0.873 |
| adaptive_l2 | 0.623 | 0.861 | 0.561 | 0.862 | 0.534 | 0.859 |
| cosine_postnorm | 0.503 | 0.670 | 0.246 | 0.795 | 0.083 | 0.861 |

### Post-norm FVE (variance explained in RMSNorm output space)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| cosine_postnorm | 0.550 | 0.560 | 0.723 |

### Norm invariance (2x input scaling, activation ratio)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 1.984 | 1.984 | 1.994 |
| cosine_l2 | 1.001 | 1.000 | 1.000 |
| cosine_cosloss | 1.000 | 1.000 | 1.000 |
| adaptive_l2 | 1.032 | 1.075 | 1.074 |
| cosine_postnorm | 1.000 | 1.000 | 1.000 |

### Ablation (cos to KL correlation)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 0.238 | 0.390 | 0.368 |
| cosine_l2 | 0.232 | 0.357 | 0.232 |
| cosine_cosloss | 0.267 | 0.397 | 0.308 |
| adaptive_l2 | 0.226 | 0.368 | 0.285 |
| cosine_postnorm | 0.204 | 0.366 | 0.336 |

### SAE-to-KL correlation

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 0.208 | 0.361 | 0.198 |
| cosine_l2 | 0.153 | 0.247 | 0.047 |
| cosine_cosloss | 0.180 | 0.288 | 0.104 |
| adaptive_l2 | 0.194 | 0.255 | 0.094 |
| cosine_postnorm | 0.141 | 0.308 | 0.252 |

### cos>inner win rate

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 23/30 (77%) | 21/30 (70%) | 23/30 (77%) |
| cosine_l2 | 25/30 (83%) | 27/30 (90%) | 24/30 (80%) |
| cosine_cosloss | 25/30 (83%) | 18/30 (60%) | 21/30 (70%) |
| adaptive_l2 | 24/30 (80%) | 25/30 (83%) | 21/30 (70%) |
| cosine_postnorm | 20/30 (67%) | 25/30 (83%) | 18/30 (60%) |

### Dead features

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 71.5% | 91.9% | 83.6% |
| cosine_l2 | 65.9% | 88.0% | 85.6% |
| cosine_cosloss | 77.7% | 92.8% | 76.3% |
| adaptive_l2 | 66.9% | 88.6% | 75.0% |
| cosine_postnorm | 73.9% | 97.2% | 81.9% |

### Scale factor

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| cosine_l2 | 68.2 | 73.5 | 70.7 |
| cosine_cosloss | 62.3 | 63.0 | 63.1 |
| cosine_postnorm | 62.8 | 63.3 | 62.6 |
