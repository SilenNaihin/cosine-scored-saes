# Experiment 13: Cosine-based ablation procedure

Standard ablation removes the inner-product projection `(x @ f) * f`, whose magnitude scales with ||x||. This introduces a structural bias favoring standard SAE activations (which also scale with ||x||) on the SAE-to-KL correlation metric. This experiment introduces a cosine-based ablation that removes `cos(x, f) * f * C` (where C = median(||x||) per layer), making the perturbation norm-invariant, and re-evaluates all SAE variants under both ablation modes.

## Results

Model: Qwen3-8B, 5 SAE variants from exp10/12, 30 features x 50 samples per variant per layer, 500K eval tokens.

### Per-layer median norms (C values)

| Layer | C |
|-------|------|
| 9 | 58.0 |
| 18 | 99.2 |
| 27 | 410.5 |

### SAE-to-KL correlation under both ablation modes

#### Layer 9

| Variant | IP SAE->KL | COS SAE->KL | Delta |
|---------|------------|-------------|-------|
| standard | 0.2375 | 0.2415 | +0.0040 |
| cosine_l2 | 0.1821 | 0.2133 | +0.0312 |
| cosine_cosloss | 0.2205 | 0.2349 | +0.0144 |
| adaptive_l2 | 0.2106 | 0.2169 | +0.0063 |
| adaptive_cosloss | 0.1860 | 0.1914 | +0.0054 |

#### Layer 18

| Variant | IP SAE->KL | COS SAE->KL | Delta |
|---------|------------|-------------|-------|
| standard | 0.3556 | 0.3376 | -0.0180 |
| cosine_l2 | 0.2421 | 0.2233 | -0.0188 |
| cosine_cosloss | 0.2147 | 0.2324 | +0.0176 |
| adaptive_l2 | 0.2561 | 0.2356 | -0.0206 |
| adaptive_cosloss | 0.2329 | 0.2408 | +0.0079 |

#### Layer 27

| Variant | IP SAE->KL | COS SAE->KL | Delta |
|---------|------------|-------------|-------|
| standard | 0.2161 | 0.1601 | -0.0560 |
| cosine_l2 | 0.0361 | 0.0146 | -0.0215 |
| cosine_cosloss | 0.1299 | 0.1087 | -0.0212 |
| adaptive_l2 | 0.0922 | 0.0477 | -0.0445 |
| adaptive_cosloss | 0.1451 | 0.1391 | -0.0060 |

### cos-to-KL correlation under both ablation modes

#### Layer 9

| Variant | IP cos->KL | COS cos->KL | IP cos>inn | COS cos>inn |
|---------|------------|-------------|------------|-------------|
| standard | 0.2817 | 0.2909 | 21/30 | 25/30 |
| cosine_l2 | 0.2275 | 0.2572 | 23/30 | 26/30 |
| cosine_cosloss | 0.2625 | 0.2579 | 27/30 | 28/30 |
| adaptive_l2 | 0.2559 | 0.2655 | 25/30 | 26/30 |
| adaptive_cosloss | 0.2330 | 0.2203 | 24/30 | 25/30 |

#### Layer 18

| Variant | IP cos->KL | COS cos->KL | IP cos>inn | COS cos>inn |
|---------|------------|-------------|------------|-------------|
| standard | 0.3712 | 0.3562 | 22/30 | 28/30 |
| cosine_l2 | 0.3165 | 0.3024 | 24/30 | 28/30 |
| cosine_cosloss | 0.3170 | 0.3068 | 23/30 | 25/30 |
| adaptive_l2 | 0.3176 | 0.3129 | 24/30 | 27/30 |
| adaptive_cosloss | 0.3166 | 0.3054 | 24/30 | 27/30 |

#### Layer 27

| Variant | IP cos->KL | COS cos->KL | IP cos>inn | COS cos>inn |
|---------|------------|-------------|------------|-------------|
| standard | 0.3786 | 0.3557 | 26/30 | 26/30 |
| cosine_l2 | 0.2114 | 0.2001 | 21/30 | 24/30 |
| cosine_cosloss | 0.3060 | 0.2736 | 26/30 | 27/30 |
| adaptive_l2 | 0.2869 | 0.2571 | 20/30 | 28/30 |
| adaptive_cosloss | 0.2907 | 0.2842 | 25/30 | 28/30 |

### norm-to-KL correlation under both modes

| Layer | Std IP | Std COS | CosL2 IP | CosL2 COS | CosCos IP | CosCos COS | AdpL2 IP | AdpL2 COS | AdpCos IP | AdpCos COS |
|-------|--------|---------|----------|----------|-----------|----------|----------|----------|-----------|----------|
| 9 | -0.154 | -0.192 | -0.187 | -0.234 | -0.184 | -0.224 | -0.178 | -0.209 | -0.160 | -0.182 |
| 18 | -0.065 | -0.141 | -0.123 | -0.175 | -0.110 | -0.176 | -0.094 | -0.164 | -0.102 | -0.153 |
| 27 | -0.095 | -0.219 | -0.092 | -0.218 | -0.162 | -0.276 | -0.095 | -0.222 | -0.127 | -0.206 |

### Standard's advantage gap reduction under cosine ablation

| Layer | IP gap (std - #2) | COS gap (std - #2) | Gap reduction |
|-------|-------------------|---------------------|---------------|
| 9 | 0.017 | 0.007 | 59% |
| 18 | 0.100 | 0.097 | 3% |
| 27 | 0.071 | 0.021 | 70% |
