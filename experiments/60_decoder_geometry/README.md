# Experiment 60: Decoder MMC and effective dimensionality

This experiment measures decoder disentanglement via mean-max-cosine similarity (MMC), effective dimensionality (participation ratio of singular values), and near-duplicate pair counts across 30 SAE checkpoints spanning multiple expansion ratios (4x/8x/16x), model sizes (1.7B/4B/8B), and training budgets (50M/500M tokens).

## Results

### Production Scale (d_sae=65536, 500M tokens, Qwen-8B)

| Architecture | MMC | p95 | Eff Dim | Pairs > 0.5 |
|---|---|---|---|---|
| adaptive_l2 | 0.293 | 0.438 | 353 | 570 |
| perfeature_l2 | 0.308 | 0.480 | 344 | 1,852 |
| standard | 0.311 | 0.474 | 340 | 1,419 |

### Low Expansion (d_sae=16384, 500M tokens, Qwen-8B)

| Layer | Standard MMC | Adaptive MMC | Delta (std - adp) |
|---|---|---|---|
| L9 | 0.163 | 0.203 | -0.040 |
| L18 | 0.147 | 0.159 | -0.012 |
| L27 | 0.134 | 0.184 | -0.049 |

### Scaling Pattern (exp57 checkpoints, 50M tokens)

| Model | 4x (std/adp) | 8x (std/adp) | 16x (std/adp) |
|---|---|---|---|
| Qwen-1.7B | 0.198/0.214 | 0.191/0.218 | 0.180/0.213 |
| Qwen-4B | 0.208/-- | 0.199/0.207 | 0.187/0.196 |
| Qwen-8B | 0.196/0.201 | 0.182/0.182 | 0.311/0.293 (500M) |

### Architecture Comparison at Matched Alive (d_sae=65536, 50M tokens)

| Architecture | Alive | MMC | Eff Dim |
|---|---|---|---|
| adaptive_l2 | 65,536 | 0.180 | 452 |
| perfeature_base_delta | 65,536 | 0.193 | 447 |
| perfeature_gaussian | 7,825 | 0.184 | 409 |
| perfeature_original | 10,684 | 0.205 | 395 |
| perfeature_var_reg | 10,941 | 0.205 | 395 |

### Detailed Metric Comparison (d_sae=65536, 500M tokens)

| Metric | Standard | Adaptive | Perfeature |
|---|---|---|---|
| MMC mean | 0.311 | 0.293 | 0.308 |
| MMC p95 | 0.474 | 0.438 | 0.480 |
| MMC std | 0.092 | 0.085 | 0.096 |
| Pairs > 0.5 | 1,419 | 570 | 1,852 |
| Mean neighbors > 0.5 | 0.043 | 0.017 | 0.057 |
| Max neighbors > 0.5 | 12 | 10 | 25 |
| MMC bin 0.2-0.3 (bulk) | 25,523 | 28,648 | 26,275 |
