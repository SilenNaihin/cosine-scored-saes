# Experiment 10: First cosine SAE training (L9/L18/L27)

Three iterations of a cosine-normalized SAE encoder are trained and compared against a standard inner-product SAE. V1 uses naive cosine similarity (bounded [-1,1], fails to reconstruct high-norm activations). V2 adds a learnable global scale factor with L2 loss. V3 uses cosine reconstruction loss instead of L2, making the scale factor irrelevant. All variants are then expanded to three layers for cross-layer comparison.

## Results

Model: Qwen3-8B, d_sae=16384, k=80, 5M tokens/layer, layers 9/18/27.

### V1: Naive cosine encoder (no scale)

| Metric | Standard | Cosine V1 |
|--------|----------|-----------|
| FVE (L27) | 0.564 | 0.009 |
| cos recon (L27) | 0.867 | 0.385 |
| Dead features (L27) | 83.5% | 97.0% |
| FVE (L9) | 0.601 | 0.139 |

### V2: Scaled cosine + L2 loss

| Metric | Standard | Cosine V2 |
|--------|----------|-----------|
| FVE (L27) | 0.564 | 0.449 |
| cos recon (L27) | 0.867 | 0.830 |
| Dead features (L27) | 83.6% | 85.6% |
| Learned scale | -- | 70.7 |
| SAE>inner (L27) | 3/30 (10%) | 12/30 (40%) |

### V3: Scaled cosine + cosine loss

| Metric | Standard | Cosine V2 (L2) | Cosine V3 (cos loss) |
|--------|----------|----------------|----------------------|
| FVE (L27) | 0.564 | 0.449 | 0.124 |
| cos recon (L27) | 0.867 | 0.830 | 0.873 |
| Dead features (L27) | 83.6% | 85.6% | 76.3% |
| Learned scale | -- | 70.7 | 63.1 (frozen) |
| cos>inner (L27) | 23/30 (77%) | 24/30 (80%) | 21/30 (70%) |
| cos>SAE (L27) | 29/30 (97%) | 23/30 (77%) | 27/30 (90%) |

### Multi-layer reconstruction

| Layer | Std FVE | Std cos | CosL2 FVE | CosL2 cos | CosCos FVE | CosCos cos |
|-------|---------|---------|-----------|-----------|------------|------------|
| 9 | 0.601 | 0.852 | 0.623 | 0.861 | 0.534 | 0.851 |
| 18 | 0.534 | 0.853 | 0.551 | 0.858 | 0.327 | 0.856 |
| 27 | 0.564 | 0.867 | 0.449 | 0.830 | 0.124 | 0.873 |

### Multi-layer ablation (cos to KL correlation)

| Layer | Std cos->KL | CosL2 cos->KL | CosCos cos->KL |
|-------|-------------|---------------|----------------|
| 9 | 0.238 | 0.232 | 0.267 |
| 18 | 0.390 | 0.357 | 0.397 |
| 27 | 0.368 | 0.232 | 0.308 |

### cos>inner win rate

| Layer | Standard | CosL2 | CosCos |
|-------|----------|-------|--------|
| 9 | 23/30 (77%) | 25/30 (83%) | 25/30 (83%) |
| 18 | 21/30 (70%) | 27/30 (90%) | 18/30 (60%) |
| 27 | 23/30 (77%) | 24/30 (80%) | 21/30 (70%) |

### Scale factor behavior

| Layer | CosL2 scale | CosCos scale |
|-------|-------------|--------------|
| 9 | 68.2 | 62.3 |
| 18 | 73.5 | 63.0 |
| 27 | 70.7 | 63.1 |

### Norm invariance (2x input scaling)

- All cosine variants: activation ratio = 1.000 (perfect invariance)
- Standard: activation ratio = 1.984 (linear scaling)
