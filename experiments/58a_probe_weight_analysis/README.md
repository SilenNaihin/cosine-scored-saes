# Experiment 58a: Probe weight decoder geometry

This experiment tests whether the TPP gap between standard and cosine SAEs originates from differences in how probe weight vectors project onto decoder directions. For 10 trained sparse probing probes, the probe weight vector is projected onto all alive decoder vectors and the resulting distribution is compared across architectures.

## Results

### Aggregate (10 probes)

| Metric | Standard | Cosine |
|--------|----------|--------|
| Entropy mean | 10.7508 | 10.7509 |
| Gini mean | 0.4486 | 0.4484 |
| Top-10 fraction mean | 0.00177 | 0.00177 |
| Top-50 fraction mean | 0.00651 | 0.00652 |
| Effective dimensionality mean | 46,669 | 46,672 |
| Entropy ratio (cos/std) | 1.0000 | |
| Eff dim ratio (cos/std) | 1.0001 | |

### Per-Dataset Effective Dimensionality

| Dataset | Standard Eff Dim | Cosine Eff Dim |
|---------|-----------------|----------------|
| bias_in_bios (probe 0) | 47,038 | 47,036 |
| bias_in_bios (probe 1) | 46,890 | 47,004 |
| bias_in_bios (probe 2) | 46,718 | 46,673 |
| bias_in_bios (probe 6) | 46,970 | 46,893 |
| bias_in_bios (probe 9) | 46,269 | 46,294 |
| amazon_reviews (probe 1) | 46,316 | 46,319 |
| amazon_reviews (probe 2) | 46,297 | 46,324 |
| amazon_reviews (probe 3) | 46,354 | 46,452 |
| amazon_reviews (probe 5) | 46,528 | 46,350 |
| amazon_reviews (probe 6) | 47,311 | 47,382 |
