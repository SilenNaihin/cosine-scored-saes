# Experiment 34: Multi-Seed Reproducibility (3 Seeds)

This experiment trains standard, adaptive_l2, and perfeature_l2 SAEs with 3 random seeds each (42, 123, 456) on Qwen3-8B layer 27 for 50M tokens to estimate run-to-run variance and confirm statistical significance of the cosine advantage. Seeds control both weight initialization and data ordering; evaluation uses a fixed 1M-token set across all runs.

## Results

### Primary Comparison (sqrt(d) init, matching prior experiments)

| Variant | FVE | Dead% | Alive | SAE->KL | cos>inner | scale_a |
|---------|-----|-------|-------|---------|-----------|---------|
| standard | 0.657 +/- 0.001 | 78.5% +/- 0.3% | 3,525 +/- 54 | 0.380 +/- 0.006 | 78% | -- |
| adaptive_l2 | 0.737 +/- 0.001 | 29.8% +/- 1.3% | 11,496 +/- 205 | 0.326 +/- 0.006 | 78% | 0.207 |
| perfeature_l2 | 0.732 +/- 0.001 | 28.0% +/- 0.6% | 11,804 +/- 100 | 0.358 +/- 0.001 | 77% | 0.113 |

### Statistical Significance (adaptive_l2 vs standard)

| Metric | Gap | Significance |
|--------|-----|-------------|
| FVE | +0.080 | 41.7 sigma |
| Dead% | -48.7pp | 37.5 sigma |
| Alive ratio | 3.26x | -- |

### scale_a Per Seed

| Variant | seed 42 | seed 123 | seed 456 |
|---------|---------|----------|----------|
| adaptive_l2 | 0.206 | 0.210 | 0.206 |
| perfeature_l2 (mean) | 0.112 | 0.113 | 0.113 |

### Norm-Adaptive Init Comparison (included for completeness)

| Variant | FVE | Dead% | Alive | scale_a |
|---------|-----|-------|-------|---------|
| standard | 0.657 +/- 0.001 | 78.5% +/- 0.3% | 3,525 +/- 54 | -- |
| adaptive_l2 (norm-adaptive) | 0.687 +/- 0.001 | 65.0% +/- 0.4% | 5,728 +/- 66 | 0.013 |
| perfeature_l2 (norm-adaptive) | 0.699 +/- 0.001 | 55.1% +/- 0.3% | 7,353 +/- 41 | 0.007 |
