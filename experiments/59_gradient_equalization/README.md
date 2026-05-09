# Experiment 59: Gradient equalization falsification

This experiment trains a standard SAE with per-token loss reweighting by 1/||x|| (mild) and 1/||x||^2 (strong) to equalize gradient magnitudes across norm quartiles, testing whether gradient imbalance is the causal mechanism behind standard SAEs' norm-detector failure mode. All variants trained on Qwen3-8B L18, d_sae=65536, k=80, 50M tokens.

## Results

### Sparse Probing (8 datasets)

| Variant | Top-1 | Top-2 | Top-5 | Dead% | FVE |
|---------|-------|-------|-------|-------|-----|
| standard | 0.5299 | 0.5431 | 0.5486 | 0% | 0.977 |
| grad_eq_mild (1/||x||) | 0.5321 | 0.5440 | 0.5527 | 0% | 0.981 |
| grad_eq_strong (1/||x||^2) | 0.5357 | 0.5487 | 0.5594 | 0% | 0.984 |
| perfeature_l2 (cosine) | 0.6479 | 0.6610 | 0.7079 | 0% | 0.946 |

### Gap Closure

| k | Gap (cos - std) | Mild closure | Strong closure |
|---|-----------------|--------------|----------------|
| top-1 | +11.80pp | 1.9% | 4.9% |
| top-2 | +11.79pp | 0.8% | 4.7% |
| top-5 | +15.93pp | 2.6% | 6.8% |

### Per-Dataset Breakdown (top-1)

| Dataset | Standard | Mild | Strong | Cosine |
|---------|----------|------|--------|--------|
| bias_in_bios_set1 | 0.543 | 0.531 | 0.531 | 0.625 |
| bias_in_bios_set2 | 0.557 | 0.553 | 0.549 | 0.651 |
| bias_in_bios_set3 | 0.528 | 0.526 | 0.545 | 0.649 |
| amazon_reviews | 0.501 | 0.505 | 0.504 | 0.577 |
| amazon_sentiment | 0.500 | 0.500 | 0.500 | 0.738 |
| github-code | 0.521 | 0.524 | 0.544 | 0.657 |
| ag_news | 0.547 | 0.540 | 0.553 | 0.645 |
| europarl | 0.544 | 0.578 | 0.560 | 0.641 |
