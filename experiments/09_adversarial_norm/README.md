# Experiment 9: High-cosine tokens have 10x more causal impact than high-norm

For each SAE feature, active tokens are split into adversarial groups based on median splits of cosine similarity and activation norm. High-cosine/low-norm tokens are those the SAE underestimates; high-norm/low-cosine tokens are those the SAE overestimates. The feature is ablated from both groups and KL divergence is measured at the output logits to determine which group has greater causal impact.

## Results

Model: Qwen3-8B, layers 9/18/27, 30 features per layer, 50 ablation samples per group per feature.

### Cross-layer summary

| Layer | n_feat | RNH correct | KL ratio (HC/HN) | HC KL | HN KL | HC SAE act | HN SAE act |
|-------|--------|-------------|-------------------|-------|-------|------------|------------|
| 9 | 30 | 28/30 (93%) | 20.1x | 4.21e-1 | 5.72e-2 | 3.21 | 1.81 |
| 18 | 30 | 25/30 (83%) | 4.4x | 1.27e-1 | 6.69e-2 | 6.24 | 3.84 |
| 27 | 30 | 28/30 (93%) | 6.7x | 1.51e-1 | 7.75e-2 | 20.15 | 12.19 |
| Overall | 90 | 81/90 (90%) | 10.4x | -- | -- | -- | -- |

### Extreme individual features (layer 9)

| Feature | HC KL | HN KL | Ratio | HC SAE act | HN SAE act |
|---------|-------|-------|-------|------------|------------|
| 9135 | 0.400 | 0.004 | 103x | 1.96 | 1.61 |
| 56119 | 0.652 | 0.007 | 90x | 3.23 | 1.68 |
