# Experiment 56a: Feature co-activation patterns

This experiment tests whether cosine SAEs distribute concept information across more features than standard SAEs. For each concept class in sparse probing datasets (bias_in_bios, github-code, ag_news), we compute per-feature activation rates across SAE architectures and measure concept support size (number of discriminative features per concept), broad support, and activation entropy.

## Results

### Concept Support Size (features with >30% activation on positive, <10% on negative)

| SAE | Mean Support | Median Support | Std Support |
|-----|-------------|----------------|-------------|
| perfeature_l2 | 5.6 | 3.0 | 7.9 |
| standard | 5.5 | 3.0 | 8.3 |
| no_C | 5.2 | 3.0 | 7.5 |
| adaptive_l2 | 4.7 | 3.0 | 6.6 |

### Broad Support (features active on >20% of positive examples)

| SAE | Mean Broad Support |
|-----|-------------------|
| no_C | 46.7 |
| standard | 44.8 |
| adaptive_l2 | 42.6 |
| perfeature_l2 | 39.8 |

### Activation Entropy

| SAE | Mean Normalized Entropy | Mean Gini |
|-----|------------------------|-----------|
| adaptive_l2 | 0.862 | 0.592 |
| standard | 0.861 | 0.591 |
| perfeature_l2 | 0.860 | 0.587 |
| no_C | 0.727 | 0.688 |
