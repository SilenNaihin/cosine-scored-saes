# Experiment 4: Testing for magnitude-based duplicate directions in SAE decoder

This experiment tests whether the SAE (65,536 features, d_model=4096, Qwen3-8B layer 18) learns near-duplicate decoder directions that differ primarily in magnitude sensitivity. Pairwise cosine similarity is computed across all decoder directions, and the 18 pairs above cos > 0.70 are analyzed for encoder similarity and activation co-occurrence on 328 tokens from 12 prompts.

## Results

Pairwise decoder cosine similarity (sampled 5000 features, 12.5M pairs):

| Threshold | Pairs found |
|---|---|
| cos > 0.95 | 0 |
| cos > 0.90 | 0 |
| cos > 0.80 | 1 |
| cos > 0.70 | 18 |

Distribution statistics:

| Statistic | Value |
|---|---|
| Mean | 0.002 |
| Std | 0.029 |
| 90th percentile | 0.034 |
| 99th percentile | 0.091 |
| 99.9th percentile | 0.178 |
| Max (in sample) | 0.621 |

Encoder analysis of the 18 pairs with decoder cos > 0.70:

| Metric | Mean | Range |
|---|---|---|
| Encoder cosine | 0.208 | [-0.15, 0.41] |
| Bias difference | 0.957 | [0.03, 2.88] |
| Encoder pairs with cos > 0.7 | 0/18 | - |

- Jaccard activation overlap: ~0 for all pairs (only 1 pair had any co-occurrence, jaccard=0.06)
