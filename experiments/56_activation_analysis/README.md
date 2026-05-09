# Experiment 56: Activation magnitude and decoder analysis

Analysis of activation magnitudes, decoder geometry, encoder norms, feature co-activation patterns, and feature overlap between standard and cosine SAE architectures at L18 on Qwen3-8B (500M tokens, d_sae=65,536). Sub-experiments test whether the sparse probing advantage is driven by distributed encoding (56a, refuted) or feature discovery (56b, confirmed).

## Results

Non-zero activation statistics (50K tokens from FineWeb):

| SAE | Mean Act | Median Act | P95 Act | Total/Token | Max/Token |
|-----|----------|-----------|---------|-------------|-----------|
| standard | 4.081 | 2.890 | 10.37 | 323.3 | 22.44 |
| adaptive_l2 | 4.003 | 2.891 | 10.05 | 321.7 | 20.01 |
| perfeature_l2 | 3.953 | 2.832 | 9.75 | 317.4 | 22.40 |
| no_C | 0.028 | 0.016 | 0.06 | 2.2 | 0.46 |

Decoder norms: all architectures have exactly unit-norm decoders (mean=1.0000, std=0.0000).

Encoder norms:

| SAE | Mean | Std |
|-----|------|-----|
| adaptive_l2 | 1.750 | 0.203 |
| perfeature_l2 | 1.740 | 0.172 |
| standard | 1.360 | 0.215 |
| independent_ref | 1.318 | 0.179 |
| no_C | 1.000 | 0.000 |

Pairwise decoder orthogonality:

| SAE | Mean Pairwise Cosine |
|-----|---------------------|
| standard | 0.00287 |
| adaptive_l2 | 0.00234 |
| perfeature_l2 | 0.00251 |
| no_C | 0.03561 |
| independent_ref | 0.00218 |

Encoder-decoder alignment:

| SAE | Mean | Std |
|-----|------|-----|
| no_C | 0.681 | 0.067 |
| perfeature_l2 | 0.669 | 0.077 |
| adaptive_l2 | 0.654 | 0.054 |
| independent_ref | 0.626 | 0.067 |
| standard | 0.607 | 0.078 |

Concept support size (discriminative features per concept, >30% positive, <10% negative):

| SAE | Mean Support | Median Support |
|-----|-------------|----------------|
| perfeature_l2 | 5.6 | 3.0 |
| standard | 5.5 | 3.0 |
| no_C | 5.2 | 3.0 |
| adaptive_l2 | 4.7 | 3.0 |

Activation entropy (normalized):

| SAE | Mean Entropy | Mean Gini |
|-----|-------------|-----------|
| adaptive_l2 | 0.862 | 0.592 |
| standard | 0.861 | 0.591 |
| perfeature_l2 | 0.860 | 0.587 |
| no_C | 0.727 | 0.688 |

Feature overlap (standard vs perfeature_l2, 100K tokens):

| Direction | Strong match (>=0.7) | Weak (0.3-0.7) | Unmatched (<0.3) |
|-----------|---------------------|----------------|------------------|
| standard -> cosine | 41.1% (7,114) | 41.7% (7,216) | 17.1% (2,958) |
| cosine -> standard | 38.5% (7,147) | 40.5% (7,515) | 21.0% (3,908) |

Decoder similarity for 7,114 strongly-matched pairs: mean=0.913, median=0.943, 96.5% above 0.7.

Unmatched feature frequencies:

| | Mean freq (unmatched) | Mean freq (matched) |
|---|---|---|
| Standard | 0.0025 | 0.0052 |
| perfeature_l2 | 0.0025 | 0.0049 |
