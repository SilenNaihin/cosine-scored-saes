# Experiment 39: Norm-preserving SAE (NoC) architecture

A new SAE architecture that performs all sparse coding on the unit sphere and re-injects the input norm at output. The minimal recipe (no_C): unit-norm encoder rows with cosine scoring, unit-norm decoder rows, MSE loss on norm-restored output. Trained at 4x expansion, k=80, 5M FineWeb tokens on Qwen3-8B (L9/L18/L27) and Gemma-2-2B (L13).

## Results

### Main Comparison

| Metric | standard | adaptive_cosine | no_C |
|---|---|---|---|
| Qwen3-8B L9 FVE | 0.598 | 0.622 | 0.619 |
| Qwen3-8B L18 FVE | 0.534 | 0.559 | 0.585 |
| Qwen3-8B L27 FVE | 0.566 | 0.578 | 0.610 |
| Gemma-2-2B L13 FVE | 0.592 | 0.624 | 0.634 |
| Qwen3-8B L9 dead% | 55.4% | 55.0% | 0.0% |
| Qwen3-8B L18 dead% | 79.8% | 81.5% | 4.2% |
| Qwen3-8B L27 dead% | 74.5% | 61.7% | 0.0% |
| Gemma-2-2B L13 dead% | 62.1% | 39.2% | 0.0% |
| Qwen3-8B L9 substitution KL | 3.21 | 2.86 | 2.83 |

### Component Ablation (Qwen3-8B L9)

Components: A = unit-norm decoder rows, B = cosine encoder, C = cosine loss, D = norm-restoring output projection.

| Variant | A | B | C | D | Dead% | fire>1e-4 | Gini |
|---|---|---|---|---|---|---|---|
| standard | -- | -- | -- | -- | 55.4% | 4,652 | 0.964 |
| +unit_dec | Y | -- | -- | -- | 49.0% | 6,129 | 0.943 |
| +cos_loss | -- | -- | Y | -- | 43.8% | 6,034 | 0.973 |
| +norm_out | -- | -- | -- | Y | 36.0% | 6,662 | 0.970 |
| A+C | Y | -- | Y | -- | 20.1% | 10,603 | 0.932 |
| A+B (no C, no D) | Y | Y | -- | -- | 86.8% | 2,049 | 0.975 |
| A+C+D (no_B) | Y | -- | Y | Y | 19.9% | 10,615 | 0.932 |
| A+B+D (no_C) | Y | Y | -- | Y | 0.0% | 16,346 | 0.875 |
| A+B+C (no_D) | Y | Y | Y | -- | 0.1% | 16,035 | 0.887 |
| A+B+C+D (full) | Y | Y | Y | Y | 0.1% | 16,041 | 0.887 |

### Decoder Direction Overlap (Qwen3-8B L9)

| Variant | n_alive | dec p99 | dec p999 | dec max |
|---|---|---|---|---|
| standard | 4,612 | 0.315 | 0.366 | 0.415 |
| naive_cosine | 2,030 | 0.963 | 0.967 | 0.967 |
| adaptive_cosine | 5,320 | 0.325 | 0.380 | 0.432 |
| norm_preserve | 16,040 | 0.299 | 0.378 | 0.446 |
| no_C | 16,344 | 0.292 | 0.351 | 0.467 |

### Token Consistency (Qwen3-8B L9)

| Variant | modal_share | n_unique_tokens (median) |
|---|---|---|
| naive_cosine | 0.70 | 6 |
| standard | 0.16 | 47 |
| adaptive_cosine | 0.13 | 49 |
| norm_preserve | 0.08 | 69 |
| no_C | 0.10 | 65 |

### Substitution KL (Qwen3-8B L9)

| Variant | mean KL | median KL | p90 | p99 |
|---|---|---|---|---|
| standard | 3.21 | 2.59 | 6.80 | 12.13 |
| adaptive_cosine | 2.86 | 2.27 | 6.10 | 10.87 |
| norm_preserve | 3.26 | 2.50 | 7.09 | 13.71 |
| no_C | 2.83 | 2.23 | 6.14 | 11.37 |

### K-Sweep (Qwen3-8B L9)

| k | Variant | FVE | Dead% | fire>1e-4 | Gini |
|---|---|---|---|---|---|
| 32 | standard | 0.467 | 87.1% | 1,443 | 0.989 |
| 32 | no_C | 0.547 | 1.1% | 8,903 | 0.935 |
| 80 | standard | 0.598 | 55.4% | 4,650 | 0.964 |
| 80 | no_C | 0.619 | 0.0% | 16,346 | 0.875 |
| 160 | standard | 0.682 | 17.3% | 11,293 | 0.912 |
| 160 | no_C | 0.664 | 0.0% | 16,382 | 0.822 |

### Scale Verification (Qwen3-8B L9, 5M vs 15M tokens)

| Variant | Tokens | FVE | FVE_dir | Dead% | fire>1e-4 |
|---|---|---|---|---|---|
| standard | 5M | 0.598 | 0.581 | 55.4% | 4,652 |
| standard | 15M | 0.647 | 0.633 | 63.6% | 5,242 |
| no_C | 5M | 0.619 | 0.624 | 0.0% | 16,346 |
| no_C | 15M | 0.691 | 0.695 | 0.0% | 16,217 |

### Cross-Model (Gemma-2-2B L13)

| Variant | FVE | Dead% | fire>1e-4 | Gini |
|---|---|---|---|---|
| standard | 0.592 | 62.1% | 2,407 | 0.960 |
| adaptive_cosine | 0.624 | 39.2% | 2,826 | 0.961 |
| no_C | 0.634 | 0.0% | 9,212 | 0.904 |
