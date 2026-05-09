# Experiment 54: Gradient analysis with saprmarks recipe

Standard and adaptive cosine SAEs are trained for 10M tokens at L9, L18, and L27 on Qwen3-8B with the saprmarks recipe (d_sae=65,536, k=80, lr=5e-5, batch=2048, auxk). Per-feature encoder gradients are logged every 50 steps and stratified by input norm quartile (Q1=lowest, Q4=highest) to measure gradient imbalance across architectures and depths.

## Results

Median per-feature Q4/Q1 gradient ratio (main loss):

| | L9 | L18 | L27 |
|---|---|---|---|
| standard | 1.98x | 1.64x | 1.63x |
| adaptive_l2 | 0.90x | 0.78x | 1.00x |

Percentage of features with >2x Q4 dominance:

| | L9 | L18 | L27 |
|---|---|---|---|
| standard | 49.4% | 39.8% | 43.0% |
| adaptive_l2 | 8.1% | 14.7% | 25.9% |

Q1-specialized features (Q1 grad > Q4 grad):

| | L9 | L18 | L27 |
|---|---|---|---|
| standard | 10,522 | 17,352 | 19,354 |
| adaptive_l2 | 37,625 | 39,649 | 32,832 |

Training quality (10M tokens):

| | L9 | L18 | L27 |
|---|---|---|---|
| standard FVE | 0.675 | 0.626 | 0.656 |
| adaptive_l2 FVE | 0.699 | 0.641 | 0.698 |
| dead% | 0% | 0% | 0% |

Q4-only features (active on Q4 but not Q1):

| | L9 | L18 | L27 |
|---|---|---|---|
| standard | 2,844 | 1,274 | 33 |
| adaptive_l2 | 7,346 | 958 | 417 |

Norm CV (std/mean) by layer: L9=0.110, L18=0.101, L27=0.087.

Aux-k gradient: aux-k barely activated (<4 dead features across full 500M training), confirming the imbalance resides entirely in the main reconstruction loss.
