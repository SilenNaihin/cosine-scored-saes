# Experiment 40: saprmarks recipe 4-architecture comparison

Multi-layer verification and auto-interpretability evaluation of the norm-preserving (no_C) architecture against standard, adaptive_cosine, and the published Karvonen BatchTopK baseline. Covers substitution KL, k-sweeps, 15M-token scaling, SAEBench (core + sparse probing), decoder geometry, matched-alive controls, and LLM-scored auto-interpretability (1000 stratified features, Sonnet). Tested on Qwen3-8B (L9/L18/L27) and Gemma-2-2B (L13, 16k dictionary, up to 150M tokens).

## Results

### SAEBench Head-to-Head (no_C 100M vs standard 100M, Gemma-2-2B L13 16k)

| Metric | no_C | standard |
|---|---|---|
| EV | 0.867 | 0.832 |
| KL_score | 0.985 | 0.973 |
| CE_score | 0.985 | 0.972 |
| top_1 | 0.763 | 0.698 |
| top_5 | 0.793 | 0.810 |
| frac_alive | 99.7% | 21.7% |

### Auto-Interpretability (Sonnet, 1000 stratified features)

| Variant | n_scored | score >=4 | alive | est. total >=4 features |
|---|---|---|---|---|
| our_standard 100M | 931 | 82.1% | 3,529 | ~2,900 |
| our_no_C 100M | 986 | 80.1% | 16,332 | ~13,100 |
| published 500M (100-feature subsample) | 97 | 53.1% | 15,450 | ~8,200 |

### vs Published Karvonen BatchTopK (Gemma-2-2B, matched L0~80)

| SAE | tokens | EV | KL_score | CE_score | top_1 | top_5 | frac_alive |
|---|---|---|---|---|---|---|---|
| pub. BatchTopK 16k L12 | 500M | 0.867 | 0.988 | 0.989 | 0.738 | 0.867 | 97.2% |
| our standard 9k L13 | 50M | 0.820 | 0.962 | 0.961 | 0.698 | 0.798 | 31.3% |
| our no_C 9k L13 | 50M | 0.844 | 0.979 | 0.980 | 0.746 | 0.797 | 100% |
| our no_C 16k L13 | 150M | 0.867 | 0.985 | 0.985 | 0.767 | 0.792 | 99.8% |

### Substitution KL (Qwen3-8B, median per-token KL)

| Layer | standard | adaptive_cosine | no_C | norm_preserve |
|---|---|---|---|---|
| L9 | 2.59 | 2.27 | 2.23 | 2.50 |
| L18 | 1.07 | 1.00 | 0.89 | 0.89 |
| L27 | 1.27 | 1.12 | 0.99 | 0.99 |

### Alive Features (Qwen3-8B, out of 16,384)

| Layer | standard | adaptive_cosine | no_C | norm_preserve |
|---|---|---|---|---|
| L9 | 6,786 (41.4%) | 6,692 (40.8%) | 16,384 (100%) | 16,346 (99.8%) |
| L18 | 2,927 (17.9%) | 2,382 (14.5%) | 14,884 (90.8%) | 13,720 (83.7%) |
| L27 | 3,910 (23.9%) | 5,225 (31.9%) | 16,383 (100%) | 16,384 (100%) |

### Per-Feature Ablation KL (Qwen3-8B, median across 30 features)

| Layer | standard | adaptive_cosine | no_C | norm_preserve |
|---|---|---|---|---|
| L9 | 0.00341 | 0.00040 | 0.00020 | 0.00021 |
| L18 | 0.00123 | 0.00046 | 0.00013 | 0.00010 |
| L27 | 0.00099 | 0.00071 | 0.00005 | 0.00002 |

### K-Sweep (Qwen3-8B L18, 5M tokens)

| k | std FVE | std dead% | std alive | no_C FVE | no_C dead% | no_C alive |
|---|---|---|---|---|---|---|
| 32 | 0.391 | 97.7% | 376 | 0.456 | 89.9% | 1,660 |
| 80 | 0.537 | 92.0% | 1,310 | 0.589 | 59.4% | 6,655 |
| 160 | 0.644 | 77.0% | 3,768 | 0.657 | 13.8% | 14,124 |

### K-Sweep (Qwen3-8B L27, 5M tokens)

| k | std FVE | std dead% | std alive | no_C FVE | no_C dead% | no_C alive |
|---|---|---|---|---|---|---|
| 32 | 0.418 | 94.5% | ~900 | 0.520 | 66.4% | 5,500 |
| 80 | 0.571 | 83.1% | 2,769 | 0.617 | 29.0% | 11,633 |
| 160 | 0.677 | 57.3% | 6,995 | 0.674 | 4.0% | 15,729 |

### Scale (Qwen3-8B, 15M tokens, k=80)

| Layer | Variant | FVE | Dead% | Alive |
|---|---|---|---|---|
| L18 | standard | 0.584 | 86.1% | 1,396 |
| L18 | no_C | 0.647 | 7.9% | 7,064 |
| L27 | standard | 0.606 | 79.8% | 2,393 |
| L27 | no_C | 0.676 | 0.0% | 12,145 |

### SAEBench Core (Gemma-2-2B L13, 5M tokens)

| Variant | EV | KL-score | CE-score | frac_alive | l2_ratio |
|---|---|---|---|---|---|
| standard | 0.758 | 0.880 | 0.876 | 37.7% | 0.859 |
| adaptive_cosine | 0.777 | 0.907 | 0.904 | 65.9% | 0.871 |
| no_C | 0.773 | 0.924 | 0.923 | 100% | 1.000 |

### SAEBench Sparse Probing (Gemma-2-2B L13, 5M tokens)

| Variant | sae_full | top_1 | top_2 | top_5 |
|---|---|---|---|---|
| standard | 0.937 | 0.687 | 0.734 | 0.775 |
| adaptive_cosine | 0.942 | 0.700 | 0.724 | 0.758 |
| no_C | 0.890 | 0.734 | 0.747 | 0.766 |

### Matched-Alive Sparse Probing (Gemma-2-2B L13, 5M tokens)

| Variant | n_alive | sae_full | top_1 | top_2 | top_5 |
|---|---|---|---|---|---|
| standard | 3,410 | 0.937 | 0.687 | 0.734 | 0.775 |
| no_C_full | 9,208 | 0.890 | 0.734 | 0.747 | 0.766 |
| no_C_top3410 | 3,410 | 0.882 | 0.730 | 0.746 | 0.765 |

### Decoder Direction Overlap (Gemma-2-2B L13, 5M tokens)

| Variant | alive | mean overlap | p99 | max | pairs > 0.5 |
|---|---|---|---|---|---|
| standard | 3,410 | 0.114 | 0.213 | 0.296 | 0.0% |
| adaptive_cosine | 2,752 | 0.126 | 0.254 | 0.474 | 0.0% |
| no_C | 9,208 | 0.165 | 0.285 | 0.459 | 0.0% |

### no_C 16k Training Trajectory (Gemma-2-2B L13)

| Tokens | EV | KL | CE | alive | top_1 | top_5 |
|---|---|---|---|---|---|---|
| 50M | 0.859 | 0.984 | 0.984 | 99.1% | 0.737 | 0.785 |
| 100M | 0.867 | 0.985 | 0.985 | 99.7% | 0.763 | 0.793 |
| 150M | 0.867 | 0.985 | 0.985 | 99.8% | 0.767 | 0.792 |
