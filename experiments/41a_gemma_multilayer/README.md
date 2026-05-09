# Experiment 41a: Gemma-2-2B 50M multilayer

50M-token streaming training on Gemma-2-2B at layers 7, 13, and 19, comparing standard BatchTopK against no_C (norm-preserving cosine). Dictionary size 9216 (4x expansion), k=80, sqrt(d) initialization. Includes reconstruction quality eval, substitution KL, SAEBench (core + sparse probing), and decoder direction overlap.

## Results

### Reconstruction Quality

| Layer | Variant | FVE | CosSim | Dead% | Alive | norm_ratio |
|-------|---------|-----|--------|-------|-------|------------|
| L7 | standard | 0.758 | 0.917 | 54.2% | 4,223 | 0.917 |
| L7 | no_C | 0.798 | 0.935 | 0.1% | 9,208 | 0.999 |
| L13 | standard | 0.696 | 0.899 | 68.8% | 2,873 | 0.899 |
| L13 | no_C | 0.747 | 0.921 | 0.0% | 9,212 | 1.000 |
| L19 | standard | 0.787 | 0.881 | 60.2% | 3,672 | 0.881 |
| L19 | no_C | 0.855 | 0.925 | 0.0% | 9,213 | 1.000 |

### Substitution KL (50M checkpoints)

| Layer | std mean | std median | no_C mean | no_C median |
|-------|----------|------------|-----------|-------------|
| L7 | 0.589 | 0.335 | 0.484 | 0.246 |
| L13 | 0.547 | 0.300 | 0.481 | 0.252 |
| L19 | 0.596 | 0.258 | 0.211 | 0.115 |

### SAEBench Core Eval (50M)

| Layer | Variant | EV | KL-score | CE-score | frac_alive | l2_ratio |
|-------|---------|-----|----------|----------|------------|----------|
| L7 | standard | 0.836 | 0.980 | 0.979 | 45.9% | 0.914 |
| L7 | no_C | 0.859 | 0.988 | 0.989 | 99.9% | 1.000 |
| L13 | standard | 0.820 | 0.962 | 0.961 | 31.3% | 0.895 |
| L13 | no_C | 0.844 | 0.979 | 0.980 | 100.0% | 1.000 |
| L19 | standard | 0.793 | 0.927 | 0.925 | 22.1% | 0.887 |
| L19 | no_C | 0.855 | 0.978 | 0.979 | 100.0% | 1.000 |

### SAEBench Sparse Probing (50M)

| Layer | Variant | sae_full | top_1 | top_2 | top_5 |
|-------|---------|----------|-------|-------|-------|
| L7 | standard | 0.949 | 0.760 | 0.783 | 0.859 |
| L7 | no_C | 0.918 | 0.751 | 0.778 | 0.806 |
| L13 | standard | 0.952 | 0.698 | 0.736 | 0.798 |
| L13 | no_C | 0.905 | 0.746 | 0.770 | 0.797 |
| L19 | standard | 0.957 | 0.813 | 0.850 | 0.890 |
| L19 | no_C | 0.930 | 0.830 | 0.837 | 0.865 |

### Decoder Direction Overlap (50M)

| Layer | Variant | alive | mean | p99 | pairs > 0.5 | pairs > 0.8 |
|-------|---------|-------|------|-----|-------------|-------------|
| L7 | standard | 4,832 | 0.181 | 0.353 | 0.03% | 0.0% |
| L7 | no_C | 9,145 | 0.220 | 0.382 | 0.10% | 0.0% |
| L13 | standard | 3,792 | 0.185 | 0.335 | 0.05% | 0.0% |
| L13 | no_C | 9,000 | 0.216 | 0.373 | 0.07% | 0.0% |
| L19 | standard | 3,616 | 0.135 | 0.314 | 0.03% | 0.0% |
| L19 | no_C | 9,126 | 0.196 | 0.364 | 0.04% | 0.0% |

### Matched-Alive Sparse Probing (top_1, 50M)

| Layer | standard | no_C_full | no_C_topN (matched) |
|-------|----------|-----------|---------------------|
| L7 | 0.760 | 0.751 | 0.734 |
| L13 | 0.698 | 0.746 | 0.690 |
| L19 | 0.813 | 0.830 | 0.804 |
