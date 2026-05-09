# Experiment 35: Gemma-2-2B full SAEBench (6 evals)

Production-scale (50M token) SAE training on Gemma-2-2B at layers 6, 13, and 20 with all six SAEBench benchmarks (Core, Sparse Probing, Absorption, SCR, TPP, RAVEL). Three architectures compared: standard BatchTopK, adaptive_l2, and perfeature_l2. Dictionary size 9216 (4x expansion), k=80, norm-adaptive initialization.

## Results

### Reconstruction Quality

| Layer | Variant | FVE | CosSim | Dead% | Alive |
|-------|---------|-----|--------|-------|-------|
| 6 | standard | 0.785 | 0.922 | 36.7% | 5,804 |
| 6 | adaptive_l2 | 0.800 | 0.927 | 29.3% | 6,312 |
| 6 | perfeature_l2 | 0.810 | 0.931 | 11.3% | 8,102 |
| 13 | standard | 0.699 | 0.900 | 66.9% | 3,015 |
| 13 | adaptive_l2 | 0.711 | 0.904 | 61.2% | 3,269 |
| 13 | perfeature_l2 | 0.721 | 0.907 | 45.9% | 4,752 |
| 20 | standard | 0.774 | 0.885 | 57.8% | 3,812 |
| 20 | adaptive_l2 | 0.831 | 0.916 | 31.8% | 6,024 |
| 20 | perfeature_l2 | 0.836 | 0.919 | 20.8% | 7,086 |

### SAEBench Core Eval

| Layer | Variant | KL Score | CE Score | FVE | Alive% |
|-------|---------|----------|----------|-----|--------|
| 6 | standard | 0.9852 | 0.9852 | 0.836 | 63.7% |
| 6 | adaptive_l2 | 0.9875 | 0.9885 | 0.848 | 74.6% |
| 6 | perfeature_l2 | 0.9884 | 0.9885 | 0.856 | 90.3% |
| 13 | standard | 0.9703 | 0.9703 | 0.820 | 33.5% |
| 13 | adaptive_l2 | 0.9728 | 0.9720 | 0.824 | 43.3% |
| 13 | perfeature_l2 | 0.9736 | 0.9736 | 0.832 | 57.5% |
| 20 | standard | 0.9422 | 0.9407 | 0.793 | 26.3% |
| 20 | adaptive_l2 | 0.9738 | 0.9736 | 0.844 | 69.3% |
| 20 | perfeature_l2 | 0.9750 | 0.9753 | 0.848 | 79.3% |

### Sparse Probing

| Layer | Variant | Full Acc | Top-1 | Top-2 | Top-5 |
|-------|---------|----------|-------|-------|-------|
| 6 | standard | 0.946 | 0.720 | 0.755 | 0.803 |
| 6 | adaptive_l2 | 0.946 | 0.749 | 0.791 | 0.862 |
| 6 | perfeature_l2 | 0.945 | 0.670 | 0.711 | 0.805 |
| 13 | standard | 0.950 | 0.696 | 0.742 | 0.794 |
| 13 | adaptive_l2 | 0.952 | 0.750 | 0.807 | 0.861 |
| 13 | perfeature_l2 | 0.953 | 0.669 | 0.697 | 0.748 |
| 20 | standard | 0.956 | 0.786 | 0.836 | 0.892 |
| 20 | adaptive_l2 | 0.957 | 0.830 | 0.855 | 0.904 |
| 20 | perfeature_l2 | 0.958 | 0.733 | 0.786 | 0.876 |

### Absorption

| Layer | Variant | Absorb Frac | Full Absorb | Num Split |
|-------|---------|-------------|-------------|-----------|
| 6 | standard | 0.045 | 0.029 | 1.08 |
| 6 | adaptive_l2 | 0.039 | 0.024 | 1.19 |
| 6 | perfeature_l2 | 0.043 | 0.003 | 1.08 |
| 13 | standard | 0.046 | 0.035 | 1.27 |
| 13 | adaptive_l2 | 0.040 | 0.015 | 1.12 |
| 13 | perfeature_l2 | 0.024 | 0.014 | 1.04 |
| 20 | standard | 0.031 | 0.029 | 1.19 |
| 20 | adaptive_l2 | 0.051 | 0.047 | 1.50 |
| 20 | perfeature_l2 | 0.056 | 0.063 | 1.42 |

### SCR (Sparse Circuit Recovery)

| Layer | Variant | N=2 | N=10 | N=20 | N=50 | N=100 |
|-------|---------|-----|------|------|------|-------|
| 6 | standard | 0.072 | 0.145 | 0.077 | 0.136 | 0.247 |
| 6 | adaptive_l2 | 0.060 | 0.159 | 0.228 | 0.279 | 0.286 |
| 6 | perfeature_l2 | 0.088 | 0.188 | 0.187 | 0.271 | 0.281 |
| 13 | standard | 0.091 | 0.259 | 0.279 | 0.415 | 0.270 |
| 13 | adaptive_l2 | 0.117 | 0.276 | 0.359 | 0.449 | 0.364 |
| 13 | perfeature_l2 | 0.096 | 0.260 | 0.326 | 0.315 | 0.322 |
| 20 | standard | 0.323 | 0.537 | 0.536 | 0.547 | 0.530 |
| 20 | adaptive_l2 | 0.234 | 0.493 | 0.530 | 0.515 | 0.595 |
| 20 | perfeature_l2 | 0.236 | 0.472 | 0.545 | 0.478 | 0.513 |

### TPP (Token Prediction via Patching)

| Layer | Variant | N=2 | N=10 | N=20 | N=50 | N=100 |
|-------|---------|-----|------|------|------|-------|
| 6 | standard | 0.022 | 0.128 | 0.242 | 0.341 | 0.373 |
| 6 | adaptive_l2 | 0.017 | 0.051 | 0.084 | 0.231 | 0.354 |
| 6 | perfeature_l2 | 0.011 | 0.069 | 0.136 | 0.293 | 0.355 |
| 13 | standard | 0.011 | 0.137 | 0.259 | 0.329 | 0.355 |
| 13 | adaptive_l2 | 0.016 | 0.046 | 0.092 | 0.190 | 0.288 |
| 13 | perfeature_l2 | 0.012 | 0.107 | 0.181 | 0.304 | 0.356 |
| 20 | standard | 0.037 | 0.181 | 0.282 | 0.371 | 0.384 |
| 20 | adaptive_l2 | 0.022 | 0.094 | 0.147 | 0.317 | 0.387 |
| 20 | perfeature_l2 | 0.028 | 0.120 | 0.212 | 0.349 | 0.398 |

### RAVEL (Causal Feature Disentanglement)

| Layer | Variant | Disentangle | Cause | Isolation |
|-------|---------|-------------|-------|-----------|
| 6 | standard | 0.649 | 0.631 | 0.668 |
| 6 | adaptive_l2 | 0.655 | 0.634 | 0.676 |
| 6 | perfeature_l2 | 0.646 | 0.627 | 0.665 |
| 13 | standard | 0.545 | 0.488 | 0.603 |
| 13 | adaptive_l2 | 0.589 | 0.559 | 0.619 |
| 13 | perfeature_l2 | 0.597 | 0.542 | 0.651 |
| 20 | standard | 0.494 | 0.009 | 0.979 |
| 20 | adaptive_l2 | 0.497 | 0.020 | 0.974 |
| 20 | perfeature_l2 | 0.498 | 0.022 | 0.974 |

### Scorecard Summary

| Eval | L6 Winner | L13 Winner | L20 Winner |
|------|-----------|------------|------------|
| Core (KL, CE, FVE, MSE) | perfeature (4/4) | perfeature (4/4) | perfeature (4/4) |
| Sparse probing (top-k) | adaptive (3/3) | adaptive (3/3) | adaptive (3/3) |
| Sparse probing (full-feat) | tied | perfeature | perfeature |
| Absorption | no signal | no signal | no signal |
| SCR (low-N) | perfeature | adaptive | standard |
| SCR (high-N) | adaptive | adaptive | standard |
| TPP (low-N) | standard | standard | standard |
| TPP (high-N) | tied | tied | perfeature |
| RAVEL | adaptive | perfeature | collapse (all) |

Cosine variants win 14 of 22 scorable categories; standard wins 6.
