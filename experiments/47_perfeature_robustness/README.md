# Experiment 47: Per-feature variant robustness

Five per-feature cosine SAE variants test structural fixes for the winner-take-all dead feature cascade at Qwen3-8B layer 27 (norm ~405, norm/sqrt(d) = 6.33x) where original per-feature loses 83% of features by step 5,500. Variants include a base+delta parameterization, variance regularization, Gaussian initialization, and the adaptive_l2 reference, all at 50M tokens with the saprmarks recipe (d_sae=65536, k=80, lr=5e-5).

## Results

### Final metrics

| Variant | FVE | Dead % | Alive | cos>inner | cos_kl | inner_kl |
|---|---|---|---|---|---|---|
| perfeature_original | 0.7199 | 83.8% | 10,625 | 71/100 | 0.285 | 0.270 |
| perfeature_base_delta | 0.7721 | 0.4% | 65,288 | 74/100 | 0.307 | 0.292 |
| perfeature_var_reg | 0.7213 | 83.4% | 10,891 | 82/100 | 0.269 | 0.253 |
| perfeature_gaussian | 0.7086 | 88.2% | 7,737 | 77/100 | 0.284 | 0.268 |
| adaptive_l2 | 0.7695 | 0.1% | 65,489 | 71/100 | 0.333 | 0.320 |

### Cascade timing

| Variant | First dead (>1%) | Dead at step 5500 | Final dead |
|---|---|---|---|
| perfeature_original | step 5500 | 67.1% (44K) | 83.8% (54.9K) |
| perfeature_var_reg | step 5500 | 65.8% (43K) | 83.4% (54.6K) |
| perfeature_gaussian | step 5000 | 86.3% (56.6K) | 88.2% (57.7K) |

### Scale parameter dynamics (base_delta vs adaptive)

| Step | a_base | a (adaptive) | ratio |
|---|---|---|---|
| 500 | 0.006 | 0.006 | 1.00 |
| 2000 | 0.074 | 0.075 | 0.99 |
| 5000 | 0.157 | 0.180 | 0.87 |
| 10000 | 0.191 | 0.217 | 0.88 |
| 20000 | 0.228 | 0.254 | 0.90 |
| 24414 | 0.231 | 0.257 | 0.90 |

### Scale parameter divergence (failing variants)

| Variant | Final a_mean | Final a_std | a_std/a_mean |
|---|---|---|---|
| perfeature_original | 0.021 | 0.084 | 4.0x |
| perfeature_var_reg | 0.022 | 0.085 | 3.9x |
| perfeature_gaussian | -0.008 | 0.101 | -- |
| perfeature_base_delta (effective) | 0.235 | 0.029 | 0.12x |
| adaptive_l2 | 0.257 | 0 | 0 |
