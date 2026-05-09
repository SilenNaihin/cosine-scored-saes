# Experiment 43c: 4-architecture 50M at L9

Four SAE architectures (standard, adaptive cosine, per-feature cosine, NoC) are trained on Qwen3-8B layer 9 (mean activation norm ~58, norm/sqrt(d) = 0.91x) for 50M tokens with the saprmarks recipe (d_sae=65536, k=80, lr=5e-5, aux k-loss). At this shallow layer the initialization is nearly matched to activations, testing whether architecture differences vanish when the norm mismatch is minimal.

## Results

| Architecture | FVE | Dead % | Alive | cos>inner | scale_a |
|---|---|---|---|---|---|
| standard | 0.784 | 0.0% | 65,536 | 44/100 | -- |
| adaptive_l2 | 0.786 | 0.0% | 65,536 | 38/100 | 0.025 |
| perfeature_l2 | 0.782 | 0.0% | 65,536 | 37/100 | mean=-0.011 |
| no_C | 0.780 | 0.0% | 65,536 | 33/100 | -- |
