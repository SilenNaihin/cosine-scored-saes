# Experiment 43d: 4-architecture 50M at L18

Four SAE architectures (standard, adaptive cosine, per-feature cosine, NoC) are trained on Qwen3-8B layer 18 (mean activation norm ~98, norm/sqrt(d) = 1.53x) for 50M tokens with the saprmarks recipe (d_sae=65536, k=80, lr=5e-5, aux k-loss). This layer represents the transition zone between shallow (L9, where architectures are equivalent) and deep (L27, where they diverge dramatically).

## Results

| Architecture | FVE | Dead % | Alive | cos>inner | cos_kl | inner_kl | scale_a |
|---|---|---|---|---|---|---|---|
| standard | 0.723 | 0.02% | 65,523 | 52/100 | 0.287 | 0.288 | -- |
| adaptive_l2 | 0.724 | 0.01% | 65,532 | 50/100 | 0.270 | 0.273 | 0.080 |
| perfeature_l2 | 0.726 | 0.0% | 65,533 | 57/100 | 0.294 | 0.292 | mean=-0.0004 |
| no_C | 0.707 | 3.0% | 63,548 | 52/100 | 0.286 | 0.286 | -- |
