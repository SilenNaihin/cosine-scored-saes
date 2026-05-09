# Experiment 43: 4-architecture 50M at L27

Four SAE architectures (standard, adaptive cosine, per-feature cosine, NoC) are trained on Qwen3-8B layer 27 (mean activation norm ~405) for 50M tokens with the saprmarks recipe (d_sae=65536, k=80, lr=5e-5, aux k-loss). This layer has a norm/sqrt(d) ratio of 6.33x, creating the most extreme initialization mismatch in the Qwen depth triptych.

## Results

| Architecture | FVE | Dead % | Alive | cos>inner | scale_a |
|---|---|---|---|---|---|
| standard | 0.765 | 0.01% | 65,531 | 67/100 | -- |
| adaptive_l2 | 0.770 | 0.05% | 65,502 | 67/100 | 0.257 |
| perfeature_l2 | 0.721 | 83.4% | 10,886 | 74/100 | mean=0.022 |
| no_C | 0.751 | 0.9% | 64,946 | 78/100 | -- |

### Per-feature collapse dynamics

| Step | Tokens | Dead features | Dead % |
|---|---|---|---|
| 4,500 | 9.2M | 0 | 0% |
| 5,000 | 10.2M | 0 | 0% |
| 5,500 | 11.3M | 43,098 | 65.8% |
| 6,000 | 12.3M | 52,728 | 80.4% |
| final | 50M | 54,650 | 83.4% |
