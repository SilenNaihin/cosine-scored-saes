# Experiment 42c: NoC 500M Qwen training and SAEBench

Four SAE architectures (standard BatchTopK, adaptive cosine, per-feature cosine, and NoC) are trained on Qwen3-8B layer 18 for 500M tokens using an identical production recipe with auxiliary k-loss. SAEBench sparse probing evaluates feature quality beyond reconstruction metrics.

## Results

### Reconstruction quality (500M tokens, L18)

| Metric | Standard | Adaptive L2 | Per-Feature L2 | NoC |
|---|---|---|---|---|
| FVE | 0.770 | 0.769 | 0.771 | 0.767 |
| Dead % | 0.0% | 0.0% | 0.0% | 4.3% |
| Alive features | 65,529 | 65,535 | 65,536 | 62,707 |
| L0 | 80.0 | 80.1 | 80.0 | 80.0 |
| Cosine recon | 0.931 | 0.931 | 0.931 | 0.930 |

### RNH diagnostic (cos>inner causal prediction)

| Metric | Standard | Adaptive L2 | Per-Feature L2 | NoC |
|---|---|---|---|---|
| cos > inner | 62/100 | 63/100 | 62/100 | 69/100 |
| cos-KL corr | 0.075 | 0.066 | 0.088 | 0.042 |
| inner-KL corr | 0.070 | 0.062 | 0.086 | 0.037 |

### SAEBench core metrics

| Metric | Standard | Adaptive L2 | Per-Feature L2 | NoC |
|---|---|---|---|---|
| KL div score | 0.985 | 0.985 | 0.985 | 0.984 |
| CE loss score | 0.993 | 0.993 | 0.993 | 0.991 |
| FVE (SAEBench) | 0.707 | 0.726 | 0.728 | 0.723 |
| Cosine sim | 0.925 | 0.925 | 0.925 | 0.923 |
| L2 ratio | 0.927 | 0.923 | 0.924 | 0.941 |
| L0 | 86.7 | 85.6 | 84.5 | 85.2 |

### SAEBench sparse probing

| Metric | Standard | Adaptive L2 | Per-Feature L2 | NoC |
|---|---|---|---|---|
| SAE test accuracy | 0.944 | 0.957 | 0.959 | 0.914 |
| SAE top-1 | 0.667 | 0.800 | 0.815 | 0.783 |
| SAE top-2 | 0.731 | 0.853 | 0.853 | 0.798 |
| SAE top-5 | 0.789 | 0.891 | 0.883 | 0.805 |
| LLM test accuracy | 0.960 | 0.960 | 0.960 | 0.960 |
| LLM top-1 | 0.590 | 0.590 | 0.590 | 0.590 |

### Per-dataset sparse probing (top-1 accuracy)

| Dataset | Standard | Adaptive | PerFeature | NoC |
|---|---|---|---|---|
| europarl (languages) | 0.644 | 0.986 | 0.992 | 0.978 |
| github-code (programming) | 0.515 | 0.700 | 0.813 | 0.784 |
| bias_in_bios set1 | 0.669 | 0.798 | 0.862 | 0.786 |
| bias_in_bios set2 | 0.559 | 0.762 | 0.758 | 0.766 |
| bias_in_bios set3 | 0.612 | 0.778 | 0.766 | 0.763 |
| amazon_reviews (categories) | 0.712 | 0.713 | 0.715 | 0.675 |
| amazon_sentiment | 0.915 | 0.924 | 0.880 | 0.795 |
| ag_news (topics) | 0.705 | 0.736 | 0.735 | 0.720 |

### Learned scale parameters

| Architecture | scale_a |
|---|---|
| Standard | N/A (inner product) |
| Adaptive L2 | 0.258 (global) |
| Per-Feature L2 | mean=0.076, 23% near zero |
| NoC | N/A (fixed at 0) |

### NoC training dynamics

| Step | Tokens | NoC FVE | NoC dead% | NoC n_dead |
|---|---|---|---|---|
| 1,000 | 2M | 0.325 | 0.0% | 0 |
| 5,000 | 10M | 0.609 | 0.0% | 0 |
| 10,000 | 20M | 0.672 | 0.0% | 0 |
| 25,000 | 51M | 0.723 | 1.6% | 1,045 |
| 50,000 | 102M | 0.753 | 4.2% | 2,758 |
| 100,000 | 205M | 0.768 | 4.2% | 2,732 |
| 150,000 | 307M | 0.772 | 4.0% | 2,653 |
| 200,000 | 410M | 0.773 | 4.0% | 2,611 |
| 244,140 | 500M | 0.774 | 3.9% | 2,549 |
