# Experiment 2: Cosine vs inner product vs SAE vs norm for ablation prediction

This experiment measures which similarity metric best predicts the causal effect of feature ablation at layer 18 of Qwen3-8B. For 50 top-frequency SAE features, correlations between each metric and the SAE's own activation are computed. For 20 of those features, each metric is correlated with the post-RMSNorm representation change induced by ablating that feature direction from the residual stream (25k tokens from synthetic prompts).

## Results

Correlation with SAE activation (50 features):

| Measure | Mean correlation |
|---|---|
| cos(x, f) | 0.751 |
| inner product | 0.786 |
| norm | 0.165 |
| within-bin corr(norm, SAE) | 0.260 |

Correlation with ablation effect (20 features):

| Predictor | Mean correlation |
|---|---|
| cos(x, f) | 0.720 |
| inner product | 0.644 |
| SAE activation | 0.576 |
| norm | -0.028 |

- cos > inner for 17/20 features
- cos > norm for 20/20 features
- cos: mean=0.7199, std=0.1826
- inner: mean=0.6444, std=0.2105
- SAE: mean=0.5758, std=0.2194
- norm: mean=-0.0278, std=0.2049
