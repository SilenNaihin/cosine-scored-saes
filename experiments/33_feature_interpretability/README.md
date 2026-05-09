# Experiment 33: LLM Auto-Interp at 50M Tokens

This experiment measures feature interpretability rates using a describe-then-predict protocol with an LLM judge. 200 features per SAE are stratified-sampled (50 low-frequency, 100 medium, 50 high-frequency) from the standard and adaptive cosine SAEs trained at 50M tokens on Qwen3-8B layer 27. Each feature is shown 10 activating contexts for description, then the judge predicts activating tokens in 10 held-out contexts.

## Results

### Overall Interpretability

| Metric | Standard | Cosine |
|--------|----------|--------|
| Alive features | 3,308 | 8,830 |
| Features sampled | 200 | 200 |
| Interpretability rate (>=50% acc) | 40.0% | 37.0% |
| Mean prediction accuracy | 43.3% | 42.8% |
| Median prediction accuracy | 35.0% | 30.0% |
| Perfect scores (100%) | 30 | 26 |
| Zero scores (0%) | 33 | 27 |
| Est. total interpretable features | 1,323 | 3,267 |

Statistical comparison (rate difference):
- Two-sample t-test: p=0.877
- Mann-Whitney U: p=0.973
- Cohen's d: 0.016

### Frequency-Stratified Rates

| Band | Standard rate | Cosine rate | Standard mean acc | Cosine mean acc |
|------|--------------|-------------|-------------------|-----------------|
| Low (bottom 25%) | 58.0% | 48.0% | 0.576 | 0.505 |
| Medium (25-75%) | 40.0% | 40.0% | 0.429 | 0.473 |
| High (top 25%) | 22.0% | 20.0% | 0.299 | 0.260 |

### Accuracy Distribution

| Threshold | Standard >= | Cosine >= |
|-----------|-------------|-----------|
| 10% | 167 (84%) | 173 (86%) |
| 30% | 121 (60%) | 119 (60%) |
| 50% | 80 (40%) | 74 (37%) |
| 70% | 59 (30%) | 54 (27%) |
| 90% | 39 (20%) | 43 (22%) |
| 100% | 30 (15%) | 26 (13%) |
