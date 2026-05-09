# Experiment 58b: Matched-feature subset probing

This experiment isolates the "discovery" vs "separability" components of the cosine sparse probing advantage by masking both SAEs to only their 8,661 matched features (correlation >= 0.7 and decoder cosine similarity > 0.7) and running sparse probing. If the gap persists on matched features, it indicates separability; if it disappears, it indicates pure feature discovery.

## Results

### Sparse Probing Accuracy

| Condition | top-1 | top-2 | top-5 | full |
|-----------|-------|-------|-------|------|
| Standard (matched only, 8,661 features) | 0.6674 | 0.7035 | 0.7837 | 0.9396 |
| Cosine (matched only, 8,661 features) | 0.6869 | 0.7380 | 0.8108 | 0.9387 |
| Standard (all 64,450 alive) | 0.6632 | 0.7328 | 0.7831 | 0.9466 |
| Cosine (all 64,672 alive) | 0.8119 | 0.8572 | 0.8888 | 0.9588 |

### Gap Decomposition

| k | Full gap | Matched gap | Discovery % | Separability % |
|---|----------|-------------|-------------|----------------|
| top-1 | +14.87pp | +1.95pp | 87% | 13% |
| top-2 | +12.44pp | +3.46pp | 72% | 28% |
| top-5 | +10.56pp | +2.70pp | 74% | 26% |

### Per-Dataset Highlights (top-5)

| Dataset | Matched gap | Full gap |
|---------|-------------|----------|
| github-code | +23.2pp | +30.1pp |
| europarl | +2.3pp | +19.3pp |
| bias_in_bios set2 | +1.3pp | +12.4pp |
| amazon_reviews_sentiment | -2.8pp | -0.1pp |
