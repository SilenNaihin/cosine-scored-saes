# Experiment 56b: Feature overlap analysis

This experiment measures feature overlap between standard and cosine (perfeature_l2) SAEs by computing activation correlations on 100K FineWeb tokens. Features are classified as strongly matched (correlation >= 0.7), weakly matched (0.3-0.7), or unmatched (< 0.3), and matched pairs are further validated by decoder cosine similarity.

## Results

### Alive Feature Counts

| SAE | Alive | % of 65536 |
|-----|-------|-----------|
| perfeature_l2 | 18,570 | 28.3% |
| standard | 17,288 | 26.4% |

### Feature Matching (standard to perfeature_l2)

| Category | Count | % of standard alive |
|----------|-------|-------------------|
| Strong match (corr >= 0.7) | 7,114 | 41.1% |
| Weak match (0.3-0.7) | 7,216 | 41.7% |
| Unmatched (corr < 0.3) | 2,958 | 17.1% |

### Feature Matching (perfeature_l2 to standard)

| Category | Count | % of perfeature_l2 alive |
|----------|-------|------------------------|
| Strong match (corr >= 0.7) | 7,147 | 38.5% |
| Weak match (0.3-0.7) | 7,515 | 40.5% |
| Unmatched (corr < 0.3) | 3,908 | 21.0% |

### Decoder Similarity for Strongly-Matched Pairs (7,114 pairs)

| Metric | Value |
|--------|-------|
| Mean decoder cosine similarity | 0.913 |
| Median decoder cosine similarity | 0.943 |
| Pairs with decoder cos_sim > 0.9 | 69.6% |
| Pairs with decoder cos_sim > 0.7 | 96.5% |

### Unmatched Feature Characteristics

| | Unmatched Freq Mean | Matched Freq Mean |
|---|-----------|---------|
| Standard | 0.0025 | 0.0052 |
| perfeature_l2 | 0.0025 | 0.0049 |
