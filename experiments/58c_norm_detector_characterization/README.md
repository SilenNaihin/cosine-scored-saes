# Experiment 58c: Norm-detector feature characterization

This experiment characterizes the unique features of each architecture by analyzing what input norm quartile they activate on. Features are classified as matched (correlation >= 0.7, decoder cos_sim > 0.7) or unique to one architecture, then their activation patterns are stratified by token norm quartile.

## Results

### Feature Counts (100K tokens, dual filter)

| | Standard | Cosine |
|---|---------|--------|
| Alive | 64,450 | 64,672 |
| Matched | 8,661 | 9,506 |
| Unique | 55,789 | 55,166 |

### Norm Quartile Distribution of Activations

| Feature Set | Q1 | Q2 | Q3 | Q4 (highest) | Mean Activation Norm |
|-------------|-----|-----|-----|------|-----|
| Standard-unique | 4.0% | 4.6% | 5.1% | 86.3% | 9,689 |
| Cosine-unique | 12.1% | 14.4% | 16.7% | 56.9% | 4,432 |
| Cosine-matched | 14.6% | 17.7% | 20.2% | 47.5% | 2,837 |
| Standard-matched | 20.4% | 23.5% | 25.8% | 30.3% | 270 |
| Uniform baseline | 25% | 25% | 25% | 25% | 203 |

### Additional Characteristics

- Token norm distribution: Q1=94.1, Q2=100.3, Q3=107.4, mean=203.5 (max=16,851)
- Standard-unique mean frequency: 0.54% (vs cosine-unique 0.17%)
- On tokens where cosine-unique features fire: standard fires 328 features/token vs cosine 122 features/token
