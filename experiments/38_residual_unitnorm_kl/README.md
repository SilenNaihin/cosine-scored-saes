# Experiment 38: Residual unit-norm KL sweep across layers

Measures how much the model's output distribution changes when residual-stream magnitude is stripped at each layer by forcing tokens to unit norm. Applied to all 36 transformer layers of Qwen3-8B one layer at a time. Metric: masked token-level KL(original || intervened) averaged over valid positions, evaluated on FineWeb data (100k tokens, 112 batches).

## Results

### Depth Trend Summary

- Early-layer mean KL (L0-L11): 8.8217
- Mid-layer mean KL (L12-L23): 10.3918
- Late-layer mean KL (L24-L35): 10.8683
- Peak layer: L34, weighted mean KL 16.2137
- Correlation (layer index vs KL): 0.2866 (all layers), 0.6831 (excluding L35)
- L35 KL: 0.000336 (effectively zero due to final RMSNorm)

### Top 5 Layers by Weighted Mean KL

| Layer | Weighted mean KL |
|---|---|
| 34 | 16.2137 |
| 20 | 13.3320 |
| 28 | 13.1408 |
| 31 | 12.8470 |
| 27 | 12.7470 |

### Confidence Intervals (50k vs 100k token runs)

| Layer | 50k KL | 100k KL | 50k CI half-width | 100k CI half-width |
|---|---|---|---|---|
| 20 | 13.3607 | 13.3320 | 0.2817 | 0.1947 |
| 27 | 12.8020 | 12.7470 | 0.1404 | 0.0920 |
| 28 | 13.1568 | 13.1408 | 0.1395 | 0.0956 |
| 31 | 12.8594 | 12.8470 | 0.2861 | 0.1756 |
| 34 | 16.3070 | 16.2137 | 0.1686 | 0.1274 |

Mean 95% CI half-width across L0-L34: 0.1077 (100k run).
