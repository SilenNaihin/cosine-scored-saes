# Experiment 17: 50M token production-scale comparison

The three best SAE architectures (standard, adaptive_l2, perfeature_l2) are trained at 10x the prior token budget (50M tokens per layer vs 5M) to test whether findings from toy-scale experiments hold with more data. Evaluation uses 100 features x 200 ablation samples per condition, and mid-training checkpoints are saved at 20/40/60/80/100% of training to track convergence.

## Results

Model: Qwen3-8B, d_sae=16384, k=80, 50M tokens/layer, 1M eval tokens, layers 9/18/27.

### Reconstruction

| Variant | L9 FVE | L18 FVE | L27 FVE | L9 cos | L18 cos | L27 cos | L9 L2 | L18 L2 | L27 L2 |
|---------|--------|---------|---------|--------|---------|---------|-------|--------|--------|
| standard | 0.679 | 0.638 | 0.657 | 0.883 | 0.887 | 0.897 | 771.6 | 2155.4 | 31620.8 |
| adaptive_l2 | 0.702 | 0.636 | 0.737 | 0.892 | 0.886 | 0.923 | 714.9 | 2167.4 | 24246.4 |
| perfeature_l2 | 0.714 | 0.654 | 0.733 | 0.896 | 0.892 | 0.921 | 687.8 | 2056.9 | 24703.5 |

### FVE improvement from 5M to 50M tokens

| Variant | L9 @5M | L9 @50M | L27 @5M | L27 @50M |
|---------|--------|---------|---------|----------|
| standard | 0.601 | 0.679 (+0.078) | 0.564 | 0.657 (+0.093) |
| adaptive_l2 | 0.623 | 0.702 (+0.079) | 0.534 | 0.737 (+0.203) |
| perfeature_l2 | 0.626 | 0.714 (+0.088) | 0.530 | 0.733 (+0.203) |

### Dead features

| Variant | L9 dead | L9 alive | L18 dead | L18 alive | L27 dead | L27 alive |
|---------|---------|----------|----------|-----------|----------|-----------|
| standard | 61.0% | 6,183 | 82.8% | 2,738 | 78.3% | 3,414 |
| adaptive_l2 | 49.0% | 7,950 | 85.3% | 2,176 | 29.7% | 11,349 |
| perfeature_l2 | 41.1% | 9,371 | 72.4% | 4,227 | 28.1% | 11,731 |

### Dead feature change from 5M to 50M tokens

| Variant | L9 @5M | L9 @50M | L27 @5M | L27 @50M |
|---------|--------|---------|---------|----------|
| standard | 71.5% | 61.0% | 83.6% | 78.3% |
| adaptive_l2 | 66.9% | 49.0% | 75.0% | 29.7% |
| perfeature_l2 | 65.2% | 41.1% | 79.6% | 28.1% |

### Ablation correlations (100 features x 200 samples)

| Variant | L9 cos->KL | L9 inner->KL | L9 SAE->KL | L18 cos->KL | L18 inner->KL | L18 SAE->KL | L27 cos->KL | L27 inner->KL | L27 SAE->KL |
|---------|------------|--------------|------------|-------------|---------------|-------------|-------------|---------------|-------------|
| standard | 0.275 | 0.255 | 0.208 | 0.396 | 0.374 | 0.335 | 0.421 | 0.407 | 0.385 |
| adaptive_l2 | 0.226 | 0.200 | 0.177 | 0.373 | 0.347 | 0.311 | 0.394 | 0.380 | 0.330 |
| perfeature_l2 | 0.292 | 0.273 | 0.227 | 0.402 | 0.379 | 0.354 | 0.415 | 0.396 | 0.360 |

### cos>inner win rates (n=100 features)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 87/100 | 78/100 | 73/100 |
| adaptive_l2 | 90/100 | 84/100 | 79/100 |
| perfeature_l2 | 86/100 | 81/100 | 80/100 |

### SAE>inner win rates (n=100 features)

| Variant | L9 | L18 | L27 |
|---------|------|------|------|
| standard | 23/100 | 24/100 | 41/100 |
| adaptive_l2 | 40/100 | 32/100 | 24/100 |
| perfeature_l2 | 30/100 | 35/100 | 35/100 |

### Norm invariance (2x input scaling)

| Variant | L9 ratio | L18 ratio | L27 ratio | L9 agree | L18 agree | L27 agree |
|---------|----------|-----------|-----------|----------|-----------|-----------|
| standard | 1.991 | 1.995 | 2.000 | 99.5% | 99.4% | 99.4% |
| adaptive_l2 | 1.050 | 1.117 | 1.162 | 100% | 99.9% | 99.9% |
| perfeature_l2 | 1.018 | 1.092 | 1.147 | 100% | 100% | 99.9% |

### Norm invariance (5x input scaling)

| Variant | L9 ratio | L18 ratio | L27 ratio |
|---------|----------|-----------|-----------|
| standard | 4.965 | 4.978 | 5.000 |
| adaptive_l2 | 1.107 | 1.279 | 1.412 |
| perfeature_l2 | 1.039 | 1.214 | 1.375 |

### Global scale_a (adaptive_l2)

| Layer | Learned a | a @5M (exp12) |
|-------|-----------|---------------|
| L9 | 0.048 | 0.044 |
| L18 | 0.138 | 0.103 |
| L27 | 0.208 | 0.103 |

### Per-feature scale_a distribution (perfeature_l2)

| Layer | Mean | Std | Median | 5th-95th pctile |
|-------|------|------|--------|-----------------|
| L9 | -0.015 | 0.031 | -0.013 | [-0.048, 0.016] |
| L18 | 0.009 | 0.032 | -0.006 | [-0.012, 0.086] |
| L27 | 0.112 | 0.079 | 0.154 | [-0.006, 0.195] |

### Per-feature scale_a categories

| Layer | Near-zero (|a|<0.05) | Low (0.05-0.2) | Medium (0.2-0.5) | High (>0.5) |
|-------|----------------------|----------------|-------------------|-------------|
| L9 | 94.8% | 5.2% | 0.0% | 0.0% |
| L18 | 87.1% | 12.9% | 0.0% | 0.0% |
| L27 | 31.2% | 65.2% | 3.6% | 0.0% |

### Convergence trajectory (FVE at 20% vs 100% training)

| Variant | L9 @20% | L9 @100% | L18 @20% | L18 @100% | L27 @20% | L27 @100% |
|---------|---------|----------|----------|-----------|----------|-----------|
| standard | 0.633 | 0.679 | 0.574 | 0.638 | 0.586 | 0.657 |
| adaptive_l2 | 0.664 | 0.702 | 0.600 | 0.636 | 0.688 | 0.737 |
| perfeature_l2 | 0.682 | 0.714 | 0.602 | 0.654 | 0.687 | 0.733 |

### Convergence trajectory (dead features at 20% vs 100% training)

| Variant | L9 @20% | L9 @100% | L18 @20% | L18 @100% | L27 @20% | L27 @100% |
|---------|---------|----------|----------|-----------|----------|-----------|
| standard | 75.3% | 61.0% | 92.4% | 82.8% | 88.6% | 78.3% |
| adaptive_l2 | 65.8% | 49.0% | 89.8% | 85.3% | 47.4% | 29.7% |
| perfeature_l2 | 59.6% | 41.1% | 87.5% | 72.4% | 57.6% | 28.1% |
