# Experiment 44: Per-quartile FVE across architectures

Five targeted analyses on the 500M L18 checkpoints characterize how the four SAE architectures differ beyond reconstruction metrics: compute overhead, decoder direction overlap, norm-stratified FVE, feature activation overlap (Jaccard), and feature steering. Eval tokens are split into four quartiles by activation norm to test whether architectures handle high-norm tokens differently.

## Results

### Compute overhead

| Architecture | Forward (ms) | Backward (ms) | Total (ms) | Relative |
|---|---|---|---|---|
| Standard | 51.1 | 99.9 | 151.0 | 1.00x |
| Adaptive L2 | 53.3 | 109.4 | 162.7 | 1.08x |
| Per-Feature L2 | 54.6 | 111.1 | 165.6 | 1.10x |
| NoC | 54.1 | 118.6 | 172.7 | 1.14x |

### Decoder direction overlap

| Pair | Mutual best matches | Mutual % | >0.90 cosine | >0.95 cosine |
|---|---|---|---|---|
| Standard vs Adaptive | 49,673 | 75.8% | 19.1% | 8.0% |
| Standard vs PerFeature | 49,755 | 75.9% | 20.3% | 9.1% |
| Standard vs NoC | 47,769 | 72.9% | 15.6% | 6.1% |
| Adaptive vs PerFeature | 50,985 | 77.8% | 23.3% | 10.5% |
| Adaptive vs NoC | 47,778 | 72.9% | 15.5% | 6.0% |
| PerFeature vs NoC | 47,596 | 72.6% | 15.5% | 5.7% |

### Norm-stratified FVE

| Architecture | Q1 FVE (low norm) | Q2 FVE | Q3 FVE | Q4 FVE (high norm) |
|---|---|---|---|---|
| Standard | 0.738 | 0.763 | 0.778 | -197.3 |
| Adaptive L2 | 0.732 | 0.760 | 0.777 | 0.329 |
| Per-Feature L2 | 0.733 | 0.762 | 0.778 | 0.252 |
| NoC | 0.738 | 0.762 | 0.775 | -0.200 |

### Feature activation overlap (Jaccard, top 1000 features)

| Pair | Mean Jaccard | Median Jaccard |
|---|---|---|
| Standard vs Adaptive | 0.066 | 0.064 |
| Standard vs PerFeature | 0.059 | 0.057 |
| Standard vs NoC | 0.059 | 0.058 |
| Adaptive vs PerFeature | 0.174 | 0.172 |
| Adaptive vs NoC | 0.111 | 0.109 |
| PerFeature vs NoC | 0.091 | 0.090 |

### Feature steering (KL divergence from unsteered baseline)

| Architecture | KL at 0.5x | KL at 1.0x | KL at 2.0x | KL at 5.0x |
|---|---|---|---|---|
| Standard | 0.0005 | 0.0008 | 0.0004 | 0.0013 |
| Adaptive L2 | 0.0007 | 0.0007 | 0.0008 | 0.0010 |
| Per-Feature L2 | 0.0003 | 0.0010 | 0.0005 | 0.0012 |
| NoC | 0.0006 | 0.0005 | 0.0013 | 0.0009 |
