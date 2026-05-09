# Experiment 27: Norm-Adaptive Initialization

This experiment tests whether initializing the cosine encoder's scale parameter from observed activation norms (scale_b = log(mean(||x||))) rather than the default sqrt(d_model) fixes performance degradation at layers where activation norms diverge significantly from sqrt(d). Four variants are trained on Qwen3-8B at layers 9, 18, and 27 for 5M tokens each with BatchTopK (k=80, d_sae=16384).

## Results

### Activation Norm Statistics

| Layer | mean(||x||) | sqrt(d_model) | Ratio |
|-------|-------------|---------------|-------|
| L9 | 58.1 | 64.0 | 0.91 |
| L18 | 99.9 | 64.0 | 1.56 |
| L27 | 407.2 | 64.0 | 6.36 |

### Reconstruction (FVE)

| Layer | Standard | Cosine (sqrt(d)) | Cosine (adaptive) | Adaptive_l2 |
|-------|----------|------------------|-------------------|-------------|
| L9 | 0.599 | 0.626 | 0.627 | 0.625 |
| L18 | 0.533 | 0.555 | 0.558 | 0.559 |
| L27 | 0.564 | 0.446 | 0.583 | 0.581 |

### Dead Features

| Layer | Standard | Cosine (sqrt(d)) | Cosine (adaptive) | Adaptive_l2 |
|-------|----------|------------------|-------------------|-------------|
| L9 | 54.2% | 49.3% | 49.8% | 53.1% |
| L18 | 76.6% | 72.0% | 79.1% | 79.7% |
| L27 | 72.7% | 74.4% | 57.3% | 59.9% |

### Learned Scale (final exp(scale_b))

| Layer | mean(||x||) | sqrt(d) init final | Adaptive init final |
|-------|-------------|-------------------|---------------------|
| L9 | 58.1 | 68.2 | 64.7 |
| L18 | 99.9 | 73.6 | 112.2 |
| L27 | 407.2 | 70.5 | 441.2 |

### Ablation (cos>inner)

| Layer | Standard | Cosine (sqrt(d)) | Cosine (adaptive) | Adaptive_l2 |
|-------|----------|------------------|-------------------|-------------|
| L9 | 26/30 (87%) | 23/30 (77%) | 20/30 (67%) | 23/30 (77%) |
| L18 | 25/30 (83%) | 23/30 (77%) | 22/30 (73%) | 24/30 (80%) |
| L27 | 21/30 (70%) | 25/30 (83%) | 24/30 (80%) | 25/30 (83%) |

### Learned scale_a (Adaptive_l2)

| L9 | L18 | L27 |
|----|-----|-----|
| 0.065 | 0.070 | 0.040 |
