# Experiment 31: Synthetic Norm Noise Injection

This experiment injects random per-token magnitude scaling (Uniform(0.5, 2.0)) during training and evaluates on clean data, testing whether the cosine encoder's norm-invariance provides robustness to magnitude perturbations. Standard and adaptive SAEs are trained in both clean and noised conditions on Qwen3-8B at layers 9, 18, and 27 for 5M tokens each.

## Results

### FVE Degradation (noised training vs clean training, evaluated on clean data)

| Layer | Standard delta | Cosine delta |
|-------|---------------|--------------|
| L9 | -0.032 (-5.3%) | -0.058 (-9.3%) |
| L18 | -0.034 (-6.3%) | -0.055 (-9.7%) |
| L27 | -0.018 (-3.2%) | -0.045 (-7.7%) |

### Dead Feature Change (noised - clean)

| Layer | Standard delta | Cosine delta |
|-------|---------------|--------------|
| L9 | +5.6pp | -3.0pp |
| L18 | +8.6pp | -5.2pp |
| L27 | -2.8pp | -5.4pp |

### Ablation (cos>inner)

| Layer | Std clean | Std noised | Cos clean | Cos noised |
|-------|-----------|------------|-----------|------------|
| L9 | 26/30 (87%) | 23/30 (77%) | 23/30 (77%) | 23/30 (77%) |
| L18 | 25/30 (83%) | 22/30 (73%) | 24/30 (80%) | 21/30 (70%) |
| L27 | 21/30 (70%) | 20/30 (67%) | 25/30 (83%) | 22/30 (73%) |

### Feature Overlap (clean vs noised, same architecture)

| Layer | Standard Jaccard | Cosine Jaccard |
|-------|-----------------|----------------|
| L9 | 0.450 | 0.466 |
| L18 | 0.279 | 0.241 |
| L27 | 0.320 | 0.225 |

### Adaptive Scale Parameter (L27)

| Condition | scale_a |
|-----------|---------|
| Clean | ~0.10 |
| Noised | 0.14 |
