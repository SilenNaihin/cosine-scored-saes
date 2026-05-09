# Experiment 25c: Mistral-7B RMSNorm Confirmation

This experiment trains standard, cosine, and adaptive SAEs on Mistral-7B-v0.1 (RMSNorm) at layers 8, 16, and 24 to test whether the cosine scoring advantage generalizes to a second RMSNorm model. Training uses BatchTopK (k=80), d_sae=4x d_model, 5M tokens per layer, with evaluation including FVE, dead feature rates, norm invariance, and the cos>inner ablation diagnostic.

## Results

### cos>inner Diagnostic (min-max across variants)

| Layer | Range |
|-------|-------|
| L8 | 47-60% |
| L16 | 53-67% |
| L24 | 73-77% |

### Reconstruction (FVE)

| Layer | Standard | Cosine | Adaptive |
|-------|----------|--------|----------|
| L8 | 0.567 | -0.000 | -0.000 |
| L16 | 0.577 | 0.025 | 0.013 |
| L24 | 0.554 | 0.422 | 0.423 |

### Dead Features

| Layer | Standard | Cosine | Adaptive |
|-------|----------|--------|----------|
| L8 | 54.0% | 97.6% | 97.6% |
| L16 | 57.5% | 0.6% | 2.9% |
| L24 | 52.7% | 42.6% | 41.0% |

### Activation Norm Scale

| Layer | Mean norm | sqrt(d_model) | Ratio |
|-------|-----------|---------------|-------|
| L8 | 6.3 | 64 | 0.10x |
| L16 | ~12 | 64 | ~0.19x |
| L24 | 18.0 | 64 | 0.28x |

### Learned scale_a (Adaptive)

| L8 | L16 | L24 |
|----|-----|-----|
| -0.002 | -0.002 | -0.007 |
