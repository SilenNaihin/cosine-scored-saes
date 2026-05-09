# Experiment 57c: Gemma-2-2B sparse probing replication

This experiment replicates the feature overlap, bootstrap sparse probing, and feature steering analyses on Gemma-2-2B (Layer 13, d_model=2304, d_sae=9216, k=80, 50M tokens) to test whether the cosine encoding advantage generalizes across model families and scales.

## Results

### Feature Overlap

| Metric | Gemma-2-2B | Qwen3-8B |
|--------|-----------|----------|
| Standard alive | 9,216 (100%) | 17,288 (26.4%) |
| Cosine alive | 9,216 (100%) | 18,570 (28.3%) |
| std-to-cos strong match | 66.1% (6,088) | 41.1% (7,114) |
| std-to-cos unmatched | 11.1% (1,027) | 17.1% (2,958) |
| cos-to-std strong match | 16.3% (1,502) | 38.5% (7,147) |
| cos-to-std unmatched | 33.3% (3,066) | 21.0% (3,908) |

### Bootstrap Sparse Probing (top-1)

| Metric | Standard | perfeature_l2 | Gap |
|--------|----------|--------------|-----|
| Mean | 0.7537 | 0.7879 | +3.42pp |
| Std | 0.0020 | 0.0020 | 0.29pp |
| t-statistic | | | 12 sigma |

### Feature Steering (KL divergence)

| Factor | Standard KL | Cosine KL | Ratio (std/cos) |
|--------|------------|-----------|-------|
| x0.0 | 0.0010 | 0.0012 | 0.83x |
| x0.5 | 0.0006 | 0.0007 | 0.85x |
| x2.0 | 0.0010 | 0.0012 | 0.81x |
| x5.0 | 0.0084 | 0.0125 | 0.67x |
