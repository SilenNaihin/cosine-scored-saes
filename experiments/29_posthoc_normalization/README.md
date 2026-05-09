# Experiment 29: Post-hoc Cosine on Trained Standard SAE

This experiment tests whether applying cosine similarity encoding at inference time to a trained standard SAE can recover cosine-level performance. A standard SAE's weights (from 50M-token training on Qwen3-8B) are used with a post-hoc cosine encoder (replacing inner product with scale * cos_sim), with the global scale optimized via grid search. Evaluation covers FVE, dead features, L0, and feature overlap at layers 9, 18, and 27.

## Results

### Reconstruction (FVE)

| Layer | Standard | PostHoc Cosine | Adaptive_l2 |
|-------|----------|----------------|-------------|
| L9 | 0.679 | 0.486 | 0.703 |
| L18 | 0.638 | 0.306 | 0.636 |
| L27 | 0.659 | 0.478 | 0.739 |

### Dead Features and L0

| Layer | Standard dead% | PostHoc dead% | Adaptive dead% |
|-------|---------------|---------------|----------------|
| L9 | 62.1% | 15.3% | 50.9% |
| L18 | 83.1% | 30.3% | 86.8% |
| L27 | 79.0% | 6.5% | 30.7% |

| Layer | Standard L0 | PostHoc L0 | Adaptive L0 |
|-------|-------------|------------|-------------|
| L9 | 79 | 302 | 79 |
| L18 | 80 | 319 | 80 |
| L27 | 80 | 501 | 80 |

### Ablation (cos>inner)

| Layer | Standard | PostHoc Cosine | Adaptive_l2 |
|-------|----------|----------------|-------------|
| L9 | 18/30 (60%) | 23/30 (77%) | 19/30 (63%) |
| L18 | 23/30 (77%) | 25/30 (83%) | 26/30 (87%) |
| L27 | 25/30 (83%) | 22/30 (73%) | 17/30 (57%) |

### Feature Overlap (Standard vs PostHoc, same weights)

| Layer | Std alive | PostHoc alive | Both alive | Jaccard |
|-------|-----------|---------------|------------|---------|
| L18 | 2,697 | 9,658 | 2,667 | 0.275 |
| L27 | 3,369 | 15,022 | 3,364 | 0.224 |

### Feature Overlap (Standard vs Adaptive_l2, different training)

| Layer | Std alive | Ada alive | Both alive | Jaccard | Decoder cos (mean) |
|-------|-----------|-----------|------------|---------|-------------------|
| L18 | 2,697 | 2,073 | 458 | 0.106 | 0.576 |
| L27 | 3,369 | 11,206 | 2,978 | 0.257 | 0.686 |
