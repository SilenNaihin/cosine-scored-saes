# Experiment 28: Gradient Q4/Q1 Ratio Analysis

This experiment measures per-quartile encoder gradient norms during training to test whether inner-product gradients are dominated by high-norm tokens while cosine gradients are balanced. Standard and cosine SAEs are trained on Qwen3-8B layer 27 for 2M tokens, with W_enc gradient norms logged every 10 steps, separated by input norm quartile (Q1=lowest, Q4=highest).

## Results

### Activation Norm Distribution (L27)

| Stat | Value |
|------|-------|
| Mean | 407.1 |
| Q1 upper bound | 387.6 |
| Q2 upper bound | 411.1 |
| Q3 upper bound | 431.3 |
| Q4/Q1 boundary ratio | 1.11x |

### Per-Quartile W_enc Gradient Norms (mean across features and steps)

| Quartile | Standard | Cosine |
|----------|----------|--------|
| Q1 (low-norm) | 5.09 | 23.05 |
| Q2 | 3.98 | 13.70 |
| Q3 | 4.83 | 14.39 |
| Q4 (high-norm) | 7.87 | 23.73 |
| Q4/Q1 ratio | 1.55x | 1.03x |

### Per-Feature Gradient Domination Distribution

| Threshold | Standard | Cosine | Ratio |
|-----------|----------|--------|-------|
| % features with Q4/Q1 > 2x | 35.3% | 13.5% | 2.6x |
| % features with Q4/Q1 > 5x | 21.8% | 9.1% | 2.4x |
| % features with Q4/Q1 > 10x | 18.1% | 8.3% | 2.2x |

### Features Alive Per Quartile (out of 16,384)

| Quartile | Standard | Cosine |
|----------|----------|--------|
| Q1 (low-norm) | 3,187 (19.5%) | 3,761 (23.0%) |
| Q2 | 3,384 (20.7%) | 4,200 (25.6%) |
| Q3 | 3,572 (21.8%) | 4,421 (27.0%) |
| Q4 (high-norm) | 4,582 (28.0%) | 5,025 (30.7%) |

### Cross-Quartile Feature Specialization

| Category | Standard | Cosine |
|----------|----------|--------|
| Alive in both Q1+Q4 | 3,169 | 3,199 |
| Q4-only (alive in Q4, dead in Q1) | 1,413 | 1,826 |
| Q1-only (alive in Q1, dead in Q4) | 18 | 562 |
| Dead in both | 11,784 | 10,797 |
