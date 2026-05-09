# Experiment 8: SAE-free direction vs norm patching

This experiment tests the Relative Norm Hypothesis directly on the residual stream without any SAE involvement. For each pair of token activations at a given layer, the vector is decomposed into direction and magnitude; one component is patched while the other is held fixed, and KL divergence from the unpatched output is measured. A supplementary scaling test perturbs individual tokens by multiplying norm (0.5x/2x/5x) or rotating direction to a random unit vector.

## Results

Model: Qwen3-8B (RMSNorm), layers 9/18/27, 500 random cross-prompt pairs per layer, 200 controlled pairs, 100 scaling-test tokens.

### Random pair patching

| Layer | n | dir_kl | norm_kl | dir > norm | median ratio | 95% CI (diff) |
|-------|---|--------|---------|------------|--------------|---------------|
| 9 | 500 | 6.5499 | 0.3644 | 497/500 (99%) | 86.7x | [5.9091, 6.4618] |
| 18 | 500 | 9.0966 | 0.2035 | 499/500 (100%) | 219.3x | [8.4918, 9.2944] |
| 27 | 500 | 15.4802 | 0.0454 | 500/500 (100%) | 2559.5x | [14.8264, 16.0433] |

### Dose-response correlations

| Layer | corr(cos_distance, dir_kl) | corr(|log_norm_ratio|, norm_kl) |
|-------|----------------------------|----------------------------------|
| 9 | 0.298 | 0.402 |
| 18 | 0.321 | 0.425 |
| 27 | 0.212 | 0.514 |

### Same-norm controlled pairs (cos(A,B) < 0.5, |norm_ratio - 1| < 0.1)

| Layer | n | dir_kl | norm_kl |
|-------|---|--------|---------|
| 9 | 200 | 6.6988 | 0.0516 |
| 18 | 200 | 9.3709 | 0.0264 |
| 27 | 200 | 16.2999 | 0.0065 |

### High-cosine controlled pairs (cos(A,B) > 0.7, norm ratio > 1.5x)

| Layer | n | norm_kl | dir_kl |
|-------|---|---------|--------|
| 27 | 6 | 0.4021 | 12.6375 |

### Scaling vs rotation (single-token test)

| Layer | n | scale_kl (avg) | rotation_kl | ratio |
|-------|---|----------------|-------------|-------|
| 9 | 100 | 2.9695 | 6.6482 | 2.2x |
| 18 | 100 | 3.0360 | 10.5580 | 3.5x |
| 27 | 100 | 1.1752 | 13.6706 | 11.6x |

### Per-scale breakdown

| Layer | 0.5x | 2.0x | 5.0x | rotation |
|-------|------|------|------|----------|
| 9 | 3.1063 | 1.6722 | 4.1300 | 6.6482 |
| 18 | 3.3978 | 1.6530 | 4.0573 | 10.5580 |
| 27 | 1.7127 | 0.4020 | 1.4109 | 13.6706 |
