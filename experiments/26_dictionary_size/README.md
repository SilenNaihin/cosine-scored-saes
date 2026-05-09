# Experiment 26: 3x Dictionary Size Control

This experiment tests whether giving a standard SAE a 3x larger dictionary (49,152 features) can match the cosine SAE's alive feature count and FVE at the same 16,384 dictionary. Two conditions are tested: 3x dictionary at matched L0 (k=80) and 3x dictionary with proportional L0 (k=240). All variants train on Qwen3-8B layer 27 for 50M tokens with BatchTopK.

## Results

### Summary

| Variant | d_sae | k | FVE | Dead% | Alive | cos>inner |
|---------|-------|---|-----|-------|-------|-----------|
| standard_16k | 16,384 | 80 | 0.658 | 77.4% | 3,709 | 79/100 |
| adaptive_16k | 16,384 | 80 | 0.737 | 28.2% | 11,768 | 84/100 |
| standard_49k | 49,152 | 80 | 0.673 | 89.4% | 5,226 | 86/100 |
| standard_49k_k240 | 49,152 | 240 | 0.751 | 31.2% | 33,811 | 76/100 |

### Matched L0 Comparison (standard_16k vs standard_49k)

| Metric | standard_16k | standard_49k | Change |
|--------|--------------|--------------|--------|
| Alive features | 3,709 | 5,226 | +40% |
| Dead% | 77.4% | 89.4% | +12pp |
| FVE | 0.658 | 0.673 | +1.5pp |
| Alive/d_sae | 22.6% | 10.6% | halved |

### Efficiency (adaptive_16k vs standard_49k_k240)

| Metric | adaptive_16k | standard_49k_k240 |
|--------|--------------|-------------------|
| Parameters | 134M | 402M |
| FVE | 0.737 | 0.751 |
| Alive/d_sae | 71.8% | 68.8% |
| Training time | 61 min | 91 min |

### Norm Invariance (output scaling ratios)

| Variant | 0.5x | 2x | 5x |
|---------|------|-----|-----|
| standard_16k | 0.500 | 2.000 | 4.999 |
| adaptive_16k | 0.857 | 1.161 | 1.408 |
| standard_49k | 0.500 | 2.000 | 4.998 |
| standard_49k_k240 | 0.560 | 1.896 | 4.590 |

### Convergence (FVE at training progress %)

| Variant | 20% | 40% | 60% | 80% | 100% |
|---------|-----|-----|-----|-----|------|
| standard_16k | 0.585 | 0.635 | 0.653 | 0.661 | 0.663 |
| adaptive_16k | 0.690 | 0.724 | 0.734 | 0.740 | 0.741 |
| standard_49k | 0.601 | 0.644 | 0.665 | 0.674 | 0.677 |
| standard_49k_k240 | 0.688 | 0.684 | 0.548 | 0.752 | 0.759 |
