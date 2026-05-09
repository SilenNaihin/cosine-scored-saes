# Experiment 57: Dictionary size x model size 3x3 matrix

This experiment sweeps expansion ratio (4x/8x/16x) across three Qwen model sizes (1.7B/4B/8B) to disentangle whether the cosine sparse probing advantage scales with dictionary size, model size, or both. All cells use the same training recipe (k=80, 50M tokens, aux-k) with standard and adaptive_l2 architectures.

## Results

### Sparse Probing Top-1 Accuracy

| Model | Expansion | d_sae | Adaptive | Standard | Gap (pp) |
|-------|-----------|-------|----------|----------|----------|
| Qwen3-1.7B | 4x | 8,192 | 0.7727 | 0.7237 | +4.9 |
| Qwen3-1.7B | 8x | 16,384 | 0.7885 | 0.7231 | +6.5 |
| Qwen3-1.7B | 16x | 32,768 | 0.7563 | 0.6883 | +6.8 |
| Qwen3-4B | 4x | 10,240 | 0.7712 | 0.6425 | +12.9 |
| Qwen3-4B | 8x | 20,480 | 0.7743 | 0.6510 | +12.3 |
| Qwen3-4B | 16x | 40,960 | 0.8016 | 0.7071 | +9.5 |
| Qwen3-8B | 4x | 16,384 | 0.7995 | 0.6924 | +10.7 |
| Qwen3-8B | 8x | 32,768 | 0.7779 | 0.7049 | +7.3 |
| Qwen3-8B | 16x | 65,536 | 0.800 | ~0.660 | ~+14 |

### Gap Summary (pp)

|  | 4x | 8x | 16x | Row Mean |
|--|----|----|-----|----------|
| 1.7B | +4.9 | +6.5 | +6.8 | +6.1 |
| 4B | +12.9 | +12.3 | +9.5 | +11.6 |
| 8B | +10.7 | +7.3 | +14.0 | +10.7 |
| Col Mean | +9.5 | +8.7 | +10.1 | |

### FVE (0% dead features across all cells)

| Model | Expansion | Adaptive FVE | Standard FVE |
|-------|-----------|-------------|-------------|
| 1.7B | 4x | 0.721 | 0.715 |
| 1.7B | 8x | 0.735 | 0.727 |
| 1.7B | 16x | 0.744 | 0.735 |
| 4B | 4x | 0.734 | 0.734 |
| 4B | 8x | 0.737 | 0.735 |
| 4B | 16x | 0.746 | 0.744 |
| 8B | 4x | 0.706 | 0.704 |
| 8B | 8x | 0.716 | 0.715 |

### LLM Baselines (sparse probing top-1 on raw model activations)

| Model | LLM Baseline |
|-------|-------------|
| Qwen3-1.7B | 0.622 |
| Qwen3-4B | 0.603 |
| Qwen3-8B | 0.600 |
