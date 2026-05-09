# Experiment 55b: Bootstrap sparse probing (5 seeds)

SAEBench sparse probing is run 5 times with different random seeds (42, 123, 456, 789, 1337) for probe training on the same SAE checkpoints (standard and perfeature_l2 at L18, Qwen3-8B, 500M tokens). This measures evaluation variance from probe initialization, not SAE training variance.

## Results

Per-seed top-1 accuracy:

| Seed | Standard | perfeature_l2 | Gap |
|------|----------|---------------|-----|
| 42 | 0.6817 | 0.8128 | 13.11pp |
| 123 | 0.6737 | 0.8136 | 13.99pp |
| 456 | 0.6630 | 0.8129 | 14.98pp |
| 789 | 0.6765 | 0.8137 | 13.73pp |
| 1337 | 0.6758 | 0.8108 | 13.50pp |

Aggregate statistics:

| Metric | Standard | perfeature_l2 | Gap |
|--------|----------|---------------|-----|
| Mean | 0.6741 | 0.8128 | 13.86pp |
| Std | 0.0061 | 0.0010 | 0.63pp |
| Min | 0.6630 | 0.8108 | 13.11pp |
| Max | 0.6817 | 0.8137 | 14.98pp |
