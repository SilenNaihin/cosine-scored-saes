# Experiment 57a: TPP on matched feature pairs

This experiment attempts to run per-feature TPP (Targeted Perturbation Probing) restricted to the 7,114 matched feature pairs identified in exp56b, to test whether the TPP power gap persists when comparing equivalent features across architectures. The experiment failed due to insufficient token count for matching and inability to restrict SAEBench's TPP pipeline to a feature subset.

## Results

### Feature Matching Quality (50K tokens vs required 100K)

| Metric | exp57a (50K tokens) | exp56b (100K tokens) |
|--------|-------------------|---------------------|
| Matched pairs found | 52,237 | 7,114 |
| Decoder cos_sim mean | 0.158 | 0.913 |
| Decoder cos_sim median | 0.009 | 0.943 |

- 50K tokens produced inflated spurious matches (52,237 vs expected 7,114)
- SAEBench TPP pipeline is not feature-restrictable from the outside; ran on all 65K features regardless of matching
