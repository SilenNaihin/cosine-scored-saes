# Experiment 34b: Multi-seed with sqrt(d) initialization

Three-seed variance estimation for cosine-scored SAEs at Layer 27 of Pythia-70M using sqrt(d) initialization. Each seed controls both weight initialization and data ordering; the eval set is seed-independent. Architectures compared: standard BatchTopK, adaptive_l2 (2 extra parameters), and perfeature_l2 (2x d_sae extra parameters). Training: 50M tokens, 16384 dictionary size.

## Results

| Variant | FVE | Dead% | Alive | SAE-to-KL | cos>inner | scale_a |
|---------|-----|-------|-------|-----------|-----------|---------|
| standard | 0.657 +/- 0.001 | 78.5% +/- 0.3% | 3,525 +/- 54 | 0.380 +/- 0.006 | 78% | -- |
| adaptive_l2 | 0.737 +/- 0.001 | 29.8% +/- 1.3% | 11,496 +/- 205 | 0.326 +/- 0.006 | 78% | 0.207 |
| perfeature_l2 | 0.732 +/- 0.001 | 28.0% +/- 0.6% | 11,804 +/- 100 | 0.358 +/- 0.001 | 77% | 0.113 |

Statistical significance (adaptive_l2 vs standard):
- FVE gap: +0.080 at 41.7 sigma
- Dead feature gap: +48.7pp at 37.5 sigma
- Alive feature ratio: 3.26x

scale_a consistency across seeds:

| | seed42 | seed123 | seed456 |
|--|--------|---------|---------|
| adaptive_l2 | 0.206 | 0.210 | 0.206 |
| perfeature_l2 (mean) | 0.112 | 0.113 | 0.113 |
