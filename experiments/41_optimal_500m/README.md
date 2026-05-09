# Experiment 41: Optimal 500M configuration

500M-token SAE training on Qwen3-8B at layers 9, 18, 27 with correct sqrt(d) initialization, comparing adaptive_l2 and group_G4 against standard baselines (reused from exp36). This experiment resolves the init contamination from exp36 (which used suboptimal norm-adaptive init) and tests whether the group_G4 architecture from exp37 generalizes to Qwen3-8B at production scale.

## Results

### Full Comparison

| Variant | Init | Layer | FVE | Dead% | Alive | scale_a | cos>inner |
|---------|------|-------|-----|-------|-------|---------|-----------|
| standard | -- | L9 | 0.711 | 59.6% | 6,621 | -- | 89/100 |
| standard | -- | L18 | 0.657 | 82.7% | 2,840 | -- | 79/100 |
| standard | -- | L27 | 0.686 | 67.8% | 5,275 | -- | 62/100 |
| adaptive_l2 (exp36, norm-adp) | norm-adp | L9 | 0.749 | 30.5% | 11,388 | 0.348 | 87/100 |
| adaptive_l2 (exp36, norm-adp) | norm-adp | L18 | 0.665 | 77.2% | 3,733 | 0.268 | 88/100 |
| adaptive_l2 (exp36, norm-adp) | norm-adp | L27 | 0.720 | 48.9% | 8,378 | 0.201 | 84/100 |
| adaptive_l2 | sqrt(d) | L9 | 0.749 | 29.8% | 11,502 | 0.318 | 86/100 |
| adaptive_l2 | sqrt(d) | L18 | 0.661 | 80.8% | 3,146 | 0.332 | 89/100 |
| adaptive_l2 | sqrt(d) | L27 | 0.748 | 23.2% | 12,576 | 0.362 | 76/100 |
| group_G4 | sqrt(d) | L9 | 0.748 | 29.9% | 11,491 | 0.270+/-0.023 | 92/100 |
| group_G4 | sqrt(d) | L18 | 0.662 | 80.8% | 3,143 | 0.303+/-0.016 | 83/100 |
| group_G4 | sqrt(d) | L27 | 0.749 | 22.6% | 12,687 | 0.348+/-0.019 | 87/100 |

### Init Fix Impact at L27

| Metric | norm-adaptive (exp36) | sqrt(d) (exp41) |
|--------|----------------------|-----------------|
| FVE | 0.720 | 0.748 |
| Dead% | 48.9% | 23.2% |
| Alive | 8,378 | 12,576 |
| scale_a | 0.201 | 0.362 |

### Cosine vs Standard Gap (sqrt(d) init, best cosine variant)

| Layer | FVE gap | Dead% gap | Alive ratio |
|-------|---------|-----------|-------------|
| L9 | +3.8pp | -29.8pp | 1.74x |
| L18 | +0.5pp | -1.9pp | 1.11x |
| L27 | +6.3pp | -44.6pp | 2.39x |

### scale_a with sqrt(d) Init

| Layer | Mean norm | adaptive_l2 a | group_G4 a (mean+/-std) |
|-------|-----------|---------------|------------------------|
| L9 | 57.5 | 0.318 | 0.270+/-0.023 |
| L18 | 97.7 | 0.332 | 0.303+/-0.016 |
| L27 | 404.7 | 0.362 | 0.348+/-0.019 |
