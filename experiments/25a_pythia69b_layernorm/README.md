# Experiment 25a: Pythia-6.9B LayerNorm test

This experiment trains cosine SAE variants on Pythia-6.9B-deduped (LayerNorm, d_model=4096) at layers 8, 16, and 24 to test whether the cosine advantage generalizes to a larger LayerNorm model. Three variants (standard, cosine, adaptive_l2) are trained with d_sae=16384, k=80, 5M tokens.

## Results

| Layer | Variant | FVE | Dead% | scale_a |
|---|---|---|---|---|
| 8 | standard | 0.310 | 93.7% | - |
| 8 | cosine | 0.325 | 91.7% | - |
| 8 | adaptive_l2 | 0.329 | 91.6% | 0.048 |
| 16 | standard | 0.307 | 94.2% | - |
| 16 | cosine | 0.138 | 98.7% | - |
| 16 | adaptive_l2 | 0.147 | 98.7% | 0.032 |
| 24 | standard | 0.783 | 90.6% | - |
| 24 | cosine | 0.293 | 99.0% | - |
| 24 | adaptive_l2 | 0.819 | 95.9% | 0.146 |
