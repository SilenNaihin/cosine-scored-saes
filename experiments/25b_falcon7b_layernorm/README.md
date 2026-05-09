# Experiment 25b: Falcon-7B LayerNorm test

This experiment trains cosine SAE variants on Falcon-7B (LayerNorm, d_model=4544) at layers 8, 16, and 24 to test whether the cosine advantage extends to a different LayerNorm architecture. Three variants (standard, cosine, adaptive_l2) are trained with d_sae=4x d_model, k=80, 5M tokens.

## Results

| Layer | Variant | FVE | Dead% | scale_a |
|---|---|---|---|---|
| 8 | standard | 0.548 | 66.2% | - |
| 8 | cosine | 0.549 | 62.3% | - |
| 8 | adaptive_l2 | 0.551 | 61.0% | -0.015 |
| 16 | standard | 0.529 | 70.8% | - |
| 16 | cosine | 0.561 | 64.6% | - |
| 16 | adaptive_l2 | 0.563 | 64.3% | -0.005 |
| 24 | standard | 0.541 | 77.0% | - |
| 24 | cosine | 0.595 | 58.0% | - |
| 24 | adaptive_l2 | 0.594 | 61.5% | 0.075 |
