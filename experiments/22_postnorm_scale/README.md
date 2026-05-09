# Experiment 22: Post-norm loss at production scale

This experiment tests whether the post-RMSNorm loss SAE->KL advantage found at 5M tokens (exp14) persists at production scale (50M tokens). Three variants are trained on Qwen3-8B layer 27 with d_sae=16384, k=80. Evaluation uses 1M tokens and 100 features x 200 ablation samples.

## Results

Summary:

| Variant | Loss | FVE | pnFVE | Dead% | Alive | cos>inner | cos->KL | SAE->KL | scale_a |
|---|---|---|---|---|---|---|---|---|---|
| standard | L2 | 0.659 | 0.740 | 77.4% | 3709 | 79/100 | 0.415 | 0.367 | - |
| adaptive_l2 | L2 | 0.738 | 0.792 | 28.2% | 11768 | 84/100 | 0.397 | 0.352 | 0.207 |
| postnorm | postnorm | 0.005 | 0.756 | 83.2% | 2745 | 72/100 | 0.379 | 0.380 | -0.401 |

Replication check vs exp17 (same scale):

| Metric | exp22 standard | exp17 standard | exp22 adaptive_l2 | exp17 adaptive_l2 |
|---|---|---|---|---|
| FVE | 0.659 | 0.657 | 0.738 | 0.737 |
| Dead% | 77.4% | 78.3% | 28.2% | 29.7% |
| Alive | 3709 | 3558 | 11768 | 11517 |
| cos>inner | 79/100 | 73/100 | 84/100 | 79/100 |
| SAE->KL | 0.367 | 0.385 | 0.352 | 0.330 |
| scale_a | - | - | 0.207 | 0.208 |

Postnorm scale_a trajectory during training:

| Step | scale_a | scale_b | FVE | Dead% |
|---|---|---|---|---|
| 2000 | -0.034 | 60.6 | 0.064 | 86.4% |
| 4000 | -0.145 | 53.1 | 0.028 | 86.4% |
| 6000 | -0.271 | 45.8 | 0.012 | 86.1% |
| 8000 | -0.351 | 41.4 | 0.007 | 85.8% |
| 10000 | -0.394 | 39.3 | 0.005 | 85.6% |
| 12207 | -0.401 | 38.9 | 0.005 | 85.6% |

Scaling comparison (5M to 50M tokens):

| Metric | exp14 (5M) | exp22 (50M) |
|---|---|---|
| Postnorm SAE->KL | 0.252 | 0.380 |
| Standard SAE->KL | 0.198 | 0.367 |
| Postnorm advantage | +27% | +3.5% |
| Postnorm pnFVE | 0.723 | 0.756 |
| Postnorm dead% | - | 83.2% |
| Postnorm scale_a | ~0 (frozen) | -0.401 |

Norm invariance (output ratio when input scaled):

| Variant | 2x scale ratio | 5x scale ratio |
|---|---|---|
| standard | 2.000 | 4.999 |
| adaptive_l2 | 1.161 | 1.408 |
| postnorm | 0.780 | 0.572 |
