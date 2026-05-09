# Experiment 18: Adaptive + post-norm combination

This experiment tests whether adaptive scaling (from Exp 12) composes with post-RMSNorm loss (from Exp 14). The adaptive encoder learns a per-token scale parameter `a` that interpolates between cosine and inner-product scoring; post-norm loss optimizes reconstruction of the post-LayerNorm signal. The experiment runs on Qwen3-8B layers 9, 18, 27 with d_sae=16384, k=80, and 5M tokens per layer.

## Results

Learned adaptive parameter `a` across loss functions:

| Layer | Gain CV | adaptive_l2 `a` (exp12) | adaptive_postnorm `a` | adaptive_cosloss `a` (control) |
|---|---|---|---|---|
| 9 | 25.1% | +0.044 | -0.011 | -0.019 |
| 18 | 40.5% | +0.103 | -0.003 | -0.008 |
| 27 | 30.7% | +0.103 | -0.015 | -0.006 |

Comparison of adaptive_postnorm vs cosine_postnorm (exp14):

| Metric | adaptive_postnorm | cosine_postnorm (exp14) | Difference |
|---|---|---|---|
| L9 FVE | 0.496 | 0.503 | -0.007 |
| L18 FVE | 0.243 | 0.246 | -0.003 |
| L27 FVE | 0.076 | 0.083 | -0.007 |
| L9 cos recon | 0.670 | 0.670 | 0.000 |
| L18 cos recon | 0.796 | 0.795 | +0.001 |
| L27 cos recon | 0.861 | 0.861 | 0.000 |
| L18 dead | 97.3% | 97.2% | +0.1% |
| L27 dead | 81.9% | 81.9% | 0.0% |
| L27 SAE->KL | 0.252 | 0.252 | 0.000 |
| L27 cos->KL | 0.327 | 0.336 | -0.009 |
| L27 pnFVE | 0.724 | 0.723 | +0.001 |

Full reconstruction comparison across variants:

| Variant | L9 FVE | L9 cos | L18 FVE | L18 cos | L27 FVE | L27 cos |
|---|---|---|---|---|---|---|
| standard | 0.601 | 0.852 | 0.534 | 0.853 | 0.564 | 0.867 |
| adaptive_l2 (exp12) | 0.623 | 0.861 | 0.561 | 0.862 | 0.534 | 0.859 |
| cosine_postnorm (exp14) | 0.503 | 0.670 | 0.246 | 0.795 | 0.083 | 0.861 |
| adaptive_postnorm | 0.496 | 0.670 | 0.243 | 0.796 | 0.076 | 0.861 |
| adaptive_cosloss (control) | 0.517 | 0.851 | 0.319 | 0.856 | 0.120 | 0.874 |

Norm invariance test (output ratio when input scaled 2x):

| Variant | L9 ratio | L18 ratio | L27 ratio |
|---|---|---|---|
| standard | 1.984 | 1.984 | 1.994 |
| adaptive_l2 (exp12) | 1.032 | 1.075 | 1.074 |
| cosine_postnorm (exp14) | 1.000 | 1.000 | 1.000 |
| adaptive_postnorm | 0.992 | 0.997 | 0.990 |
| adaptive_cosloss (control) | 0.985 | 0.996 | 0.996 |

Ablation SAE->KL (causal faithfulness):

| Variant | L9 | L18 | L27 |
|---|---|---|---|
| standard | 0.208 | 0.361 | 0.198 |
| adaptive_l2 | 0.194 | 0.255 | 0.094 |
| cosine_postnorm (exp14) | 0.141 | 0.308 | 0.252 |
| adaptive_postnorm | 0.135 | 0.285 | 0.252 |
| adaptive_cosloss (control) | 0.141 | 0.182 | 0.084 |

Dead features:

| Variant | L9 | L18 | L27 |
|---|---|---|---|
| standard | 71.5% | 91.9% | 83.6% |
| adaptive_l2 | 66.9% | 88.6% | 75.0% |
| cosine_postnorm (exp14) | 73.9% | 97.2% | 81.9% |
| adaptive_postnorm | 73.9% | 97.3% | 81.9% |
| adaptive_cosloss (control) | 77.6% | 92.8% | 76.2% |
