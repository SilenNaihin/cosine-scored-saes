# Experiment 25: 5-model cross-architecture matrix

This experiment extends the LayerNorm control (exp24) into a 5-model matrix spanning 2 normalization types to test whether the cosine SAE advantage is normalization-dependent. Three variants (standard, cosine, adaptive_l2) are trained at 3 depth-matched layers on each model with d_sae=4x d_model, k=80, 5M tokens. Models include Qwen3-8B (RMSNorm), Mistral-7B (RMSNorm), Pythia-2.8B (LayerNorm), Pythia-6.9B (LayerNorm), and Falcon-7B (LayerNorm).

## Results

cos>inner diagnostic by model and depth:

| Model | Norm | L_shallow | L_mid | L_deep | Pattern |
|---|---|---|---|---|---|
| Qwen3-8B | RMSNorm | 80% (L9) | 76% (L18) | 74% (L27) | Gentle decline |
| Mistral-7B | RMSNorm | 47-60% (L8) | 53-67% (L16) | 73-77% (L24) | Inverted (increases) |
| Pythia-2.8B | LayerNorm | 100% (L8) | 90% (L16) | 40-57% (L24) | Steep collapse |
| Pythia-6.9B | LayerNorm | 83-90% (L8) | 73-80% (L16) | 50-70% (L24) | Steep collapse |
| Falcon-7B | LayerNorm | 90-100% (L8) | 50-70% (L16) | 40-47% (L24) | Steep collapse |

FVE (adaptive - standard gap):

| Model | Norm | Layer | Standard | Cosine | Adaptive | Ada-Std |
|---|---|---|---|---|---|---|
| Pythia-6.9B | LN | L8 | 0.310 | 0.324 | 0.328 | +1.8pp |
| Pythia-6.9B | LN | L16 | 0.307 | 0.139 | 0.149 | -15.9pp |
| Pythia-6.9B | LN | L24 | 0.801 | 0.300 | 0.834 | +3.3pp |
| Falcon-7B | LN | L8 | 0.544 | 0.544 | 0.546 | +0.2pp |
| Falcon-7B | LN | L16 | 0.520 | 0.551 | 0.552 | +3.2pp |
| Falcon-7B | LN | L24 | 0.536 | 0.588 | 0.587 | +5.0pp |
| Mistral-7B | RMS | L8 | 0.567 | -0.000 | -0.000 | -56.7pp |
| Mistral-7B | RMS | L16 | 0.577 | 0.025 | 0.013 | -56.4pp |
| Mistral-7B | RMS | L24 | 0.554 | 0.422 | 0.423 | -13.1pp |

Dead features:

| Model | Norm | Layer | Standard | Cosine | Adaptive |
|---|---|---|---|---|---|
| Pythia-6.9B | LN | L8 | 87.1% | 90.4% | 89.4% |
| Pythia-6.9B | LN | L16 | 91.3% | 98.6% | 98.6% |
| Pythia-6.9B | LN | L24 | 90.6% | 98.9% | 90.6% |
| Falcon-7B | LN | L8 | 49.5% | 11.0% | 10.0% |
| Falcon-7B | LN | L16 | 58.7% | 39.4% | 39.0% |
| Falcon-7B | LN | L24 | 68.1% | 29.1% | 32.8% |
| Mistral-7B | RMS | L8 | 54.0% | 97.6% | 97.6% |
| Mistral-7B | RMS | L16 | 57.5% | 0.6% | 2.9% |
| Mistral-7B | RMS | L24 | 52.7% | 42.6% | 41.0% |

Learned scale_a (adaptive variant):

| Model | Norm | L_shallow | L_mid | L_deep |
|---|---|---|---|---|
| Qwen3-8B | RMSNorm | 0.044 | 0.103 | 0.103 |
| Mistral-7B | RMSNorm | -0.002 | -0.002 | -0.007 |
| Pythia-2.8B | LayerNorm | 0.052 | 0.189 | 0.118 |
| Pythia-6.9B | LayerNorm | 0.048 | 0.032 | 0.146 |
| Falcon-7B | LayerNorm | -0.015 | -0.005 | 0.075 |

Activation norm scale (root cause of Mistral failure):

| Model | L8 norm mean | L24 norm mean | init scale (sqrt(d)) |
|---|---|---|---|
| Qwen3-8B | ~125 | ~200 | 64 |
| Pythia-6.9B | 105 | ~200 | 64 |
| Falcon-7B | ~92 | ~120 | 67 |
| Mistral-7B | 6.3 | 18.0 | 64 |

Cross-norm summary (cos>inner):

| | Shallow (25%) | Deep (75%) |
|---|---|---|
| RMSNorm (Qwen) | 80% | 74% |
| RMSNorm (Mistral) | 47-60% | 73-77% |
| LayerNorm (3 models) | 83-100% | 40-70% |

Cross-norm summary (alive feature ratio, adaptive/standard):

| | Falcon-7B (LN) | Pythia-6.9B (LN) | Mistral-7B (RMS) |
|---|---|---|---|
| L_shallow | 4.5x more alive | 0.9x | 0.05x (catastrophic) |
| L_deep | 2.3x more alive | 1.0x | 1.2x (recovered) |
