# Experiment 24: LayerNorm control experiment

This experiment tests the cosine SAE on a LayerNorm model (Pythia-2.8B-deduped) as a negative control. The RNH predicts cosine advantage should weaken on LayerNorm because LayerNorm erases both magnitude and mean, while cosine only corrects for magnitude. Three variants (standard, cosine, adaptive_l2) are trained at layers 8, 16, 24 with d_sae=10240, k=50, 5M tokens.

## Results

Full summary:

| Layer | Variant | FVE | Dead% | Alive | cos->KL | inner->KL | cos>inner | scale_a |
|---|---|---|---|---|---|---|---|---|
| 8 | standard | 0.359 | 77.2% | 2335 | 0.662 | 0.569 | 30/30 | - |
| 8 | cosine | 0.405 | 78.1% | 2237 | 0.644 | 0.542 | 30/30 | - |
| 8 | adaptive_l2 | 0.408 | 75.2% | 2538 | 0.655 | 0.554 | 30/30 | 0.052 |
| 16 | standard | 0.495 | 82.8% | 1765 | 0.585 | 0.554 | 27/30 | - |
| 16 | cosine | 0.341 | 86.3% | 1403 | 0.443 | 0.363 | 27/30 | - |
| 16 | adaptive_l2 | 0.541 | 85.1% | 1525 | 0.395 | 0.266 | 28/30 | 0.189 |
| 24 | standard | 0.565 | 95.0% | 507 | 0.271 | 0.441 | 17/30 | - |
| 24 | cosine | 0.393 | 97.7% | 235 | 0.177 | 0.076 | 12/30 | - |
| 24 | adaptive_l2 | 0.572 | 88.4% | 1187 | 0.134 | 0.063 | 12/30 | 0.118 |

Cross-model comparison (RMSNorm vs LayerNorm):

| Metric | Qwen L9 (RMS) | Qwen L27 (RMS) | Pythia L8 (LN) | Pythia L24 (LN) |
|---|---|---|---|---|
| cos>inner (std features) | 80% | 74% | 100% | 57% |
| cos>inner (ada features) | ~80% | ~73% | 100% | 40% |
| FVE gap (ada - std) | +2.3pp | +8.0pp | +5.0pp | +0.8pp |
| Alive ratio (ada/std) | 1.3x | 3.3x | 1.1x | 2.3x |
| adaptive scale_a | 0.044 | 0.103 | 0.052 | 0.118 |

Learned scale_a by depth:

| Layer depth | Qwen (RMSNorm) scale_a | Pythia (LayerNorm) scale_a |
|---|---|---|
| 25% | 0.044 | 0.052 |
| 50% | 0.103 | 0.189 |
| 75% | 0.103 | 0.118 |
