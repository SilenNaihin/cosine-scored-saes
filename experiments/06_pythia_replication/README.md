# Experiment 6: Replication on Pythia-70M with SAE-Lens

This experiment replicates the ablation-prediction test on Pythia-70M-deduped (6 layers, d_model=512, LayerNorm, GPT-NeoX architecture) using EleutherAI TopK SAEs (32,768 features, k=16). The methodology is identical to Experiment 5: correlating cosine, inner product, SAE activation, and norm with logit-level KL divergence from feature ablation across layers 1, 3, and 5.

## Results

| Layer | n_feat | cos-KL | norm-KL | inner-KL | SAE-KL | cos>inner | cos>SAE |
|---|---|---|---|---|---|---|---|
| 1 (early) | 50 | 0.481 | -0.113 | 0.370 | 0.257 | 31/50 (62%) | 38/50 (76%) |
| 3 (mid) | 50 | 0.334 | 0.260 | 0.501 | 0.427 | 28/50 (56%) | 31/50 (62%) |
| 5 (late) | 50 | 0.721 | -0.332 | 0.405 | 0.557 | 44/50 (88%) | 45/50 (90%) |
| Average | 150 | 0.512 | -0.062 | 0.425 | 0.414 | 103/150 (69%) | 114/150 (76%) |

Cross-model comparison:

| Metric | Qwen3-8B (Exp 5) | Pythia-70M (Exp 6) |
|---|---|---|
| cos-KL (avg) | 0.269 | 0.512 |
| norm-KL (avg) | -0.101 | -0.062 |
| inner-KL (avg) | 0.238 | 0.425 |
| SAE-KL (avg) | 0.205 | 0.414 |
| cos>inner (%) | 80% | 69% |
| cos>SAE (%) | 77% | 76% |
| cos>inner at best layer | 90% (L9) | 88% (L5) |
