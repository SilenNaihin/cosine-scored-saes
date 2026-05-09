# Experiment 7b: Pythia-2.8B LayerNorm model test

This experiment tests the ablation-prediction methodology on Pythia-2.8B-deduped (32 layers, d_model=2560, LayerNorm) to disentangle whether Experiment 7's weak results were due to LayerNorm's mean subtraction or the small model size. Uses TopK SAEs (k=60, 61,440 features) on resid_pre at layers 8, 16, and 24, with FineWeb (500 samples, ~63k tokens per layer) and logit-level KL divergence.

## Results

| Layer | cos-KL | norm-KL | inner-KL | SAE-KL | cos>inner | cos>SAE |
|---|---|---|---|---|---|---|
| 8 | 0.319 | -0.206 | 0.291 | 0.342 | 45/50 (90%) | 38/50 (76%) |
| 16 | 0.268 | -0.038 | 0.306 | 0.341 | 34/50 (68%) | 27/50 (54%) |
| 24 | 0.300 | 0.212 | 0.476 | 0.506 | 26/50 (52%) | 19/50 (38%) |
| Total | - | - | - | - | 105/150 (70%) | 84/150 (56%) |

Scale comparison on LayerNorm models:

| Metric | Pythia-70M (Exp 7) | Pythia-2.8B (Exp 7b) |
|---|---|---|
| cos > inner | 71/150 (47%) | 105/150 (70%) |
| d_model | 512 | 2560 |
| Layers | 6 | 32 |
