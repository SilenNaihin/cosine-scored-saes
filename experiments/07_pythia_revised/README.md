# Experiment 7: Revised Pythia-70M test with LayerNorm mean subtraction

This experiment repeats the Pythia-70M ablation-prediction test using SAE-Lens standard ReLU SAEs (32k features) on FineWeb (500 samples, ~63k tokens per layer). It tests whether the weaker results on LayerNorm models (compared to RMSNorm) are attributable to LayerNorm's mean subtraction step, which changes direction in addition to erasing magnitude.

## Results

| Layer | cos-KL | norm-KL | inner-KL | SAE-KL | cos>inner | cos>SAE |
|---|---|---|---|---|---|---|
| 1 | 0.320 | 0.072 | 0.306 | 0.230 | 27/50 (54%) | 31/50 (62%) |
| 3 | 0.605 | 0.405 | 0.746 | 0.517 | 5/50 (10%) | 30/50 (60%) |
| 5 | 0.403 | -0.063 | -0.022 | 0.222 | 39/50 (78%) | 43/50 (86%) |
| Total | - | - | - | - | 71/150 (47%) | 104/150 (69%) |

Cross-architecture comparison:

| Metric | Qwen3-8B (RMSNorm) | Pythia-70M (LayerNorm) |
|---|---|---|
| cos > inner | 80-83% | 47% |
| cos > SAE | 77-81% | 69% |
| d_model | 3584 | 512 |
