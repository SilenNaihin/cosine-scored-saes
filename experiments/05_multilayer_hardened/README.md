# Experiment 5: Cosine vs inner product at layers 9, 18, 27 with KL divergence

This experiment is a hardened replication of Experiment 2 across three layers of Qwen3-8B (layers 9, 18, 27) using logit-level KL divergence as the ablation measure. It tests 50 features per layer with 100 ablation samples each on 40 diverse real-text passages, using a full forward pass through remaining layers rather than next-layer cosine distance.

## Results

| Layer | n_feat | cos-KL | norm-KL | inner-KL | SAE-KL | cos>inner | cos>SAE |
|---|---|---|---|---|---|---|---|
| 9 (early) | 50 | 0.193 | -0.146 | 0.152 | 0.136 | 45/50 (90%) | 40/50 (80%) |
| 18 (mid) | 50 | 0.288 | -0.047 | 0.261 | 0.249 | 38/50 (76%) | 35/50 (70%) |
| 27 (late) | 50 | 0.327 | -0.111 | 0.301 | 0.230 | 37/50 (74%) | 40/50 (80%) |
| Average | 150 | 0.269 | -0.101 | 0.238 | 0.205 | 120/150 (80%) | 115/150 (77%) |
