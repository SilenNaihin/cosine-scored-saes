# Experiment 32: Validation Against Open-Source Reference SAE

This experiment compares our standard and cosine SAEs (50M tokens) against a community-standard reference BatchTopK SAE trained on 500M tokens using an established open-source codebase. The reference uses the same model (Qwen3-8B), same architecture (BatchTopK, d_sae=16384, k=80), and same layers (9, 18, 27), providing a baseline to validate implementation correctness and measure sample efficiency.

## Results

### Reconstruction (FVE)

| Layer | Reference (500M) | Our Standard (50M) | Our Cosine (50M) |
|-------|-----------------|-------------------|-----------------|
| L9 | 0.765 | 0.679 | 0.703 |
| L18 | 0.729 | 0.638 | 0.636 |
| L27 | 0.767 | 0.659 | 0.739 |

### Dead Features

| Layer | Reference (500M) | Our Standard (50M) | Our Cosine (50M) |
|-------|-----------------|-------------------|-----------------|
| L9 | 0.0% (16,382 alive) | 62.1% (6,203 alive) | 50.9% (8,050 alive) |
| L18 | 0.0% (16,383 alive) | 83.1% (2,768 alive) | 86.8% (2,158 alive) |
| L27 | 0.0% (16,384 alive) | 79.0% (3,447 alive) | 30.7% (11,356 alive) |

### Ablation (cos>inner)

| Layer | Reference (500M) | Our Standard (50M) | Our Cosine (50M) |
|-------|-----------------|-------------------|-----------------|
| L9 | 25/30 (83%) | 23/30 (77%) | 24/30 (80%) |
| L18 | 23/30 (77%) | 24/30 (80%) | 20/30 (67%) |
| L27 | 24/30 (80%) | 19/30 (63%) | 18/30 (60%) |

### Feature Direction Overlap (Reference vs Our Standard)

| Layer | Ref alive | Std alive | Jaccard | Max cos mean | >0.9 match % |
|-------|-----------|-----------|---------|--------------|--------------|
| L9 | 16,382 | 6,203 | 0.374 | 0.463 | 3.7% |
| L18 | 16,383 | 2,768 | 0.166 | 0.390 | 0.3% |
| L27 | 16,384 | 3,447 | 0.206 | 0.410 | 1.0% |

### Feature Direction Overlap (Reference vs Our Cosine)

| Layer | Ref alive | Cos alive | Jaccard | Max cos mean | >0.9 match % |
|-------|-----------|-----------|---------|--------------|--------------|
| L9 | 16,382 | 8,050 | 0.478 | 0.527 | 7.0% |
| L18 | 16,383 | 2,158 | 0.128 | 0.375 | 0.4% |
| L27 | 16,384 | 11,356 | 0.688 | 0.621 | 10.5% |
