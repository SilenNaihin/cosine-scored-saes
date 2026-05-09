# Experiment 11: Cross-model replication on Mistral-7B

The cosine-vs-inner-product causal prediction comparison is replicated on Mistral-7B-v0.1, a different RMSNorm model family with a standard-architecture SAE (65,536 features, resid_pre hook). This tests whether the directional advantage generalizes beyond Qwen3-8B and BatchTopK SAEs. Results are compared against prior Pythia-2.8B (LayerNorm) data at equivalent layers and hook points.

## Results

Model: Mistral-7B-v0.1, SAE: mistral-7b-res-wg (65k features), 50 features/layer, FineWeb corpus (~109k tokens/layer).

### Per-layer ablation correlations

| Layer | cos->KL | norm->KL | inner->KL | SAE->KL | cos>inner | cos>SAE |
|-------|---------|----------|-----------|---------|-----------|---------|
| 8 | 0.505 | -0.094 | 0.484 | 0.396 | 37/50 (74%) | 44/50 (88%) |
| 16 | 0.550 | -0.126 | 0.506 | 0.390 | 48/50 (96%) | 47/50 (94%) |
| 24 | 0.400 | -0.138 | 0.333 | 0.347 | 46/50 (92%) | 37/50 (74%) |
| Total | | | | | 131/150 (87%) | 128/150 (85%) |

### Cross-model comparison (RMSNorm)

| Model | SAE | cos > inner | cos > SAE | Layer gradient |
|-------|-----|-------------|-----------|----------------|
| Mistral-7B | standard (65k) | 87% | 85% | No |
| Qwen3-8B | BatchTopK (65k) | 80-83% | 77-81% | No |

### Cross-model comparison (LayerNorm)

| Model | SAE | cos > inner | cos > SAE | Layer gradient |
|-------|-----|-------------|-----------|----------------|
| Pythia-2.8B | TopK (61k) | 70% | 56% | Yes (90% early, 52% late) |
| Pythia-70M | standard (32k) | 47% | 69% | Mixed |
