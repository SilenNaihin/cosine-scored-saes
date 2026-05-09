# Experiment 1: Per-layer magnitude invariance under LayerNorm/RMSNorm

This experiment tests whether RMSNorm renders the residual stream magnitude-invariant at each layer, and whether this invariance persists through full forward passes. Part A isolates RMSNorm on Qwen3-8B, confirming the mathematical identity RMSNorm(ax) = RMSNorm(x). Part B scales the residual stream at a given layer and measures output logit cosine similarity after the remaining forward pass.

## Results

Part A: Cosine similarity between RMSNorm(x) and RMSNorm(scale * x) is exactly 1.0 across all layers, scale factors (0.1x to 100x), and prompts. Bfloat16 noise floor: ~1e-2 L2 distance at extreme scales.

Part B: Full-model logit cosine similarity after residual stream scaling:

| Scale Factor | Layer 9 cos | Layer 18 cos | Layer 27 cos |
|---|---|---|---|
| 0.1x | -0.46 | 0.87 | -0.06 |
| 0.5x | 0.90 | 0.96 | 0.97 |
| 2.0x | 0.99 | 0.99 | 0.99 |
| 5.0x | 0.66 | 0.96 | 0.89 |
| 10.0x | 0.50 | 0.85 | 0.30 |
| 100.0x | -0.29 | -0.43 | -0.39 |
