# Cosine-Scored Sparse Autoencoders

Code supplement for **"Size Doesn't Matter: Cosine-Scored Sparse Autoencoders"**.

## Overview

Standard sparse autoencoders (SAEs) score features by inner product: a feature's activation scales with both its directional alignment and the input token's norm. In models with pre-layer RMSNorm, downstream sublayers strip magnitude before reading the residual stream — so the inner-product score encodes information the model has already discarded. Under BatchTopK selection, this causes high-norm tokens to inflate all pre-activations, driving ~86% of features to converge to norm detectors rather than content encoders.

We replace the inner-product score with a cosine score scaled by a learned norm-dependence exponent that interpolates between cosine (a=0) and inner product (a=1). On Qwen3-8B with 500M FineWeb tokens, the cosine encoder matches reconstruction (FVE ≈ 0.77) and improves single-feature sparse-probing top-1 by **+14.9%**. A matched-feature decomposition shows ~87% of the gap comes from features the standard encoder fails to learn, not from better separability. The learned exponent consistently converges below 0.3, confirming the optimizer discounts magnitude.

## Repository Structure

```
cosine-scored-saes/
├── benchmarks/          # SAEBench adapter and evaluation infrastructure
├── experiments/         # All 60+ experiments (scripts, results, logs)
│   ├── 01_layernorm_erasure/
│   ├── 02_magnitude_confound/
│   ├── ...
│   └── 60_decoder_geometry/
├── README.md
└── experiments.md       # Experiment index with results summaries
```

## Key Results

| Metric | Standard SAE | Cosine SAE | Gap |
|--------|-------------|-----------|-----|
| Sparse probing top-1 | 0.667 | 0.815 | **+14.9%** |
| Sparse probing top-5 | 0.783 | 0.889 | **+10.6%** |
| FVE (reconstruction) | 0.770 | 0.772 | matched |
| Q4 FVE (high-norm tokens) | -184 | +0.33 | fixed |
| Content-encoding features | 13.4% | ~100% | 7.5× |
| Per-feature interpretability | 82.1% | 80.1% | matched (p=0.88) |

*Qwen3-8B, layer 18, 500M FineWeb tokens, d_sae=65,536, BatchTopK k=80.*

## Models and SAEs

- **Primary model:** Qwen3-8B (RMSNorm, d_model=4096)
- **Cross-model:** Gemma-2-2B, Mistral-7B, Pythia-70M/2.8B/6.9B, Falcon-7B
- **SAE training:** BatchTopK (k=80), Adam (lr=5e-5), aux-k dead-feature loss (α=1/32), decoder unit-norm + gradient projection, 500M FineWeb tokens
- **Evaluation:** SAEBench (sparse probing, absorption, SCR, TPP, core metrics)
- **Reference SAE:** [adamkarvonen/qwen3-8b-saes](https://huggingface.co/adamkarvonen/qwen3-8b-saes)

## Architecture

The cosine-scored encoder replaces the standard pre-activation:

```
Standard:  s_i(x) = <w_i, x_c> + b_i         = ||x_c|| · cos(x_c, w_i) + b_i
Cosine:    s_i(x) = e^b · ||x_c||^a · cos(x_c, w_i) + b_i
```

where `a` interpolates between pure cosine (a=0) and inner product (a=1), and `b` is a global scale. Encoder rows are unit-normalized. A per-feature extension parameterizes `a_i = a_base + δ_i`.

## Reproducing

Each experiment folder contains the training/evaluation script and its results. Experiments are designed to run on a single GPU (A100 80GB or H100 80GB). Key dependencies:

```
torch >= 2.1
transformers
datasets
sae-bench  # for SAEBench evaluations
```

## Experiment Progression

See [experiments.md](experiments.md) for the full experiment index with results summaries.

## License

MIT
