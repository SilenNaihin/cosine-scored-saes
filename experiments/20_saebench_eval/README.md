# Experiment 20: SAEBench evaluation across architectures

This experiment evaluates exp17 production-scale SAE checkpoints (50M tokens, Qwen3-8B, d_sae=16384, k=80) on the SAEBench suite: core reconstruction metrics, sparse probing, absorption, sparse circuit recovery (SCR), and token prediction probing (TPP). Three variants (standard, adaptive_l2, perfeature_l2) are evaluated at layers 9, 18, and 27.

## Results

Core eval, Layer 9:

| Variant | KL Score | KL w/ SAE | CE Score | CosSim | Expl Var | MSE | L0 | Alive % |
|---|---|---|---|---|---|---|---|---|
| standard | 0.9739 | 0.260 | 0.9874 | 0.871 | 0.520 | 4.78 | 81.5 | 62.2% |
| adaptive_l2 | 0.9770 | 0.229 | 0.9874 | 0.883 | 0.258 | 7.41 | 81.0 | 53.9% |
| perfeature_l2 | 0.9793 | 0.206 | 0.9892 | 0.887 | 0.258 | 7.38 | 80.5 | 60.9% |

Core eval, Layer 18:

| Variant | KL Score | KL w/ SAE | CE Score | CosSim | Expl Var | MSE | L0 | Alive % |
|---|---|---|---|---|---|---|---|---|
| standard | 0.9613 | 0.385 | 0.9820 | 0.879 | 0.762 | 7.03 | 81.3 | 48.5% |
| adaptive_l2 | 0.9623 | 0.375 | 0.9803 | 0.879 | 0.293 | 20.88 | 80.5 | 15.2% |
| perfeature_l2 | 0.9664 | 0.334 | 0.9820 | 0.883 | 0.277 | 21.38 | 80.3 | 28.2% |

Core eval, Layer 27:

| Variant | KL Score | KL w/ SAE | CE Score | CosSim | Expl Var | MSE | L0 | Alive % |
|---|---|---|---|---|---|---|---|---|
| standard | 0.9454 | 0.543 | 0.9605 | 0.898 | 0.824 | 31.38 | 78.9 | 44.6% |
| adaptive_l2 | 0.9689 | 0.309 | 0.9803 | 0.922 | 0.805 | 35.00 | 78.6 | 70.8% |
| perfeature_l2 | 0.9672 | 0.326 | 0.9803 | 0.918 | 0.789 | 38.00 | 79.9 | 71.9% |

Sparse probing, Layer 9:

| Variant | SAE Acc | SAE Top-1 | SAE Top-2 | SAE Top-5 | LLM Baseline |
|---|---|---|---|---|---|
| standard | 0.9443 | 0.636 | 0.710 | 0.793 | 0.9550 |
| adaptive_l2 | 0.9491 | 0.796 | 0.823 | 0.860 | 0.9560 |
| perfeature_l2 | 0.9488 | 0.681 | 0.792 | 0.876 | 0.9558 |

Sparse probing, Layer 18:

| Variant | SAE Acc | SAE Top-1 | SAE Top-2 | SAE Top-5 | LLM Baseline |
|---|---|---|---|---|---|
| standard | 0.9457 | 0.656 | 0.725 | 0.797 | 0.9596 |
| adaptive_l2 | 0.9526 | 0.783 | 0.817 | 0.852 | 0.9578 |
| perfeature_l2 | 0.9546 | 0.797 | 0.814 | 0.856 | 0.9592 |

Sparse probing, Layer 27:

| Variant | SAE Acc | SAE Top-1 | SAE Top-2 | SAE Top-5 | LLM Baseline |
|---|---|---|---|---|---|
| standard | 0.9532 | 0.703 | 0.812 | 0.865 | 0.9607 |
| adaptive_l2 | 0.9582 | 0.775 | 0.843 | 0.885 | 0.9604 |
| perfeature_l2 | 0.9572 | 0.764 | 0.797 | 0.893 | 0.9595 |

Absorption, Layer 9:

| Variant | Absorption Frac | Full Absorption | Split Features |
|---|---|---|---|
| standard | 0.0897 | 0.0302 | 1.12 |
| adaptive_l2 | 0.0610 | 0.0112 | 1.12 |
| perfeature_l2 | 0.0673 | 0.0165 | 1.19 |

Absorption, Layer 18:

| Variant | Absorption Frac | Full Absorption | Split Features |
|---|---|---|---|
| standard | 0.0419 | 0.0136 | 1.19 |
| adaptive_l2 | 0.0621 | 0.0111 | 1.27 |
| perfeature_l2 | 0.1016 | 0.0205 | 1.15 |

Absorption, Layer 27:

| Variant | Absorption Frac | Full Absorption | Split Features |
|---|---|---|---|
| standard | 0.0353 | 0.0135 | 1.23 |
| adaptive_l2 | 0.0794 | 0.0177 | 1.31 |
| perfeature_l2 | 0.1224 | 0.0372 | 1.35 |

SCR (Sparse Circuit Recovery), Layer 9:

| Variant | SCR@10 | SCR@20 | SCR@50 | SCR@100 |
|---|---|---|---|---|
| standard | 0.240 | 0.307 | 0.263 | 0.362 |
| adaptive_l2 | 0.274 | 0.354 | 0.295 | 0.455 |
| perfeature_l2 | 0.262 | 0.334 | 0.442 | 0.517 |

SCR, Layer 18:

| Variant | SCR@10 | SCR@20 | SCR@50 | SCR@100 |
|---|---|---|---|---|
| standard | 0.210 | 0.226 | 0.199 | 0.193 |
| adaptive_l2 | 0.357 | 0.417 | 0.122 | 0.266 |
| perfeature_l2 | 0.285 | 0.275 | 0.331 | 0.361 |

SCR, Layer 27:

| Variant | SCR@10 | SCR@20 | SCR@50 | SCR@100 |
|---|---|---|---|---|
| standard | 0.401 | 0.484 | 0.568 | 0.336 |
| adaptive_l2 | 0.225 | 0.159 | 0.119 | 0.163 |
| perfeature_l2 | 0.198 | 0.325 | 0.282 | 0.360 |

TPP (Token Prediction Probing), Layer 9:

| Variant | TPP@10 | TPP@20 | TPP@50 | TPP@100 | TPP@500 |
|---|---|---|---|---|---|
| standard | 0.104 | 0.264 | 0.366 | 0.358 | 0.297 |
| adaptive_l2 | 0.104 | 0.216 | 0.328 | 0.371 | 0.327 |
| perfeature_l2 | 0.060 | 0.146 | 0.273 | 0.332 | 0.360 |

TPP, Layer 18:

| Variant | TPP@10 | TPP@20 | TPP@50 | TPP@100 | TPP@500 |
|---|---|---|---|---|---|
| standard | 0.209 | 0.306 | 0.355 | 0.313 | 0.240 |
| adaptive_l2 | 0.168 | 0.297 | 0.365 | 0.363 | 0.333 |
| perfeature_l2 | 0.155 | 0.304 | 0.375 | 0.375 | 0.343 |

TPP, Layer 27:

| Variant | TPP@10 | TPP@20 | TPP@50 | TPP@100 | TPP@500 |
|---|---|---|---|---|---|
| standard | 0.138 | 0.238 | 0.368 | 0.380 | 0.361 |
| adaptive_l2 | 0.125 | 0.182 | 0.271 | 0.334 | 0.375 |
| perfeature_l2 | 0.116 | 0.218 | 0.331 | 0.374 | 0.383 |

Summary scorecard (wins out of total layer comparisons):

| Metric Category | Standard Wins | Cosine Wins |
|---|---|---|
| KL Score | 0/9 | 9/9 |
| CE Score | 0/9 | 6/9 |
| CosSim | 0/9 | 6/9 |
| MSE | 6/9 | 0/9 |
| Explained Var | 4/9 | 2/9 |
| Alive % | 2/9 | 4/9 |
| Sparse Probe Acc | 0/9 | 9/9 |
| Sparse Probe Top-1 | 0/9 | 9/9 |
| Sparse Probe Top-5 | 0/9 | 9/9 |
| Absorption Frac | 1/9 | 5/9 |
| SCR@100 | 1/3 | 2/3 |
| SCR@50 | 1/3 | 1/3 |
| TPP@20 | 3/3 | 0/3 |
| TPP@100 | 1/3 | 2/3 |
| TPP@500 | 0/3 | 3/3 |

Depth robustness (L9 to L27 change):

| Metric | Standard L9 to L27 | Cosine (best) L9 to L27 |
|---|---|---|
| KL Score | 0.974 to 0.945 (-2.9pp) | 0.979 to 0.969 (-1.0pp) |
| Sparse Probe Top-1 | 0.636 to 0.703 (+6.7pp) | 0.796 to 0.775 (-2.1pp) |
| Sparse Probe Top-5 | 0.793 to 0.865 (+7.2pp) | 0.876 to 0.893 (+1.7pp) |
