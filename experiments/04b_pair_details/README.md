# Experiment 4b: Activation pattern analysis of near-similar decoder pairs

This experiment provides detailed activation pattern analysis for the 18 decoder direction pairs with cosine similarity > 0.70 identified in Experiment 4 (Qwen3-8B, layer 18, 65,536 features). For each pair, encoder cosine, encoder norms, bias values, and token-level firing co-occurrence statistics are computed on 328 tokens from 12 prompts.

## Results

| Feature i | Feature j | Dec cos | Enc cos | Bias diff | Jaccard |
|---|---|---|---|---|---|
| 10132 | 55507 | 0.809 | 0.166 | 0.688 | 0.0 |
| 16103 | 42018 | 0.793 | 0.238 | 1.063 | 0.0 |
| 24852 | 37448 | 0.785 | -0.013 | 0.813 | 0.0 |
| 8114 | 25105 | 0.770 | 0.338 | 0.500 | 0.0 |
| 12399 | 63545 | 0.742 | 0.083 | 0.438 | 0.0 |
| 22631 | 30600 | 0.742 | -0.152 | 0.719 | 0.0 |
| 1280 | 37506 | 0.738 | 0.326 | 1.375 | 0.0 |
| 25105 | 26891 | 0.738 | 0.322 | 0.938 | 0.0 |
| 8114 | 26891 | 0.734 | 0.254 | 0.438 | 0.0 |
| 13254 | 63541 | 0.730 | 0.157 | 1.406 | 0.063 |
| 20321 | 31291 | 0.730 | 0.131 | 0.438 | 0.0 |
| 12714 | 26891 | 0.727 | 0.295 | 0.031 | 0.0 |
| 19314 | 64376 | 0.723 | 0.408 | 2.875 | 0.0 |
| 407 | 37919 | 0.719 | 0.092 | 0.250 | 0.0 |
| 27121 | 44963 | 0.715 | 0.275 | 2.000 | 0.0 |
| 2483 | 12044 | 0.707 | 0.318 | 0.500 | 0.0 |
| 14245 | 34559 | 0.703 | 0.350 | 0.906 | 0.0 |
| 41938 | 50407 | 0.703 | 0.154 | 1.844 | 0.0 |

- Only 1/18 pairs shows any activation co-occurrence (features 13254 and 63541, jaccard=0.063)
- Mean encoder cosine across pairs: 0.208
- No pair has encoder cosine > 0.41
- Norm separation for the one co-firing pair: 0.603
