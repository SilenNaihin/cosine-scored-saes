# Experiment 15: Feature overlap between standard and cosine SAEs

Features shared across SAE architectures are identified by decoder cosine similarity (threshold > 0.9) and head-to-head activation-to-KL comparisons are run on those shared features. This eliminates the feature selection confound present in all prior aggregate comparisons, where each architecture was evaluated on its own (different) feature set.

## Results

Model: Qwen3-8B, SAE checkpoints from exp10/12 (5M tokens, d_sae=16384, k=80), 100 ablation samples per feature, 500K eval tokens.

### Shared feature counts (L18)

| SAE Pair | Shared Features |
|----------|-----------------|
| standard - cosine_l2 | 3 |
| standard - cosine_cosloss | 6 |
| standard - adaptive_l2 | 3 |
| standard - adaptive_cosloss | 5 |
| cosine_l2 - adaptive_l2 | 615 |
| cosine_cosloss - adaptive_cosloss | 299 |
| cosine_l2 - cosine_cosloss | 66 |
| cosine_l2 - adaptive_cosloss | 65 |
| cosine_cosloss - adaptive_l2 | 67 |
| adaptive_l2 - adaptive_cosloss | 71 |

### Cross-layer shared feature counts by pair type

| Pair type | L9 | L18 | L27 |
|-----------|------|------|------|
| standard - any cosine | 14-20 | 3-6 | 18-25 |
| cosine_l2 - adaptive_l2 (same loss) | 878 | 615 | 223 |
| cosine_cosloss - adaptive_cosloss (same loss) | 605 | 299 | 262 |
| cross-loss cosine variants | 61-73 | 65-71 | 35-56 |

### Standard vs cosine head-to-head (% won by standard, n in parentheses)

| Layer | std vs cos_l2 | std vs cos_cos | std vs adp_l2 | std vs adp_cos |
|-------|---------------|----------------|---------------|----------------|
| 9 | 53% (17) | 47% (15) | 55% (20) | 50% (14) |
| 18 | 33% (3) | 33% (6) | 67% (3) | 40% (5) |
| 27 | 100% (18) | 64% (25) | 77% (22) | 54% (24) |

### cosine_cosloss vs cosine_l2 (shared features)

| Layer | cos_cos wins | N | cos_cos mean->KL | cos_l2 mean->KL |
|-------|--------------|------|------------------|-----------------|
| 9 | 66% | 61 | 0.278 | 0.262 |
| 18 | 55% | 66 | 0.342 | 0.342 |
| 27 | 84% | 38 | 0.236 | 0.146 |

### cosine_l2 vs adaptive_l2 (shared features)

| Layer | cos_l2 wins | N | cos_l2 mean->KL | adp_l2 mean->KL |
|-------|-------------|------|-----------------|-----------------|
| 9 | 52% | 878 | 0.228 | 0.227 |
| 18 | 55% | 615 | 0.377 | 0.369 |
| 27 | 19% | 223 | 0.315 | 0.370 |

### cosine_cosloss vs adaptive_l2 (shared features)

| Layer | cos_cos wins | N |
|-------|--------------|------|
| 9 | 67% | 69 |
| 18 | 63% | 67 |
| 27 | 66% | 56 |

### cosine_cosloss vs adaptive_cosloss (same loss, different encoder)

| Layer | cos_cos wins | N |
|-------|--------------|------|
| 9 | 47% | 605 |
| 18 | 51% | 299 |
| 27 | 51% | 262 |

### adaptive_cosloss vs adaptive_l2 (shared features)

| Layer | adp_cos wins | N |
|-------|--------------|------|
| 9 | 58% | 73 |
| 18 | 58% | 71 |
| 27 | 73% | 55 |

### cos>inner on shared features

| Layer | Median cos>inner across pairs | Range |
|-------|-------------------------------|-------|
| 9 | 77% | 60-88% |
| 18 | 80% | 33-87% |
| 27 | 80% | 72-95% |
