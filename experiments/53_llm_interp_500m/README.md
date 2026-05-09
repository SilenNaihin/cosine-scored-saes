# Experiment 53: LLM auto-interp at 500M tokens

Five SAE architectures (standard, adaptive cosine, per-feature cosine, NoC, and an independent reference) are evaluated using a describe-then-predict LLM interpretability protocol at L18 on Qwen3-8B. For each SAE, 200 features are sampled (stratified by frequency), described by an LLM judge from 10 activating contexts, then scored on 10 held-out prediction trials (interpretable if >=50% prediction accuracy).

## Results

| SAE | Alive | Sampled | Interpretable | Rate |
|-----|-------|---------|---------------|------|
| standard | 17,316 | 200 | 50 | 0.250 |
| adaptive_l2 | 17,826 | 200 | 38 | 0.190 |
| perfeature_l2 | 18,651 | 200 | 48 | 0.240 |
| no_C | 18,400 | 200 | 47 | 0.235 |
| independent_ref | 16,801 | 200 | 39 | 0.195 |

Per frequency band:

| SAE | Low | Medium | High |
|-----|-----|--------|------|
| standard | 13/51 (0.255) | 25/99 (0.253) | 12/50 (0.240) |
| adaptive_l2 | 11/51 (0.216) | 19/99 (0.192) | 8/50 (0.160) |
| perfeature_l2 | 19/50 (0.380) | 22/100 (0.220) | 7/50 (0.140) |
| no_C | 17/51 (0.333) | 20/98 (0.204) | 10/51 (0.196) |
| independent_ref | 15/51 (0.294) | 18/99 (0.182) | 6/50 (0.120) |
