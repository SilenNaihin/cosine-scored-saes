# Experiment 55a: Absorption/SCR/TPP evaluation

Five SAE architectures (standard, adaptive cosine, per-feature cosine, NoC, and an independent reference) are evaluated on three additional SAEBench metrics at L18 on Qwen3-8B: absorption (task-relevant information in individual features), SCR (spurious correlation removal at varying feature counts), and TPP (targeted probe perturbation with intended/unintended decomposition).

## Results

Absorption (first-letter task):

| SAE | Absorption Fraction | Full Absorption | Split Features |
|-----|---------------------|-----------------|----------------|
| perfeature_l2 | 0.170 | 0.097 | 1.12 |
| independent_ref | 0.144 | 0.077 | 1.00 |
| standard | 0.125 | 0.065 | 1.19 |
| adaptive_l2 | 0.098 | 0.049 | 1.12 |
| no_C | 0.000 | 0.000 | 1.04 |

SCR (spurious correlation removal) by number of features ablated:

| SAE | @2 | @5 | @10 | @20 | @50 | @100 | @500 |
|-----|-----|-----|------|------|------|------|------|
| standard | 0.108 | 0.168 | 0.192 | 0.226 | 0.303 | 0.213 | -0.182 |
| adaptive_l2 | 0.092 | 0.141 | 0.182 | 0.230 | 0.311 | 0.365 | 0.379 |
| perfeature_l2 | 0.109 | 0.148 | 0.188 | 0.230 | 0.297 | 0.364 | 0.380 |
| no_C | 0.091 | 0.132 | 0.184 | 0.232 | 0.321 | 0.369 | 0.340 |
| independent_ref | 0.095 | 0.150 | 0.183 | 0.191 | 0.202 | 0.119 | -0.215 |

TPP (targeted probe perturbation) total metric:

| SAE | @2 | @5 | @10 | @20 | @50 |
|-----|-----|-----|------|------|------|
| independent_ref | 0.016 | 0.060 | 0.124 | 0.254 | 0.277 |
| standard | 0.013 | 0.038 | 0.079 | 0.193 | 0.295 |
| no_C | 0.007 | 0.010 | 0.023 | 0.042 | 0.138 |
| adaptive_l2 | 0.006 | 0.010 | 0.019 | 0.053 | 0.097 |
| perfeature_l2 | 0.005 | 0.006 | 0.018 | 0.052 | 0.094 |

TPP intended vs unintended decomposition (@20):

| SAE | Intended | Unintended | Ratio (precision) | Total |
|-----|----------|------------|-------------------|-------|
| independent_ref | 0.293 | 0.039 | 7.4x | 0.254 |
| standard | 0.227 | 0.034 | 6.7x | 0.193 |
| adaptive_l2 | 0.058 | 0.006 | 10.4x | 0.053 |
| perfeature_l2 | 0.057 | 0.005 | 11.4x | 0.052 |
| no_C | 0.044 | 0.002 | 20.0x | 0.042 |
