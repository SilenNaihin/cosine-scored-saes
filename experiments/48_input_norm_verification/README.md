# Experiment 48: Input normalization compensation verification

This experiment tests which mechanisms compensate for the dead feature problem caused by input normalization in cosine SAEs. Eight architectural variants are trained at L27 on Qwen3-8B with 50M tokens (d_sae=65,536, k=80, lr=5e-5, auxk), systematically toggling input normalization, encoder weight normalization, norm-restoration, and scale parameter structure (E0/E1/E3/E4).

## Results

| Variant | FVE | Dead % | Alive | cos>inner | W_enc norm | scale_a |
|---------|-----|--------|-------|-----------|------------|---------|
| standard_inputnorm | 0.297 | 93.5% | 4,284 | 58/100 | 2.48+/-5.68 | -- |
| unnormed_perfeature_b | 0.770 | 0.4% | 65,275 | 61/100 | 1.22+/-0.20 | 0.201 |
| adaptive_l2 | 0.770 | 0.0% | 65,504 | 54/100 | 1.15+/-0.03 | 0.257 |
| standard | 0.765 | 0.0% | 65,530 | 56/100 | 0.95+/-0.08 | -- |
| perfeature_base_delta | 0.772 | 0.3% | 65,335 | 61/100 | 1.13+/-0.03 | base=0.231 |
| noc_baseline | 0.751 | 0.3% | 65,310 | 60/100 | 1.00+/-0.00 | -- |
| noc_enc_free | 0.754 | 0.3% | 65,313 | 67/100 | 1.03+/-0.11 | -- |
| perfeature_bd_no_enc_norm | 0.740 | 20.0% | 52,401 | 61/100 | 1.04+/-0.08 | base=-0.574 |

Scale parameter structure summary:

| Scale structure | Enc norm | Dead % |
|---|---|---|
| E0 (none) + norm-restore | off | 0.3% |
| E1 (global a,b) + enc norm | on | 0.0% |
| E3 (base+delta) + enc norm | on | 0.3% |
| E4 (global a + per-feat b_i) | off | 0.4% |
| E3 (base+delta) + norm-restore | off | 20.0% |
| E0 (none), no compensation | off | 93.5% |
