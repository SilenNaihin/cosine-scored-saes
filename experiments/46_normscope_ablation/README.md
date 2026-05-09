# Experiment 46: Full normalization scope factorial

A full 2x2x2 factorial ablation of NoC's three architectural knobs (encoder unit-norm, decoder unit-norm, post-decode norm restoration) crossed with aux-k {on, off} yields 12 variants trained from a single 5M-token bf16 activation cache at Qwen3-8B layer 27 (d_sae=65536, k=80, lr=5e-5). All variants see identical data in identical order on A100 80GB.

## Results

### Reconstruction (500K-token held-out eval, aux-k enabled)

| Variant | FVE | dead% | alive | mean L0 |
|---|---:|---:|---:|---:|
| noc_baseline_aux | 0.5537 | 0.0 | 65,536 | 66.5 |
| noc_dec_free_restore_aux | 0.5499 | 0.0 | 65,536 | 67.4 |
| noc_dec_free_no_restore_aux | 0.0818 | 88.5 | 7,505 | 118.6 |
| noc_enc_free_aux | 0.5582 | 0.0 | 65,536 | 70.7 |
| noc_input_only_restore_aux | 0.5540 | 0.0 | 65,536 | 71.7 |
| noc_input_only_no_restore_aux | 0.1378 | 89.3 | 7,001 | 136.7 |

### Reconstruction (500K-token held-out eval, aux-k disabled)

| Variant | FVE | dead% | alive | mean L0 |
|---|---:|---:|---:|---:|
| noc_baseline_noaux | 0.5537 | 0.0 | 65,536 | 66.5 |
| noc_dec_free_restore_noaux | 0.5499 | 0.0 | 65,536 | 67.4 |
| noc_dec_free_no_restore_noaux | 0.0818 | 88.5 | 7,505 | 118.6 |
| noc_enc_free_noaux | 0.5582 | 0.0 | 65,536 | 70.7 |
| noc_input_only_restore_noaux | 0.5540 | 0.0 | 65,536 | 71.7 |
| noc_input_only_no_restore_noaux | 0.1378 | 89.3 | 7,001 | 136.7 |

### RNH cos>inner diagnostic (top-30 features x 20 samples, aux-k enabled)

| Variant | cos>inner | cos-KL | inner-KL |
|---|---|---:|---:|
| noc_baseline_aux | 13/30 (43%) | 0.322 | 0.334 |
| noc_dec_free_restore_aux | 16/30 (53%) | 0.254 | 0.258 |
| noc_dec_free_no_restore_aux | 28/30 (93%) | 0.195 | 0.149 |
| noc_enc_free_aux | 8/30 (27%) | 0.272 | 0.282 |
| noc_input_only_restore_aux | 13/30 (43%) | 0.302 | 0.301 |
| noc_input_only_no_restore_aux | 19/30 (63%) | 0.515 | 0.489 |

### RNH cos>inner diagnostic (aux-k disabled)

| Variant | cos>inner | cos-KL | inner-KL |
|---|---|---:|---:|
| noc_baseline_noaux | 13/30 (43%) | 0.322 | 0.334 |
| noc_dec_free_restore_noaux | 16/30 (53%) | 0.254 | 0.258 |
| noc_dec_free_no_restore_noaux | 27/30 (90%) | 0.200 | 0.147 |
| noc_enc_free_noaux | 8/30 (27%) | 0.272 | 0.282 |
| noc_input_only_restore_noaux | 13/30 (43%) | 0.302 | 0.301 |
| noc_input_only_no_restore_noaux | 18/30 (60%) | 0.525 | 0.499 |

### Training trajectory (group 1, selected variants)

| Step | Tokens | baseline FVE | dec_free_restore FVE | dec_free_no_restore FVE | dec_free_no_restore dead% | input_only_no_restore FVE | input_only_no_restore dead% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.4M | -0.159 | -0.159 | 0.001 | 0.0 | 0.001 | 0.0 |
| 400 | 0.8M | -0.037 | -0.037 | 0.005 | 0.0 | 0.005 | 0.0 |
| 600 | 1.2M | 0.201 | 0.204 | 0.025 | 0.0 | 0.026 | 0.0 |
| 800 | 1.6M | 0.327 | 0.327 | 0.037 | 0.0 | 0.043 | 0.0 |
| 1000 | 2.0M | 0.402 | 0.400 | 0.045 | 87.5 | 0.056 | 87.8 |
| 1200 | 2.5M | 0.443 | 0.440 | 0.051 | 88.4 | 0.068 | 88.9 |
| 1400 | 2.9M | 0.499 | 0.494 | 0.057 | 88.5 | 0.081 | 89.0 |
| 1600 | 3.3M | 0.526 | 0.521 | 0.059 | 88.6 | 0.089 | 89.1 |
| 1800 | 3.7M | 0.550 | 0.546 | 0.064 | 88.6 | 0.099 | 89.2 |
| 2000 | 4.1M | 0.565 | 0.561 | 0.067 | 88.6 | 0.109 | 89.3 |
| 2200 | 4.5M | 0.575 | 0.570 | 0.073 | 88.6 | 0.123 | 89.5 |
