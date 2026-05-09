# Experiment 45: Encoder normalization ablation

This experiment tests whether encoder weight normalization (F.normalize on W_enc) is cosmetic or load-bearing by comparing standard adaptive cosine SAEs against a variant where encoder rows are free to drift from unit norm. Two variants are trained at layers 9, 18, and 27 of Qwen3-8B for 50M tokens each (d_sae=65536, k=80, lr=5e-5, saprmarks recipe), sharing cached activations per layer for an identical-batch comparison.

## Results

### Final metrics

| Layer | Variant | FVE | cos_recon | Dead% | Alive | L0 | cos>inner | scale_a |
|---|---|---|---|---|---|---|---|---|
| L9 | adaptive_l2 | 0.786 | 0.927 | 0.0% | 65,536 | 80.1 | 34/100 | 0.025 |
| L9 | adaptive_l2_unnormed | 0.665 | 0.886 | 90.8% | 6,058 | 80.0 | 27/100 | 0.589 |
| L18 | adaptive_l2 | 0.723 | 0.916 | 0.0% | 65,534 | 80.1 | 51/100 | 0.081 |
| L18 | adaptive_l2_unnormed | 0.623 | 0.885 | 94.6% | 3,553 | 80.0 | 36/100 | 0.595 |
| L27 | adaptive_l2 | 0.770 | 0.935 | 0.0% | 65,504 | 80.1 | 54/100 | 0.257 |
| L27 | adaptive_l2_unnormed | 0.614 | 0.888 | 93.2% | 4,462 | 79.7 | 55/100 | 0.620 |

### Gaps

| Layer | FVE gap | Dead% gap | Alive ratio |
|---|---|---|---|
| L9 | -12.0pp | +90.8pp | 0.09x |
| L18 | -10.0pp | +94.6pp | 0.05x |
| L27 | -15.6pp | +93.1pp | 0.07x |

### Encoder weight norm divergence (unnormed variant)

| Layer | norm median (final) | norm max (final) | max/median ratio | max growth from init |
|---|---|---|---|---|
| L9 | 1.014 | 6.80 | 6.7x | 6.1x |
| L18 | 1.013 | 8.53 | 8.4x | 7.6x |
| L27 | 1.010 | 11.86 | 11.7x | 10.6x |

### scale_a comparison

| Layer | Normed a | Unnormed a | Ratio |
|---|---|---|---|
| L9 | 0.025 | 0.589 | 23.6x |
| L18 | 0.081 | 0.595 | 7.4x |
| L27 | 0.257 | 0.620 | 2.4x |

### Post-hoc verification: standard SAE encoder norms (from checkpoints)

| Architecture | Layer | norm max/median | norm std | Dead% |
|---|---|---|---|---|
| Standard | L9 | 1.68x | 0.060 | 0.0% |
| Standard | L18 | 1.65x | 0.067 | 0.0% |
| Standard | L27 | 1.76x | 0.079 | 0.01% |
| Adaptive (normed) | L9 | 1.08x | 0.035 | 0.0% |
| Adaptive (normed) | L18 | 1.11x | 0.059 | 0.0% |
| Adaptive (normed) | L27 | 1.06x | 0.033 | 0.0% |
| Adaptive (unnormed) | L9 | 6.71x | 1.048 | 90.8% |
| Adaptive (unnormed) | L18 | 8.43x | 0.914 | 94.6% |
| Adaptive (unnormed) | L27 | 11.74x | 1.500 | 93.2% |
