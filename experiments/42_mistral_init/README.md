# Experiment 42: Mistral initialization fix

This experiment tests whether norm-adaptive initialization (scale_b = log(mean_norm)) resolves the catastrophic failure of cosine SAEs on Mistral-7B, where sqrt(d) initialization overshoots activation norms by 3.6-28x. Five variants (standard, adaptive with sqrt(d), adaptive with norm-adaptive, group_G4 with sqrt(d), group_G4 with norm-adaptive) are trained at layers 8, 16, and 24 for 50M tokens each on a single H100 GPU.

## Results

### Activation norms

| Layer | mean_norm | sqrt(d) | Mismatch |
|---|---|---|---|
| L8 | 2.3 | 64.0 | 28.0x overshoot |
| L16 | 6.3 | 64.0 | 10.2x overshoot |
| L24 | 18.0 | 64.0 | 3.6x overshoot |

### Full results (15 runs)

| Variant | Init | Layer | FVE | Dead% | Alive | cos>inner | scale_a | b(exp) |
|---|---|---|---|---|---|---|---|---|
| standard | -- | L8 | 0.6101 | 58.9% | 6,731 | 81/100 | -- | -- |
| adaptive_sqrtd | sqrt(d) | L8 | 0.0588 | 64.7% | 5,779 | 3/5 | -0.0009 | 59.3 |
| adaptive_norm | norm | L8 | 0.6289 | 40.2% | 9,794 | 82/100 | 0.3708 | 1.3 |
| group_G4_sqrtd | sqrt(d) | L8 | 0.0588 | 65.8% | 5,611 | 3/5 | -0.0009 | 59.3 |
| group_G4_norm | norm | L8 | 0.6287 | 39.9% | 9,851 | 87/100 | 0.3552 | 1.3 |
| standard | -- | L16 | 0.6139 | 69.0% | 5,077 | 91/100 | -- | -- |
| adaptive_sqrtd | sqrt(d) | L16 | 0.3399 | 92.0% | 1,307 | 78/100 | -0.0174 | 58.1 |
| adaptive_norm | norm | L16 | 0.6201 | 59.2% | 6,677 | 94/100 | 0.1022 | 4.2 |
| group_G4_sqrtd | sqrt(d) | L16 | 0.3360 | 91.4% | 1,416 | 76/100 | -0.0171 | 58.1 |
| group_G4_norm | norm | L16 | 0.6193 | 59.4% | 6,650 | 94/100 | 0.0914 | 4.2 |
| standard | -- | L24 | 0.6268 | 52.8% | 7,734 | 76/100 | -- | -- |
| adaptive_sqrtd | sqrt(d) | L24 | 0.5339 | 54.1% | 7,522 | 76/100 | -0.1363 | 50.3 |
| adaptive_norm | norm | L24 | 0.6570 | 28.2% | 11,761 | 88/100 | 0.0498 | 11.4 |
| group_G4_sqrtd | sqrt(d) | L24 | 0.5343 | 55.2% | 7,340 | 76/100 | -0.1360 | 50.4 |
| group_G4_norm | norm | L24 | 0.6570 | 30.1% | 11,446 | 87/100 | 0.0435 | 11.7 |

### Initialization comparison (norm-adaptive vs sqrt(d))

| Architecture | Layer | Mismatch | norm FVE | sqrt FVE | Delta FVE | Delta dead | Delta alive |
|---|---|---|---|---|---|---|---|
| adaptive | L8 | 28x | 0.6289 | 0.0588 | +0.5701 | -24.5pp | 1.69x |
| group_G4 | L8 | 28x | 0.6287 | 0.0588 | +0.5699 | -25.9pp | 1.76x |
| adaptive | L16 | 10x | 0.6201 | 0.3399 | +0.2802 | -32.8pp | 5.11x |
| group_G4 | L16 | 10x | 0.6193 | 0.3360 | +0.2833 | -31.9pp | 4.70x |
| adaptive | L24 | 3.6x | 0.6570 | 0.5339 | +0.1230 | -25.9pp | 1.56x |
| group_G4 | L24 | 3.6x | 0.6570 | 0.5343 | +0.1227 | -25.1pp | 1.56x |

### Cosine vs standard (norm-adaptive init only)

| Layer | standard FVE | cosine FVE | Delta FVE | standard dead% | cosine dead% | Delta dead |
|---|---|---|---|---|---|---|
| L8 | 0.6101 | 0.6289 | +1.9pp | 58.9% | 40.2% | -18.7pp |
| L16 | 0.6139 | 0.6201 | +0.6pp | 69.0% | 59.2% | -9.8pp |
| L24 | 0.6268 | 0.6570 | +3.0pp | 52.8% | 28.2% | -24.6pp |

### Convergence trajectories (L8)

sqrt(d) init (28x overshoot):

| Checkpoint | FVE | Dead% |
|---|---|---|
| 2% (244 steps) | -0.016 | 0% |
| 5% | -0.010 | 0% |
| 10% | 0.024 | 1% |
| 20% | 0.048 | 13% |
| 40% | 0.055 | 55% |
| 60% | 0.057 | 72% |
| 80% | 0.058 | 74% |

norm-adaptive init:

| Checkpoint | FVE | Dead% |
|---|---|---|
| 2% (244 steps) | 0.287 | 0% |
| 5% | 0.495 | 1% |
| 10% | 0.559 | 8% |
| 20% | 0.586 | 22% |
| 40% | 0.604 | 33% |
| 60% | 0.618 | 38% |
| 80% | 0.626 | 40% |

### Learned scale_a (norm-adaptive init)

| Layer | scale_a |
|---|---|
| L8 | +0.371 |
| L16 | +0.102 |
| L24 | +0.050 |

### scale_b convergence

| Variant | Layer | Init b(exp) | Final b(exp) | Movement |
|---|---|---|---|---|
| adaptive_sqrtd | L8 | 64.0 | 59.3 | -7% |
| adaptive_norm | L8 | 2.3 | 1.3 | -43% |
| adaptive_sqrtd | L16 | 64.0 | 58.1 | -9% |
| adaptive_norm | L16 | 6.3 | 4.2 | -33% |
| adaptive_sqrtd | L24 | 64.0 | 50.3 | -21% |
| adaptive_norm | L24 | 18.0 | 11.4 | -37% |

### Ablation cos>inner (norm-adaptive runs)

| Layer | adaptive cos>inner | group_G4 cos>inner |
|---|---|---|
| L8 | 82/100 | 87/100 |
| L16 | 94/100 | 94/100 |
| L24 | 88/100 | 87/100 |
