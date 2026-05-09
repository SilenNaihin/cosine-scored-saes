# Experiment 41d: Gemma-2-2B 4-architecture comparison with aux-k

Five architectures (standard, adaptive_l2, perfeature_l2, perfeature_bd, no_C) trained with and without aux-k loss on Gemma-2-2B at layers 7, 13, and 19 (30 total SAEs). Uses the saprmarks recipe (Adam lr=5e-5, batch 2048, decoder grad projection, geometric median init). 50M tokens, d_sae=9216, k=80, aux-k alpha=1/32.

## Results

### SAEBench (No Aux-k)

| Layer | Variant | FVE | Alive% | SP top-1 | SP top-5 | SP full | KL w/ SAE |
|-------|---------|-----|--------|----------|----------|---------|-----------|
| L7 | standard | 0.862 | 100.0% | 0.7977 | 0.8637 | 0.9543 | 0.125 |
| L7 | adaptive_l2 | 0.863 | 100.0% | 0.7889 | 0.8790 | 0.9529 | 0.122 |
| L7 | perfeature_l2 | 0.864 | 100.0% | 0.7838 | 0.8713 | 0.9529 | 0.120 |
| L7 | perfeature_bd | 0.863 | 100.0% | 0.7788 | 0.8796 | 0.9519 | 0.122 |
| L7 | no_C | 0.855 | 100.0% | 0.7872 | 0.8324 | 0.9228 | 0.124 |
| L13 | standard | 0.849 | 100.0% | 0.7524 | 0.8503 | 0.9543 | 0.189 |
| L13 | adaptive_l2 | 0.851 | 99.9% | 0.7822 | 0.8731 | 0.9557 | 0.192 |
| L13 | perfeature_l2 | 0.852 | 99.9% | 0.7892 | 0.8617 | 0.9544 | 0.193 |
| L13 | perfeature_bd | 0.852 | 100.0% | 0.7891 | 0.8602 | 0.9548 | 0.190 |
| L13 | no_C | 0.839 | 100.0% | 0.7700 | 0.8334 | 0.9144 | 0.199 |
| L19 | standard | 0.857 | 100.0% | 0.8468 | 0.9139 | 0.9571 | 0.249 |
| L19 | adaptive_l2 | 0.864 | 99.8% | 0.8519 | 0.9100 | 0.9571 | 0.235 |
| L19 | perfeature_l2 | 0.846 | 66.4% | 0.8272 | 0.8972 | 0.9572 | 0.305 |
| L19 | perfeature_bd | 0.865 | 100.0% | 0.8472 | 0.9077 | 0.9562 | 0.234 |
| L19 | no_C | 0.854 | 100.0% | 0.8267 | 0.8747 | 0.9302 | 0.223 |

### SAEBench (With Aux-k)

| Layer | Variant | FVE | Alive% | SP top-1 | SP top-5 | SP full | KL w/ SAE |
|-------|---------|-----|--------|----------|----------|---------|-----------|
| L7 | standard | 0.862 | 100.0% | 0.7875 | 0.8740 | 0.9522 | 0.127 |
| L7 | adaptive_l2 | 0.863 | 100.0% | 0.7821 | 0.8739 | 0.9524 | 0.125 |
| L7 | perfeature_l2 | 0.864 | 100.0% | 0.7867 | 0.8768 | 0.9514 | 0.123 |
| L7 | perfeature_bd | 0.863 | 100.0% | 0.7819 | 0.8765 | 0.9536 | 0.123 |
| L7 | no_C | 0.855 | 100.0% | 0.7754 | 0.8243 | 0.9241 | 0.121 |
| L13 | standard | 0.849 | 100.0% | 0.7746 | 0.8593 | 0.9544 | 0.193 |
| L13 | adaptive_l2 | 0.851 | 100.0% | 0.7817 | 0.8494 | 0.9544 | 0.190 |
| L13 | perfeature_l2 | 0.852 | 99.9% | 0.7768 | 0.8621 | 0.9535 | 0.190 |
| L13 | perfeature_bd | 0.852 | 100.0% | 0.7914 | 0.8728 | 0.9546 | 0.190 |
| L13 | no_C | 0.839 | 100.0% | 0.7823 | 0.8262 | 0.9096 | 0.199 |
| L19 | standard | 0.857 | 100.0% | 0.8309 | 0.9124 | 0.9554 | 0.250 |
| L19 | adaptive_l2 | 0.864 | 99.8% | 0.8535 | 0.9094 | 0.9578 | 0.231 |
| L19 | perfeature_l2 | 0.847 | 65.6% | 0.8462 | 0.9014 | 0.9554 | 0.307 |
| L19 | perfeature_bd | 0.865 | 100.0% | 0.8104 | 0.9042 | 0.9567 | 0.227 |
| L19 | no_C | 0.854 | 100.0% | 0.8288 | 0.8794 | 0.9320 | 0.221 |

### Aux-k Effect (delta from adding aux-k)

| Layer | Variant | dAlive% | dSP top-1 | dFVE |
|-------|---------|---------|-----------|------|
| L7 | standard | +0.0pp | -1.0pp | -0.000 |
| L7 | adaptive_l2 | +0.0pp | -0.7pp | -0.000 |
| L7 | perfeature_l2 | +0.0pp | +0.3pp | +0.000 |
| L7 | perfeature_bd | +0.0pp | +0.3pp | +0.000 |
| L7 | no_C | +0.0pp | -1.2pp | +0.000 |
| L13 | standard | +0.0pp | +2.2pp | -0.001 |
| L13 | adaptive_l2 | +0.0pp | -0.1pp | +0.000 |
| L13 | perfeature_l2 | -0.0pp | -1.2pp | -0.000 |
| L13 | perfeature_bd | +0.0pp | +0.2pp | -0.000 |
| L13 | no_C | +0.0pp | +1.2pp | -0.000 |
| L19 | standard | +0.0pp | -1.6pp | -0.001 |
| L19 | adaptive_l2 | -0.0pp | +0.2pp | +0.000 |
| L19 | perfeature_l2 | -0.8pp | +1.9pp | +0.000 |
| L19 | perfeature_bd | +0.0pp | -3.7pp | -0.000 |
| L19 | no_C | +0.0pp | +0.2pp | +0.000 |

### Recipe Comparison (exp41c old recipe vs exp41d saprmarks)

| Variant | Layer | FVE old / new | SP top-1 old / new | Alive% old / new |
|---------|-------|---------------|--------------------|--------------------|
| standard | L7 | 0.836 / 0.862 | 0.760 / 0.798 | 45.9% / 100.0% |
| no_C | L7 | 0.859 / 0.855 | 0.751 / 0.787 | 99.9% / 100.0% |
| standard | L13 | 0.820 / 0.849 | 0.698 / 0.752 | 31.3% / 100.0% |
| no_C | L13 | 0.844 / 0.839 | 0.746 / 0.770 | 100.0% / 100.0% |
| standard | L19 | 0.793 / 0.857 | 0.813 / 0.847 | 22.1% / 100.0% |
| no_C | L19 | 0.855 / 0.854 | 0.830 / 0.827 | 100.0% / 100.0% |

### Perfeature_bd vs Perfeature_l2 at L19

| Metric | perfeature_l2 L19 | perfeature_bd L19 |
|--------|-------------------|-------------------|
| Alive% | 66.4% | 100.0% |
| FVE | 0.846 | 0.865 |
| SP top-1 | 0.827 | 0.847 |
| SP top-5 | 0.897 | 0.908 |
