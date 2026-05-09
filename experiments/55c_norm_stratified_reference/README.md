# Experiment 55c: Norm-stratified FVE on reference SAE

Three SAEs (standard, adaptive cosine, and an independently trained reference) are evaluated using norm-stratified FVE at L9, L18, and L27 on Qwen3-8B. Inputs are split into quartiles by activation norm (Q1=lowest, Q4=highest) and FVE is computed per quartile to test whether the high-norm reconstruction catastrophe generalizes to an independent standard SAE and varies with depth.

## Results

Q4 (highest-norm quartile) FVE:

| SAE | L9 Q4 | L18 Q4 | L27 Q4 |
|-----|--------|--------|--------|
| standard | -185.1 | -183.5 | 0.25 |
| adaptive_l2 | 0.05 | 0.33 | 0.76 |
| independent_ref | 1.00 | -136.3 | -7.0 |

Full per-quartile FVE, Layer 9:

| SAE | Q1 (low) | Q2 | Q3 | Q4 (high) |
|-----|----------|----|----|-----------|
| standard | 0.732 | 0.774 | 0.789 | -185.1 |
| adaptive_l2 | 0.724 | 0.775 | 0.793 | 0.045 |
| independent_ref | 0.773 | 0.812 | 0.826 | 1.000 |

Full per-quartile FVE, Layer 18:

| SAE | Q1 (low) | Q2 | Q3 | Q4 (high) |
|-----|----------|----|----|-----------|
| standard | 0.742 | 0.763 | 0.777 | -183.5 |
| adaptive_l2 | 0.737 | 0.760 | 0.777 | 0.328 |
| independent_ref | 0.776 | 0.806 | 0.819 | -136.3 |

Full per-quartile FVE, Layer 27:

| SAE | Q1 (low) | Q2 | Q3 | Q4 (high) |
|-----|----------|----|----|-----------|
| standard | 0.747 | 0.763 | 0.776 | 0.247 |
| adaptive_l2 | 0.748 | 0.770 | 0.781 | 0.760 |
| independent_ref | 0.779 | 0.800 | 0.810 | -7.0 |

L2 ratio (reconstruction norm / input norm) at Q4:

| SAE | L9 Q4 | L18 Q4 | L27 Q4 |
|-----|--------|--------|--------|
| standard | 10.82 | 9.54 | 1.14 |
| adaptive_l2 | 0.26 | 0.78 | 0.81 |
| independent_ref | 0.98 | 5.87 | 1.67 |
