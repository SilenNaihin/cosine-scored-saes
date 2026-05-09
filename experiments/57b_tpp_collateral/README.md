# Experiment 57b: TPP collateral damage analysis

This experiment measures TPP (Targeted Perturbation Probing) across varying numbers of ablated features (N=2 to N=500) to characterize the intended vs unintended effect decomposition for standard and cosine SAEs. It also analyzes decoder projection geometry to test whether the TPP gap originates from decoder space differences.

## Results

### TPP by Number of Ablated Features

| N | Standard Total | Cosine Total | Ratio (cos/std) | Std Intended | Cos Intended | Std Unintended | Cos Unintended |
|---|---------------|-------------|-------|-------------|-------------|----------------|----------------|
| 2 | 0.0128 | 0.0052 | 0.41x | 0.0156 | 0.0059 | 0.0028 | 0.0007 |
| 5 | 0.0381 | 0.0060 | 0.16x | 0.0479 | 0.0066 | 0.0098 | 0.0007 |
| 10 | 0.0792 | 0.0179 | 0.23x | 0.0953 | 0.0200 | 0.0161 | 0.0021 |
| 20 | 0.1931 | 0.0522 | 0.27x | 0.2269 | 0.0572 | 0.0338 | 0.0050 |
| 50 | 0.2951 | 0.0944 | 0.32x | 0.3748 | 0.1016 | 0.0797 | 0.0072 |
| 100 | 0.2496 | 0.1630 | 0.65x | 0.4148 | 0.1756 | 0.1652 | 0.0126 |
| 500 | 0.1342 | 0.3532 | 2.63x | 0.4387 | 0.3702 | 0.3046 | 0.0170 |

### Precision Ratio (intended / unintended)

| N | Standard | Cosine |
|---|----------|--------|
| 2 | 5.6x | 8.4x |
| 5 | 4.9x | 9.4x |
| 10 | 5.9x | 9.5x |
| 20 | 6.7x | 11.4x |
| 50 | 4.7x | 14.1x |
| 100 | 2.5x | 13.9x |
| 500 | 1.4x | 21.8x |

### Decoder Projection Analysis (100 random probe directions)

| SAE | Alive | Mean Total Projection | Effective Dim | Concentration top-5 | Concentration top-100 |
|-----|-------|--------------------|--------------|--------------------|--------------------|
| standard | 65,536 | 815.6 +/- 14.3 | 49,343 +/- 74 | 0.040% | 0.66% |
| perfeature_l2 | 65,536 | 815.6 +/- 14.4 | 49,359 +/- 67 | 0.040% | 0.66% |
