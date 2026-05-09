# Experiment 36: 500M token production training

500M token SAE training on Qwen3-8B at layers 9, 18, and 27 comparing standard BatchTopK against adaptive_l2 (cosine-scored). Dictionary size 16,384 (4x expansion), k=80, LR=3e-4 with cosine decay. Mid-training checkpoints at 10/20/40/60/80/100% capture convergence trajectories. Note: L18 and L27 cosine runs used norm-adaptive init (suboptimal at this scale); L9 is a clean comparison.

## Results

### Final Metrics

| Run | FVE | Dead% | Alive | cos_recon | cos>inner | scale_a |
|-----|-----|-------|-------|-----------|-----------|---------|
| standard/L9 | 0.711 | 59.6% | 6,621 | 0.901 | 89/100 | -- |
| standard/L18 | 0.657 | 82.7% | 2,840 | 0.895 | 79/100 | -- |
| standard/L27 | 0.686 | 67.8% | 5,275 | 0.909 | 62/100 | -- |
| adaptive/L9 | 0.749 | 30.5% | 11,388 | 0.914 | 87/100 | 0.348 |
| adaptive/L18 | 0.665 | 77.2% | 3,733 | 0.897 | 88/100 | 0.268 |
| adaptive/L27 | 0.720 | 48.9% | 8,378 | 0.919 | 84/100 | 0.201 |

### Convergence Trajectories (Standard)

| Checkpoint | L9 FVE | L18 FVE | L27 FVE |
|---|---|---|---|
| 10% (50M) | 0.649 | 0.607 | 0.625 |
| 20% (100M) | 0.669 | 0.633 | 0.662 |
| 40% (200M) | 0.697 | 0.651 | 0.681 |
| 60% (300M) | 0.708 | 0.656 | 0.686 |
| 80% (400M) | 0.711 | 0.656 | 0.684 |
| 100% (500M) | 0.711 | 0.657 | 0.686 |

### Convergence Trajectories (Cosine)

| Checkpoint | L9 FVE | L18 FVE | L27 FVE |
|---|---|---|---|
| 10% (50M) | 0.717 | 0.616 | 0.678 |
| 20% (100M) | 0.735 | 0.644 | 0.704 |
| 40% (200M) | 0.745 | 0.665 | 0.717 |
| 60% (300M) | 0.748 | 0.667 | 0.720 |
| 80% (400M) | 0.749 | 0.668 | 0.720 |
| 100% (500M) | 0.749 | 0.665 | 0.720 |

### Dead Feature Trajectories

| Checkpoint | std/L9 dead | adp/L9 dead | std/L27 dead | adp/L27 dead |
|---|---|---|---|---|
| 10% (50M) | 70.2% | 40.6% | 79.0% | 58.3% |
| 20% (100M) | 64.7% | 34.6% | 73.8% | 54.4% |
| 40% (200M) | 61.8% | 31.7% | 70.8% | 51.0% |
| 60% (300M) | 59.7% | 30.9% | 69.8% | 49.6% |
| 80% (400M) | 59.7% | 30.6% | 67.8% | 49.0% |
| 100% (500M) | 59.6% | 30.5% | 67.8% | 48.9% |

### Learned Scale Parameters

| Layer | Mean norm | sqrt(d) | Ratio | scale_a |
|---|---|---|---|---|
| L9 | 57.5 | 64.0 | 0.90 | 0.348 |
| L18 | 97.7 | 64.0 | 1.53 | 0.268 |
| L27 | 404.7 | 64.0 | 6.32 | 0.201 |
