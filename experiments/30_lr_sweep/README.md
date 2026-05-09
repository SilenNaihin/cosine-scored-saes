# Experiment 30: Learning Rate Sweep Across Architectures

This experiment sweeps learning rates {1e-4, 3e-4, 1e-3} for both standard and adaptive cosine encoders to test whether the two architectures prefer different LRs, which would confound prior comparisons run at LR=3e-4. All runs train on Qwen3-8B layer 27 for 50M tokens with BatchTopK (k=80, d_sae=16384).

## Results

### Reconstruction

| Encoder | LR | FVE | Dead% | Alive |
|---------|-----|-----|-------|-------|
| standard | 1e-4 | 0.582 | 88.5% | 1,888 |
| standard | 3e-4 | 0.659 | 77.4% | 3,709 |
| standard | 1e-3 | 0.581 | 58.6% | 6,790 |
| adaptive | 1e-4 | 0.711 | 26.0% | 12,125 |
| adaptive | 3e-4 | 0.738 | 28.2% | 11,768 |
| adaptive | 1e-3 | 0.716 | 49.3% | 8,307 |

### Optimal LR

| Encoder | Best LR | Best FVE | FVE spread across sweep |
|---------|---------|----------|------------------------|
| Standard | 3e-4 | 0.659 | 7.8pp |
| Adaptive | 3e-4 | 0.738 | 2.7pp |

### Gap at Each LR

| LR | Standard FVE | Adaptive FVE | Gap |
|----|--------------|--------------|-----|
| 1e-4 | 0.582 | 0.711 | +12.9pp |
| 3e-4 | 0.659 | 0.738 | +7.9pp |
| 1e-3 | 0.581 | 0.716 | +13.5pp |

### Ablation

| Encoder | LR | cos_KL | inner_KL | sae_KL | cos>inner |
|---------|-----|--------|----------|--------|-----------|
| standard | 1e-4 | 0.383 | 0.371 | 0.363 | 74/100 |
| standard | 3e-4 | 0.415 | 0.391 | 0.367 | 79/100 |
| standard | 1e-3 | 0.440 | 0.423 | 0.403 | 73/100 |
| adaptive | 1e-4 | 0.413 | 0.392 | 0.367 | 84/100 |
| adaptive | 3e-4 | 0.397 | 0.379 | 0.352 | 84/100 |
| adaptive | 1e-3 | 0.385 | 0.366 | 0.306 | 83/100 |

### Learned scale_a

| LR | scale_a |
|----|---------|
| 1e-4 | 0.236 |
| 3e-4 | 0.207 |
| 1e-3 | 0.178 |

### Training Dynamics (FVE at step checkpoints, of 12,207 total)

Standard encoder:

| Step | LR=1e-4 | LR=3e-4 | LR=1e-3 |
|------|---------|---------|---------|
| 600 | 0.260 | 0.427 | 0.551 |
| 6400 | 0.569 | 0.651 | 0.430 |
| 12207 | 0.585 | 0.663 | 0.576 |

Adaptive encoder:

| Step | LR=1e-4 | LR=3e-4 | LR=1e-3 |
|------|---------|---------|---------|
| 600 | 0.108 | 0.293 | 0.547 |
| 6400 | 0.699 | 0.734 | 0.715 |
| 12207 | 0.715 | 0.741 | 0.720 |
