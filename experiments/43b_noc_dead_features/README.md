# Experiment 43b: NoC dead feature investigation

Five empirical investigations on the 500M-token final checkpoints determine why NoC has 3.9% persistently dead features while all other architectures reach ~0%. The root cause is the interaction between NoC's bounded [0,1] cosine activation space and the BatchTopK + aux-k loss mechanisms, which assume activations can scale freely.

## Results

### Activation distribution mismatch

| Architecture | Alive pre-topk mean | Alive pre-topk max | TopK threshold | Dead pre-topk max |
|---|---|---|---|---|
| Standard | 0.0125 | 77.83 | 1.908 | 1.047 |
| Adaptive L2 | 0.0104 | 73.63 | 1.877 | -- |
| Per-Feature L2 | 0.0136 | 83.96 | 1.922 | -- |
| NoC | 0.0001 | 0.580 | 0.0105 | 0.016 |

### Aux loss gradient magnitude

| Architecture | Mean aux grad norm (dead enc) | Aux loss value |
|---|---|---|
| Standard | 5.4e-7 | 1.011 |
| Adaptive L2 | 2.0e-6 | 1.008 |
| Per-Feature L2 | 2.0e-6 | 1.008 |
| NoC | 1.3e-12 | 1.023 |

### Aux reconstruction quality

| Architecture | Aux recon norm | Residual norm | Ratio | Aux FVE |
|---|---|---|---|---|
| Standard | 0.0003 | 35.51 | 8.5e-6 | ~0 |
| Adaptive L2 | 0.876 | 35.64 | 0.025 | 0.002 |
| Per-Feature L2 | 0.919 | 35.52 | 0.026 | 0.002 |
| NoC | 7.9e-7 | 35.94 | 2.2e-8 | 0 |

### Weight structure (encoder-decoder alignment)

| Architecture | Dead enc-dec alignment | Alive enc-dec alignment |
|---|---|---|
| Standard | 0.551 | 0.607 |
| NoC | 0.808 | 0.676 |

### Training dynamics (dead feature count)

| Checkpoint | NoC dead | Standard dead |
|---|---|---|
| 50M | 938 | 1 |
| 100M | 2,741 | 0 |
| 200M | 2,762 | 1 |
| 300M | 2,647 | 1 |
| 400M | 2,618 | 4 |
| 500M | 2,549 | 3 |
