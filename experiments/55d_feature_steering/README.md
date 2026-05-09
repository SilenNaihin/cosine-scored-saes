# Experiment 55d: Feature steering behavioral test

For standard and per-feature cosine SAEs at L18 (Qwen3-8B, 500M tokens), the top-10 most consistently active features across 50 diverse test prompts are identified. Each feature is amplified/ablated at four factors (x0.0, x0.5, x2.0, x5.0) and the resulting KL divergence and top-5 probability shift of the output distribution are measured to compare per-feature behavioral power across architectures.

## Results

Aggregate steering effects:

| Factor | Standard KL | Cosine KL | Ratio | Standard Shift | Cosine Shift |
|--------|------------|-----------|-------|----------------|--------------|
| x0.0 (ablation) | 0.0024 | 0.0026 | 0.94x | 0.0177 | 0.0180 |
| x0.5 (dampen) | 0.0009 | 0.0009 | 0.96x | 0.0121 | 0.0117 |
| x2.0 (amplify) | 0.0024 | 0.0023 | 1.00x | 0.0181 | 0.0174 |
| x5.0 (strong) | 0.0287 | 0.0287 | 1.00x | 0.0560 | 0.0555 |
