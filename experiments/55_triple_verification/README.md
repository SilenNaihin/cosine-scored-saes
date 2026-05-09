# Experiment 55: Triple verification meta-experiment

A battery of independent verification tests for four headline claims: concept alignment (+14pp sparse probing), high-norm catastrophe (Q4 FVE=-197), sample efficiency (50M cosine > 500M standard), and feature interpretability. Sub-experiments span bootstrap probing, norm-stratified FVE on reference SAEs, additional SAEBench metrics (absorption/SCR/TPP), feature steering, and cross-budget sparse probing comparisons.

## Results

Verification matrix for Claim 1 (concept alignment):

| Verification | Result |
|---|---|
| Depth generalization (L9/L27) | +12.0pp L9, +8.4pp L27 |
| Independent reference baseline | ref matches standard within 0.3pp at L18/L27 |
| Cross-model (Gemma-2-2b) | +3.42pp +/- 0.29pp (12 sigma) at L13 |
| Bootstrap sparse probing (5 seeds) | 13.86pp +/- 0.63pp (22 sigma) |
| LLM auto-interp (500M) | standard 25.0% rate vs cosine 19-24%; no cosine advantage |
| Absorption | perfeature_l2 0.170, standard 0.125, adaptive_l2 0.098; no consistent advantage |
| SCR @100-@500 | cosine 0.34-0.38 vs standard/ref collapse to -0.18/-0.22 |
| TPP @20 | standard 0.193 vs cosine 0.052; cosine precision 10-20x vs standard 6.7x |

Verification matrix for Claim 2 (high-norm catastrophe):

| Verification | Result |
|---|---|
| Independent reference L18 | Q4 FVE = -136.3, L2 ratio 5.87x |
| Depth generalization | standard Q4: -185.1 (L9), -183.5 (L18), 0.25 (L27) |
| Cosine all depths | adaptive_l2 Q4: 0.05 (L9), 0.33 (L18), 0.76 (L27) |

Verification matrix for Claim 3 (sample efficiency):

| Comparison | Cosine 50M | Standard 500M (ref) | Delta |
|---|---|---|---|
| L9 sparse probing | 0.786 | 0.613 | +17.3pp |
| L27 sparse probing | 0.849 | 0.762 | +8.7pp |

Verification matrix for Claim 4 (feature steering):

| Amplification factor | Standard KL | Cosine KL | Ratio |
|---|---|---|---|
| x0.0 (ablation) | 0.0024 | 0.0026 | 0.94x |
| x0.5 (dampen) | 0.0009 | 0.0009 | 0.96x |
| x2.0 (amplify) | 0.0024 | 0.0023 | 1.00x |
| x5.0 (strong) | 0.0287 | 0.0287 | 1.00x |
