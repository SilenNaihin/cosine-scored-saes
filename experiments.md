# Experiments

All experiments conducted during this research, with results summaries.

## Diagnostic Evidence (Exp 1–9)

| # | Name | Key Result |
|---|------|-----------|
| 1 | LayerNorm Erasure | Per-layer magnitude invariance confirmed; cross-layer composition matters |
| 2 | Magnitude Confound | cos (0.720) > inner (0.644) > SAE (0.576) > norm for ablation prediction |
| 2b | Outlier Analysis | Cosine correct 86.7% in adversarial pairs |
| 3 | LLM Interpretability | Cosine-detected tokens 3.6x more interpretable (31.4% vs 8.7%) |
| 4 | Feature Splitting | Null result — no magnitude-based duplicate directions in SAE decoder |
| 4b | Pair Details | 18 near-similar pairs have completely different activation patterns |
| 5 | Multi-Layer Hardened | cos wins at layers 9, 18, 27 (80%, 76%, 74%) with logit KL divergence |
| 6 | Pythia-70M Replication | RNH replicates on different model/SAE family; layer 5 cos→KL = 0.721 |
| 7 | Pythia-70M (Revised) | Weak replication (47% cos>inner); LayerNorm mean subtraction complicates |
| 7b | Pythia-2.8B LayerNorm | Larger LayerNorm model tested to separate confounds |
| 8 | Activation Patching | SAE-free: direction patching produces 10x more KL change than norm patching |
| 9 | Adversarial Norm | Cosine correct 90%; high-cosine tokens have 10x more causal impact |

## Cosine SAE Development (Exp 10–22)

| # | Name | Key Result |
|---|------|-----------|
| 10 | Cosine SAE Training | Cosine encoder beats standard at L9/L18; breaks at L27 |
| 11 | Mistral-7B Replication | cos>inner 74–88% across layers; RNH holds across RMSNorm families |
| 12 | Adaptive Scaling | Optimizer picks a=0.1 (magnitude ~90% noise); fixes L27 |
| 13 | Cosine-Based Ablation | Standard's L27 lead shrinks 70% under cosine ablation |
| 14 | Post-RMSNorm Loss | First cosine variant to beat standard on SAE→KL (0.252 vs 0.198 at L27) |
| 15 | Shared Feature Eval | Standard shares <1% features with cosine; loss function matters more |
| 16 | Per-Feature Scaling | 88–99% of features want a=0 (pure cosine); none wants inner-product |
| 17 | Production Scale | Cosine beats standard at all layers at 50M; +8 FVE at L27; 3.3x more alive |
| 18 | Adaptive + Post-Norm | Two improvements don't compose; a froze |
| 19 | Norm-Stratified | Cosine advantage uniform across norm quartiles; within-quartile ~50% (Simpson's paradox) |
| 20 | SAEBench Eval | Cosine wins at all layers on KL, CE, CosSim; +2.5pp KL at L27 |
| 21 | Composite Loss | FVE-causality tension is fundamental; no α bridges the gap |
| 22 | Postnorm Scale | Postnorm advantage narrows from +27% (5M) to +3.5% (50M); adaptive_l2 clear winner |

## Architecture Ablations (Exp 23–31)

| # | Name | Key Result |
|---|------|-----------|
| 23 | Input Norm Ablation | Advantage is from geometry (both norms), not input normalization alone |
| 24 | LayerNorm Control | cos>inner 100% at L8, collapses to 40% at L24; advantage is RMSNorm-specific at depth |
| 25 | Multi-Model Matrix | 5-model 2×2 confirms: LN cos>inner collapse replicates (3 models) |
| 25a | Pythia-6.9B | LayerNorm depth collapse confirmed |
| 25b | Falcon-7B | LayerNorm depth collapse confirmed |
| 25c | Mistral-7B | RMSNorm advantage confirmed |
| 26 | Dictionary Size | 3x dictionary does NOT match cosine at same L0; standard_49k has 89% dead |
| 27 | Norm-Adaptive Init | scale_b = log(mean(‖x‖)) fixes cosine SAE: +13.6pp FVE at L27 |
| 28 | Gradient Analysis | Standard W_enc gradients are Q4-dominated (1.55x); cosine balanced (1.03x) |
| 29 | Post-Hoc Normalization | Post-hoc cosine on standard SAE destroys FVE; advantage is training-time |
| 30 | LR Sweep | Both prefer LR=3e-4; cosine 2.9x more LR-robust |
| 31 | Norm Injection | Cosine FVE degrades more under noise but keeps more features alive |

## Production Validation (Exp 32–38)

| # | Name | Key Result |
|---|------|-----------|
| 32 | Open-Source Baseline | Our standard valid (-9pp vs 500M ref); cosine 50M within 2.9pp of ref 500M |
| 33 | Feature Interpretability | Rates indistinguishable (40% vs 37%, p=0.88); cosine 2.5x more total |
| 34 | Multi-Seed | All gaps >37σ; FVE 0.737±0.001 vs 0.657±0.001; 3.26x alive |
| 34b | Multi-Seed (sqrt(d)) | sqrt(d) init is correct at 50M |
| 35 | Gemma SAEBench | Full 6-eval SAEBench on Gemma-2-2B; cosine wins 14/22 categories |
| 36 | Production 500M | Cosine beats standard at all layers at 500M; 50M cosine ≈ 500M standard |
| 37 | Regularized Training | Goldilocks variants tested |
| 38 | Residual Unit-Norm KL | Norm sensitivity rises with depth, peaks at L34; final layer collapses |

## saprmarks Recipe (Exp 39–48)

| # | Name | Key Result |
|---|------|-----------|
| 39 | Norm-Preserving SAE (NoC) | NoC architecture development and validation |
| 40 | 4-Arch 500M (saprmarks) | All 4 architectures match FVE (0.767–0.771) at 500M; NoC wins RNH diagnostic |
| 41 | Optimal 500M | sqrt(d) init fixes L27; group_G4 is best (cos>inner 87%) |
| 41a | Gemma Multilayer | Gemma-2-2B 50M across layers |
| 41d | Gemma 4-Architecture | 5-architecture controlled comparison on Gemma; gap is 0.3–3.3pp at 4× expansion |
| 42 | Mistral Init | Norm-adaptive fixes all layers; cosine +1.9–3.0pp FVE |
| 42c | NoC 500M (saprmarks) | NoC matches standard: FVE 0.767 vs 0.770; cos>inner 69% vs 62% |
| 43 | 4-Arch 50M L27 | 50M at L27 with all 4 architectures |
| 43b | NoC Dead Features | Bounded activation space starving aux loss |
| 43c | 4-Arch 50M L9 | All architectures at L9 |
| 43d | 4-Arch 50M L18 | All architectures at L18 |
| 44 | Norm-Stratified FVE | Standard Q4 FVE = -184; cosine = +0.33; 10x norm amplification |
| 45 | Encoder Norm Ablation | Removing encoder norm kills adaptive cosine (90% dead) |
| 46 | Normscope Ablation | Full factorial of normalization knobs |
| 47 | Per-Feature Robustness | Per-feature variant collapse investigation |
| 48 | Input Norm Verification | Input norm alone → 93.5% dead; compensation required |

## Mechanism Investigation (Exp 51–60)

| # | Name | Key Result |
|---|------|-----------|
| 51 | SAEBench L9/L18/L27 | +14.9pp (L18), +12pp (L9), +8.4pp (L27) sparse probing |
| 53 | LLM Auto-Interp 500M | Per-feature rates matched at 500M; no cosine quality advantage |
| 54 | Gradient (saprmarks) | Gradient equalization persists with aux-k loss |
| 55 | Triple Verification | Meta-experiment confirming all headline claims from multiple angles |
| 55a | Additional SAEBench | Absorption: perfeature_l2 best (0.170); TPP/SCR precision-vs-power tradeoff |
| 55b | Bootstrap Probing | Gap = 13.86pp ± 0.63pp (22σ) over 5 probe seeds |
| 55c | Norm-Stratified Reference | Q4 catastrophe confirmed on independent reference SAE (-136) |
| 55d | Feature Steering | Per-feature power identical (ratio 0.94–1.00x); gap is which features exist |
| 56 | Activation Analysis | Activation magnitude and decoder analysis |
| 56a | Co-Activation | Feature co-activation patterns |
| 56b | Feature Overlap | Feature overlap quantification |
| 57 | Scaling Matrix | 3×3 (1.7B/4B/8B × 4x/8x/16x): advantage universal; dictionary size not the driver |
| 57a | TPP Matched Pairs | TPP on matched feature pairs |
| 57b | TPP Collateral | Standard collateral damage explodes at high N (0.305 vs 0.017 for cosine) |
| 57c | Gemma Replication | +3.42pp ± 0.29pp (12σ) on Gemma-2-2B |
| 58a | Probe Weight Analysis | Decoder geometry identical (entropy ratio 1.0000) |
| 58b | Matched-Feature Probing | 87% of gap = feature discovery; 13% = separability |
| 58c | Norm-Detector Characterization | 86.3% of standard's unmatched features fire on Q4; 48× activation magnitude |
| 59 | Gradient Equalization | Loss reweighting closes only 1.9–6.8% of gap; forward pass is the lever |
| 60 | Decoder Geometry | Cosine: lower MMC (0.293 vs 0.311), 2.5× fewer tight pairs at 16× expansion |
| 61 | Multi-Seed 500M | n=3 seeds at headline: FVE matched ±0.0002; top-1 gap stable +14.1% (global) / +14.6% (per-feature); a seed-invariant |
| 65 | Salient-Concept Control | Both arms cleanly represent salient concepts (Python/French); cosine advantage is aggregate, not single-concept |
| 66 | Sparsity-Budget Sweep | k∈{10..160}: probing advantage robust (+8 to +11.4%) at all k; auto-interp parity holds at all k |

## Camera-Ready Revision (Exp 61+)

| # | Name | Key Result |
|---|------|-----------|
| 62 | Headline Interp + Causal (`62_interp_causal_headline`) | 62a DONE (1000 feat, single-seed HF ckpts): per-feature interp MATCHED at scale (std 20.1 / global 21.3 / per-feature 19.2%); advantage is discovery/volume not legibility; exp53 freq-crossover does NOT replicate. Replaces exp33/exp40. A4 causal spec ready (Chat 1 runs). |
