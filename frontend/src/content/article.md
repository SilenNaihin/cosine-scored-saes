<Abstract>
Sparse autoencoders (SAEs) detect features via inner product, so a feature's activation scales with both its directional alignment and the input's norm. Under BatchTopK, high-norm tokens inflate all pre-activations simultaneously, claiming dictionary slots regardless of content alignment. This matters because sublayer normalization has already discarded the magnitude the score measures, so the encoder detects a quantity the model does not read. We replace the score with a learned blend of cosine similarity and input magnitude, letting the optimizer choose how much norm to use; a per-feature extension lets each feature decide independently. In both regimes, training is free to recover inner product but never does, with no feature ever choosing more than half-magnitude dependence. At matched reconstruction, the cosine encoder learns features that align with human-recognizable concepts far more often than standard, filling dictionary slots that inner product wastes on norm detectors. Loss reweighting that equalizes gradients barely closes the gap, confirming forward-pass score geometry as the lever. The advantage is not universal across tasks or depths, but we believe cosine scoring should be the default for dictionary learning on normalized representations.
</Abstract>

<ScoreGeometry />

<Mechanism />

## 1  Introduction

Sparse autoencoders are a standard dictionary-learning tool for mechanistic interpretability. 

The interpretation depends on how the encoder detects those directions. The standard rule is an inner product $\langle w_i, x\rangle$, so a feature fires in proportion to both its directional alignment with $x$ and the magnitude $\lVert x\rVert$.

This is the wrong scoring geometry for transformers with pre-sublayer normalization. The inner-product score mixes content alignment with token norm, but normalization strips that norm before the model reads the activation. <DetailNote tip="BatchTopK uses one batch-wide activation budget. High-norm tokens inflate many pre-activations at once, so they can claim sparse-code slots even when their directions are less aligned.">BatchTopK detail.</DetailNote>

Over training, this selection pressure starves content-encoding features: they rarely win TopK slots, receive no learning signal, and never develop. The result is a dictionary dominated by features that fire on magnitude rather than meaning. This is not a minor calibration issue. If dictionary elements fire on norm rather than content, they cannot reliably be used for sparse probing, feature-based steering, or circuit analysis; the practitioner interprets them as model concepts, but the features encode a quantity the model discards.

We replace the inner-product score with a cosine score scaled by a learned exponent $a$ on the input norm:
$$
s_i(x) = \lVert x\rVert^a \cdot \cos(x, w_i) + b_{\mathrm{enc},i}.
$$

This interpolates between cosine ($a=0$) and inner product ($a=1$). The key property is that the optimizer is free to recover inner product if magnitude is genuinely useful, but is not forced to use it by default. <DetailNote tip="We test both a single global exponent shared across features and a per-feature extension where each feature can learn its own norm dependence.">Variants: global and per-feature.</DetailNote>

Through experiments on Qwen3-8B and Gemma-2-2B, we demonstrate that cosine-scored SAEs match standard BatchTopK on reconstruction fidelity, model-behavior preservation, and per-feature interpretability, while producing dictionaries that align far more often with human-recognizable concepts. A matched-feature decomposition reveals that most of the probing gap comes not from improving shared features, but from features the standard encoder never learns at all; its dictionary slots are instead occupied by norm detectors.

## 2  Headline result

<ProbeWin />

<Hero />

At matched reconstruction (<DetailNote tip="FVE means fraction of variance explained: 1 minus reconstruction MSE divided by the activation variance. A value near 0.77 means the SAE explains about 77% of the residual-stream variance under this metric.">FVE $\approx 0.77$</DetailNote>), the per-feature cosine encoder raises sparse-probing top-1 by +14.9% over the standard SAE. It wins 7 of 8 SAEBench probing datasets; sentiment is the exception.

The failure mode is visible in the high-norm tail. Standard features concentrate on high-norm tokens and over-reconstruct them at 9.5x input norm, while cosine stays close to scale at 0.55x.

## 3  Where the gap comes from

<Decomposition />

Pairing features by activation correlation $\geq 0.7$ and decoder cosine $> 0.7$ matches 8,661 features. Of standard's 64,450 alive features, only 13.4% direction-encode under this criterion; the rest are norm-conditioned.

Restricting probing to shared features drops the top-1 gap from +14.9% to $\approx +2\%$. The gain is mostly feature discovery: cosine learns content features that standard never develops.

## 4  Robustness and mechanism checks

<ProbingTable />

<GradientFig />

FVE is fraction of variance explained:
$$
\mathrm{FVE} = 1 - \frac{\mathbb{E}\lVert x - \hat{x}\rVert^2}{\mathbb{E}\lVert x - \bar{x}\rVert^2}.
$$
We use it as a reconstruction-control metric: cosine and standard are compared at the same FVE, so the probing gap is not coming from one SAE simply reconstructing better.

Multi-seed runs ($n = 3$) keep FVE matched within 0.4%, and learned $a$ remains far below $1$ (global $a \approx 0.26$; per-feature mean $a \approx 0.076$). The advantage replicates across layers and on Gemma-2-2B.

Cosine is also more sample-efficient: trained on 10x fewer tokens, it matches the standard 500M-token reference on probing.

The mechanism checks point to the forward pass. Reweighting the reconstruction loss to equalize high- and low-norm gradient contributions barely changes probing: the mild $1/\lVert x\rVert$ reweighting closes only 1.9% of the gap in the [gradient-equalizing experiment](#fig-9), and even $1/\lVert x\rVert^2$ remains far below cosine. In score-swap tests, probing changes with the score used at evaluation; reconstruction mostly tracks the score used during training.

## 5  Feature browser and scope

<FeatureBrowser />

The browser shows top-activating contexts from the published checkpoints, with standard and cosine features side by side. Per-feature LLM-judged interpretability is matched; the advantage is in which features are discovered.

<AutoInterp />

For more details, including additional investigations into training dynamics, hyperparameters, and model-architecture effects, see the [paper](https://arxiv.org/abs/2606.15054).
