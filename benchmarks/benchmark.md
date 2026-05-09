# Benchmark Strategy for RNH Validation

## Why Standard Benchmarks Matter

Our experiments (1-6) show cosine similarity outperforms inner product as a feature detector across models, layers, and SAE families. But these are custom metrics (correlation with ablation KL divergence). Standard MI benchmarks test whether this advantage translates to practical tasks: feature disentanglement, concept detection, causal variable isolation, and sparse probing accuracy.

If a cosine-based SAE scores higher on SAEBench's disentanglement metrics (SCR, TPP, RAVEL) than an inner-product SAE with the same architecture, that's publishable evidence that RNH has practical implications — not just a correlation improvement.

## Benchmarks

### SAEBench (Primary) — ICML 2025

**Paper:** Karvonen, Rager, Lin, Tigges, Bloom, Chanin, Nanda et al.
**Repo:** github.com/adamkarvonen/SAEBench

The community standard for SAE evaluation. Tests 8 metrics across interpretability, disentanglement, and downstream utility. We use it as our primary benchmark.

**Re<author>ant metrics for RNH:**

| Metric | What it tests | Why it matters for RNH |
|--------|--------------|----------------------|
| **Sparse Probing** | Can SAE features classify concepts? | If cosine-detected features are more semantically meaningful, probing accuracy should improve |
| **SCR** (Spurious Correlation Removal) | Can ablating irre<author>ant SAE features remove spurious cues? | Depends on correctly identifying which features are active — cosine should help |
| **TPP** (Targeted Probe Perturbation) | Can SAE features disentangle similar concepts? | Direct test of feature detection quality |
| **RAVEL** | Can features isolate independent knowledge dimensions? | Disentanglement requires accurate feature activation |
| **Absorption** | Do hierarchical features get improperly absorbed? | If magnitude noise causes absorption, cosine-based detection may reduce it |
| **Core** (L0, loss recovered) | Basic reconstruction and sparsity | Sanity check — cosine SAE should not degrade reconstruction |
| **AutoInterp** | LLM-generated feature explanations | Aligns with our Exp 3 (cosine-detected tokens are 3.6x more interpretable) |

**Model:** Qwen/Qwen3-8B (supported by TransformerLens v2.16+)
**Evals we run:** core, sparse_probing, absorption (safe defaults). Then ravel, scr, tpp after validating Qwen3 compatibility. AutoInterp requires an OpenAI API key.

**Key insight:** SAEBench found that proxy metrics (L0, loss recovered) do NOT reliably predict practical performance. This aligns with our thesis — standard SAEs may score well on proxies while encoding magnitude noise.

### MIB Track 2 (Secondary) — ICML 2025

**Paper:** Mueller, Geiger, Wiegreffe et al.
**Repo:** github.com/aaronmueller/MIB

Tests whether featurized representations can isolate causal variables via interchange interventions. The key negative result from MIB: **SAE features provide no advantage over raw neurons** for causal variable localization. If cosine-based detection improves this, it directly validates RNH's practical importance.

**Metric:** Interchange intervention accuracy (0.0-1.0) — swap features between base/source inputs that differ on a causal variable, check if model output changes correctly.

**Constraint:** Supports Qwen2.5-0.5B, not Qwen3-8B. To use this benchmark, we need SAEs trained on Qwen2.5-0.5B. The training pipeline (`src/custom_encoder/train.py`) supports any HF model.

**Tasks:** IOI (indirect object identification), MCQA (multiple choice QA), ARC.

### AxBench (Excluded for Now)

**Paper:** Wu, Arora, Geiger, Manning, Potts
**Repo:** github.com/stanfordnlp/axbench

Tests concept detection and steering. Found that SAEs are not competitive — even difference-in-means beats them. Highly re<author>ant for RNH, but **concepts are defined by GemmaScope SAE labels from Neuronpedia**, making it Gemma-specific. Cannot use with Qwen without creating entirely new concept definitions.

**Future work:** If we expand to Gemma-2-2B (which SAEBench also supports), AxBench becomes viable.

## Adapter Design

The `BenchSAE` class in `adapter.py` wraps any SAE into SAEBench's `BaseSAE` interface. Factory functions handle different SAE sources:

```python
# Existing inner-product baseline
sae = BenchSAE.from_adamkarvonen(hook_layer=18, device="cuda")

# Custom-trained cosine SAE (plug-in point for other chat)
sae = BenchSAE.from_custom_checkpoint("path/to/checkpoint.pt", hook_layer=18, device="cuda")

# sae-lens SAE
sae = BenchSAE.from_saelens("pythia-70m-deduped-res-sm", sae_id="...", hook_layer=3, device="cuda")
```

All factories ensure decoder rows are unit-normalized (required by SAEBench's SCR/TPP/absorption evals).

The `CosineAutoEncoder` returns `(features, x_norm)` from encode — the adapter strips the tuple and caches the norm for decode. The `CosineBatchTopKAutoEncoder` uses standard single-tensor encode/decode, so it's a clean drop-in.

## Running Instructions

### SAEBench (GPU required)

```bash
# Install benchmark deps
pip install sae-bench transformer-lens

# Run core eval with existing inner-product SAE (fastest, ~10 min on A100)
python benchmarks/run_saebench.py --layer 18 --evals core

# Run full safe eval suite
python benchmarks/run_saebench.py --layer 18 --evals core sparse_probing absorption

# Run with custom SAE from Python
python -c "
from benchmarks.adapter import BenchSAE
from benchmarks.run_saebench import run_saebench
sae = BenchSAE.from_custom_checkpoint('path/to/cosine_sae.pt', hook_layer=18, device='cuda')
run_saebench(sae, 'rnh-cosine-l18', eval_types=['core', 'sparse_probing', 'absorption'])
"
```

Results saved to `benchmarks/eval_results/{eval_type}/`.

### MIB Track 2 (GPU required, Qwen2.5-0.5B SAE needed)

```bash
# Clone MIB
git clone https://github.com/aaronmueller/MIB.git benchmarks/MIB
cd benchmarks/MIB && git submodule update --init --recursive
pip install -r MIB-causal-variable-track/requirements.txt

# Prepare submission
python -c "
from benchmarks.adapter import BenchSAE
from benchmarks.run_mib import prepare_submission
sae = BenchSAE.from_custom_checkpoint('path/to/qwen2.5_sae.pt', hook_layer=12, model_name='Qwen/Qwen2.5-0.5B', device='cuda')
prepare_submission(sae, 'rnh-cosine')
"

# Evaluate
python benchmarks/MIB/MIB-causal-variable-track/evaluate_submission.py \
    --submission_folder benchmarks/mib_submissions/rnh-cosine
```

### Smoke Tests (CPU, no model download needed)

```bash
python benchmarks/test_harness.py
```

## Interpreting Results for RNH

The core comparison: run identical SAEBench evals on two SAEs that differ only in their encoder (inner product vs cosine similarity), trained on the same data with the same architecture otherwise.

**Strong RNH evidence** if the cosine SAE shows:
- Higher sparse probing accuracy (features detect the right concepts)
- Higher SCR/TPP scores (features disentangle concepts better)
- Higher RAVEL disentanglement
- Lower or equal absorption (cosine doesn't make absorption worse)
- Comparable core metrics (reconstruction quality isn't sacrificed)

**Particularly strong** if sparse probing improves while L0 stays similar — this means the cosine SAE is activating the _right_ features more often, not just more features.

**For MIB:** If cosine-based featurization improves interchange intervention accuracy above the SAE baseline (which currently matches raw neurons), that demonstrates RNH has causal re<author>ance — not just correlational.
