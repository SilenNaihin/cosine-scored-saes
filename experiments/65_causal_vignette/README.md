# Experiment 65: Salient-concept causal control

A scoping control for the causal/probing advantage (A4). The aggregate results
(sparse probing +14.6%, ablation precision exp57b) raise a natural over-reading:
"standard scoring can't represent concepts like code or French." This experiment
shows that reading is **false** and locates where the advantage actually lives.

## Method

For each content concept, rank every feature in a dictionary by selectivity
(mean activation on concept-positive minus concept-negative prompts), ablate the
top-K as a set (sweeping K), and measure the within-arm precision:

    precision(K) = (mean ablation KL on ON-concept prompts)
                 / (mean ablation KL on OFF-concept prompts)

The ratio is within-arm, so cross-architecture activation-magnitude differences
cancel (avoiding the norm confound). Inference-only on the 500M checkpoints
(seed 123); standard vs per-feature cosine; concepts: Python code, French text.

## Result

For both salient concepts, **both** dictionaries contain a cleanly selective
top concept feature: it fires strongly on-concept and is silent off-concept
(off-concept ablation KL ≈ 0 for K ≤ 10 under either score). Neither dictionary
lacks the concept. The precision ratio is degenerate (inf, near-zero denominator)
at small K for both arms and only becomes finite at K ≥ 20, where the two arms
are within noise of each other.

**Conclusion:** the cosine advantage is **not** visible at the single-salient-
concept level. It is distributional — it emerges in the aggregate over the eight
SAEBench probing categories and the long feature tail (see exp58b matched-feature
decomposition: +14.9% full vs +2.0% on matched features) and in aggregate
ablation cleanliness (exp57b: precision 8.4×→21.8× for cosine vs 5.6×→1.4× for
standard as N grows). This experiment is reported in the paper as a salient-
concept control (Appendix "Salient-Concept Control"), guarding against the
over-claim that standard scoring cannot represent obvious concepts.

## Files

- `exp65_causal_vignette.py` — ranking + top-K set ablation + within-arm precision.
- `results.json` — per-concept, per-arm K-sweep (K ∈ {1,3,5,10,20,50}).

## Provenance note

v1 used single-best-feature ablation with a raw cross-arm KL metric; it was
confounded (single feature too easy; cross-arm magnitude not comparable). v2
(this version) fixes both with top-K set ablation and a within-arm precision
ratio. The qualitative conclusion (both arms represent salient concepts; the
advantage is aggregate) is robust across both versions.
