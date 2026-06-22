# Experiment 61: Multi-seed reproducibility at the 500M headline

Reviewer item (EZEE, 762k): the 500M-token headline result was single-seed at the
SAE-training level. This experiment re-runs the exact exp40 recipe (Qwen3-8B layer 18,
500M FineWeb tokens, `d_sae = 65,536`, BatchTopK `k = 80`, AuxK dead-feature loss) at
two additional SAE-training seeds (123, 456) and combines them with the published
seed-42 run for **n = 3**.

`exp61_multiseed_500m.py` is a thin wrapper that imports exp40 and overrides only
`SEED`, `SAVE_DIR`, and `RESULTS_PATH` (via env vars) so the recipe is identical to the
headline. All three arms (standard, adaptive_l2 / global-`a`, perfeature_l2) are trained
per seed.

`exp61_saebench_posthoc.py` recovers the SAEBench `core` + `sparse_probing` metrics from
each saved `*_final.pt`. It uses **per-seed `sae_name` and `output_dir` with
`force_rerun=True`**, so each seed is an independent SAEBench computation (verified by
distinct `eval_id`s); the reusable model-activation caches under `artifacts/` are
model+layer keyed and shared safely.

## Results (n = 3 seeds {42, 123, 456})

Mean ± SD. FVE matched to ±0.0002; the top-1 gap is stable across seeds; the learned
norm-dependence exponent `a` stays far below the inner-product limit with negligible
seed variance.

| Variant | FVE | Probing top-1 | Top-1 gap | Learned `a` |
|---------|-----|---------------|-----------|-------------|
| Standard | 0.7702 ± 0.0002 | 0.6669 ± 0.0026 | — | — |
| Global `a` (adaptive_l2) | 0.7690 ± 0.0000 | 0.8081 ± 0.0059 | **+14.1%** | 0.2577 ± 0.0007 |
| Per-feature (perfeature_l2) | 0.7707 ± 0.0002 | 0.8128 ± 0.0126 | **+14.6%** | 0.0759 ± 0.0001 |

Per-seed numbers and the full breakdown are in `results_summary.json`. The seed-42 run is
the one reported in the paper's Table 1; it sits within the n = 3 spread.

The trained checkpoints (best seed per arm) are released at
[huggingface.co/Silen/cosine-scored-saes-qwen3-8b](https://huggingface.co/Silen/cosine-scored-saes-qwen3-8b).

## Files

- `exp61_multiseed_500m.py` — training wrapper (imports exp40, per-seed overrides).
- `exp61_saebench_posthoc.py` — post-hoc SAEBench probing recovery (per-seed, force_rerun).
- `results_summary.json` — per-seed and n = 3 summary metrics.
