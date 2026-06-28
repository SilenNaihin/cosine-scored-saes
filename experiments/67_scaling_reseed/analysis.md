# Experiment 67 — Scaling-Matrix Reseed: results

## TL;DR
Grounds exp57's single-seed, partially-eyeballed scaling matrix with a full **27-cell run
(Qwen3 1.7B/4B/8B x 4x/8x/16x expansion x 3 seeds {42,123,456}, 50M tokens, k=80, aux-on,
hook L14/L18/L18)**. All cells completed with real SAEBench probing (0 errors). **exp57's two
claims hold up: (1) model dimension drives the gap, (2) expansion ratio is flat.** The trend is
non-monotonic (4B >= 8B), seed variance is modest (~2pp), and the one genuinely-wrong exp57
value (the hand-eyeballed 8B/16x `~+14%` cell) is corrected to +7.6%.

## Metric note (IMPORTANT — parsing correction)
The reported number is the **aggregate** `eval_result_metrics/sae/sae_top_1_test_accuracy`
(mean over the 8 SAEBench probing datasets). An earlier analysis pass mistakenly read the
**last per-dataset** value from `eval_result_details[7]` (a near-ceiling outlier dataset),
which inflated cosine top-1 to ~0.99 and produced bogus gaps of +18-30pp. Those numbers were
wrong and never entered the paper; the corrected aggregates below are the real result.

## Results: gap = cosine(adaptive_l2) - standard, top-1 sparse probing (pp)

| model | d_model | 4x | 8x | 16x | **row mean** |
|-------|---------|-----|-----|-----|----------|
| 1.7B  | 2048 | +6.1 ± 1.1 | +5.0 ± 2.1 | +4.7 ± 1.8 | **+5.3** |
| 4B    | 2560 | +11.2 ± 2.2 | +12.5 ± 0.7 | +11.4 ± 2.6 | **+11.7** |
| 8B    | 4096 | +10.9 ± 1.0 | +9.0 ± 1.6 | +7.6 ± 2.4 | **+9.2** |

Column (expansion) means across all models+seeds: **4x +9.4, 8x +8.9, 16x +7.9** (sd ~3,
overlapping → flat). FVE matched at every model (std/cos: 1.7B 0.726/0.733, 4B 0.735/0.736,
8B 0.714/0.715), so the gap is a feature-use difference, not a reconstruction artifact.

## Verdict vs predictions (plan H1/H2/H3)
- **H1 (model size drives it):** CONFIRMED. 1.7B (+5.3) << 4B (+11.7), 8B (+9.2). The 1.7B
  row is clearly weakest; the jump to >=4B is real and exceeds seed noise.
- **trend shape:** non-monotonic — 4B >= 8B, i.e. "rises then plateaus/slightly dips," matching
  exp57's "+6.1 -> +11.6 -> plateau." NOT a clean monotone scaling law. The §4.5 body wording
  already hedges to "tracks model dimension... flat across expansion," which is correct as-is.
- **expansion flat (exp57's other claim):** CONFIRMED. Column means 9.4/8.9/7.9 overlap within sd.
- **H3 (eyeballed cell):** the exp57 8B/16x `~+14%` was the one materially wrong value; real
  +7.6 ± 2.4. Corrected.
- **seed variance:** modest (~2pp sd typical), NOT large. exp57 single-seed was noisier than
  ideal but not misleading. (An earlier note claimed "sd 7.4, large hidden variance" — that was
  the parsing bug, not real.)

## Bottom line
exp67 **grounds and confirms** exp57 rather than overturning it. Net change to the paper is a
figure refresh (n=3 + error bars) + correcting one eyeballed cell, not a rewrite. No correctness
liability remains in the scaling claim.

---

## Appendix: infrastructure saga (run 1-4, for the record)
The run took 4 launches across 3 boxes before completing; documented so the next person doesn't
repeat it.
- **Run 1 (a100-backup-1, 2026-06-17):** SIGABRT — root disk hit 100%. SAEBench writes a ~17G
  probe-activation cache to repo-relative `artifacts/` that `HF_HOME` does NOT redirect.
- **Runs 2-3 (a100, box-5, 2026-06-22):** repeated disk-full corruption (`/mnt` filled). Root
  cause: the probe cache accumulates per-model and never self-evicts (44G 1.7B + 37G 8B). Fix:
  added `--evict-others` (purge other models' caches between sequential runs) + a measured disk
  guard (abort cleanly if free space < 60G rather than truncating a `.pt`).
- **box-8 (2026-06-22):** trained fine but every probing failed `No module named 'benchmarks'` —
  exp67 relied on exp57's `sys.path` insert, which resolves to `experiments/` (not repo root) in
  the public-repo layout. Fix: exp67 inserts the repo root itself.
- **Guard bug:** first version stat'd cwd (root) not the data volume (`/mnt` via the `artifacts`
  symlink), causing a false abort. Fixed to resolve the symlink target.
- **Run 4 (box-5 GPU1, 2026-06-22 18:17, completed 2026-06-25):** clean, 27/27 cells, 0 errors,
  after one transient FineWeb-stream hang cleared by relaunch with `python -u`.
- Script + all fixes: `experiments/exp67_scaling_reseed.py`.
