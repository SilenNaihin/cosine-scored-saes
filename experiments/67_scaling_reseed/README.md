# Experiment 67: Scaling-matrix reseed (model size vs expansion ratio)

Grounds the paper's claim (§4.5, Appendix cross-model) that the cosine advantage
**grows with model dimension and is flat across expansion ratio**. The original
exp57 matrix supporting this was single-seed, lacked raw results for several
cells, and had one hand-eyeballed cell (8B/16x ≈ +14%). This experiment reruns it
with proper seeds and real SAEBench probing.

## Setup

- 27-cell matrix: Qwen3 **1.7B / 4B / 8B** × **4× / 8× / 16×** expansion × **3 seeds {42,123,456}**.
- 50M FineWeb tokens, BatchTopK k=80, aux loss on, saprmarks recipe (reuses exp57's
  exact recipe by importing the module; only seed + on-disk paths change).
- Two SAEs per cell: standard + adaptive_l2 (global cosine). Hook layers match exp57
  (1.7B=L14, 4B=L18, 8B=L18) so results stay comparable to the headline.
- Metric: aggregate `sae_top_1_test_accuracy` (mean over 8 SAEBench probing datasets).

## Results — gap = cosine(adaptive_l2) − standard, top-1 (pp), mean ± SD over 3 seeds

| model | d_model | 4× | 8× | 16× | **row mean** |
|-------|---------|-----|-----|-----|----------|
| 1.7B  | 2048 | +6.1 ± 1.1 | +5.0 ± 2.1 | +4.7 ± 1.8 | **+5.3** |
| 4B    | 2560 | +11.2 ± 2.2 | +12.5 ± 0.7 | +11.4 ± 2.6 | **+11.7** |
| 8B    | 4096 | +10.9 ± 1.0 | +9.0 ± 1.6 | +7.6 ± 2.4 | **+9.2** |

Column (expansion) means: 4× +9.4, 8× +8.9, 16× +7.9 (sd ~3, overlapping → flat).
FVE matched at every model (std/cos: 1.7B 0.726/0.733, 4B 0.735/0.736, 8B 0.714/0.715),
so the gap is a feature-use difference, not a reconstruction artifact.

## Findings

1. **Model dimension drives the gap (confirmed).** 1.7B (+5.3) ≪ 4B (+11.7), 8B (+9.2).
   The jump from 1.7B to ≥4B is real and exceeds seed noise. Trend is non-monotonic
   (4B ≥ 8B): "rises then plateaus," not a clean monotone scaling law.
2. **Expansion ratio is flat (confirmed).** Column means overlap within SD.
3. **Seed variance is modest (~2pp).** The original single-seed matrix was noisier than
   ideal but not misleading; the one materially wrong value (eyeballed 8B/16× ≈ +14%) is
   corrected to +7.6 ± 2.4.

**Bottom line:** exp67 grounds and confirms exp57 rather than overturning it. The net
change to the paper is a figure refresh (n=3 + error bars) plus correcting one eyeballed
cell. See `fig:scaling-matrix` and §4.5 in the paper.

## Metric note

The reported number is the aggregate top-1 over the 8 probing datasets. An earlier
analysis pass mistakenly read the last per-dataset value (a near-ceiling outlier),
producing inflated +18–30pp gaps; those were a parsing bug and never entered the paper.
The corrected aggregates above are the real result. Full detail and the run-1..4
infrastructure log are in `analysis.md`.

## Files

- `exp67_scaling_reseed.py` — the reseed driver (imports exp57's recipe; per-seed paths).
- `plan.md` — motivation, design, predictions, block ordering.
- `analysis.md` — full results, verdict vs predictions, and infrastructure saga.
