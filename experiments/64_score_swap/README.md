# Experiment 64: Direct score-swap mechanism test

Answers reviewer 762k: "Add a direct score swap experiment. Train a standard SAE
and a cosine SAE, then evaluate what happens when the score geometry is swapped or
partially swapped at inference and during continued training. This would test the
mechanism more directly than gradient reweighting alone."

exp29 did only the forward direction (post-hoc cosine on a trained STANDARD SAE
destroys FVE; advantage is training-time). This completes the picture symmetrically:
both swap directions at inference, plus continued-training swaps.

Setting: Qwen3-8B L18, d_sae=65536, k=80 (BatchTopK), 50M FineWeb tokens, saprmarks
recipe. A SwappableBatchTopKSAE shares weights and hot-swaps inner<->cosine scoring
(adaptive-cosine recovers inner product at a=1). Both arms init W_enc=W_dec
(matching exp43d; the 0.1x init from exp59/60 caused ~79% dead for the standard arm).

## Results

### Part 1 — inference-time score swap

| Checkpoint | Scored as | FVE | Dead% | Probe top-1 |
|---|---|---|---|---|
| standard | inner (native) | 0.709 | 0.1 | 0.694 |
| standard | **cosine (swap)** | 0.599 | 1.0 | **0.763** |
| cosine | cosine (native) | 0.707 | 0.1 | 0.766 |
| cosine | **inner (swap)** | 0.612 | 0.0 | **0.698** |

Two separable effects:
1. **FVE is destroyed by the swap in BOTH directions** (~0.71 -> 0.60). Reconstruction
   quality is bound to the training-time score; you cannot recover it by changing the
   score at inference. Symmetric extension of exp29.
2. **Probing top-1 tracks the score used at READ TIME, not the checkpoint.** Standard
   weights read with cosine -> 0.763 (~= native cosine 0.766); cosine weights read with
   inner -> 0.698 (~= native inner 0.694). Probing quality is governed by the input-norm
   channel in the score (cosine strips ||x||), applied at read time, while FVE is baked
   into the trained weights. Consistent with the exp63 two-channel decomposition.

### Part 2 — continued-training swap (10M tokens)

Take each trained checkpoint, swap the score, continue training 10M tokens.

| Run | FVE (new score) | Jaccard vs orig alive | decoder cos vs orig |
|---|---|---|---|
| standard -> cosine-ft | 0.714 | 0.998 | 0.973 |
| cosine -> inner-ft | 0.713 | 0.999 | 0.981 |

**Near-null: the dictionary barely reorganizes.** FVE recovers (0.60 -> 0.71) but the
alive feature set stays ~identical (Jaccard ~0.998) and decoder directions barely move
(cos ~0.97-0.98). At a 10M-token continued-training budget the swapped-score finetune
re-fits magnitudes/thresholds but does NOT re-sort the dictionary toward the other
score's solution. Reported as minimal-at-this-budget; could reflect sticky dictionaries
or a too-short finetune (vs 50M training) — not isolated here.

## Files
- `exp64_score_swap.py` — SwappableBatchTopKSAE, train pair + inference swap + continued-training swap + SAEBench probing.
- `results.json` — full metrics.
