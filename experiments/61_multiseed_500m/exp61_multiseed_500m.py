#!/usr/bin/env python3
"""Experiment 61: Multi-seed replication of the 500M headline (exp40).

Reviewer item A1 (EZEE, 762k): the headline 500M Qwen3-8B L18 / d_sae=65,536
result is single-seed at the SAE-training level. This wrapper re-runs the
EXACT exp40 recipe at additional seeds so we can report mean +/- SD on FVE and
sparse-probing top-1 across n>=3 SAE-training seeds (combining with the
published seed-42 numbers).

It does NOT duplicate any recipe logic: it imports exp40's module, overrides
the three module-level constants that must differ per seed (SEED, SAVE_DIR,
RESULTS_PATH) from the environment, then calls exp40.main(). Everything else
(architecture, aux loss, k, lr, token budget, all three arms standard /
adaptive_l2 / perfeature_l2) is inherited verbatim from exp40.

Aux loss is ON in exp40 (auxk_alpha=1/32), which drives both arms to ~0% dead
and is the variance-suppressing mechanism; this is intentional, because the
reviewers asked for variance AT THE HEADLINE SETTING, and the headline is
aux-on.

Usage (one GPU per seed):
    CUDA_VISIBLE_DEVICES=0 EXP61_SEED=123 \
        EXP61_SAVE_DIR=checkpoints/exp61_seed123 \
        EXP61_RESULTS=experiments/exp61_seed123_results.json \
        python3 experiments/exp61_multiseed_500m.py

    CUDA_VISIBLE_DEVICES=1 EXP61_SEED=456 \
        EXP61_SAVE_DIR=checkpoints/exp61_seed456 \
        EXP61_RESULTS=experiments/exp61_seed456_results.json \
        python3 experiments/exp61_multiseed_500m.py
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# exp40 lives flat in the working repo and under 40_saprmarks_recipe/ in the
# public repo; support both so this runs in either checkout.
_CANDIDATES = [
    HERE / "exp40_karvonen_recipe.py",
    HERE / "40_saprmarks_recipe" / "exp40_karvonen_recipe.py",
]
exp40_path = next((p for p in _CANDIDATES if p.exists()), None)
if exp40_path is None:
    sys.exit(f"Could not locate exp40_karvonen_recipe.py in {_CANDIDATES}")

spec = importlib.util.spec_from_file_location("exp40", exp40_path)
exp40 = importlib.util.module_from_spec(spec)
sys.modules["exp40"] = exp40
spec.loader.exec_module(exp40)


def _require(name):
    val = os.environ.get(name)
    if not val:
        sys.exit(f"Missing required env var {name}")
    return val


def main():
    seed = int(_require("EXP61_SEED"))
    save_dir = _require("EXP61_SAVE_DIR")
    results_path = _require("EXP61_RESULTS")

    # Override the three per-seed constants; leave the recipe untouched.
    exp40.SEED = seed
    exp40.SAVE_DIR = save_dir
    exp40.RESULTS_PATH = results_path

    print(f"[exp61] Replicating exp40 headline with overrides:")
    print(f"[exp61]   SEED         = {exp40.SEED}")
    print(f"[exp61]   SAVE_DIR     = {exp40.SAVE_DIR}")
    print(f"[exp61]   RESULTS_PATH = {exp40.RESULTS_PATH}")
    print(f"[exp61]   model={exp40.MODEL_NAME} L{exp40.LAYER} "
          f"d_sae={exp40.D_SAE} k={exp40.K} tokens={exp40.N_TRAIN_TOKENS:,}")
    print(f"[exp61]   variants={[v[0] for v in exp40.VARIANTS]}")

    exp40.main()


if __name__ == "__main__":
    main()
