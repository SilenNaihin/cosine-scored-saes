#!/usr/bin/env python3
"""Experiment 66 (auto-interp eval): describe-then-predict across the k-sweep.

pYoQ asked specifically whether the AUTO-INTERP gains persist at low sparsity.
This runs the exp53 protocol (collect activating contexts -> LLM describes the
feature -> LLM predicts activating tokens on held-out contexts -> interpretable
if >=50% prediction accuracy) on every exp66 k-checkpoint (standard +
per-feature, k in {10,20,40,80,160}).

Reuses exp53's functions verbatim (collect_contexts, score_features, load_sae,
the SAE classes, the Bedrock judge) by overriding its module globals per (k,arm)
so the protocol is byte-identical to the published 500M auto-interp; only the
checkpoint and the output paths change.

Per (k,arm) we set exp53.CKPTS / SAE_CLASSES / VARIANT_NAMES to a single tagged
entry and per-k CONTEXTS_PATH / RESULTS_PATH (separate files; no cross-run
contamination). collect (GPU) then score (Bedrock API) run back-to-back.

Env:
    CUDA_VISIBLE_DEVICES=0 EXP66_CKPT_DIR=checkpoints/exp66 \
        EXP66_ARMS=standard,perfeature_l2 EXP66_KS=10,20,40,80,160 \
        EXP66_OUT=experiments/exp66_autointerp \
        PYTHONUNBUFFERED=1 python3 experiments/exp66_autointerp.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))


def _load(modname, *cands):
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        sys.exit(f"cannot locate {modname} in {cands}")
    spec = importlib.util.spec_from_file_location(modname, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m

exp53 = _load("exp53", HERE / "exp53_llm_interp_500m.py",
              HERE / "53_llm_interp_500m" / "exp53_llm_interp_500m.py")
exp43d = _load("exp43d", HERE / "exp43d_50m_l18.py",
               HERE / "43d_4arch_50m_l18" / "exp43d_50m_l18.py")

# Map arm -> the exp53 SAE class (same architectures, weights match exp66 ckpts).
ARM_CLASS = {
    "standard": exp53.BatchTopKSAE,
    "perfeature_l2": exp53.PerFeatureAdaptiveCosineSAE,
}


def main():
    ckpt_dir = Path(os.environ.get("EXP66_CKPT_DIR", "checkpoints/exp66"))
    arms = os.environ.get("EXP66_ARMS", "standard,perfeature_l2").split(",")
    ks = [int(x) for x in os.environ.get("EXP66_KS", "10,20,40,80,160").split(",")]
    out_dir = Path(os.environ.get("EXP66_OUT", "experiments/exp66_autointerp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    layer = exp43d.LAYER

    summary = {}
    for k in ks:
        exp53.K = k  # load_sae reads this; SAE owns topk
        for arm in arms:
            tag = f"{arm}_k{k}"
            ckpt = ckpt_dir / f"{arm}_L{layer}_k{k}_final.pt"
            if not ckpt.exists():
                print(f"[exp66-interp] {tag}: no ckpt {ckpt}, skip"); continue

            res_path = out_dir / f"{tag}_results.json"

            def _rate_from(path):
                """exp53 stores per-variant {features:[{interpretable:bool}...]}; compute rate."""
                if not Path(path).exists():
                    return None
                d = json.load(open(path))
                v = d.get(tag, {})
                feats = v.get("features", [])
                if not feats:
                    return None
                n_int = sum(1 for f in feats if f.get("interpretable"))
                return {"k": k, "arm": arm, "n_features": len(feats),
                        "n_interpretable": n_int, "interpretable_rate": n_int / len(feats),
                        "alive_count": v.get("alive_count")}

            done = _rate_from(res_path)
            if done and done["n_features"] >= exp53.N_FEATURES_PER_SAE:
                print(f"[exp66-interp] {tag}: already scored ({done['interpretable_rate']:.3f}), skip")
                summary[tag] = done
                continue

            # Point exp53 at this single checkpoint, per-k IO paths.
            exp53.CKPTS = {tag: str(ckpt)}
            exp53.SAE_CLASSES = {tag: ARM_CLASS[arm]}
            exp53.VARIANT_NAMES = [tag]
            exp53.CONTEXTS_PATH = str(out_dir / f"{tag}_contexts.json")
            exp53.RESULTS_PATH = str(res_path)

            phase = os.environ.get("EXP66_PHASE", "both")  # collect | score | both
            if phase in ("collect", "both"):
                if not Path(exp53.CONTEXTS_PATH).exists():
                    print(f"\n{'='*70}\n[exp66-interp] {tag}: collect contexts (GPU)\n{'='*70}")
                    exp53.collect_contexts()
                else:
                    print(f"[exp66-interp] {tag}: contexts exist, skip collect")
            if phase in ("score", "both"):
                print(f"[exp66-interp] {tag}: score (Bedrock)")
                exp53.score_features()
            if phase == "collect":
                continue  # scoring deferred (needs Bedrock creds)

            summary[tag] = _rate_from(res_path) or {"k": k, "arm": arm, "interpretable_rate": None}
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"[exp66-interp] {tag}: rate={summary[tag].get('interpretable_rate')}")

    print(f"\n[exp66-interp] done -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
