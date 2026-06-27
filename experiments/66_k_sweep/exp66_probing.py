#!/usr/bin/env python3
"""Experiment 66 (probing eval): SAEBench sparse-probing across the k-sweep.

Runs core + sparse_probing on every exp66 k-checkpoint (standard + per-feature,
k in {10,20,40,80,160}) so we can report top-1 vs k -- the probing half of
pYoQ's low-k question. Auto-interp (the metric pYoQ named explicitly) is a
separate script (exp66_autointerp.py) to decouple API cost.

CRITICAL (A1 contamination lesson): per-(k,arm) sae_name AND output_dir, with
force_rerun=True, so SAEBench never reloads another run's cached result.
Reuses exp40's run_saebench wrapper via the benchmarks package (repo root on
sys.path). Drops "absorption" (SAEBench-internal float32/bf16 dtype bug; not
needed). SAE classes imported from exp43d so they match the trained weights.

Env (one GPU):
    CUDA_VISIBLE_DEVICES=0 EXP66_CKPT_DIR=checkpoints/exp66 \
        EXP66_ARMS=standard,perfeature_l2 EXP66_KS=10,20,40,80,160 \
        EXP66_RESULTS=experiments/exp66_probing_results.json \
        PYTHONUNBUFFERED=1 python3 experiments/exp66_probing.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

# Locate exp43d (SAE classes) and exp40 (run_saebench wrapper / MODEL/LAYER).
def _load(modname, *cands):
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        sys.exit(f"cannot locate {modname} in {cands}")
    spec = importlib.util.spec_from_file_location(modname, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m

exp43d = _load("exp43d", HERE / "exp43d_50m_l18.py",
               HERE / "43d_4arch_50m_l18" / "exp43d_50m_l18.py")
exp40 = _load("exp40", HERE / "exp40_karvonen_recipe.py",
              HERE / "40_saprmarks_recipe" / "exp40_karvonen_recipe.py")

ARM_CLASSES = {
    "standard": exp43d.BatchTopKSAE,
    "perfeature_l2": exp43d.PerFeatureAdaptiveCosineSAE,
}


def run_probing(name, sae, tag, out_dir):
    from benchmarks.adapter import BenchSAE
    from benchmarks.run_saebench import run_saebench
    _sae = sae.eval()
    bench = BenchSAE(
        W_enc=sae.W_enc.detach().T,
        W_dec=F.normalize(sae.W_dec.detach(), dim=1),
        b_enc=sae.b_enc.detach(),
        b_dec=sae.b_dec.detach(),
        encode_fn=lambda x: _sae.encode(x),
        decode_fn=lambda f: _sae.decode(f),
        model_name=exp40.MODEL_NAME,
        hook_layer=exp40.LAYER,
        device=exp40.DEVICE,
        dtype=exp40.DTYPE,
    )
    return run_saebench(
        bench, sae_name=tag, eval_types=["core", "sparse_probing"],
        output_dir=out_dir, llm_batch_size=4, device=exp40.DEVICE, force_rerun=True,
    )


def main():
    ckpt_dir = Path(os.environ.get("EXP66_CKPT_DIR", "checkpoints/exp66"))
    arms = os.environ.get("EXP66_ARMS", "standard,perfeature_l2").split(",")
    ks = [int(x) for x in os.environ.get("EXP66_KS", "10,20,40,80,160").split(",")]
    results_path = os.environ.get("EXP66_RESULTS", "experiments/exp66_probing_results.json")

    results = {"runs": {}}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    d_model, d_sae, layer = exp43d.D_MODEL, exp43d.D_SAE, exp43d.LAYER
    for arm in arms:
        cls = ARM_CLASSES[arm]
        for k in ks:
            run_key = f"{arm}_k{k}"
            if run_key in results["runs"]:
                print(f"[exp66-probe] {run_key} done, skip"); continue
            ckpt = ckpt_dir / f"{arm}_L{layer}_k{k}_final.pt"
            if not ckpt.exists():
                print(f"[exp66-probe] {run_key}: no ckpt {ckpt}, skip"); continue
            print(f"\n[exp66-probe] {run_key}: loading {ckpt}")
            sae = cls(d_model, d_sae, k).to(exp40.DEVICE)
            obj = torch.load(ckpt, map_location=exp40.DEVICE, weights_only=False)
            sae.load_state_dict(obj["state_dict"] if "state_dict" in obj else obj)
            tag = f"exp66-{arm}-k{k}"
            sb = run_probing(arm, sae, tag, f"benchmarks/eval_results/exp66_{arm}_k{k}")
            sp = sb.get("sparse_probing", {}).get("eval_result_metrics", {}).get("sae", {})
            results["runs"][run_key] = {
                "arm": arm, "k": k,
                "top_1": sp.get("sae_top_1_test_accuracy"),
                "top_2": sp.get("sae_top_2_test_accuracy"),
                "top_5": sp.get("sae_top_5_test_accuracy"),
            }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"[exp66-probe] {run_key}: top-1={results['runs'][run_key]['top_1']}")

    print(f"\n[exp66-probe] done -> {results_path}")


if __name__ == "__main__":
    main()
