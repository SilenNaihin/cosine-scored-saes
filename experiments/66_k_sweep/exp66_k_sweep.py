#!/usr/bin/env python3
"""Experiment 66: Sparsity-budget (k) sweep at low k (A6).

Reviewer item A6 (pYoQ): "it would be interesting to see whether the autointerp
score gains persist at low sparsity, for example [low k]." The body only states
the probing gap "narrows at higher k"; this measures the LOW-k end, for both
sparse probing AND auto-interp.

Design: train standard + per-feature cosine at k in {10,20,40,80,160} at 50M
tokens / L18 / d_sae=65,536 (the exp43d cheap recipe), then evaluate FVE +
SAEBench sparse probing on each. Auto-interp (exp53 protocol) is run separately
on the resulting checkpoints (exp66_autointerp.py) to keep API cost decoupled.

This is a thin orchestrator over exp43d's building blocks (SAE classes,
train_sae, ActivationStream, collect_eval_data, evaluate_reconstruction) so the
training recipe is identical to the established 50M baseline; only k and the arm
set change. k is set on the SAE constructor per iteration (the SAE owns its k).

Env (one GPU):
    CUDA_VISIBLE_DEVICES=0 EXP66_KS=10,20,40,80,160 \
        EXP66_SAVE_DIR=checkpoints/exp66 \
        EXP66_RESULTS=experiments/exp66_results.json \
        PYTHONUNBUFFERED=1 python3 experiments/exp66_k_sweep.py

Notes:
  - Trains only standard + perfeature_l2 (the headline pair) to bound compute;
    adaptive_l2 / no_C are omitted (add to ARMS if wanted).
  - Resumable: skips (k, arm) already present in the results JSON.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    HERE / "exp43d_50m_l18.py",
    HERE / "43d_4arch_50m_l18" / "exp43d_50m_l18.py",
]
exp43d_path = next((p for p in _CANDIDATES if p.exists()), None)
if exp43d_path is None:
    sys.exit(f"Could not locate exp43d_50m_l18.py in {_CANDIDATES}")

spec = importlib.util.spec_from_file_location("exp43d", exp43d_path)
exp43d = importlib.util.module_from_spec(spec)
sys.modules["exp43d"] = exp43d
spec.loader.exec_module(exp43d)

# Headline pair only (bound compute); add adaptive_l2 / no_C here if desired.
_ALL_ARMS = [
    ("standard", exp43d.BatchTopKSAE),
    ("perfeature_l2", exp43d.PerFeatureAdaptiveCosineSAE),
]
# EXP66_ARMS lets each GPU own one arm (separate results file) to avoid
# two processes racing on the same JSON (the A1 contamination lesson).
_want = os.environ.get("EXP66_ARMS")
ARMS = ([a for a in _ALL_ARMS if a[0] in _want.split(",")] if _want else _ALL_ARMS)


def _env(name, default=None):
    return os.environ.get(name, default)


def main():
    ks = [int(x) for x in _env("EXP66_KS", "10,20,40,80,160").split(",")]
    save_dir = Path(_env("EXP66_SAVE_DIR", "checkpoints/exp66"))
    results_path = _env("EXP66_RESULTS", "experiments/exp66_results.json")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp66] k-sweep {ks} | arms {[a for a, _ in ARMS]} | "
          f"L{exp43d.LAYER} d_sae={exp43d.D_SAE} {exp43d.N_TRAIN_TOKENS:,} tokens")

    # Load model + eval data once (shared across all k/arm).
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(exp43d.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        exp43d.MODEL_NAME, dtype=torch.float32, device_map=exp43d.DEVICE)
    model.eval()
    eval_data = exp43d.collect_eval_data(
        model, tokenizer, exp43d.LAYER, exp43d.N_EVAL_TOKENS)
    if isinstance(eval_data, tuple):
        eval_data = eval_data[0]

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {"config": {"layer": exp43d.LAYER, "d_sae": exp43d.D_SAE,
                              "n_train_tokens": exp43d.N_TRAIN_TOKENS,
                              "ks": ks, "arms": [a for a, _ in ARMS]},
                   "runs": {}}

    checkpoint_steps = set(exp43d.CHECKPOINT_STEPS)
    for k in ks:
        exp43d.K = k  # SAE constructor reads this; topk is owned by the SAE
        for name, cls in ARMS:
            run_key = f"{name}_k{k}"
            if run_key in results["runs"]:
                print(f"[exp66] {run_key} already done, skipping")
                continue
            print(f"\n{'='*70}\n[exp66] {run_key}  (k={k})\n{'='*70}")
            torch.manual_seed(exp43d.SEED)
            sae = cls(exp43d.D_MODEL, exp43d.D_SAE, k).to(exp43d.DEVICE)
            stream = exp43d.ActivationStream(model, tokenizer, exp43d.LAYER, seed=exp43d.SEED)
            exp43d.train_sae(name, sae, stream, save_dir, checkpoint_steps)
            recon = exp43d.evaluate_reconstruction(name, sae, eval_data)

            # save final checkpoint named by (arm, k) for downstream auto-interp
            ckpt_path = save_dir / f"{name}_L{exp43d.LAYER}_k{k}_final.pt"
            torch.save({"state_dict": sae.state_dict(), "k": k}, ckpt_path)

            results["runs"][run_key] = {
                "arm": name, "k": k,
                "reconstruction": recon,
                "checkpoint": str(ckpt_path),
            }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            fve = recon.get("fve") if isinstance(recon, dict) else None
            print(f"[exp66] {run_key} done  FVE={fve}")

    print(f"\n[exp66] done -> {results_path}")


if __name__ == "__main__":
    main()
