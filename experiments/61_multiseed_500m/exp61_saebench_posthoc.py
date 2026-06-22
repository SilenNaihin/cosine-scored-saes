#!/usr/bin/env python3
"""Post-hoc SAEBench eval for exp61 multi-seed finals.

Why this exists: exp40's inline SAEBench eval inserts `Path(__file__).parent.parent`
on sys.path to import `benchmarks/`. In the PUBLIC repo layout the script lives at
experiments/40_saprmarks_recipe/exp40_karvonen_recipe.py, so parent.parent resolves
to experiments/ (NOT the repo root where benchmarks/ lives), and the import silently
fails -> sparse_probing is skipped. (In the flat working-repo layout it happened to
resolve correctly.) FVE/reconstruction are unaffected and already in the results JSON;
only the SAEBench probing numbers are missing.

This script loads each saved *_final.pt, rebuilds the matching SAE class from exp40,
and runs exp40's OWN run_saebench_eval (so the probing protocol is identical to the
headline), with the repo root on sys.path so `benchmarks` imports. It writes the
sparse_probing/core/absorption block back into the per-seed results JSON under the
arm's "saebench" key.

Usage (one GPU per seed, after training finishes):
    CUDA_VISIBLE_DEVICES=0 EXP61_SEED=123 \
        EXP61_SAVE_DIR=checkpoints/exp61_seed123 \
        EXP61_RESULTS=experiments/exp61_seed123_results.json \
        python3 experiments/exp61_saebench_posthoc.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # so `benchmarks` imports regardless of layout
sys.path.insert(0, str(REPO_ROOT))

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


def _run_eval_no_absorption(vname, sae, seed):
    """Same BenchSAE construction as exp40.run_saebench_eval, but eval_types =
    ["core", "sparse_probing"] only. We drop "absorption": it crashes inside
    SAEBench (k_sparse_probing builds a float32 probe but the cached embeddings
    are bfloat16 -> "mat1 and mat2 must have the same dtype"), and it is not a
    metric A1 needs (the headline number is sparse_probing top-1). Skipping it
    also lets `run_saebench` return so the block gets written.

    CRITICAL: sae_name AND output_dir are seed-specific, and force_rerun=True.
    SAEBench keys result files by sae_name in output_dir and, with
    force_rerun=False, RELOADS an existing file instead of recomputing. Earlier
    both seeds shared name `exp40-standard-L18` and dir `eval_results/exp40`, so
    seed 456 silently loaded seed 123's result (identical eval_id) -> the
    per-seed probing numbers were NOT independent. Per-seed name+dir+force_rerun
    guarantees each seed is its own computation. (The big reusable model-
    activation caches under artifacts/ are model+layer keyed, not SAE-specific,
    so they are still safely shared; only the per-SAE result JSONs must differ.)
    """
    import torch.nn.functional as F
    from benchmarks.adapter import BenchSAE
    from benchmarks.run_saebench import run_saebench

    _sae = sae.eval()
    bench_sae = BenchSAE(
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
        bench_sae,
        sae_name=f"exp61-seed{seed}-{vname}-L{exp40.LAYER}",
        eval_types=["core", "sparse_probing"],
        output_dir=f"benchmarks/eval_results/exp61_seed{seed}",
        llm_batch_size=4,
        device=exp40.DEVICE,
        force_rerun=True,
    )


def main():
    seed = int(_require("EXP61_SEED"))
    save_dir = Path(_require("EXP61_SAVE_DIR"))
    results_path = _require("EXP61_RESULTS")

    # Sanity: confirm benchmarks now imports from repo root.
    try:
        from benchmarks.adapter import BenchSAE  # noqa: F401
        from benchmarks.run_saebench import run_saebench  # noqa: F401
    except ImportError as e:
        sys.exit(f"benchmarks still not importable from {REPO_ROOT}: {e}")

    force = os.environ.get("EXP61_FORCE", "").lower() in ("1", "true", "yes")

    with open(results_path) as f:
        all_results = json.load(f)

    layer = exp40.LAYER
    for vname, vcls in exp40.VARIANTS:
        run_name = f"{vname}_L{layer}"
        run = all_results.get("runs", {}).get(run_name)
        if run is None:
            print(f"[exp61-sb] seed{seed} {run_name}: no results entry yet, skipping")
            continue
        if isinstance(run.get("saebench"), dict) and not force:
            print(f"[exp61-sb] seed{seed} {run_name}: saebench already present, skipping")
            continue

        ckpt_path = save_dir / f"{vname}_L{layer}_final.pt"
        if not ckpt_path.exists():
            print(f"[exp61-sb] seed{seed} {run_name}: no final checkpoint at {ckpt_path}, skipping")
            continue

        print(f"\n[exp61-sb] seed{seed} {run_name}: loading {ckpt_path}")
        sae = vcls(exp40.D_MODEL, exp40.D_SAE, exp40.K).to(exp40.DEVICE)
        ckpt = torch.load(ckpt_path, map_location=exp40.DEVICE, weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        sae.load_state_dict(sd)

        sb = _run_eval_no_absorption(vname, sae, seed)
        run["saebench"] = sb

        # Persist after each arm (crash-safe).
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"[exp61-sb] seed{seed} {run_name}: saebench written")

    print(f"\n[exp61-sb] seed{seed}: done -> {results_path}")


if __name__ == "__main__":
    main()
