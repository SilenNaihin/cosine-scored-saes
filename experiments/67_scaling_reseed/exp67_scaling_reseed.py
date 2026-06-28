#!/usr/bin/env python3
"""Experiment 67: Scaling-matrix reseed to ground the d_model claim.

Camera-ready B6/A3. The paper claims the cosine advantage grows with model
dimension and is flat across expansion ratio. The supporting exp57 matrix is
single-seed (SEED=42), is missing raw results for several cells, and its
8B/16x cell is a hand-eyeballed `~+14`. This experiment re-runs selected cells
with real SAEBench evals and multiple seeds so the trend carries error bars and
no estimated cells.

It does NOT duplicate any recipe logic: it imports exp57's module and reuses its
architectures (BatchTopKSAE, AdaptiveCosineBatchTopKSAE), training loop
(train_sae), recon eval (evaluate_reconstruction), SAEBench wrapper
(run_sparse_probing internals), activation streaming, and MODEL_CONFIGS verbatim.
The ONLY things reimplemented here are the seed plumbing and the on-disk paths,
because exp57 hardcodes SEED=42 and bakes the literal "exp57" (no seed) into its
checkpoint dir, SAEBench output dir, and RESULTS_PATH, so a pure constant-override
wrapper would make seeds collide on disk.

Machine: a100-backup-1 (single A100 80GB PCIe). Activations are streamed on the
fly (no cache exists for these models); 8B cells dominate wall-clock.

Usage (run cheap models first so the trend lands before the expensive tail):
    # Block B, cheap end: 1.7B and 4B at 8x, seeds 123 + 456
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp67_scaling_reseed.py \
        --model 1.7b --expansion 8 --seeds 123,456
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp67_scaling_reseed.py \
        --model 4b --expansion 8 --seeds 123,456

    # Block A: complete the 8B row at the base seed (replaces eyeballed 8b/16x)
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp67_scaling_reseed.py \
        --model 8b --all --seeds 42

    # Block B, 8B end: 8B at 8x, seeds 123 + 456 (combine w/ block-A seed-42 8b/8x)
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp67_scaling_reseed.py \
        --model 8b --expansion 8 --seeds 123,456

Results: experiments/exp67_results.json, keyed by "{model}_{expansion}x_seed{seed}".
Checkpoints: checkpoints/exp67/{cell}_seed{seed}/. Existing checkpoints/results are
skipped, so the script is resumable.
"""
import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

# Put the REPO ROOT on sys.path so `from benchmarks...` resolves regardless of
# cwd. exp67 lives at experiments/exp67_*.py, so the repo root is HERE.parent.
# We cannot rely on exp57's own sys.path insert: in the public-repo layout exp57
# sits at experiments/57_scaling_matrix/, whose parent.parent is experiments/
# (NOT the repo root), so benchmarks/ would be unimportable and SAEBench probing
# silently fails (the run-on-box-8 "No module named 'benchmarks'" bug).
_REPO_ROOT = HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# exp57 lives flat in the working repo and under 57_scaling_matrix/ in the
# public repo; support both so this runs in either checkout.
_CANDIDATES = [
    HERE / "exp57_scaling_matrix.py",
    HERE / "57_scaling_matrix" / "exp57_scaling_matrix.py",
]
exp57_path = next((p for p in _CANDIDATES if p.exists()), None)
if exp57_path is None:
    sys.exit(f"Could not locate exp57_scaling_matrix.py in {_CANDIDATES}")

spec = importlib.util.spec_from_file_location("exp57", exp57_path)
exp57 = importlib.util.module_from_spec(spec)
sys.modules["exp57"] = exp57
spec.loader.exec_module(exp57)

# Pull the pieces we reuse verbatim from exp57.
MODEL_CONFIGS = exp57.MODEL_CONFIGS
EXPANSION_RATIOS = exp57.EXPANSION_RATIOS
DEVICE = exp57.DEVICE
DTYPE = exp57.DTYPE
K = exp57.K
N_TRAIN_TOKENS = exp57.N_TRAIN_TOKENS
N_EVAL_TOKENS = exp57.N_EVAL_TOKENS

RESULTS_PATH = Path("experiments/exp67_results.json")


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def run_sparse_probing_seeded(name, sae, model_name, layer, model_tag, expansion, seed):
    """exp57.run_sparse_probing, but with a seed-namespaced output dir + sae_name."""
    from benchmarks.run_saebench import run_saebench

    out_dir = f"benchmarks/eval_results/exp67/{model_tag}_{expansion}x_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)

    bench_sae = exp57.wrap_for_saebench(name, sae, model_name, layer, DEVICE, DTYPE)
    sae_name = f"exp67-{name}-{model_tag}-{expansion}x-seed{seed}"

    print(f"    Running SAEBench sparse probing for {sae_name}...")
    return run_saebench(
        bench_sae,
        sae_name=sae_name,
        eval_types=["sparse_probing"],
        output_dir=out_dir,
        llm_batch_size=4,
        device=DEVICE,
    )


def run_cell_seeded(model_tag, expansion, seed, model, tokenizer):
    """Seed-aware port of exp57.run_cell.

    Reuses exp57's architectures, train_sae, evaluate_reconstruction, and
    ActivationStream verbatim; only seeds the RNG/stream and namespaces paths
    by seed. Resumable: existing checkpoints are loaded, not retrained.
    """
    cfg = MODEL_CONFIGS[model_tag]
    model_name = cfg["model_name"]
    d_model = cfg["d_model"]
    layer = cfg["hook_layer"]
    d_sae = d_model * expansion

    cell_key = f"{model_tag}_{expansion}x_seed{seed}"
    save_dir = Path(f"checkpoints/exp67/{cell_key}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Cell: {cell_key} | {model_name} L{layer} | d_sae={d_sae} | seed={seed}")
    print(f"{'='*70}")

    variants = [
        ("standard", exp57.BatchTopKSAE),
        ("adaptive_l2", exp57.AdaptiveCosineBatchTopKSAE),
    ]

    cell_results = {}
    eval_data = None
    stream = None

    for name, cls in variants:
        print(f"\n  --- {name} ({cell_key}) ---")
        final_path = save_dir / f"{name}_final.pt"

        # Collect eval data + stream once per cell, seeded.
        if eval_data is None:
            eval_data, _mean_norm = exp57.collect_eval_data(
                model, tokenizer, layer, N_EVAL_TOKENS
            )
            stream = exp57.ActivationStream(model, tokenizer, layer, seed=seed)

        if final_path.exists():
            print(f"    Checkpoint exists, loading: {final_path}")
            sae = cls(d_model, d_sae, K).to(DEVICE)
            ckpt = torch.load(final_path, map_location=DEVICE, weights_only=False)
            sae.load_state_dict(ckpt["state_dict"])
            sae.eval()
        else:
            torch.manual_seed(seed)
            sae = cls(d_model, d_sae, K).to(DEVICE)
            exp57.train_sae(name, sae, stream, save_dir, d_model, d_sae, layer)

        recon = exp57.evaluate_reconstruction(name, sae, eval_data, layer, d_sae)
        cell_results.setdefault(name, {})["reconstruction"] = recon

        try:
            sp = run_sparse_probing_seeded(
                name, sae, model_name, layer, model_tag, expansion, seed
            )
            cell_results[name]["sparse_probing"] = sp.get("sparse_probing", {})
        except Exception as e:  # noqa: BLE001 - mirror exp57's defensive eval
            print(f"    WARNING: sparse probing failed: {e}")
            cell_results[name]["sparse_probing"] = {"error": str(e)}

        if hasattr(sae, "scale_a"):
            cell_results[name]["scale_a"] = sae.scale_a.item()

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    return cell_results


def _dir_size_gb(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / 1e9


def evict_other_models(keep_model_tag):
    """Delete SAEBench probe-activation caches and checkpoints for every model
    EXCEPT keep_model_tag.

    The probe cache (artifacts/sparse_probing/<HF-model>/) is ~37-44G per model
    and never self-evicts; left alone it accumulates across the matrix and fills
    the disk (the run-1/2/3 failure mode). It is keyed by model+layer and is
    identical across a model's expansions/seeds, so we evict only when SWITCHING
    models, never within one. Checkpoints (~0.3-0.5G/cell) are kept for the
    current model so re-runs resume; other models' are dropped to reclaim space.
    """
    import shutil
    keep_hf = MODEL_CONFIGS[keep_model_tag]["model_name"]  # e.g. Qwen/Qwen3-1.7B
    keep_leaf = keep_hf.split("/")[-1]
    freed = 0.0
    # 1) probe-activation caches under artifacts/sparse_probing/<org>/<model>
    sp_root = Path("artifacts/sparse_probing")
    if sp_root.exists():
        for org_dir in sp_root.iterdir():
            if not org_dir.is_dir():
                continue
            for model_dir in org_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != keep_leaf:
                    sz = _dir_size_gb(model_dir)
                    print(f"    [evict] probe cache {model_dir} ({sz:.1f}G)")
                    shutil.rmtree(model_dir, ignore_errors=True)
                    freed += sz
    # 2) checkpoints for other models (checkpoints/exp67/<tag>_<exp>x_seed<seed>)
    ckpt_root = Path("checkpoints/exp67")
    if ckpt_root.exists():
        for cell_dir in ckpt_root.iterdir():
            tag = cell_dir.name.split("_")[0]  # "1.7b" / "4b" / "8b"
            if cell_dir.is_dir() and tag != keep_model_tag:
                sz = _dir_size_gb(cell_dir)
                print(f"    [evict] checkpoint {cell_dir} ({sz:.1f}G)")
                shutil.rmtree(cell_dir, ignore_errors=True)
                freed += sz
    print(f"  [evict] freed {freed:.1f}G (kept model {keep_model_tag})")


def main():
    parser = argparse.ArgumentParser(
        description="Exp67: scaling-matrix reseed (grounds the d_model claim)"
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--expansion", type=int, choices=EXPANSION_RATIOS, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Run all expansion ratios for the given model")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds, e.g. 123,456")
    parser.add_argument("--evict-others", action="store_true",
                        help="Before running, delete probe caches + checkpoints "
                             "for all OTHER models to reclaim disk. Use when "
                             "running models sequentially on a tight disk.")
    args = parser.parse_args()

    if not args.all and args.expansion is None:
        parser.error("Must specify --expansion or --all")

    expansions = EXPANSION_RATIOS if args.all else [args.expansion]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    model_tag = args.model
    cfg = MODEL_CONFIGS[model_tag]

    print("Experiment 67: Scaling Matrix Reseed")
    print(f"  Model: {cfg['model_name']} (tag={model_tag})")
    print(f"  Layer: {cfg['hook_layer']}, d_model: {cfg['d_model']}")
    print(f"  Expansions: {expansions}  Seeds: {seeds}")
    print(f"  Tokens: {N_TRAIN_TOKENS:,}")
    print()

    if args.evict_others:
        print(f"Evicting other models' caches/checkpoints (keeping {model_tag})...")
        evict_other_models(model_tag)

    # Disk guard: SAEBench probe caches are tens of GB; abort early with a clear
    # message rather than truncating a .pt mid-write (the run-1/2/3 corruption).
    # Measure the volume where data ACTUALLY lands: artifacts/ and checkpoints/
    # are typically symlinked to a scratch mount (e.g. /mnt), which is NOT the
    # same filesystem as cwd (often a near-full root). Resolve the symlink and
    # stat that volume, not "." (the earlier false-abort bug).
    import shutil as _sh
    artifacts_dir = Path("artifacts").resolve()  # follows symlink to /mnt if set
    guard_path = artifacts_dir if artifacts_dir.exists() else Path(".").resolve()
    free_gb = _sh.disk_usage(str(guard_path)).free / 1e9
    MIN_FREE_GB = 60.0
    print(f"  Free disk on data volume ({guard_path}): {free_gb:.1f}G "
          f"(need >= {MIN_FREE_GB:.0f}G)")
    if free_gb < MIN_FREE_GB:
        sys.exit(f"ABORT: only {free_gb:.1f}G free on {guard_path}; need >= "
                 f"{MIN_FREE_GB:.0f}G for {model_tag} probe cache + checkpoints. "
                 f"Free space or use --evict-others, then retry.")

    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"  Model loaded: {cfg['model_name']}")

    all_results = load_results()

    for seed in seeds:
        for expansion in expansions:
            cell_key = f"{model_tag}_{expansion}x_seed{seed}"
            if cell_key in all_results and all_results[cell_key].get("architectures"):
                print(f"\n  SKIP {cell_key}: already in results")
                continue

            t0 = time.time()
            cell_results = run_cell_seeded(model_tag, expansion, seed, model, tokenizer)
            all_results[cell_key] = {
                "model": cfg["model_name"],
                "model_tag": model_tag,
                "d_model": cfg["d_model"],
                "layer": cfg["hook_layer"],
                "expansion": expansion,
                "d_sae": cfg["d_model"] * expansion,
                "k": K,
                "n_train_tokens": N_TRAIN_TOKENS,
                "seed": seed,
                "elapsed_sec": round(time.time() - t0, 1),
                "architectures": cell_results,
            }
            save_results(all_results)
            print(f"\n  Results saved for {cell_key} "
                  f"({all_results[cell_key]['elapsed_sec']/3600:.2f}h)")

    print(f"\n  Full results: {RESULTS_PATH}")
    print(f"  Checkpoints: checkpoints/exp67/")


if __name__ == "__main__":
    main()
