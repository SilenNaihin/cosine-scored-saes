"""
Experiment 20: SAEBench Evaluation on Production Checkpoints
=============================================================
Runs industry-standard SAEBench evals (core, sparse_probing, absorption) on
the 9 exp17 checkpoints: {standard, adaptive_l2, perfeature_l2} x L{9,18,27}.

Why: Our custom metrics (cos→KL correlation, FVE, alive features) showed cosine
SAEs beating standard, but exp19 revealed Simpson's paradox in the cos>inner
diagnostic. We need task-based benchmarks that aren't confounded by norm.

Usage:
    # Sanity check — L18 only, core eval
    python3 experiments/exp20_saebench.py --layers 18 --evals core

    # Full run — all layers, all evals
    python3 experiments/exp20_saebench.py

    # Specific combo
    python3 experiments/exp20_saebench.py --layers 18 27 --evals core sparse_probing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.adapter import BenchSAE

# =============================================================================
# Exp17 SAE architectures (must match training code exactly)
# =============================================================================

class BatchTopKSAE(nn.Module):
    """Standard BatchTopK SAE with inner-product encoder."""

    def __init__(self, d_model: int, d_sae: int, k: int = 80):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.zeros(()))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
            batch_size = max(acts.shape[0], 1)
            total_k = min(self.k * batch_size, acts.numel())
            flat = acts.reshape(-1)
            values, indices = torch.topk(flat, total_k)
            sparse = torch.zeros_like(flat)
            sparse[indices] = values
            return sparse.view_as(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


class AdaptiveCosineBatchTopKSAE(nn.Module):
    """BatchTopK SAE with per-token adaptive-scale cosine encoder."""

    def __init__(self, d_model: int, d_sae: int, k: int = 80):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.zeros(()))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
            batch_size = max(acts.shape[0], 1)
            total_k = min(self.k * batch_size, acts.numel())
            flat = acts.reshape(-1)
            values, indices = torch.topk(flat, total_k)
            sparse = torch.zeros_like(flat)
            sparse[indices] = values
            return sparse.view_as(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


class PerFeatureAdaptiveCosineSAE(nn.Module):
    """BatchTopK SAE with per-feature adaptive-scale cosine encoder."""

    def __init__(self, d_model: int, d_sae: int, k: int = 80):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.zeros(d_sae))
        self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.zeros(()))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        log_norm = torch.log(input_norm)
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
            batch_size = max(acts.shape[0], 1)
            total_k = min(self.k * batch_size, acts.numel())
            flat = acts.reshape(-1)
            values, indices = torch.topk(flat, total_k)
            sparse = torch.zeros_like(flat)
            sparse[indices] = values
            return sparse.view_as(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


# =============================================================================
# Checkpoint loading + BenchSAE wrapping
# =============================================================================

VARIANTS = {
    "standard": BatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
    "perfeature_l2": PerFeatureAdaptiveCosineSAE,
}

D_MODEL = 4096
D_SAE = 16384
K = 80
CHECKPOINT_DIR = Path("checkpoints/exp17")


def load_exp17_sae(
    variant: str,
    layer: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> BenchSAE:
    """Load an exp17 checkpoint and wrap it for SAEBench."""
    path = CHECKPOINT_DIR / f"{variant}_L{layer}_final.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    cls = VARIANTS[variant]
    sae = cls(D_MODEL, D_SAE, K)
    sae.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    sae = sae.to(device=device, dtype=dtype).eval()

    # Build BenchSAE wrapper
    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        return sae.encode(x)

    def decode_fn(f: torch.Tensor) -> torch.Tensor:
        return sae.decode(f)

    W_enc = sae.W_enc.detach().T  # (d_sae, d_model) -> (d_model, d_sae)
    W_dec = F.normalize(sae.W_dec.detach(), dim=1)  # (d_sae, d_model), unit-norm rows
    b_enc = sae.b_enc.detach()
    b_dec = sae.b_dec.detach()

    return BenchSAE(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name="Qwen/Qwen3-8B",
        hook_layer=layer,
        device=device,
        dtype=dtype,
    )


# =============================================================================
# Per-layer eval runner (one model load per layer, all variants share it)
# =============================================================================

def run_layer_evals(
    layer: int,
    variants: list[str],
    eval_types: list[str],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    llm_batch_size: int = 8,
    output_dir: str = "experiments/exp20_results",
    force_rerun: bool = False,
) -> dict:
    """Run SAEBench evals for all variants at a single layer.

    Core eval is special — it uses multiple_evals() which loads the model once
    and evaluates all SAEs in a single pass. Other evals are run per-SAE.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Load all variants for this layer
    print(f"\n{'='*70}")
    print(f"Layer {layer}: Loading {len(variants)} SAE variants")
    print(f"{'='*70}")

    saes: list[tuple[str, BenchSAE]] = []
    for variant in variants:
        name = f"exp17-{variant}-L{layer}"
        print(f"  Loading {name}...")
        sae = load_exp17_sae(variant, layer, device=device, dtype=dtype)
        assert sae.check_decoder_norms(), f"Decoder norm check failed for {name}"
        saes.append((name, sae))
        print(f"    d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, norms OK")

    # Run each eval type
    for eval_type in eval_types:
        print(f"\n--- {eval_type} eval (layer {layer}) ---")
        eval_output = os.path.join(output_dir, eval_type)
        os.makedirs(eval_output, exist_ok=True)

        t0 = time.time()

        if eval_type == "core":
            import sae_bench.evals.core.main as core_eval

            result = core_eval.multiple_evals(
                selected_saes=saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=llm_batch_size,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder=eval_output,
                verbose=True,
                dtype="bfloat16",
                device=device,
                force_rerun=force_rerun,
            )
            for sae_result in result:
                name = sae_result.get("unique_id", sae_result.get("sae_set", "unknown"))
                results[f"{name}_core"] = sae_result

        elif eval_type == "sparse_probing":
            import sae_bench.evals.sparse_probing.main as sp_eval

            for name, sae in saes:
                config = sp_eval.SparseProbingEvalConfig(
                    model_name="Qwen/Qwen3-8B",
                    llm_batch_size=llm_batch_size,
                    llm_dtype="bfloat16",
                )
                sp_eval.run_eval(
                    config,
                    [(name, sae)],
                    device,
                    eval_output,
                    force_rerun=force_rerun,
                    clean_up_activations=True,
                    save_activations=False,
                )
                results[f"{name}_sparse_probing"] = _load_latest_result(eval_output, name)

        elif eval_type == "absorption":
            import sae_bench.evals.absorption.main as ab_eval

            for name, sae in saes:
                config = ab_eval.AbsorptionEvalConfig(
                    model_name="Qwen/Qwen3-8B",
                    llm_batch_size=llm_batch_size,
                    llm_dtype="bfloat16",
                )
                ab_eval.run_eval(
                    config,
                    [(name, sae)],
                    device,
                    eval_output,
                    force_rerun=force_rerun,
                )
                results[f"{name}_absorption"] = _load_latest_result(eval_output, name)

        elif eval_type == "scr_and_tpp":
            import sae_bench.evals.scr_and_tpp.main as scr_tpp_eval

            for name, sae in saes:
                config = scr_tpp_eval.ScrAndTppEvalConfig(
                    model_name="Qwen/Qwen3-8B",
                    llm_batch_size=llm_batch_size,
                    llm_dtype="bfloat16",
                    perform_scr=True,
                )
                scr_tpp_eval.run_eval(
                    config,
                    [(name, sae)],
                    device,
                    eval_output,
                    force_rerun=force_rerun,
                    clean_up_activations=True,
                    save_activations=False,
                )
                results[f"{name}_scr_and_tpp"] = _load_latest_result(eval_output, name)

        elif eval_type == "tpp":
            import sae_bench.evals.scr_and_tpp.main as scr_tpp_eval

            for name, sae in saes:
                config = scr_tpp_eval.ScrAndTppEvalConfig(
                    model_name="Qwen/Qwen3-8B",
                    llm_batch_size=llm_batch_size,
                    llm_dtype="bfloat16",
                    perform_scr=False,
                )
                scr_tpp_eval.run_eval(
                    config,
                    [(name, sae)],
                    device,
                    eval_output,
                    force_rerun=force_rerun,
                    clean_up_activations=True,
                    save_activations=False,
                )
                results[f"{name}_tpp"] = _load_latest_result(eval_output, name)

        elapsed = time.time() - t0
        print(f"  {eval_type} completed in {elapsed:.0f}s")

    return results


def _load_latest_result(output_dir: str, sae_name: str) -> dict:
    """Load the result JSON for a given SAE name."""
    for p in Path(output_dir).glob("*.json"):
        if sae_name in p.stem:
            with open(p) as f:
                return json.load(f)
    # Fall back to any recent JSON
    jsons = sorted(Path(output_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    if jsons:
        with open(jsons[-1]) as f:
            return json.load(f)
    return {"error": "result file not found"}


# =============================================================================
# Results summary
# =============================================================================

def print_comparison_table(all_results: dict, eval_types: list[str]):
    """Print a compact comparison table."""
    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)

    if "core" in eval_types:
        print("\n--- Core Metrics ---")
        print(f"{'SAE':<35} {'KL score':>10} {'CE score':>10} {'CosSim':>10} {'L0':>10} {'Alive%':>10}")
        print("-" * 90)
        for key, result in sorted(all_results.items()):
            if not key.endswith("_core"):
                continue
            metrics = result.get("metrics", result.get("eval_result_metrics", {}))
            if not metrics:
                continue
            mbp = metrics.get("model_behavior_preservation", {})
            mpp = metrics.get("model_performance_preservation", {})
            rq = metrics.get("reconstruction_quality", {})
            sp = metrics.get("sparsity", {})
            mm = metrics.get("misc_metrics", {})

            name = key.replace("_custom_sae_core", "").replace("_core", "")
            kl = mbp.get("kl_div_score", "?")
            ce = mpp.get("ce_loss_score", "?")
            cos = rq.get("cossim", "?")
            l0 = sp.get("l0", "?")
            alive = mm.get("frac_alive", "?")

            kl_s = f"{kl:.4f}" if isinstance(kl, float) else str(kl)
            ce_s = f"{ce:.4f}" if isinstance(ce, float) else str(ce)
            cos_s = f"{cos:.4f}" if isinstance(cos, float) else str(cos)
            l0_s = f"{l0:.1f}" if isinstance(l0, float) else str(l0)
            alive_s = f"{alive:.4f}" if isinstance(alive, float) else str(alive)

            print(f"{name:<35} {kl_s:>10} {ce_s:>10} {cos_s:>10} {l0_s:>10} {alive_s:>10}")

    if "sparse_probing" in eval_types:
        print("\n--- Sparse Probing ---")
        for key, result in sorted(all_results.items()):
            if not key.endswith("_sparse_probing"):
                continue
            name = key.replace("_sparse_probing", "")
            metrics = result.get("eval_result_metrics", result)
            print(f"\n{name}:")
            if isinstance(metrics, dict):
                for mk, mv in sorted(metrics.items()):
                    if isinstance(mv, dict):
                        for mk2, mv2 in sorted(mv.items()):
                            if isinstance(mv2, (int, float)):
                                print(f"  {mk}.{mk2}: {mv2:.4f}")
                    elif isinstance(mv, (int, float)):
                        print(f"  {mk}: {mv:.4f}")

    if "absorption" in eval_types:
        print("\n--- Absorption ---")
        for key, result in sorted(all_results.items()):
            if not key.endswith("_absorption"):
                continue
            name = key.replace("_absorption", "")
            metrics = result.get("eval_result_metrics", result)
            print(f"\n{name}:")
            if isinstance(metrics, dict):
                for mk, mv in sorted(metrics.items()):
                    if isinstance(mv, dict):
                        for mk2, mv2 in sorted(mv.items()):
                            if isinstance(mv2, (int, float)):
                                print(f"  {mk}.{mk2}: {mv2:.4f}")
                    elif isinstance(mv, (int, float)):
                        print(f"  {mk}: {mv:.4f}")

    for eval_suffix in ["scr_and_tpp", "tpp"]:
        if eval_suffix not in eval_types:
            continue
        print(f"\n--- {eval_suffix.upper()} ---")
        for key, result in sorted(all_results.items()):
            if not key.endswith(f"_{eval_suffix}"):
                continue
            name = key.replace(f"_{eval_suffix}", "")
            metrics = result.get("eval_result_metrics", result)
            print(f"\n{name}:")
            if isinstance(metrics, dict):
                for mk, mv in sorted(metrics.items()):
                    if isinstance(mv, dict):
                        for mk2, mv2 in sorted(mv.items()):
                            if isinstance(mv2, (int, float)):
                                print(f"  {mk}.{mk2}: {mv2:.4f}")
                    elif isinstance(mv, (int, float)):
                        print(f"  {mk}: {mv:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp20: SAEBench on exp17 checkpoints")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[9, 18, 27],
        help="Layers to evaluate (default: 9 18 27)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["standard", "adaptive_l2", "perfeature_l2"],
        help="SAE variants (default: all three)",
    )
    parser.add_argument(
        "--evals", nargs="+", default=["core", "sparse_probing", "absorption"],
        help="Eval types (default: core sparse_probing absorption)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--output-dir", default="experiments/exp20_results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    print("Experiment 20: SAEBench Evaluation")
    print("=" * 50)
    print(f"Layers:   {args.layers}")
    print(f"Variants: {args.variants}")
    print(f"Evals:    {args.evals}")
    print(f"Device:   {args.device}")
    print(f"Output:   {args.output_dir}")
    print()

    # Verify all checkpoints exist
    missing = []
    for variant in args.variants:
        for layer in args.layers:
            path = CHECKPOINT_DIR / f"{variant}_L{layer}_final.pt"
            if not path.exists():
                missing.append(str(path))
    if missing:
        print("ERROR: Missing checkpoints:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print(f"All {len(args.variants) * len(args.layers)} checkpoints found.\n")

    # Run evals layer by layer (shares model load within each layer)
    all_results = {}
    total_start = time.time()

    for layer in args.layers:
        layer_results = run_layer_evals(
            layer=layer,
            variants=args.variants,
            eval_types=args.evals,
            device=args.device,
            llm_batch_size=args.batch_size,
            output_dir=args.output_dir,
            force_rerun=args.force_rerun,
        )
        all_results.update(layer_results)

    total_elapsed = time.time() - total_start

    # Save combined results
    combined_path = os.path.join(args.output_dir, "exp20_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_path}")

    # Print comparison
    print_comparison_table(all_results, args.evals)

    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
