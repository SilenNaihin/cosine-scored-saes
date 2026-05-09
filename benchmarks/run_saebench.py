"""
SAEBench runner for RNH validation.

Calls individual SAEBench eval runners directly, bypassing the hardcoded
MODEL_CONFIGS in run_all_evals_custom_saes.py. This lets us use Qwen3-8B.

Usage:
    # On GPU with the existing inner-product SAE:
    python benchmarks/run_saebench.py

    # Or from Python:
    from benchmarks.adapter import BenchSAE
    from benchmarks.run_saebench import run_saebench

    sae = BenchSAE.from_adamkarvonen(hook_layer=18, device="cuda")
    run_saebench(sae, "baseline-batchtopk-l18")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from benchmarks.adapter import BenchSAE

# Lazy imports for eval modules (heavy deps, slow to load)
_EVAL_MODULES: dict = {}


def _get_eval_module(name: str):
    if name not in _EVAL_MODULES:
        if name == "core":
            import sae_bench.evals.core.main as m
        elif name == "sparse_probing":
            import sae_bench.evals.sparse_probing.main as m
        elif name == "absorption":
            import sae_bench.evals.absorption.main as m
        elif name == "ravel":
            import sae_bench.evals.ravel.main as m
        elif name == "scr":
            import sae_bench.evals.scr_and_tpp.main as m
        elif name == "tpp":
            import sae_bench.evals.scr_and_tpp.main as m
        else:
            raise ValueError(f"Unknown eval type: {name}")
        _EVAL_MODULES[name] = m
    return _EVAL_MODULES[name]


def run_saebench(
    sae: BenchSAE,
    sae_name: str,
    eval_types: list[str] | None = None,
    output_dir: str = "benchmarks/eval_results",
    llm_batch_size: int = 8,
    llm_dtype: str = "bfloat16",
    device: str = "cuda",
    force_rerun: bool = False,
) -> dict[str, dict]:
    """Run SAEBench evaluations on a wrapped SAE.

    Args:
        sae: A BenchSAE adapter wrapping any SAE source.
        sae_name: Identifier for this SAE (used in result filenames).
        eval_types: Which evals to run. Defaults to safe set (no API keys needed).
            Options: core, sparse_probing, absorption, ravel, scr, tpp.
            Note: autointerp requires OpenAI API key; unlearning is Gemma-only.
        output_dir: Where to save result JSON files.
        llm_batch_size: Batch size for model inference (8 is safe for 8B on A100 80GB).
        llm_dtype: Model dtype string ("bfloat16" or "float32").
        device: Device string.
        force_rerun: If True, rerun even if results already exist.

    Returns:
        Dict mapping eval_type -> result dict.
    """
    if eval_types is None:
        eval_types = ["core", "sparse_probing", "absorption"]

    model_name = sae.cfg.model_name
    selected_saes = [(sae_name, sae)]
    results = {}

    for eval_type in eval_types:
        print(f"\n{'='*60}")
        print(f"Running {eval_type} evaluation")
        print(f"{'='*60}\n")

        eval_output = os.path.join(output_dir, eval_type)
        os.makedirs(eval_output, exist_ok=True)

        if eval_type == "core":
            mod = _get_eval_module("core")
            result = mod.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=100,
                n_eval_sparsity_variance_batches=1000,
                eval_batch_size_prompts=llm_batch_size,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder=eval_output,
                verbose=True,
                dtype=llm_dtype,
                device=device,
                force_rerun=force_rerun,
            )
            results["core"] = result

        elif eval_type == "sparse_probing":
            mod = _get_eval_module("sparse_probing")
            config = mod.SparseProbingEvalConfig(
                model_name=model_name,
                llm_batch_size=llm_batch_size,
                llm_dtype=llm_dtype,
            )
            artifacts_dir = os.path.join(output_dir, "artifacts")
            mod.run_eval(
                config,
                selected_saes,
                device,
                eval_output,
                force_rerun=force_rerun,
                artifacts_path=artifacts_dir,
            )
            results["sparse_probing"] = _load_result(eval_output, sae_name)

        elif eval_type == "absorption":
            mod = _get_eval_module("absorption")
            config = mod.AbsorptionEvalConfig(
                model_name=model_name,
                llm_batch_size=llm_batch_size,
                llm_dtype=llm_dtype,
            )
            mod.run_eval(
                config,
                selected_saes,
                device,
                eval_output,
                force_rerun=force_rerun,
            )
            results["absorption"] = _load_result(eval_output, sae_name)

        elif eval_type == "ravel":
            mod = _get_eval_module("ravel")
            config = mod.RAVE<author>alConfig(
                model_name=model_name,
                llm_batch_size=max(1, llm_batch_size // 4),
                llm_dtype=llm_dtype,
            )
            mod.run_eval(
                config,
                selected_saes,
                device,
                eval_output,
                force_rerun=force_rerun,
            )
            results["ravel"] = _load_result(eval_output, sae_name)

        elif eval_type == "scr":
            mod = _get_eval_module("scr")
            config = mod.ScrAndTppEvalConfig(
                model_name=model_name,
                perform_scr=True,
                llm_batch_size=llm_batch_size,
                llm_dtype=llm_dtype,
            )
            mod.run_eval(
                config,
                selected_saes,
                device,
                eval_output,
                force_rerun=force_rerun,
            )
            results["scr"] = _load_result(eval_output, sae_name)

        elif eval_type == "tpp":
            mod = _get_eval_module("tpp")
            config = mod.ScrAndTppEvalConfig(
                model_name=model_name,
                perform_scr=False,
                llm_batch_size=llm_batch_size,
                llm_dtype=llm_dtype,
            )
            mod.run_eval(
                config,
                selected_saes,
                device,
                eval_output,
                force_rerun=force_rerun,
            )
            results["tpp"] = _load_result(eval_output, sae_name)

        else:
            print(f"Skipping unknown eval type: {eval_type}")

    # Save combined results
    combined_path = os.path.join(output_dir, f"{sae_name}_combined.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_path}")

    return results


def _load_result(output_dir: str, sae_name: str) -> dict:
    """Try to load the result JSON that an eval runner saved."""
    for p in Path(output_dir).glob("*.json"):
        if sae_name in p.stem:
            with open(p) as f:
                return json.load(f)
    # Fall back to loading any JSON in the directory
    for p in Path(output_dir).glob("*.json"):
        with open(p) as f:
            return json.load(f)
    return {"status": "completed, but result file not found"}


# ------------------------------------------------------------------
# CLI entry point: test with the existing adamkarvonen SAE
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SAEBench on an SAE")
    parser.add_argument(
        "--layer", type=int, default=18, help="Hook layer (default: 18)"
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=["core"],
        help="Eval types to run (default: core)",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (default: cuda)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="LLM batch size (default: 8)"
    )
    args = parser.parse_args()

    print(f"Loading adamkarvonen BatchTopK SAE for layer {args.layer}...")
    sae = BenchSAE.from_adamkarvonen(
        hook_layer=args.layer,
        device=args.device,
        dtype=torch.bfloat16,
    )

    print(f"SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    print(f"Decoder norms OK: {sae.check_decoder_norms()}")

    results = run_saebench(
        sae,
        sae_name=f"adamkarvonen-batchtopk-l{args.layer}",
        eval_types=args.evals,
        llm_batch_size=args.batch_size,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("Results summary:")
    print("=" * 60)
    for eval_type, result in results.items():
        print(f"\n{eval_type}:")
        if isinstance(result, dict):
            for k, v in list(result.items())[:10]:
                print(f"  {k}: {v}")
        else:
            print(f"  {result}")
