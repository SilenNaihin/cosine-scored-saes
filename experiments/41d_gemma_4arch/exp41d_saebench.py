"""
Experiment 41d — SAEBench (core + sparse_probing) on all 24 Gemma checkpoints
===============================================================================

Evaluates all checkpoints from exp41d_gemma_4arch_auxk.py:
  4 variants × 3 layers × 2 (±aux-k) = 24 SAEs

Skips runs where combined results already exist (resume-friendly).

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python3 -u \
        experiments/exp41d_saebench.py \
        > experiments/exp41d_saebench_output.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.adapter import BenchSAE
from benchmarks.run_saebench import run_saebench

from exp41d_gemma_4arch_auxk import (
    SAE_CLASSES, NoCBatchTopKSAE,
    D_MODEL, D_SAE, K, NORM_EPS,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "google/gemma-2-2b"
CKPT_DIR = Path("checkpoints/exp41d")

VARIANTS = ["standard", "adaptive_l2", "perfeature_l2", "perfeature_bd", "no_C"]
LAYERS = [7, 13, 19]
AUXK_OPTIONS = [False, True]


def ckpt_key(variant, layer, use_auxk):
    return f"{variant}_L{layer}{'_auxk' if use_auxk else ''}"


def load_sae(variant, layer, use_auxk):
    cls = SAE_CLASSES[variant]
    key = ckpt_key(variant, layer, use_auxk)
    path = CKPT_DIR / f"{key}_final.pt"
    if not path.exists():
        return None
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def wrap_for_saebench(sae, layer):
    encode_fn = sae.encode
    decode_fn = sae.decode

    W_dec = F.normalize(sae.W_dec.detach(), dim=1)
    b_enc = sae.b_enc.detach() if hasattr(sae, "b_enc") else torch.zeros(
        D_SAE, device=DEVICE, dtype=sae.W_enc.dtype
    )

    return BenchSAE(
        W_enc=sae.W_enc.detach().T,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=sae.b_dec.detach(),
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name=MODEL_NAME,
        hook_layer=layer,
        device=DEVICE,
        dtype=DTYPE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--evals", nargs="+", default=["core", "sparse_probing"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="experiments/exp41d_saebench_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "exp41d_saebench_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"runs": {}}

    for layer in args.layers:
        for use_auxk in AUXK_OPTIONS:
            for variant in args.variants:
                key = ckpt_key(variant, layer, use_auxk)
                sae_name = f"exp41d_{key}"

                combined_path = output_dir / f"{sae_name}_combined.json"
                if combined_path.exists():
                    print(f"\n  {key} — SKIP (results exist)")
                    continue

                print(f"\n{'='*70}\n  {key}\n{'='*70}")
                sae = load_sae(variant, layer, use_auxk)
                if sae is None:
                    print(f"  Checkpoint missing, skip")
                    summary["runs"][sae_name] = {"error": "checkpoint missing"}
                    continue

                bench = wrap_for_saebench(sae, layer)
                t0 = time.time()
                try:
                    results = run_saebench(
                        bench, sae_name=sae_name,
                        eval_types=args.evals,
                        output_dir=str(output_dir),
                        llm_batch_size=args.batch_size,
                        llm_dtype="bfloat16",
                        device=DEVICE,
                    )
                    summary["runs"][sae_name] = {
                        "elapsed_s": time.time() - t0,
                        "evals": list(results.keys()),
                        "status": "complete",
                    }
                except Exception as e:
                    print(f"  ERROR: {e}")
                    summary["runs"][sae_name] = {"error": str(e)}

                del sae, bench
                gc.collect()
                torch.cuda.empty_cache()

                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
