"""
Experiment 42c (SAEBench eval) — core + sparse_probing on the 16k Gemma L13 50M checkpoints

Mirrors exp41c_saebench_50m.py but for d_sae=16384 instead of 9216, and
loads from `checkpoints/exp42c/`. Direct comparison to the published
Karvonen BatchTopK 16k checkpoints (exp42a results).
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

from exp39_norm_preserving_sae import BatchTopKSAE, NORM_EPS
from exp39d_leave_one_out import NoC_SAE


DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "google/gemma-2-2b"
GEMMA_D_MODEL = 2304
GEMMA_D_SAE = 16384      # exp42c width
K = 80
LAYER = 13
CKPT_DIR = Path("checkpoints/exp42c")

VARIANTS = {"standard": BatchTopKSAE, "no_C": NoC_SAE}


def load_native_sae(variant):
    cls = VARIANTS[variant]
    p = CKPT_DIR / f"{variant}_L{LAYER}.pt"
    if not p.exists():
        return None
    sae = cls(GEMMA_D_MODEL, GEMMA_D_SAE, K).to(DEVICE)
    state = torch.load(p, map_location=DEVICE)
    sae.load_state_dict({k: v.float() for k, v in state.items()})
    sae.eval()
    return sae


def wrap_for_saebench(sae, variant):
    cls_name = sae.__class__.__name__
    if cls_name == "BatchTopKSAE":
        W_enc = sae.W_enc.detach()
        W_dec = sae.W_dec.detach()
        W_dec = F.normalize(W_dec, dim=1)
        b_enc = sae.b_enc.detach()
        b_dec = sae.b_dec.detach()

        def encode_fn(x):
            return sae.encode(x.to(torch.float32)).to(x.dtype)

        def decode_fn(f):
            return sae.decode(f.to(torch.float32)).to(f.dtype)
    elif cls_name == "NoC_SAE":
        W_enc = sae.W_enc.detach()
        W_dec = sae.W_dec.detach()
        W_dec = F.normalize(W_dec, dim=1)
        b_enc = torch.zeros(GEMMA_D_SAE, device=DEVICE, dtype=DTYPE)
        b_dec = sae.b_dec.detach()

        def encode_fn(x):
            x32 = x.to(torch.float32)
            f, a = sae.encode_full(x32)
            sae._sae_bench_a = a
            return f.to(x.dtype)

        def decode_fn(f):
            f32 = f.to(torch.float32)
            a = getattr(sae, "_sae_bench_a", None)
            if a is None:
                w_u = F.normalize(sae.W_dec, dim=-1, eps=NORM_EPS)
                return (f32 @ w_u + sae.b_dec).to(f.dtype)
            x_raw = sae.decode_raw(f32)
            nrm = x_raw.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS)
            return (x_raw * (a / nrm) + sae.b_dec).to(f.dtype)
    else:
        raise ValueError(f"Unknown SAE class: {cls_name}")

    return BenchSAE(
        W_enc=W_enc.T.contiguous(),
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name=MODEL_NAME,
        hook_layer=LAYER,
        device=DEVICE,
        dtype=DTYPE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["standard", "no_C"])
    parser.add_argument("--evals", nargs="+", default=["core", "sparse_probing"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="experiments/exp42c_results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary = {"runs": {}}
    for variant in args.variants:
        print(f"\n{'='*70}\n  {variant} L{LAYER} 16k\n{'='*70}")
        sae = load_native_sae(variant)
        if sae is None:
            print(f"  [{variant}] missing, skip")
            continue
        bench = wrap_for_saebench(sae, variant)
        sae_name = f"exp42c_{variant}_L{LAYER}_16k"
        t0 = time.time()
        try:
            results = run_saebench(
                bench, sae_name=sae_name,
                eval_types=args.evals,
                output_dir=args.output_dir,
                llm_batch_size=args.batch_size,
                llm_dtype="bfloat16",
                device=DEVICE,
            )
            summary["runs"][sae_name] = {"elapsed_s": time.time() - t0,
                                         "evals": list(results.keys())}
        except Exception as e:
            print(f"  ERROR: {e}")
            summary["runs"][sae_name] = {"error": str(e)}
        del sae, bench
        gc.collect()
        torch.cuda.empty_cache()

    summary_path = Path(args.output_dir) / "exp42c_saebench_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
