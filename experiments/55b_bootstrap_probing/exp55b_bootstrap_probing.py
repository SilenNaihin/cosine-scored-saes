"""
Experiment 55b: Bootstrap Sparse Probing Confidence Intervals

The +14.9pp sparse probing advantage (perfeature_l2 0.815 vs standard 0.667)
is a point estimate from a single SAEBench run. This experiment runs sparse
probing 5 times with different random seeds for the probe training to estimate
variance. This tells us if the gap is stable across probe initialization.

Run on <gpu-server> GPU 1:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python3 experiments/exp55b_bootstrap_probing.py \
        > experiments/exp55b_output.log 2>&1 &
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.backends.cuda.enable_cudnn_sdp(False)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL, D_SAE, K = 4096, 65536, 80
NORM_EPS = 1e-8
LAYER = 18

CKPTS = {
    "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
    "perfeature_l2": "/scratch/checkpoints/exp40/perfeature_l2_L18_final.pt",
}

N_BOOTSTRAP = 5
SEEDS = [42, 123, 456, 789, 1337]
OUTPUT_BASE = "/scratch/saebench_results/exp55b"
RESULTS_PATH = "experiments/exp55b_results.json"


# ─── SAE Architectures ───────────────────────────────────────────────────────

class BatchTopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.tensor(-1.0))
    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)
    def encode(self, x):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.threshold < 0: return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)
    def decode(self, f): return f @ self.W_dec + self.b_dec
    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


class PerFeatureAdaptiveCosineSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.zeros(d_sae))
        self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.tensor(-1.0))
    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)
    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        log_norm = torch.log(input_norm)
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.threshold < 0: return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)
    def decode(self, f): return f @ self.W_dec + self.b_dec
    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


SAE_CLASSES = {
    "standard": BatchTopKSAE,
    "perfeature_l2": PerFeatureAdaptiveCosineSAE,
}


def load_sae(name):
    cls = SAE_CLASSES[name]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(CKPTS[name], map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def wrap_for_saebench(name, sae):
    from benchmarks.adapter import BenchSAE
    sae_dtype = sae.W_enc.dtype

    def encode_fn(x):
        return sae.encode(x.to(dtype=sae_dtype)).to(dtype=x.dtype)

    def decode_fn(f):
        return sae.decode(f.to(dtype=sae_dtype)).to(dtype=f.dtype)

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
        hook_layer=LAYER,
        device=DEVICE,
        dtype=DTYPE,
    )


def main():
    import sae_bench.evals.sparse_probing.main as sp

    print("=" * 70)
    print("Exp 55b: Bootstrap Sparse Probing Confidence Intervals")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    all_results = {"config": {
        "model": MODEL_NAME, "layer": LAYER,
        "n_bootstrap": N_BOOTSTRAP, "seeds": SEEDS,
        "sae_names": list(CKPTS.keys()),
    }}

    for sae_name in ["standard", "perfeature_l2"]:
        print(f"\n{'='*60}")
        print(f"SAE: {sae_name}")
        print(f"{'='*60}")

        sae = load_sae(sae_name)
        bench_sae = wrap_for_saebench(sae_name, sae)

        seed_results = []
        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n  Seed {seed_idx+1}/{N_BOOTSTRAP}: {seed}")
            t0 = time.time()

            out_dir = f"{OUTPUT_BASE}/{sae_name}_seed{seed}"
            sae_label = f"exp55b-{sae_name}-seed{seed}"

            config = sp.SparseProbingEvalConfig(
                model_name=MODEL_NAME,
                llm_batch_size=4,
                llm_dtype="bfloat16",
                random_seed=seed,
            )

            sp.run_eval(
                config,
                [(sae_label, bench_sae)],
                DEVICE,
                out_dir,
                force_rerun=True,
            )

            # Load result
            result_file = None
            for p in Path(out_dir).glob("*.json"):
                if sae_label in p.stem:
                    result_file = p
                    break
            if result_file is None:
                for p in Path(out_dir).glob("*.json"):
                    result_file = p
                    break

            if result_file:
                with open(result_file) as f:
                    result_data = json.load(f)

                metrics = result_data.get("eval_result_metrics", {})
                sae_metrics = metrics.get("sae", {})
                top1 = sae_metrics.get("sae_top_1_test_accuracy", None)

                # Per-dataset breakdown
                details = result_data.get("eval_result_details", [])
                per_dataset = {}
                for det in details:
                    ds_name = det.get("dataset_name", "unknown")
                    per_dataset[ds_name] = {
                        "top_1": det.get("sae_top_1_test_accuracy"),
                        "top_5": det.get("sae_top_5_test_accuracy"),
                    }

                elapsed = time.time() - t0
                print(f"    Top-1: {top1:.4f} ({elapsed:.0f}s)")
                seed_results.append({
                    "seed": seed,
                    "top_1": top1,
                    "per_dataset": per_dataset,
                    "elapsed_s": elapsed,
                })
            else:
                print(f"    No result file found")
                seed_results.append({"seed": seed, "error": "no result file"})

        # Aggregate
        top1s = [r["top_1"] for r in seed_results if r.get("top_1") is not None]
        if top1s:
            agg = {
                "mean_top1": float(np.mean(top1s)),
                "std_top1": float(np.std(top1s)),
                "min_top1": float(np.min(top1s)),
                "max_top1": float(np.max(top1s)),
                "all_top1": top1s,
            }
            print(f"\n  {sae_name} aggregate: {agg['mean_top1']:.4f} ± {agg['std_top1']:.4f} "
                  f"(range: {agg['min_top1']:.4f} - {agg['max_top1']:.4f})")
        else:
            agg = {"error": "no valid results"}

        all_results[sae_name] = {
            "seeds": seed_results,
            "aggregate": agg,
        }

        del sae, bench_sae
        torch.cuda.empty_cache()

        # Incremental save
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Confidence interval on the gap
    std_top1s = [r["top_1"] for r in all_results["standard"]["seeds"] if r.get("top_1")]
    cos_top1s = [r["top_1"] for r in all_results["perfeature_l2"]["seeds"] if r.get("top_1")]

    if std_top1s and cos_top1s:
        gaps = [c - s for c, s in zip(cos_top1s, std_top1s)]
        print(f"\n{'='*60}")
        print(f"GAP (perfeature_l2 - standard)")
        print(f"{'='*60}")
        print(f"  Mean gap: {np.mean(gaps)*100:.2f}pp ± {np.std(gaps)*100:.2f}pp")
        print(f"  Min gap:  {np.min(gaps)*100:.2f}pp")
        print(f"  Max gap:  {np.max(gaps)*100:.2f}pp")
        print(f"  All gaps: {[f'{g*100:.2f}pp' for g in gaps]}")

        all_results["gap"] = {
            "mean_pp": float(np.mean(gaps) * 100),
            "std_pp": float(np.std(gaps) * 100),
            "min_pp": float(np.min(gaps) * 100),
            "max_pp": float(np.max(gaps) * 100),
            "all_pp": [float(g * 100) for g in gaps],
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFinal results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
