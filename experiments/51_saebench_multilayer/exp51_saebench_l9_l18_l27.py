"""
Experiment 51: SAEBench sparse probing at L9/L18/L27 + adamkarvonen reference.

Does the cosine > standard gap (+14.9pp at L18) generalize across depth?
Also evaluates adamkarvonen reference SAE (independently-trained BatchTopK, 500M)
to validate our standard baseline isn't just bad.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp51_saebench_l9_l18_l27.py \
        > experiments/exp51_output.log 2>&1 &
"""

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_cudnn_sdp(False)

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL = 4096
D_SAE = 65536
K = 80
NORM_EPS = 1e-8

CKPTS = {
    9: {
        "standard": "/scratch/checkpoints/exp43c/standard_L9_final.pt",
        "adaptive_l2": "/scratch/checkpoints/exp43c/adaptive_l2_L9_final.pt",
        "perfeature_l2": "/scratch/checkpoints/exp43c/perfeature_l2_L9_final.pt",
        "no_C": "/scratch/checkpoints/exp43c/no_C_L9_final.pt",
    },
    18: {
        "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
        "adaptive_l2": "/scratch/checkpoints/exp40/adaptive_l2_L18_final.pt",
        "perfeature_l2": "/scratch/checkpoints/exp40/perfeature_l2_L18_final.pt",
        "no_C": "/scratch/checkpoints/exp42c/no_C_L18_final.pt",
    },
    27: {
        "standard": "/scratch/checkpoints/exp43/standard_L27_final.pt",
        "adaptive_l2": "/scratch/checkpoints/exp43/adaptive_l2_L27_final.pt",
        "perfeature_l2": "/scratch/checkpoints/exp43/perfeature_l2_L27_final.pt",
        "no_C": "/scratch/checkpoints/exp43/no_C_L27_final.pt",
    },
}

OUTPUT_BASE = "/scratch/saebench_results/exp51"
RESULTS_PATH = "experiments/exp51_results.json"


# =============================================================================
# SAE Architectures (must match training definitions exactly)
# =============================================================================

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
        if self.training:
            return self._batch_topk(post_relu)
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


class AdaptiveCosineBatchTopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
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
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.training:
            return self._batch_topk(post_relu)
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

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
        if self.training:
            return self._batch_topk(post_relu)
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


class NoCBatchTopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
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
        x_c = x - self.b_dec
        x_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_u = x_c / x_norm
        w_u = F.normalize(self.W_enc, dim=-1, eps=NORM_EPS)
        post_relu = F.relu(x_u @ w_u.T)
        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)
        self._cached_x_norm = x_norm
        return encoded

    def decode(self, f, x_norm=None):
        w_u = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w_u
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        if x_norm is None:
            x_norm = getattr(self, "_cached_x_norm", None)
        if x_norm is not None:
            x_raw = x_raw * (x_norm / raw_norm)
        return x_raw + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


# =============================================================================
# Load and wrap
# =============================================================================

SAE_CLASSES = {
    "standard": BatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
    "perfeature_l2": PerFeatureAdaptiveCosineSAE,
    "no_C": NoCBatchTopKSAE,
}


def load_sae(name, layer):
    cls = SAE_CLASSES[name]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt_path = CKPTS[layer][name]
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def wrap_for_saebench(name, sae, layer):
    from benchmarks.adapter import BenchSAE

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


def load_adamkarvonen(layer):
    from benchmarks.adapter import BenchSAE
    return BenchSAE.from_adamkarvonen(hook_layer=layer, device=DEVICE, dtype=DTYPE)


def main():
    from benchmarks.run_saebench import run_saebench

    all_results = {}
    variants = ["standard", "adaptive_l2", "perfeature_l2", "no_C", "adamkarvonen_ref"]
    layers = [9, 18, 27]

    for layer in layers:
        for name in variants:
            run_key = f"{name}_L{layer}"
            print(f"\n{'='*70}")
            print(f"  SAEBench: {run_key}")
            print(f"{'='*70}")

            t0 = time.time()

            try:
                if name == "adamkarvonen_ref":
                    bench_sae = load_adamkarvonen(layer)
                else:
                    sae = load_sae(name, layer)
                    bench_sae = wrap_for_saebench(name, sae, layer)

                sae_label = f"exp51-{name}-L{layer}"
                out_dir = f"{OUTPUT_BASE}/{name}_L{layer}"

                results = run_saebench(
                    bench_sae,
                    sae_name=sae_label,
                    eval_types=["core", "sparse_probing"],
                    output_dir=out_dir,
                    llm_batch_size=4,
                    device=DEVICE,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                results = {"error": str(e)}

            elapsed = time.time() - t0
            all_results[run_key] = {
                "saebench": results,
                "elapsed_min": round(elapsed / 60, 1),
            }
            print(f"\n  {run_key} done in {elapsed/60:.1f} min")

            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            if name != "adamkarvonen_ref":
                del sae
            del bench_sae
            torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*70}")
    print("  SAEBench Summary — Sparse Probing")
    print(f"{'='*70}")
    print(f"  {'SAE':<20} {'Layer':>5} {'top-1':>8} {'top-2':>8} {'top-5':>8} {'test_acc':>10} {'time':>8}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for key, data in all_results.items():
        sb = data.get("saebench", {})
        sp = sb.get("sparse_probing", {})
        if isinstance(sp, dict) and "error" not in sb:
            top1 = sp.get("sae_top1_test_accuracy", "?")
            top2 = sp.get("sae_top2_test_accuracy", "?")
            top5 = sp.get("sae_top5_test_accuracy", "?")
            acc = sp.get("sae_test_accuracy", "?")
            parts = key.rsplit("_L", 1)
            name_part = parts[0]
            layer_part = parts[1] if len(parts) > 1 else "?"
            t = data.get("elapsed_min", "?")
            print(f"  {name_part:<20} {layer_part:>5} {top1:>8} {top2:>8} {top5:>8} {acc:>10} {t:>7}m")
        else:
            print(f"  {key}: ERROR — {sb.get('error', 'unknown')}")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
