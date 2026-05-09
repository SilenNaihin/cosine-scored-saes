"""
SAEBench evaluation for exp40 variants: standard, adaptive_l2, perfeature_l2.
Also re-runs NoC from exp42c for completeness.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp40_saebench.py \
        2>&1 | tee experiments/exp40_saebench_output.log &
"""

import json
import math
import os
import sys
import time
from pathlib import Path

# Add project root to path for benchmarks imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN SDPA backend — broken on H100 with driver 595.58 / cuDNN 9.1
torch.backends.cuda.enable_cudnn_sdp(False)

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 18
D_MODEL = 4096
D_SAE = 65536
K = 80
NORM_EPS = 1e-8

# Checkpoint paths
CKPTS = {
    "standard": "checkpoints/exp40/standard_L18_final.pt",
    "adaptive_l2": "checkpoints/exp40/adaptive_l2_L18_final.pt",
    "perfeature_l2": "checkpoints/exp40/perfeature_l2_L18_final.pt",
    "no_C": "/mnt/nvme0/checkpoints/exp42c/no_C_L18_final.pt",
}

OUTPUT_DIR = "/mnt/nvme0/saebench_results/exp40_comparison"
RESULTS_PATH = "experiments/exp40_saebench_results.json"


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


def load_sae(name):
    cls = SAE_CLASSES[name]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(CKPTS[name], map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def wrap_for_saebench(name, sae):
    """Wrap an SAE into SAEBench's BenchSAE adapter."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmarks.adapter import BenchSAE

    encode_fn = sae.encode
    decode_fn = sae.decode

    # W_dec normalized for SAEBench (it expects unit-norm rows)
    W_dec = F.normalize(sae.W_dec.detach(), dim=1)

    # b_enc: NoC has no b_enc
    b_enc = sae.b_enc.detach() if hasattr(sae, "b_enc") else torch.zeros(
        D_SAE, device=DEVICE, dtype=sae.W_enc.dtype
    )

    return BenchSAE(
        W_enc=sae.W_enc.detach().T,  # (d_model, d_sae) — BaseSAE convention
        W_dec=W_dec,                  # (d_sae, d_model)
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
    from benchmarks.run_saebench import run_saebench

    all_results = {}
    variants = ["standard", "adaptive_l2", "perfeature_l2", "no_C"]

    for name in variants:
        print(f"\n{'='*70}")
        print(f"  SAEBench: {name}")
        print(f"{'='*70}")

        t0 = time.time()
        sae = load_sae(name)
        bench_sae = wrap_for_saebench(name, sae)

        sae_label = f"exp40-{name}-L{LAYER}"
        out_dir = f"{OUTPUT_DIR}/{name}"

        try:
            results = run_saebench(
                bench_sae,
                sae_name=sae_label,
                eval_types=["core", "sparse_probing"],
                output_dir=out_dir,
                llm_batch_size=4,
                device=DEVICE,
            )
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            results = {"error": str(e)}

        elapsed = time.time() - t0
        all_results[name] = {
            "saebench": results,
            "elapsed_min": round(elapsed / 60, 1),
        }
        print(f"\n  {name} done in {elapsed/60:.1f} min")

        # Save incremental results
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Free GPU memory
        del sae, bench_sae
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*70}")
    print("  SAEBench Summary")
    print(f"{'='*70}")
    for name, data in all_results.items():
        sb = data.get("saebench", {})
        core = sb.get("core", {})
        sp = sb.get("sparse_probing", {})
        print(f"\n  {name}:")
        if isinstance(core, dict):
            # Try to extract key metrics
            for k in ["kl_div_score", "ce_loss_score", "explained_variance", "l0", "frac_alive"]:
                if k in core:
                    print(f"    {k}: {core[k]}")
        if isinstance(sp, dict):
            for k in ["sae_test_accuracy", "sae_top1_test_accuracy"]:
                if k in sp:
                    print(f"    {k}: {sp[k]}")
        print(f"    elapsed: {data.get('elapsed_min', '?')} min")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
