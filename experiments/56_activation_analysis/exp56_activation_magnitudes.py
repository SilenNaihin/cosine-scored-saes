"""
Experiment 56 — Activation Magnitude Analysis

Since all decoders have unit norms, the TPP power gap must come from
activation magnitudes. When TPP zeroes feature i, the residual stream
changes by -activation_i * W_dec[i]. With unit-norm decoders, the
perturbation magnitude is just |activation_i|.

If cosine SAEs produce smaller activations, zeroing features has less
effect on the residual stream, explaining lower TPP intended effect.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp56_activation_magnitudes.py
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_cudnn_sdp(False)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL, D_SAE, K = 4096, 65536, 80
NORM_EPS = 1e-8
LAYER = 18
N_TOKENS = 50000
COLLECTION_BATCH_SIZE = 16
CONTEXT_LENGTH = 256
SKIP_DOCS = 200000
SEED = 42
RESULTS_PATH = "experiments/exp56_activation_magnitudes.json"

CKPTS = {
    "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
    "adaptive_l2": "/scratch/checkpoints/exp40/adaptive_l2_L18_final.pt",
    "perfeature_l2": "/scratch/checkpoints/exp40/perfeature_l2_L18_final.pt",
    "no_C": "/scratch/checkpoints/exp42c/no_C_L18_final.pt",
}


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


class _EarlyStop(Exception):
    pass

def collect_layer_acts(model, layer_idx, inputs):
    captured = {}
    def hook(module, inp, out):
        captured["act"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        raise _EarlyStop
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    except _EarlyStop:
        pass
    handle.remove()
    return captured["act"]


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Exp 56: Activation Magnitude Analysis")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()

    # Collect activations
    print(f"\nCollecting {N_TOKENS:,} tokens...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= SKIP_DOCS:
            break

    all_acts = []
    tokens = 0
    batch_texts = []
    while tokens < N_TOKENS:
        try:
            row = next(text_iter)
        except StopIteration:
            break
        text = row["text"]
        if len(text) < 50:
            continue
        batch_texts.append(text[:2048])
        if len(batch_texts) >= COLLECTION_BATCH_SIZE:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=CONTEXT_LENGTH).to(DEVICE)
            acts = collect_layer_acts(model, LAYER, inputs)
            mask = inputs["attention_mask"].bool()
            flat = acts[mask]
            norms = flat.float().norm(dim=-1)
            median = norms.median()
            if median > 0:
                flat = flat[norms < median * 10.0]
            all_acts.append(flat.cpu().float())
            tokens += flat.shape[0]
            batch_texts = []
    all_acts = torch.cat(all_acts, dim=0)[:N_TOKENS]
    print(f"Collected {all_acts.shape[0]:,} tokens")

    # Input norms
    input_norms = all_acts.norm(dim=-1)
    print(f"\nInput norms: mean={input_norms.mean():.2f} std={input_norms.std():.2f} "
          f"median={input_norms.median():.2f}")

    del model
    torch.cuda.empty_cache()

    results = {"input_norm_mean": input_norms.mean().item(),
               "input_norm_std": input_norms.std().item(),
               "input_norm_median": input_norms.median().item()}

    for name in ["standard", "adaptive_l2", "perfeature_l2", "no_C"]:
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")

        sae = load_sae(name)

        # Encode in batches
        all_feats = []
        batch_size = 4096
        for i in range(0, all_acts.shape[0], batch_size):
            batch = all_acts[i:i+batch_size].to(DEVICE, dtype=DTYPE)
            with torch.no_grad():
                feats = sae.encode(batch)
            all_feats.append(feats.cpu().float())
        all_feats = torch.cat(all_feats, dim=0)

        # Activation statistics (non-zero only)
        nonzero_mask = all_feats > 0
        nonzero_vals = all_feats[nonzero_mask]

        print(f"  Active features per token: mean={nonzero_mask.float().sum(dim=1).mean():.1f}")
        print(f"  Non-zero activations: mean={nonzero_vals.mean():.4f} std={nonzero_vals.std():.4f} "
              f"median={nonzero_vals.median():.4f}")

        # Percentiles of non-zero activations
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pvals = torch.quantile(nonzero_vals, torch.tensor([p/100 for p in percentiles]))
        pct_str = " ".join(f"p{p}={v:.4f}" for p, v in zip(percentiles, pvals.tolist()))
        print(f"  Percentiles: {pct_str}")

        # Per-token reconstruction impact: sum of |activation * decoder_norm|
        # Since decoder norms are 1.0, this is just sum of activations per token
        total_act_per_token = all_feats.sum(dim=1)
        print(f"  Total activation per token: mean={total_act_per_token.mean():.2f} "
              f"std={total_act_per_token.std():.2f}")

        # Max activation per token
        max_act_per_token = all_feats.max(dim=1)[0]
        print(f"  Max activation per token: mean={max_act_per_token.mean():.4f} "
              f"std={max_act_per_token.std():.4f}")

        # Reconstruction norm — use forward() to handle NoC's norm caching
        rec_chunks = []
        for i in range(0, all_acts.shape[0], batch_size):
            batch_x = all_acts[i:i+batch_size].to(DEVICE, dtype=DTYPE)
            with torch.no_grad():
                rec, _ = sae.forward(batch_x)
            rec_chunks.append(rec.cpu().float())
        recs = torch.cat(rec_chunks, dim=0)
        rec_norms = recs.norm(dim=-1)
        error_norms = (recs - all_acts).norm(dim=-1)
        print(f"  Reconstruction norms: mean={rec_norms.mean():.2f} std={rec_norms.std():.2f}")
        print(f"  Error norms: mean={error_norms.mean():.2f}")
        print(f"  Rec/Input norm ratio: mean={(rec_norms / input_norms.clamp(min=1e-8)).mean():.4f}")

        results[name] = {
            "active_per_token_mean": nonzero_mask.float().sum(dim=1).mean().item(),
            "nonzero_act_mean": nonzero_vals.mean().item(),
            "nonzero_act_std": nonzero_vals.std().item(),
            "nonzero_act_median": nonzero_vals.median().item(),
            "nonzero_act_p25": pvals[1].item(),
            "nonzero_act_p75": pvals[3].item(),
            "nonzero_act_p95": pvals[5].item(),
            "nonzero_act_p99": pvals[6].item(),
            "total_act_per_token_mean": total_act_per_token.mean().item(),
            "total_act_per_token_std": total_act_per_token.std().item(),
            "max_act_per_token_mean": max_act_per_token.mean().item(),
            "rec_norm_mean": rec_norms.mean().item(),
            "error_norm_mean": error_norms.mean().item(),
            "rec_input_ratio_mean": (rec_norms / input_norms.clamp(min=1e-8)).mean().item(),
        }

        del sae, all_feats, recs
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'SAE':20s} {'act_mean':>10s} {'act_med':>10s} {'act_p95':>10s} {'total/tok':>10s} {'max/tok':>10s} {'rec/in':>8s}")
    for name in ["standard", "adaptive_l2", "perfeature_l2", "no_C"]:
        r = results[name]
        print(f"{name:20s} {r['nonzero_act_mean']:10.4f} {r['nonzero_act_median']:10.4f} "
              f"{r['nonzero_act_p95']:10.4f} {r['total_act_per_token_mean']:10.2f} "
              f"{r['max_act_per_token_mean']:10.4f} {r['rec_input_ratio_mean']:8.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
