"""
Experiment 55c: Norm-Stratified FVE Across Depth + Reference SAE

Per-quartile reconstruction quality for standard, best cosine, and
adamkarvonen reference at L9/L18/L27. Validates the Q4 norm catastrophe
on an independently-trained SAE.

Run on <gpu-server> GPU 1:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 experiments/exp55c_norm_stratified.py \
        2>&1 | tee experiments/exp55c_output.log
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL = 4096
D_SAE = 65536
K = 80
NORM_EPS = 1e-8

CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
N_TOKENS = 100_000
EVAL_BATCH_SIZE = 512

CKPTS = {
    9: {
        "standard": "/scratch/checkpoints/exp43c/standard_L9_final.pt",
        "adaptive_l2": "/scratch/checkpoints/exp43c/adaptive_l2_L9_final.pt",
    },
    18: {
        "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
        "adaptive_l2": "/scratch/checkpoints/exp40/adaptive_l2_L18_final.pt",
    },
    27: {
        "standard": "/scratch/checkpoints/exp43/standard_L27_final.pt",
        "adaptive_l2": "/scratch/checkpoints/exp43/adaptive_l2_L27_final.pt",
    },
}

RESULTS_PATH = "experiments/exp55c_results.json"

# ─── SAE Architectures ────────────────────────────────────────────────────────


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


# ─── SAE Loading ──────────────────────────────────────────────────────────────

SAE_CLASSES = {
    "standard": BatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
}


def load_sae(name, layer):
    cls = SAE_CLASSES[name]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(CKPTS[layer][name], map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def load_adamkarvonen(layer):
    """Load adamkarvonen reference and wrap as forward-compatible SAE."""
    from benchmarks.adapter import BenchSAE
    bench = BenchSAE.from_adamkarvonen(hook_layer=layer, device=DEVICE, dtype=DTYPE)

    class AdamkWrapper(nn.Module):
        def __init__(self, bench_sae):
            super().__init__()
            self._bench = bench_sae

        def encode(self, x):
            return self._bench.encode(x)

        def decode(self, f):
            return self._bench.decode(f)

        def forward(self, x):
            f = self.encode(x)
            return self.decode(f), f

    return AdamkWrapper(bench)


# ─── Activation Collection ────────────────────────────────────────────────────


class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
    captured = {}

    def hook(module, inp, out):
        captured["act"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        raise _EarlyStop

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    except _EarlyStop:
        pass
    handle.remove()
    return captured["act"]


def collect_activations(model, tokenizer, layer_idx, n_tokens):
    from datasets import load_dataset
    print(f"  Collecting {n_tokens:,} activations at L{layer_idx}...")
    t0 = time.time()
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)
    text_iter = iter(ds)
    all_acts = []
    collected = 0
    while collected < n_tokens:
        batch_texts = []
        for _ in range(COLLECTION_BATCH_SIZE):
            try:
                row = next(text_iter)
                if len(row["text"]) > 50:
                    batch_texts.append(row["text"][:8192])
            except StopIteration:
                break
        if not batch_texts:
            break
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=CTX_LEN).to(DEVICE)
        acts = _collect_layer_acts(model, layer_idx, inputs)
        flat = acts[inputs["attention_mask"].bool()]
        all_acts.append(flat.cpu())
        collected += flat.shape[0]
    all_acts = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"  Collected {all_acts.shape[0]:,} tokens in {time.time()-t0:.1f}s")
    return all_acts


# ─── Norm-Stratified FVE Analysis ────────────────────────────────────────────


@torch.no_grad()
def analyze_norm_sensitivity(saes, activations, layer):
    """Per-quartile FVE -- does cosine encoder help high-norm tokens?"""
    print(f"\n=== Norm-Stratified FVE at L{layer} ===")
    results = {}

    sample = activations[:100_000].float()
    norms = sample.norm(dim=-1)
    q25, q50, q75 = norms.quantile(torch.tensor([0.25, 0.50, 0.75]))
    quartile_bounds = [
        ("Q1 (low)", 0, q25),
        ("Q2", q25, q50),
        ("Q3", q50, q75),
        ("Q4 (high)", q75, float("inf")),
    ]
    print(f"  Norm quartiles: Q1<{q25:.1f}, Q2<{q50:.1f}, Q3<{q75:.1f}, Q4>{q75:.1f}")

    for name, sae in saes.items():
        print(f"\n  {name}:")
        quartile_results = {}

        for qname, lo, hi in quartile_bounds:
            all_recon = []
            all_orig = []

            for i in range(0, min(activations.shape[0], 100_000), EVAL_BATCH_SIZE):
                batch = activations[i:i+EVAL_BATCH_SIZE].to(DEVICE, dtype=torch.float32)
                batch_norms = batch.norm(dim=-1)
                mask = (batch_norms >= lo) & (batch_norms < hi)
                if mask.sum() == 0:
                    continue
                subset = batch[mask]
                x_hat, _ = sae(subset.to(DTYPE))
                x_hat = x_hat.float()
                all_recon.append((subset - x_hat).cpu())
                all_orig.append(subset.cpu())

            if not all_orig:
                continue
            orig = torch.cat(all_orig, dim=0).float()
            resid = torch.cat(all_recon, dim=0).float()
            total_var = torch.var(orig, dim=0, unbiased=False).sum()
            resid_var = torch.var(resid, dim=0, unbiased=False).sum()
            fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
            cos = F.cosine_similarity(orig, orig - resid, dim=-1).mean().item()
            l2_in = orig.norm(dim=-1).mean().item()
            l2_out = (orig - resid).norm(dim=-1).mean().item()
            l2_ratio = l2_out / l2_in if l2_in > 0 else 0
            n_tokens = orig.shape[0]

            quartile_results[qname] = {
                "fve": round(fve, 4),
                "cos_recon": round(cos, 4),
                "l2_ratio": round(l2_ratio, 4),
                "mean_norm_in": round(l2_in, 2),
                "mean_norm_out": round(l2_out, 2),
                "n_tokens": n_tokens,
            }
            print(f"    {qname}: FVE={fve:.4f} cos={cos:.4f} L2r={l2_ratio:.4f} (n={n_tokens:,})")

        results[name] = quartile_results

    # Print Q1 vs Q4 gap
    print(f"\n  Q4 FVE (the catastrophe indicator):")
    for name in saes:
        q4 = results[name].get("Q4 (high)", {}).get("fve", 0)
        q1 = results[name].get("Q1 (low)", {}).get("fve", 0)
        print(f"    {name:25s}: Q4={q4:.4f}  Q1={q1:.4f}  gap={q4-q1:+.4f}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Exp 55c: Norm-Stratified FVE Across Depth + Reference SAE")
    print("=" * 70)

    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    all_results = {}

    for layer in [9, 18, 27]:
        print(f"\n{'='*70}")
        print(f"  Layer {layer}")
        print(f"{'='*70}")

        # Collect activations
        activations = collect_activations(model, tokenizer, layer, N_TOKENS)

        # Load SAEs for this layer
        saes = {}
        saes["standard"] = load_sae("standard", layer)
        saes["adaptive_l2"] = load_sae("adaptive_l2", layer)
        saes["adamkarvonen_ref"] = load_adamkarvonen(layer)

        # Run analysis
        layer_results = analyze_norm_sensitivity(saes, activations, layer)
        all_results[f"L{layer}"] = layer_results

        # Save incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

        # Cleanup
        for s in saes.values():
            del s
        del saes, activations
        torch.cuda.empty_cache()
        gc.collect()

    # Free model
    del model
    torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*70}")
    print("  Summary: Q4 FVE Across Depth")
    print(f"{'='*70}")
    print(f"  {'SAE':25s} {'L9 Q4':>10s} {'L18 Q4':>10s} {'L27 Q4':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for sae_name in ["standard", "adaptive_l2", "adamkarvonen_ref"]:
        row = f"  {sae_name:25s}"
        for layer in [9, 18, 27]:
            q4 = all_results.get(f"L{layer}", {}).get(sae_name, {}).get("Q4 (high)", {}).get("fve", "?")
            if isinstance(q4, float):
                row += f" {q4:10.4f}"
            else:
                row += f" {'?':>10s}"
        print(row)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
