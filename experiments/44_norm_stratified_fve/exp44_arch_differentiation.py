"""
Experiment 44: Architecture Differentiation Analysis
=====================================================

Goes beyond surface metrics (FVE, dead%) to understand how the 4 architectures
differ in practice. Uses the 500M-token L18 checkpoints from exp40/42c.

Analyses:
  1. Compute per step:       Forward+backward timing for each architecture
  2. Decoder direction overlap: Pairwise cosine similarity of decoder directions
                                across architectures — do they learn the same features?
  3. Feature specialization:  Which features are unique to each architecture?
                                Activation correlation on shared eval data
  4. Norm sensitivity:        Per-quartile reconstruction quality — does cosine
                                encoder actually help low-norm tokens more?
  5. Feature steering:        Pick 10 known-interpretable directions, inject into
                                residual stream, compare behavioral consistency

Architectures:
  1. standard:       BatchTopK (inner product)
  2. adaptive_l2:    Cosine + global adaptive scale (a=0.258 at L18)
  3. perfeature_l2:  Cosine + per-feature adaptive scale (a_mean=0.076)
  4. no_C:           NoC — pure cosine, norm-preserving decode

All checkpoints from exp40 (standard, adaptive_l2, perfeature_l2) and
exp42c (no_C), trained at 500M tokens with saprmarks recipe on Qwen3-8B L18.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp44_arch_differentiation.py \
        2>&1 | tee experiments/exp44_output.log &
"""

import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN SDPA backend — broken on H100 with driver 595.58 / cuDNN 9.1
torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 18
D_MODEL = 4096
D_SAE = 65536
K = 80
NORM_EPS = 1e-8
CTX_LEN = 2048
BATCH_SIZE = 2048
COLLECTION_BATCH_SIZE = 4

N_ANALYSIS_TOKENS = 500_000
N_STEERING_SAMPLES = 50

CKPTS = {
    "standard": "checkpoints/exp40/standard_L18_final.pt",
    "adaptive_l2": "checkpoints/exp40/adaptive_l2_L18_final.pt",
    "perfeature_l2": "checkpoints/exp40/perfeature_l2_L18_final.pt",
    "no_C": "/mnt/nvme0/checkpoints/exp42c/no_C_L18_final.pt",
}

RESULTS_PATH = Path("experiments/exp44_results.json")


# =============================================================================
# SAE Architectures (eval only)
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


# =============================================================================
# Activation collection
# =============================================================================

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


def collect_activations(model, tokenizer, n_tokens):
    print(f"  Collecting {n_tokens:,} activations at L{LAYER}...")
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
        acts = _collect_layer_acts(model, LAYER, inputs)
        flat = acts[inputs["attention_mask"].bool()]
        all_acts.append(flat.cpu())
        collected += flat.shape[0]
    result = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"    {result.shape[0]:,} tokens in {time.time()-t0:.1f}s")
    return result


# =============================================================================
# Analysis 1: Compute per step (forward + backward timing)
# =============================================================================

def analyze_compute(saes, activations):
    """Time forward + backward for each architecture."""
    print("\n=== Analysis 1: Compute per Step ===")
    results = {}
    n_warmup = 5
    n_timed = 50

    for name, sae in saes.items():
        print(f"\n  {name}:")
        sae.train()  # Need backward

        # Warmup
        for _ in range(n_warmup):
            batch = activations[:BATCH_SIZE].to(DEVICE, dtype=torch.float32)
            x_hat, f = sae(batch)
            loss = (batch - x_hat).pow(2).sum(dim=-1).mean()
            loss.backward()
            sae.zero_grad()

        torch.cuda.synchronize()
        fwd_times, bwd_times = [], []

        for i in range(n_timed):
            batch = activations[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to(DEVICE, dtype=torch.float32)
            if batch.shape[0] < BATCH_SIZE:
                break

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x_hat, f = sae(batch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            loss = (batch - x_hat).pow(2).sum(dim=-1).mean()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            fwd_times.append(t1 - t0)
            bwd_times.append(t2 - t1)
            sae.zero_grad()

        sae.eval()
        fwd_ms = np.mean(fwd_times) * 1000
        bwd_ms = np.mean(bwd_times) * 1000
        total_ms = fwd_ms + bwd_ms

        results[name] = {
            "forward_ms": round(fwd_ms, 2),
            "backward_ms": round(bwd_ms, 2),
            "total_ms": round(total_ms, 2),
            "relative_to_standard": None,  # filled below
        }
        print(f"    Forward: {fwd_ms:.2f} ms")
        print(f"    Backward: {bwd_ms:.2f} ms")
        print(f"    Total: {total_ms:.2f} ms")

    std_total = results["standard"]["total_ms"]
    for name in results:
        results[name]["relative_to_standard"] = round(
            results[name]["total_ms"] / std_total, 3
        )
        print(f"  {name}: {results[name]['relative_to_standard']:.3f}x standard")

    return results


# =============================================================================
# Analysis 2: Decoder direction overlap across architectures
# =============================================================================

@torch.no_grad()
def analyze_decoder_overlap(saes):
    """Pairwise cosine similarity of decoder directions across architectures."""
    print("\n=== Analysis 2: Decoder Direction Overlap ===")
    results = {}

    # Get unit-norm decoder directions for each
    dec_dirs = {}
    for name, sae in saes.items():
        W = F.normalize(sae.W_dec.detach(), dim=1)  # (d_sae, d_model)
        dec_dirs[name] = W

    pairs = list(combinations(saes.keys(), 2))
    for name_a, name_b in pairs:
        print(f"\n  {name_a} vs {name_b}:")
        W_a = dec_dirs[name_a]  # (d_sae, d_model)
        W_b = dec_dirs[name_b]  # (d_sae, d_model)

        # For each feature in A, find its best match in B
        # Process in chunks to avoid OOM on (65536, 65536) matrix
        chunk_size = 4096
        max_sims_ab = []
        best_match_ab = []
        for i in range(0, D_SAE, chunk_size):
            chunk_a = W_a[i:i+chunk_size]  # (chunk, d_model)
            sim = chunk_a @ W_b.T  # (chunk, d_sae)
            max_vals, max_idx = sim.max(dim=1)
            max_sims_ab.append(max_vals)
            best_match_ab.append(max_idx)

        max_sims_ab = torch.cat(max_sims_ab)
        best_match_ab = torch.cat(best_match_ab)

        # Reverse: for each feature in B, find best match in A
        max_sims_ba = []
        for i in range(0, D_SAE, chunk_size):
            chunk_b = W_b[i:i+chunk_size]
            sim = chunk_b @ W_a.T
            max_vals, _ = sim.max(dim=1)
            max_sims_ba.append(max_vals)
        max_sims_ba = torch.cat(max_sims_ba)

        # Mutual best matches (feature i in A matches j in B, AND j in B matches i in A)
        reverse_match = []
        for i in range(0, D_SAE, chunk_size):
            chunk_b = W_b[i:i+chunk_size]
            sim = chunk_b @ W_a.T
            _, max_idx = sim.max(dim=1)
            reverse_match.append(max_idx)
        reverse_match = torch.cat(reverse_match)

        mutual = 0
        for feat_a in range(D_SAE):
            feat_b = best_match_ab[feat_a].item()
            if reverse_match[feat_b].item() == feat_a:
                mutual += 1

        # Thresholded matches
        thresh_90 = (max_sims_ab > 0.9).sum().item()
        thresh_95 = (max_sims_ab > 0.95).sum().item()
        thresh_99 = (max_sims_ab > 0.99).sum().item()

        pair_key = f"{name_a}_vs_{name_b}"
        results[pair_key] = {
            "mean_max_sim_ab": round(float(max_sims_ab.mean()), 4),
            "mean_max_sim_ba": round(float(max_sims_ba.mean()), 4),
            "median_max_sim": round(float(max_sims_ab.median()), 4),
            "mutual_best_matches": mutual,
            "mutual_pct": round(mutual / D_SAE * 100, 2),
            "matches_above_90": thresh_90,
            "matches_above_95": thresh_95,
            "matches_above_99": thresh_99,
            "pct_above_90": round(thresh_90 / D_SAE * 100, 2),
            "pct_above_95": round(thresh_95 / D_SAE * 100, 2),
        }

        print(f"    Mean best-match cosine: {max_sims_ab.mean():.4f} (A→B), {max_sims_ba.mean():.4f} (B→A)")
        print(f"    Mutual best matches: {mutual:,} ({mutual/D_SAE*100:.1f}%)")
        print(f"    >0.90: {thresh_90:,} ({thresh_90/D_SAE*100:.1f}%)")
        print(f"    >0.95: {thresh_95:,} ({thresh_95/D_SAE*100:.1f}%)")

    return results


# =============================================================================
# Analysis 3: Norm-stratified reconstruction quality
# =============================================================================

@torch.no_grad()
def analyze_norm_sensitivity(saes, activations):
    """Per-quartile FVE — does cosine encoder help low-norm tokens more?"""
    print("\n=== Analysis 3: Norm-Stratified Reconstruction ===")
    results = {}

    # Compute norms and quartile boundaries
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
        sae.eval()
        quartile_results = {}

        for qname, lo, hi in quartile_bounds:
            all_recon = []
            all_orig = []

            for i in range(0, min(activations.shape[0], 100_000), BATCH_SIZE):
                batch = activations[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
                batch_norms = batch.norm(dim=-1)
                mask = (batch_norms >= lo) & (batch_norms < hi)
                if mask.sum() == 0:
                    continue
                subset = batch[mask]
                x_hat, _ = sae(subset)
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
            n_tokens = orig.shape[0]

            quartile_results[qname] = {
                "fve": round(fve, 4),
                "cos_recon": round(cos, 4),
                "n_tokens": n_tokens,
            }
            print(f"    {qname}: FVE={fve:.4f} cos={cos:.4f} (n={n_tokens:,})")

        results[name] = quartile_results

    # Print Q1 vs Q4 gap
    print("\n  Q1-Q4 FVE gap (positive = Q4 better):")
    for name in saes:
        q1 = results[name].get("Q1 (low)", {}).get("fve", 0)
        q4 = results[name].get("Q4 (high)", {}).get("fve", 0)
        print(f"    {name:20s}: Q4-Q1 = {q4-q1:+.4f} (Q1={q1:.4f}, Q4={q4:.4f})")

    return results


# =============================================================================
# Analysis 4: Feature activation correlation across architectures
# =============================================================================

@torch.no_grad()
def analyze_feature_specialization(saes, activations):
    """Which tokens activate which features? Measure activation pattern overlap."""
    print("\n=== Analysis 4: Feature Activation Patterns ===")
    results = {}

    # Collect top-k active feature indices for each architecture on same data
    n_tokens = min(50_000, activations.shape[0])
    arch_active = {}  # name -> (n_tokens,) list of sets of active feature indices

    for name, sae in saes.items():
        sae.eval()
        active_sets = []
        for i in range(0, n_tokens, BATCH_SIZE):
            batch = activations[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
            _, features = sae(batch)
            # For each token, which features are active?
            active_mask = features > 0  # (batch, d_sae)
            for j in range(active_mask.shape[0]):
                active_sets.append(set(active_mask[j].nonzero(as_tuple=True)[0].cpu().tolist()))
        arch_active[name] = active_sets[:n_tokens]
        print(f"  {name}: collected {len(arch_active[name]):,} activation patterns")

    # Pairwise Jaccard similarity of active feature sets
    pairs = list(combinations(saes.keys(), 2))
    for name_a, name_b in pairs:
        sets_a = arch_active[name_a]
        sets_b = arch_active[name_b]
        jaccards = []
        for sa, sb in zip(sets_a, sets_b):
            if len(sa) == 0 and len(sb) == 0:
                continue
            intersection = len(sa & sb)
            union = len(sa | sb)
            jaccards.append(intersection / union if union > 0 else 0)

        pair_key = f"{name_a}_vs_{name_b}"
        results[pair_key] = {
            "mean_jaccard": round(float(np.mean(jaccards)), 4),
            "median_jaccard": round(float(np.median(jaccards)), 4),
            "std_jaccard": round(float(np.std(jaccards)), 4),
        }
        print(f"  {name_a} vs {name_b}: Jaccard={np.mean(jaccards):.4f} "
              f"(median={np.median(jaccards):.4f})")

    return results


# =============================================================================
# Analysis 5: Feature steering
# =============================================================================

@torch.no_grad()
def analyze_feature_steering(saes, model, tokenizer):
    """Inject top feature directions and measure behavioral change consistency."""
    print("\n=== Analysis 5: Feature Steering ===")
    results = {}

    # Collect some prompts
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In the year 2025, artificial intelligence",
        "The patient presented with symptoms of",
        "import torch\nimport numpy as np\n",
        "Once upon a time in a small village",
        "The mathematical proof relies on",
        "Breaking news: the stock market today",
        "La ciudad más grande de España es",
        "The function f(x) = x^2 has derivative",
    ]

    for name, sae in saes.items():
        print(f"\n  {name}:")
        sae.eval()

        # Find top features by mean activation
        act_sums = torch.zeros(D_SAE, device=DEVICE)
        n_counted = 0
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=CTX_LEN).to(DEVICE)
            acts = _collect_layer_acts(model, LAYER, inputs)
            flat = acts.reshape(-1, D_MODEL)
            features = sae.encode(flat)
            act_sums += features.sum(dim=0)
            n_counted += flat.shape[0]

        top_feats = act_sums.topk(20).indices.tolist()

        # For each feature, steer and measure KL divergence
        steer_strengths = [0.5, 1.0, 2.0, 5.0]
        feat_results = []

        for feat_idx in top_feats[:10]:
            feat_dir = sae.W_dec[feat_idx]
            feat_dir_unit = feat_dir / feat_dir.norm().clamp(min=1e-8)

            kl_by_strength = {}
            for strength in steer_strengths:
                kls = []
                for prompt in prompts:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=CTX_LEN).to(DEVICE)

                    # Clean logits
                    outputs_clean = model(**inputs)
                    clean_logits = outputs_clean.logits[0, -1]
                    clean_probs = F.softmax(clean_logits, dim=-1)

                    # Steered logits
                    act = _collect_layer_acts(model, LAYER, inputs)
                    steered_act = act.clone()
                    steer_vec = strength * feat_dir.norm() * feat_dir_unit
                    steered_act[0, -1] += steer_vec.to(steered_act.dtype)

                    def steering_hook(module, inp, out):
                        result = out[0] if isinstance(out, tuple) else out
                        result = result.clone()
                        result[0, -1] = steered_act[0, -1].to(result.dtype)
                        if isinstance(out, tuple):
                            return (result,) + out[1:]
                        return result

                    handle = model.model.layers[LAYER].register_forward_hook(steering_hook)
                    outputs_steer = model(**inputs)
                    steer_logits = outputs_steer.logits[0, -1]
                    steer_probs = F.softmax(steer_logits, dim=-1)
                    handle.remove()

                    kl = F.kl_div(steer_probs.log(), clean_probs, reduction="sum").item()
                    kls.append(kl)

                kl_by_strength[str(strength)] = round(float(np.mean(kls)), 4)

            feat_results.append({
                "feature_idx": feat_idx,
                "kl_by_strength": kl_by_strength,
            })

        # Summary: mean KL at each strength across top features
        summary = {}
        for strength in steer_strengths:
            s = str(strength)
            mean_kl = np.mean([fr["kl_by_strength"][s] for fr in feat_results])
            summary[s] = round(float(mean_kl), 4)

        results[name] = {
            "per_feature": feat_results,
            "mean_kl_by_strength": summary,
        }

        print(f"    Mean KL at strength 1.0: {summary['1.0']:.4f}")
        print(f"    Mean KL at strength 5.0: {summary['5.0']:.4f}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Experiment 44: Architecture Differentiation Analysis")
    print(f"  Model: {MODEL_NAME}, Layer {LAYER}")
    print(f"  Analyzing {N_ANALYSIS_TOKENS:,} tokens\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()

    print("Collecting activations...")
    activations = collect_activations(model, tokenizer, N_ANALYSIS_TOKENS)

    print("\nLoading SAEs...")
    saes = {}
    for name in CKPTS:
        sae = load_sae(name)
        saes[name] = sae
        print(f"  {name}: loaded")

    results = {}

    # Run all analyses
    results["compute"] = analyze_compute(saes, activations)
    results["decoder_overlap"] = analyze_decoder_overlap(saes)
    results["norm_sensitivity"] = analyze_norm_sensitivity(saes, activations)
    results["feature_specialization"] = analyze_feature_specialization(saes, activations)
    results["feature_steering"] = analyze_feature_steering(saes, model, tokenizer)

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    print("\n  Compute (relative to standard):")
    for name, data in results["compute"].items():
        print(f"    {name:20s}: {data['relative_to_standard']:.3f}x ({data['total_ms']:.1f} ms)")

    print("\n  Decoder overlap (mutual best matches):")
    for pair, data in results["decoder_overlap"].items():
        print(f"    {pair:40s}: {data['mutual_pct']:.1f}% mutual, "
              f"{data['pct_above_95']:.1f}% >0.95")

    print("\n  Norm sensitivity (Q1 vs Q4 FVE gap):")
    for name, data in results["norm_sensitivity"].items():
        q1 = data.get("Q1 (low)", {}).get("fve", 0)
        q4 = data.get("Q4 (high)", {}).get("fve", 0)
        print(f"    {name:20s}: Q4-Q1 = {q4-q1:+.4f}")

    print("\n  Feature steering (mean KL at strength 1.0):")
    for name, data in results["feature_steering"].items():
        kl = data["mean_kl_by_strength"]["1.0"]
        print(f"    {name:20s}: KL={kl:.4f}")


if __name__ == "__main__":
    main()
