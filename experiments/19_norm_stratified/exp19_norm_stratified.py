"""
Experiment 19: Norm-Stratified Analysis
========================================

Post-hoc analysis of exp17 checkpoints: stratify eval tokens by activation
norm into quartiles and compute per-quartile metrics. Tests whether the
cosine SAE advantage is uniform across the norm distribution or concentrated
on low-norm tokens.

Key questions:
  1. Does cos>inner win rate vary by norm quartile?
  2. Does the standard SAE close the FVE gap on high-norm tokens?
  3. Is the L27 cosine advantage driven by low-norm tokens?
  4. Does adaptive scale_a=0.21 at L27 specifically help high-norm tokens?

Loads exp17 final checkpoints (no training). Runs ablation evaluation with
per-token norm tracking, then stratifies all metrics by norm quartile.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python experiments/exp19_norm_stratified.py
"""

import json
import math
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

# --- Model ---
MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]
D_MODEL = 4096

# --- SAE architecture (must match exp17) ---
D_SAE = 16384
K = 80

# --- Eval ---
N_EVAL_TOKENS = 1_000_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0
BATCH_SIZE = 4096

# --- Ablation ---
N_ABLATION_FEATURES = 30       # Matches exp17 scale — ~50 samples/quartile
N_ABLATION_SAMPLES = 200       # Splits to ~50/quartile, enough for correlations
N_QUARTILES = 4

# --- Paths ---
CHECKPOINT_DIR = "checkpoints/exp17"
RESULTS_PATH = "experiments/exp19_results.json"

VARIANTS = ["standard", "adaptive_l2", "perfeature_l2"]


# =============================================================================
# SAE Architectures (copied from exp17 for self-containment)
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

    def _batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
            return self._batch_topk(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


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
            return self._batch_topk(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def _batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


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

    def _batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

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
            return self._batch_topk(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


SAE_CLASSES = {
    "standard": BatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
    "perfeature_l2": PerFeatureAdaptiveCosineSAE,
}


# =============================================================================
# Activation Collection
# =============================================================================

class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
    """Capture residual stream activations at a layer via forward hook."""
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


def collect_eval_data(model, tokenizer, layer_idx, n_tokens):
    """Collect evaluation activations, skipping 500K docs to avoid train overlap."""
    print(f"  Collecting eval activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    # Skip ahead to avoid overlap with training data
    skip_count = 500_000
    for i, _ in enumerate(text_iter):
        if i >= skip_count:
            break

    all_acts = []
    tokens_collected = 0
    while tokens_collected < n_tokens:
        batch_texts = []
        for _ in range(COLLECTION_BATCH_SIZE):
            try:
                row = next(text_iter)
                if len(row["text"]) > 50:
                    batch_texts.append(row["text"][:2048])
            except StopIteration:
                break
        if not batch_texts:
            break

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=CTX_LEN,
        ).to(DEVICE)

        acts = _collect_layer_acts(model, layer_idx, inputs)
        flat = acts[inputs["attention_mask"].bool()]
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            flat = flat[norms < median * OUTLIER_MULTIPLIER]
        all_acts.append(flat.to("cpu", dtype=DTYPE))
        tokens_collected += flat.shape[0]

    result = torch.cat(all_acts, dim=0)[:n_tokens]
    norms = result.float().norm(dim=-1)
    print(f"  Layer {layer_idx}: {result.shape[0]:,} eval tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={norms.mean():.1f}, std={norms.std():.1f})")
    return result


# =============================================================================
# Load Checkpoints
# =============================================================================

def load_sae(variant, layer_idx):
    """Load a trained SAE from exp17 checkpoint."""
    cls = SAE_CLASSES[variant]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    path = Path(CHECKPOINT_DIR) / f"{variant}_L{layer_idx}_final.pt"
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    sae.load_state_dict(state)
    sae.eval()
    print(f"  Loaded {variant}/L{layer_idx} from {path}")
    if hasattr(sae, "scale_a"):
        if sae.scale_a.dim() == 0:
            print(f"    scale_a={sae.scale_a.item():.4f}")
        else:
            a = sae.scale_a.detach()
            print(f"    scale_a: mean={a.mean().item():.4f}, median={a.median().item():.4f}")
    return sae


# =============================================================================
# Norm Quartile Computation
# =============================================================================

def compute_norm_quartiles(eval_data):
    """Compute norm quartile boundaries and assign each token to a quartile."""
    norms = eval_data.float().norm(dim=-1)
    boundaries = [
        norms.quantile(q).item()
        for q in [0.0, 0.25, 0.50, 0.75, 1.0]
    ]
    # Assign quartiles: Q1=lowest norms, Q4=highest norms
    quartile_idx = torch.zeros(len(norms), dtype=torch.long)
    q25 = norms.quantile(0.25)
    q50 = norms.quantile(0.50)
    q75 = norms.quantile(0.75)
    quartile_idx[norms >= q75] = 3  # Q4
    quartile_idx[(norms >= q50) & (norms < q75)] = 2  # Q3
    quartile_idx[(norms >= q25) & (norms < q50)] = 1  # Q2
    # Q1 is default (0)

    quartile_info = {
        "boundaries": boundaries,
        "labels": ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
    }
    for q in range(4):
        mask = quartile_idx == q
        qnorms = norms[mask]
        quartile_info[f"Q{q+1}_count"] = int(mask.sum().item())
        quartile_info[f"Q{q+1}_norm_mean"] = float(qnorms.mean().item())
        quartile_info[f"Q{q+1}_norm_std"] = float(qnorms.std().item())
        quartile_info[f"Q{q+1}_norm_range"] = [float(qnorms.min().item()), float(qnorms.max().item())]

    print(f"  Norm quartile boundaries: {[f'{b:.1f}' for b in boundaries]}")
    for q in range(4):
        r = quartile_info[f"Q{q+1}_norm_range"]
        print(f"    Q{q+1}: n={quartile_info[f'Q{q+1}_count']:,}, "
              f"norm={quartile_info[f'Q{q+1}_norm_mean']:.1f}±{quartile_info[f'Q{q+1}_norm_std']:.1f} "
              f"[{r[0]:.1f}, {r[1]:.1f}]")

    return norms, quartile_idx, quartile_info


# =============================================================================
# Per-Quartile FVE
# =============================================================================

@torch.no_grad()
def compute_quartile_fve(sae, eval_data, quartile_idx):
    """Compute FVE separately for each norm quartile."""
    n = eval_data.shape[0]
    # Accumulate per-quartile variance stats
    total_var = [0.0] * N_QUARTILES
    resid_var = [0.0] * N_QUARTILES
    counts = [0] * N_QUARTILES

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        qi = quartile_idx[i:i+BATCH_SIZE]
        x_hat, _ = sae(batch)
        resid = batch - x_hat

        for q in range(N_QUARTILES):
            mask = qi == q
            if mask.sum() == 0:
                continue
            bq = batch[mask]
            rq = resid[mask]
            total_var[q] += bq.var(dim=0, unbiased=False).sum().item() * mask.sum().item()
            resid_var[q] += rq.var(dim=0, unbiased=False).sum().item() * mask.sum().item()
            counts[q] += mask.sum().item()

    fves = {}
    for q in range(N_QUARTILES):
        if counts[q] > 0 and total_var[q] > 0:
            fves[f"Q{q+1}"] = 1.0 - resid_var[q] / total_var[q]
        else:
            fves[f"Q{q+1}"] = 0.0
    return fves


# =============================================================================
# Per-Quartile Reconstruction Metrics
# =============================================================================

@torch.no_grad()
def compute_quartile_recon(sae, eval_data, quartile_idx):
    """Compute per-quartile L2 loss and cosine reconstruction."""
    n = eval_data.shape[0]
    l2_sums = [0.0] * N_QUARTILES
    cos_sums = [0.0] * N_QUARTILES
    counts = [0] * N_QUARTILES

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        qi = quartile_idx[i:i+BATCH_SIZE]
        x_hat, _ = sae(batch)

        for q in range(N_QUARTILES):
            mask = qi == q
            if mask.sum() == 0:
                continue
            bq = batch[mask]
            xq = x_hat[mask]
            l2_sums[q] += (bq - xq).pow(2).sum(dim=-1).sum().item()
            cos_sums[q] += F.cosine_similarity(bq, xq, dim=-1).sum().item()
            counts[q] += mask.sum().item()

    results = {}
    for q in range(N_QUARTILES):
        if counts[q] > 0:
            results[f"Q{q+1}"] = {
                "l2_loss": l2_sums[q] / counts[q],
                "cos_recon": cos_sums[q] / counts[q],
                "n_tokens": counts[q],
            }
    return results


# =============================================================================
# Ablation with Per-Token Norm Tracking
# =============================================================================

def ablate_feature_kl(model, activation, feature_dir, layer_idx):
    """Ablate a feature direction from the residual stream, measure KL at logits."""
    projection = (activation @ feature_dir) * feature_dir
    x = activation.unsqueeze(0).unsqueeze(0).to(DTYPE)
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0).to(DTYPE)

    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

    def make_hook(replacement):
        def hook(module, inputs, outputs):
            return (replacement,) + outputs[1:] if isinstance(outputs, tuple) else replacement
        return hook

    h = model.model.layers[layer_idx].register_forward_hook(make_hook(x))
    with torch.no_grad():
        try:
            orig_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            h.remove()
            return None
    h.remove()

    h = model.model.layers[layer_idx].register_forward_hook(make_hook(x_ablated))
    with torch.no_grad():
        try:
            abl_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            h.remove()
            return None
    h.remove()

    orig_probs = torch.softmax(orig_logits, dim=-1).clamp(min=1e-10)
    abl_log_probs = torch.log_softmax(abl_logits, dim=-1)
    kl = torch.sum(orig_probs * (orig_probs.log() - abl_log_probs)).item()
    if np.isnan(kl) or kl < 0:
        return None
    return kl


def run_ablation_stratified(name, model, sae, eval_data, norms, quartile_idx, layer_idx):
    """Ablation evaluation that tracks per-token norms for stratification.

    For each feature, samples tokens across the norm distribution and records
    per-token: norm, cosine, inner product, SAE activation, KL divergence,
    and quartile membership.
    """
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Ablation [{tag}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    # Find active features
    n_probe = min(200_000, eval_data.shape[0])
    probe = eval_data[:n_probe]
    all_feats = []
    for i in range(0, n_probe, BATCH_SIZE):
        batch = probe[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        _, f = sae(batch)
        all_feats.append(f.detach().cpu())
    all_feats = torch.cat(all_feats, dim=0)

    freq = (all_feats > 0).float().mean(dim=0)
    alive_mask = freq > 0
    n_alive = alive_mask.sum().item()
    print(f"  [{tag}] {n_alive} alive features (of {D_SAE})")

    n_to_select = min(N_ABLATION_FEATURES, n_alive)
    top_idx = freq.topk(n_to_select).indices

    # Collect per-token data for all features
    feature_results = []
    # Also collect all per-token data for quartile aggregation
    all_token_data = []  # list of dicts with norm, cos, inner, sae_act, kl, quartile

    for rank, fi in enumerate(top_idx):
        fi = fi.item()
        feat_dir = sae.W_dec[fi].float()
        feat_dir = feat_dir / feat_dir.norm()

        feat_acts = all_feats[:, fi]
        active = torch.where(feat_acts > 0)[0]
        if len(active) < 40:
            continue

        # Sample more tokens to ensure coverage across quartiles
        n_sample = min(N_ABLATION_SAMPLES, len(active))
        chosen = active[torch.randperm(len(active))[:n_sample]]

        cos_v, norm_v, inner_v, sae_v, kl_v, q_v = [], [], [], [], [], []
        for idx in chosen:
            x = probe[idx].to(DEVICE, dtype=torch.float32)
            kl = ablate_feature_kl(model, x, feat_dir, layer_idx)
            if kl is None:
                continue
            token_norm = norms[idx].item()
            token_q = quartile_idx[idx].item()
            cos_val = F.cosine_similarity(x.unsqueeze(0), feat_dir.unsqueeze(0)).item()
            inner_val = (x @ feat_dir).item()
            sae_val = feat_acts[idx].item()

            cos_v.append(cos_val)
            norm_v.append(token_norm)
            inner_v.append(inner_val)
            sae_v.append(sae_val)
            kl_v.append(kl)
            q_v.append(token_q)

            all_token_data.append({
                "feature_idx": fi,
                "norm": token_norm,
                "cos": cos_val,
                "inner": inner_val,
                "sae_act": sae_val,
                "kl": kl,
                "quartile": token_q,
            })

        if len(kl_v) < 20:
            continue

        kl_arr = np.array(kl_v)
        if kl_arr.std() < 1e-10:
            continue

        cos_arr = np.array(cos_v)
        norm_arr = np.array(norm_v)
        inner_arr = np.array(inner_v)
        sae_arr = np.array(sae_v)
        q_arr = np.array(q_v)

        # Overall correlations
        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]

        # Per-quartile correlations
        quartile_stats = {}
        for q in range(N_QUARTILES):
            qmask = q_arr == q
            nq = qmask.sum()
            if nq < 10:
                quartile_stats[f"Q{q+1}"] = {"n": int(nq), "insufficient": True}
                continue
            cos_q = cos_arr[qmask]
            inner_q = inner_arr[qmask]
            kl_q = kl_arr[qmask]
            if kl_q.std() < 1e-10:
                quartile_stats[f"Q{q+1}"] = {"n": int(nq), "insufficient": True}
                continue

            qc_cos = np.corrcoef(cos_q, kl_q)[0, 1]
            qc_inner = np.corrcoef(inner_q, kl_q)[0, 1]
            quartile_stats[f"Q{q+1}"] = {
                "n": int(nq),
                "corr_cos_kl": float(qc_cos) if not np.isnan(qc_cos) else 0.0,
                "corr_inner_kl": float(qc_inner) if not np.isnan(qc_inner) else 0.0,
                "cos_wins_inner": bool(abs(qc_cos) > abs(qc_inner)),
            }

        result = {
            "feature_idx": fi,
            "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos),
            "corr_inner_kl": float(corr_inner),
            "corr_sae_kl": float(corr_sae),
            "corr_norm_kl": float(corr_norm),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "quartile_stats": quartile_stats,
            "n_per_quartile": [int((q_arr == q).sum()) for q in range(N_QUARTILES)],
        }
        feature_results.append(result)

        if rank < 5 or rank % 20 == 0:
            q_strs = []
            for q in range(N_QUARTILES):
                qs = quartile_stats.get(f"Q{q+1}", {})
                if qs.get("insufficient"):
                    q_strs.append(f"Q{q+1}:n={qs['n']}")
                else:
                    q_strs.append(f"Q{q+1}:{qs.get('corr_cos_kl', 0):.2f}>{qs.get('corr_inner_kl', 0):.2f}")
            print(f"    feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"{' '.join(q_strs)}")

    if not feature_results:
        print(f"  [{tag}] No features with enough data")
        return {"n_features": 0}

    # Aggregate per-quartile across all features
    n_feats = len(feature_results)
    aggregate = {
        "n_features": n_feats,
        "overall": {
            "cos_kl_mean": float(np.mean([r["corr_cos_kl"] for r in feature_results])),
            "inner_kl_mean": float(np.mean([r["corr_inner_kl"] for r in feature_results])),
            "sae_kl_mean": float(np.mean([r["corr_sae_kl"] for r in feature_results])),
            "norm_kl_mean": float(np.mean([r["corr_norm_kl"] for r in feature_results])),
            "cos_wins_inner": sum(r["cos_wins_inner"] for r in feature_results),
        },
    }

    for q in range(N_QUARTILES):
        qkey = f"Q{q+1}"
        cos_kls, inner_kls, wins = [], [], 0
        n_sufficient = 0
        for r in feature_results:
            qs = r["quartile_stats"].get(qkey, {})
            if qs.get("insufficient"):
                continue
            n_sufficient += 1
            cos_kls.append(qs["corr_cos_kl"])
            inner_kls.append(qs["corr_inner_kl"])
            if qs["cos_wins_inner"]:
                wins += 1

        if n_sufficient > 0:
            aggregate[qkey] = {
                "n_features": n_sufficient,
                "cos_kl_mean": float(np.mean(cos_kls)),
                "inner_kl_mean": float(np.mean(inner_kls)),
                "cos_wins_inner": wins,
                "cos_wins_inner_pct": float(wins / n_sufficient),
            }
        else:
            aggregate[qkey] = {"n_features": 0}

    # Print summary
    print(f"\n  [{tag}] Summary ({n_feats} features):")
    print(f"    Overall: cos→KL={aggregate['overall']['cos_kl_mean']:.4f} | "
          f"inner→KL={aggregate['overall']['inner_kl_mean']:.4f} | "
          f"cos>inner={aggregate['overall']['cos_wins_inner']}/{n_feats}")
    for q in range(N_QUARTILES):
        qkey = f"Q{q+1}"
        qa = aggregate.get(qkey, {})
        if qa.get("n_features", 0) > 0:
            print(f"    {qkey}: cos→KL={qa['cos_kl_mean']:.4f} | "
                  f"inner→KL={qa['inner_kl_mean']:.4f} | "
                  f"cos>inner={qa['cos_wins_inner']}/{qa['n_features']} "
                  f"({qa['cos_wins_inner_pct']:.0%})")
        else:
            print(f"    {qkey}: insufficient data")

    return {
        "features": feature_results,
        "aggregate": aggregate,
        "n_token_samples": len(all_token_data),
    }


# =============================================================================
# Per-Layer Runner
# =============================================================================

def run_layer(model, tokenizer, layer_idx):
    """Load checkpoints, collect eval data, compute per-quartile metrics."""
    print(f"\n{'='*70}")
    print(f"  LAYER {layer_idx}")
    print(f"{'='*70}")

    # Collect eval data
    eval_data = collect_eval_data(model, tokenizer, layer_idx, N_EVAL_TOKENS)

    # Compute norm quartiles
    norms, quartile_idx, quartile_info = compute_norm_quartiles(eval_data)

    layer_results = {"quartile_info": quartile_info}

    for vname in VARIANTS:
        print(f"\n  --- {vname} ---")
        sae = load_sae(vname, layer_idx)

        # Per-quartile FVE
        print(f"  Computing per-quartile FVE...")
        fve = compute_quartile_fve(sae, eval_data, quartile_idx)
        print(f"    FVE: " + " | ".join(f"{k}={v:.4f}" for k, v in fve.items()))

        # Per-quartile reconstruction
        print(f"  Computing per-quartile reconstruction...")
        recon = compute_quartile_recon(sae, eval_data, quartile_idx)
        for q in range(N_QUARTILES):
            qkey = f"Q{q+1}"
            r = recon.get(qkey, {})
            if r:
                print(f"    {qkey}: L2={r['l2_loss']:.1f} | cos={r['cos_recon']:.4f}")

        # Norm-stratified ablation
        abl = run_ablation_stratified(
            vname, model, sae, eval_data, norms, quartile_idx, layer_idx
        )

        layer_results[vname] = {
            "quartile_fve": fve,
            "quartile_recon": recon,
            "ablation": abl,
        }

        # Log scale params
        if hasattr(sae, "scale_a"):
            if sae.scale_a.dim() == 0:
                layer_results[vname]["scale_a"] = sae.scale_a.item()
            else:
                a = sae.scale_a.detach()
                layer_results[vname]["scale_a_mean"] = a.mean().item()
                layer_results[vname]["scale_a_median"] = a.median().item()

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    del eval_data
    gc.collect()
    torch.cuda.empty_cache()

    return layer_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 19: Norm-Stratified Analysis")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Eval tokens: {N_EVAL_TOKENS:,}")
    print(f"Ablation: {N_ABLATION_FEATURES} features × {N_ABLATION_SAMPLES} samples")
    print(f"Quartiles: {N_QUARTILES}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Variants: {VARIANTS}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Check checkpoints exist
    ckpt_dir = Path(CHECKPOINT_DIR)
    for v in VARIANTS:
        for li in LAYERS:
            path = ckpt_dir / f"{v}_L{li}_final.pt"
            if not path.exists():
                print(f"  ERROR: Missing checkpoint {path}")
                return
    print("All checkpoints found.")

    # Load existing results for resume
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results: {list(all_results.get('layers', {}).keys())}")
    else:
        all_results = {
            "config": {
                "model_name": MODEL_NAME,
                "layers": LAYERS,
                "n_eval_tokens": N_EVAL_TOKENS,
                "n_ablation_features": N_ABLATION_FEATURES,
                "n_ablation_samples": N_ABLATION_SAMPLES,
                "n_quartiles": N_QUARTILES,
                "checkpoint_dir": CHECKPOINT_DIR,
                "variants": VARIANTS,
            },
            "layers": {},
        }

    # Run each layer
    for layer_idx in LAYERS:
        layer_key = str(layer_idx)
        if layer_key in all_results["layers"]:
            # Check completeness
            done = set(all_results["layers"][layer_key].keys()) - {"quartile_info"}
            if set(VARIANTS).issubset(done):
                print(f"\n  Layer {layer_idx} already complete, skipping")
                continue

        layer_results = run_layer(model, tokenizer, layer_idx)
        all_results["layers"][layer_key] = layer_results

        # Save after each layer
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to {RESULTS_PATH}")

    # Cross-layer summary
    print(f"\n{'='*70}")
    print("  CROSS-LAYER SUMMARY")
    print(f"{'='*70}")

    for li in LAYERS:
        lr = all_results["layers"].get(str(li), {})
        qi = lr.get("quartile_info", {})
        print(f"\n  Layer {li} — norm quartiles: "
              f"Q1=[{qi.get('Q1_norm_mean', 0):.1f}] "
              f"Q2=[{qi.get('Q2_norm_mean', 0):.1f}] "
              f"Q3=[{qi.get('Q3_norm_mean', 0):.1f}] "
              f"Q4=[{qi.get('Q4_norm_mean', 0):.1f}]")

        print(f"\n  FVE by quartile:")
        for v in VARIANTS:
            fve = lr.get(v, {}).get("quartile_fve", {})
            if fve:
                parts = [f"{q}={fve.get(q, 0):.4f}" for q in ["Q1", "Q2", "Q3", "Q4"]]
                print(f"    {v:>16s}: {' | '.join(parts)}")

        print(f"\n  cos>inner win rate by quartile:")
        for v in VARIANTS:
            abl = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if abl:
                parts = []
                for q in range(N_QUARTILES):
                    qkey = f"Q{q+1}"
                    qa = abl.get(qkey, {})
                    nf = qa.get("n_features", 0)
                    if nf > 0:
                        w = qa["cos_wins_inner"]
                        parts.append(f"Q{q+1}={w}/{nf} ({qa['cos_wins_inner_pct']:.0%})")
                    else:
                        parts.append(f"Q{q+1}=N/A")
                overall = abl.get("overall", {})
                ow = overall.get("cos_wins_inner", 0)
                on = abl.get("n_features", 0)
                print(f"    {v:>16s}: {' | '.join(parts)} | all={ow}/{on}")

    print(f"\nResults: {RESULTS_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
