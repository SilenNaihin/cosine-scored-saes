"""
Experiment 13: Cosine-Based Ablation
=====================================

Tests whether standard SAE's dominance on SAE→KL is an artifact of
inner-product-biased ablation, or a genuine quality difference.

Every ablation in exp2-12 removes the inner-product projection (x @ f̂) * f̂
from the residual stream. This perturbation scales with ||x||, giving standard
SAEs (whose activations also scale with ||x||) a tautological correlation
advantage over cosine SAEs (whose activations are norm-invariant).

This experiment introduces cosine-based ablation: remove cos(x, f̂) * f̂ * C
(norm-invariant perturbation), then recompute all correlations. If cosine/
adaptive SAEs win under cosine ablation, the evaluation bias is confirmed and
the ablation story across 12 experiments needs reinterpretation.

Uses existing checkpoints from exp10/12 — no retraining needed.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python experiments/exp13_cosine_ablation.py
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

# --- SAE ---
D_SAE = 16384
K = 80

# --- Data ---
N_EVAL_TOKENS = 500_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0
BATCH_SIZE = 4096

# --- Ablation ---
N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 50

# --- Paths ---
CHECKPOINT_DIR = "checkpoints/exp10"
RESULTS_PATH = "experiments/exp13_results.json"
ANALYSIS_PATH = "experiments/exp13_analysis.md"

# --- Variants ---
VARIANT_NAMES = ["standard", "cosine_l2", "cosine_cosloss", "adaptive_l2", "adaptive_cosloss"]
VARIANT_SHORT = {
    "standard": "Std", "cosine_l2": "CosL2", "cosine_cosloss": "CosCos",
    "adaptive_l2": "AdpL2", "adaptive_cosloss": "AdpCos",
}

SEED = 42


# =============================================================================
# SAE Architectures (copied from exp10 for checkpoint loading)
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
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))
            self.W_enc.copy_(self.W_dec * 0.1)

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        with torch.no_grad():
            if values.numel() > 0:
                self.threshold.copy_(values[-1].detach())
        return sparse.view_as(acts)

    def encode(self, x):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
            return self._batch_topk(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


class CosineBatchTopKSAE(nn.Module):
    """BatchTopK SAE with scaled-cosine-similarity encoder."""

    def __init__(self, d_model: int, d_sae: int, k: int = 80):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.log_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.zeros(()))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))
            self.W_enc.copy_(self.W_dec * 0.1)

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        with torch.no_grad():
            if values.numel() > 0:
                self.threshold.copy_(values[-1].detach())
        return sparse.view_as(acts)

    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        scale = self.log_scale.exp()
        pre_acts = scale * (x_unit @ w_unit.T) + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
            return self._batch_topk(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


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
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))
            self.W_enc.copy_(self.W_dec * 0.1)

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        with torch.no_grad():
            if values.numel() > 0:
                self.threshold.copy_(values[-1].detach())
        return sparse.view_as(acts)

    def encode(self, x):
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

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


# Variant → SAE class mapping
VARIANT_CLASS = {
    "standard": BatchTopKSAE,
    "cosine_l2": CosineBatchTopKSAE,
    "cosine_cosloss": CosineBatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
    "adaptive_cosloss": AdaptiveCosineBatchTopKSAE,
}


# =============================================================================
# Data Collection (reused from exp10)
# =============================================================================

class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
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


def collect_texts(n_total_tokens):
    n_docs_target = int(n_total_tokens / 150 * 1.5)
    print(f"  Downloading ~{n_docs_target:,} docs from FineWeb...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    texts = []
    for row in ds:
        if len(row["text"]) > 50:
            texts.append(row["text"][:2048])
        if len(texts) >= n_docs_target:
            break
    print(f"  Collected {len(texts):,} texts in {time.time()-t0:.1f}s")
    return texts


def texts_to_activations(model, tokenizer, texts, layer_idx, n_tokens):
    print(f"  Converting texts -> layer {layer_idx} activations (target {n_tokens:,})...")
    t0 = time.time()
    all_acts = []
    tokens_collected = 0
    text_idx = 0

    while tokens_collected < n_tokens and text_idx < len(texts):
        end = min(text_idx + COLLECTION_BATCH_SIZE, len(texts))
        batch_texts = texts[text_idx:end]
        text_idx = end

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

    all_acts = torch.cat(all_acts, dim=0)[:n_tokens]
    norms = all_acts.float().norm(dim=-1)
    print(f"  Layer {layer_idx}: {all_acts.shape[0]:,} tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={norms.mean():.1f}, std={norms.std():.1f})")
    return all_acts


# =============================================================================
# Ablation Functions
# =============================================================================

def ablate_feature_kl(model, activation, feature_dir, layer_idx, mode="inner_product",
                      median_norm=None):
    """Ablate a feature direction from the residual stream, measure KL at logits.

    Two ablation modes:
      - "inner_product": remove (x @ f̂) * f̂  [standard, scales with ||x||]
      - "cosine": remove cos(x, f̂) * f̂ * C   [norm-invariant, C = median ||x||]

    The inner-product projection magnitude is ||x|| * cos(x, f̂).
    The cosine projection magnitude is C * cos(x, f̂), where C is fixed.

    Under cosine ablation, the perturbation strength depends ONLY on the
    directional alignment between x and f̂, not on ||x||.
    """
    if mode == "inner_product":
        # Standard: projection magnitude = (x @ f̂), scales with ||x||
        projection = (activation @ feature_dir) * feature_dir
    elif mode == "cosine":
        # Cosine: projection magnitude = cos(x, f̂) * C, norm-invariant
        cos_sim = F.cosine_similarity(
            activation.unsqueeze(0), feature_dir.unsqueeze(0)
        ).squeeze()
        projection = cos_sim * feature_dir * median_norm
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    x = activation.unsqueeze(0).unsqueeze(0).to(DTYPE)
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0).to(DTYPE)

    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

    def make_hook(replacement):
        def hook(module, inputs, outputs):
            return (replacement,) + outputs[1:] if isinstance(outputs, tuple) else replacement
        return hook

    # Forward pass with original activation
    h = model.model.layers[layer_idx].register_forward_hook(make_hook(x))
    with torch.no_grad():
        try:
            orig_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            h.remove()
            return None
    h.remove()

    # Forward pass with ablated activation
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


def evaluate_ablation_dual(name, model, sae, eval_data, layer_idx, median_norm):
    """Run both inner-product and cosine ablation on the SAME features and samples.

    This is the core of Exp13: identical features, identical samples, only the
    ablation mode differs. Any change in correlation ranking is purely due to
    the ablation method, not feature selection.
    """
    tag = f"{name}/L{layer_idx}"
    print(f"\n    Dual ablation [{tag}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples, median_norm={median_norm:.1f})...")
    sae.eval()

    # Find top features by activation frequency
    n_probe = min(50_000, eval_data.shape[0])
    probe = eval_data[:n_probe]
    all_feats = []
    for i in range(0, n_probe, BATCH_SIZE):
        batch = probe[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        _, f = sae(batch)
        all_feats.append(f.detach().cpu())
    all_feats = torch.cat(all_feats, dim=0)

    freq = (all_feats > 0).float().mean(dim=0)
    top_idx = freq.topk(N_ABLATION_FEATURES).indices

    feature_results = []

    for rank, fi in enumerate(top_idx):
        fi = fi.item()
        feat_dir = sae.W_dec[fi].float()
        feat_dir = feat_dir / feat_dir.norm()

        feat_acts = all_feats[:, fi]
        active = torch.where(feat_acts > 0)[0]
        if len(active) < 20:
            continue

        n_sample = min(N_ABLATION_SAMPLES, len(active))
        torch.manual_seed(SEED + fi)  # deterministic sampling per feature
        chosen = active[torch.randperm(len(active))[:n_sample]]

        # Collect activations and run BOTH ablation modes on each sample
        cos_v, norm_v, inner_v, sae_v = [], [], [], []
        kl_inner_v, kl_cosine_v = [], []

        for idx in chosen:
            x = probe[idx].to(DEVICE, dtype=torch.float32)

            # Run both ablation modes
            kl_inner = ablate_feature_kl(
                model, x, feat_dir, layer_idx, mode="inner_product"
            )
            kl_cosine = ablate_feature_kl(
                model, x, feat_dir, layer_idx, mode="cosine",
                median_norm=median_norm
            )

            if kl_inner is None or kl_cosine is None:
                continue

            cos_v.append(F.cosine_similarity(x.unsqueeze(0), feat_dir.unsqueeze(0)).item())
            norm_v.append(x.norm().item())
            inner_v.append((x @ feat_dir).item())
            sae_v.append(feat_acts[idx].item())
            kl_inner_v.append(kl_inner)
            kl_cosine_v.append(kl_cosine)

        if len(kl_inner_v) < 10:
            continue

        kl_inner_arr = np.array(kl_inner_v)
        kl_cosine_arr = np.array(kl_cosine_v)
        if kl_inner_arr.std() < 1e-10 or kl_cosine_arr.std() < 1e-10:
            continue

        cos_arr = np.array(cos_v)
        norm_arr = np.array(norm_v)
        inner_arr = np.array(inner_v)
        sae_arr = np.array(sae_v)

        # Correlations under INNER-PRODUCT ablation (reproduces exp10/12)
        corr_cos_kl_ip = np.corrcoef(cos_arr, kl_inner_arr)[0, 1]
        corr_norm_kl_ip = np.corrcoef(norm_arr, kl_inner_arr)[0, 1]
        corr_inner_kl_ip = np.corrcoef(inner_arr, kl_inner_arr)[0, 1]
        corr_sae_kl_ip = np.corrcoef(sae_arr, kl_inner_arr)[0, 1]

        # Correlations under COSINE ablation (the new measurement)
        corr_cos_kl_cos = np.corrcoef(cos_arr, kl_cosine_arr)[0, 1]
        corr_norm_kl_cos = np.corrcoef(norm_arr, kl_cosine_arr)[0, 1]
        corr_inner_kl_cos = np.corrcoef(inner_arr, kl_cosine_arr)[0, 1]
        corr_sae_kl_cos = np.corrcoef(sae_arr, kl_cosine_arr)[0, 1]

        # Also: correlation between norm and KL under each mode
        # (should be high for inner-product, low for cosine — validates the method)

        feature_results.append({
            "feature_idx": fi,
            "n_ablated": len(kl_inner_v),
            "mean_kl_inner": float(kl_inner_arr.mean()),
            "mean_kl_cosine": float(kl_cosine_arr.mean()),
            # Inner-product ablation correlations
            "ip_corr_cos_kl": float(corr_cos_kl_ip),
            "ip_corr_norm_kl": float(corr_norm_kl_ip),
            "ip_corr_inner_kl": float(corr_inner_kl_ip),
            "ip_corr_sae_kl": float(corr_sae_kl_ip),
            "ip_cos_wins_inner": bool(abs(corr_cos_kl_ip) > abs(corr_inner_kl_ip)),
            "ip_cos_wins_sae": bool(abs(corr_cos_kl_ip) > abs(corr_sae_kl_ip)),
            "ip_sae_wins_inner": bool(abs(corr_sae_kl_ip) > abs(corr_inner_kl_ip)),
            # Cosine ablation correlations
            "cos_corr_cos_kl": float(corr_cos_kl_cos),
            "cos_corr_norm_kl": float(corr_norm_kl_cos),
            "cos_corr_inner_kl": float(corr_inner_kl_cos),
            "cos_corr_sae_kl": float(corr_sae_kl_cos),
            "cos_cos_wins_inner": bool(abs(corr_cos_kl_cos) > abs(corr_inner_kl_cos)),
            "cos_cos_wins_sae": bool(abs(corr_cos_kl_cos) > abs(corr_sae_kl_cos)),
            "cos_sae_wins_inner": bool(abs(corr_sae_kl_cos) > abs(corr_inner_kl_cos)),
        })

        if rank < 5 or rank % 10 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_inner_v):>2d} | "
                  f"IP: sae→KL={corr_sae_kl_ip:+.3f} cos→KL={corr_cos_kl_ip:+.3f} | "
                  f"COS: sae→KL={corr_sae_kl_cos:+.3f} cos→KL={corr_cos_kl_cos:+.3f}")

    if not feature_results:
        print(f"    [{tag}] No features with enough data for ablation")
        return {"n_features": 0}

    n = len(feature_results)

    def _agg(key):
        vals = [r[key] for r in feature_results]
        return float(np.mean(vals)) if vals else 0.0

    def _count(key):
        return sum(r[key] for r in feature_results)

    agg = {
        "n_features": n,
        # Inner-product ablation aggregate
        "ip_cos_kl_mean": _agg("ip_corr_cos_kl"),
        "ip_inner_kl_mean": _agg("ip_corr_inner_kl"),
        "ip_sae_kl_mean": _agg("ip_corr_sae_kl"),
        "ip_norm_kl_mean": _agg("ip_corr_norm_kl"),
        "ip_cos_wins_inner": _count("ip_cos_wins_inner"),
        "ip_cos_wins_sae": _count("ip_cos_wins_sae"),
        "ip_sae_wins_inner": _count("ip_sae_wins_inner"),
        # Cosine ablation aggregate
        "cos_cos_kl_mean": _agg("cos_corr_cos_kl"),
        "cos_inner_kl_mean": _agg("cos_corr_inner_kl"),
        "cos_sae_kl_mean": _agg("cos_corr_sae_kl"),
        "cos_norm_kl_mean": _agg("cos_corr_norm_kl"),
        "cos_cos_wins_inner": _count("cos_cos_wins_inner"),
        "cos_cos_wins_sae": _count("cos_cos_wins_sae"),
        "cos_sae_wins_inner": _count("cos_sae_wins_inner"),
        # Cross-mode comparisons
        "sae_kl_change": _agg("cos_corr_sae_kl") - _agg("ip_corr_sae_kl"),
        "cos_kl_change": _agg("cos_corr_cos_kl") - _agg("ip_corr_cos_kl"),
        "norm_kl_change": _agg("cos_corr_norm_kl") - _agg("ip_corr_norm_kl"),
    }

    print(f"    [{tag}] Summary ({n} features):")
    print(f"      IP ablation:  SAE→KL={agg['ip_sae_kl_mean']:.4f}  "
          f"cos→KL={agg['ip_cos_kl_mean']:.4f}  "
          f"norm→KL={agg['ip_norm_kl_mean']:.4f}  "
          f"SAE>inner={agg['ip_sae_wins_inner']}/{n}")
    print(f"      COS ablation: SAE→KL={agg['cos_sae_kl_mean']:.4f}  "
          f"cos→KL={agg['cos_cos_kl_mean']:.4f}  "
          f"norm→KL={agg['cos_norm_kl_mean']:.4f}  "
          f"SAE>inner={agg['cos_sae_wins_inner']}/{n}")
    print(f"      Delta:        SAE→KL={agg['sae_kl_change']:+.4f}  "
          f"cos→KL={agg['cos_kl_change']:+.4f}  "
          f"norm→KL={agg['norm_kl_change']:+.4f}")

    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_sae(variant, layer_idx, checkpoint_dir):
    """Load a trained SAE checkpoint."""
    cls = VARIANT_CLASS[variant]
    sae = cls(D_MODEL, D_SAE, K)
    path = Path(checkpoint_dir) / f"{variant}_L{layer_idx}.pt"
    if not path.exists():
        print(f"    WARNING: checkpoint not found: {path}")
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    sae.load_state_dict(state)
    sae = sae.to(DEVICE)
    sae.eval()
    print(f"    Loaded {variant}/L{layer_idx} from {path}")
    return sae


# =============================================================================
# Analysis Generation
# =============================================================================

def write_analysis(results):
    """Generate analysis markdown from results."""
    layers = results["layers"]
    L = []

    L.append("# Experiment 13: Cosine-Based Ablation — Does the Ruler Bias the Measurement?\n")

    L.append("## Why this experiment\n")
    L.append("Every ablation experiment in this project (exp2, 5, 8, 9, 10, 11, 12) uses the same "
             "procedure: remove the inner-product projection `(x @ f̂) * f̂` from the residual "
             "stream, then measure KL divergence at the logits. The perturbation magnitude is "
             "`||x|| * cos(x, f̂)` — it scales linearly with `||x||`.\n")
    L.append("When we correlate this effect with SAE activation, standard SAEs (whose activations "
             "also scale with `||x||`) get a near-tautological correlation advantage: "
             "`corr(||x|| * cos, ||x|| * cos * ||f||)`. Cosine SAEs remove the `||x||` factor "
             "from their activations, making them structurally disadvantaged under this metric.\n")
    L.append("Standard SAE has dominated SAE→KL at **every layer of every model** we've tested. "
             "We've argued this is evaluation bias, not a real quality difference. "
             "This experiment proves or disproves that claim.\n")

    L.append("## The fix: cosine-based ablation\n")
    L.append("Instead of removing `(x @ f̂) * f̂` (inner product, ∝ `||x||`), we remove "
             "`cos(x, f̂) * f̂ * C` where `C = median(||x||)` is a fixed constant per layer.\n")
    L.append("| Property | Inner-product ablation | Cosine ablation |")
    L.append("|---|---|---|")
    L.append("| Perturbation magnitude | `||x|| * cos(x, f̂)` | `C * cos(x, f̂)` |")
    L.append("| Scales with `||x||`? | Yes | No |")
    L.append("| Favors standard SAE? | Yes (shared `||x||` factor) | No (norm-invariant) |")
    L.append("| Perturbation direction | `f̂` | `f̂` |")
    L.append("")
    L.append("The constant `C = median(||x||)` ensures cosine ablation produces KL divergences "
             "of comparable magnitude to inner-product ablation (on the median token), "
             "avoiding numerical noise from tiny perturbations.\n")
    L.append("Crucially, we run **both ablation modes on the same features and same samples** "
             "for each SAE variant. The only variable is the ablation method.\n")

    L.append("## Setup\n")
    L.append("| Dimension | Value |")
    L.append("|---|---|")
    L.append(f"| Model | {MODEL_NAME} |")
    L.append(f"| Layers | {LAYERS} |")
    L.append(f"| SAE variants | {', '.join(VARIANT_NAMES)} |")
    L.append(f"| Checkpoints | exp10/12 (5M tokens each, d_sae={D_SAE}, k={K}) |")
    L.append(f"| Features per variant | {N_ABLATION_FEATURES} (top by activation frequency) |")
    L.append(f"| Samples per feature | {N_ABLATION_SAMPLES} |")
    L.append(f"| Eval tokens | {N_EVAL_TOKENS:,} |")
    L.append("| Ablation modes | inner_product (exp10/12 baseline), cosine (new) |")
    L.append("| Cosine ablation C | median(||x||) per layer |")
    L.append("")

    # --- Median norms ---
    L.append("### Per-layer median norms (C values)\n")
    L.append("| Layer | Median ||x|| (C) |")
    L.append("|---|---|")
    for li in LAYERS:
        lr = layers.get(str(li), {})
        mn = lr.get("median_norm", 0)
        L.append(f"| {li} | {mn:.1f} |")
    L.append("")

    # --- Main results: SAE→KL under both ablation modes ---
    L.append("## Results: SAE→KL under both ablation modes\n")
    L.append("This is the key table. Standard SAE has dominated SAE→KL in every prior experiment. "
             "Does it survive cosine ablation?\n")

    for li in LAYERS:
        lr = layers.get(str(li), {})
        L.append(f"### Layer {li}\n")
        L.append("| Variant | IP SAE→KL | COS SAE→KL | Delta | IP SAE>inn | COS SAE>inn |")
        L.append("|---|---|---|---|---|---|")
        for v in VARIANT_NAMES:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                n = a["n_features"]
                L.append(
                    f"| {v} | {a['ip_sae_kl_mean']:.4f} | {a['cos_sae_kl_mean']:.4f} | "
                    f"{a['sae_kl_change']:+.4f} | "
                    f"{a['ip_sae_wins_inner']}/{n} | {a['cos_sae_wins_inner']}/{n} |"
                )
            else:
                L.append(f"| {v} | — | — | — | — | — |")
        L.append("")

    # --- cos→KL under both modes ---
    L.append("## Results: cos→KL under both ablation modes\n")
    L.append("cos→KL is the metric we've argued is fairest for the RNH. "
             "How does it change under cosine ablation?\n")

    for li in LAYERS:
        lr = layers.get(str(li), {})
        L.append(f"### Layer {li}\n")
        L.append("| Variant | IP cos→KL | COS cos→KL | Delta | IP cos>inn | COS cos>inn |")
        L.append("|---|---|---|---|---|---|")
        for v in VARIANT_NAMES:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                n = a["n_features"]
                L.append(
                    f"| {v} | {a['ip_cos_kl_mean']:.4f} | {a['cos_cos_kl_mean']:.4f} | "
                    f"{a['cos_kl_change']:+.4f} | "
                    f"{a['ip_cos_wins_inner']}/{n} | {a['cos_cos_wins_inner']}/{n} |"
                )
            else:
                L.append(f"| {v} | — | — | — | — | — |")
        L.append("")

    # --- norm→KL validation ---
    L.append("## Validation: norm→KL under both modes\n")
    L.append("If cosine ablation works correctly, norm→KL should DROP (the perturbation no longer "
             "scales with ||x||, so input norm should not predict ablation effect).\n")

    L.append("| Layer |" + "".join(f" {VARIANT_SHORT[v]} IP | {VARIANT_SHORT[v]} COS |" for v in VARIANT_NAMES))
    L.append("|---|" + "---|---|" * len(VARIANT_NAMES))
    for li in LAYERS:
        lr = layers.get(str(li), {})
        cells = [f" {li} "]
        for v in VARIANT_NAMES:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                cells.append(f" {a['ip_norm_kl_mean']:.3f} | {a['cos_norm_kl_mean']:.3f} ")
            else:
                cells.append(" — | — ")
        L.append("|" + "|".join(cells) + "|")
    L.append("")

    # --- Cross-variant ranking ---
    L.append("## Cross-variant ranking: who wins under each ablation mode?\n")
    L.append("For each layer, rank variants by SAE→KL (higher = SAE activations better predict "
             "causal impact).\n")

    for li in LAYERS:
        lr = layers.get(str(li), {})
        # Collect (variant, ip_sae_kl, cos_sae_kl) tuples
        entries = []
        for v in VARIANT_NAMES:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                entries.append((v, a["ip_sae_kl_mean"], a["cos_sae_kl_mean"]))
        if entries:
            L.append(f"### Layer {li}\n")
            ip_ranked = sorted(entries, key=lambda e: e[1], reverse=True)
            cos_ranked = sorted(entries, key=lambda e: e[2], reverse=True)
            L.append("| Rank | IP ablation (variant: SAE→KL) | COS ablation (variant: SAE→KL) |")
            L.append("|---|---|---|")
            for i in range(len(entries)):
                ip_v, ip_val = ip_ranked[i][0], ip_ranked[i][1]
                cos_v, cos_val = cos_ranked[i][0], cos_ranked[i][2]
                L.append(f"| {i+1} | {ip_v}: {ip_val:.4f} | {cos_v}: {cos_val:.4f} |")
            L.append("")

    # --- Key insights (placeholder for manual fill) ---
    L.append("## Key Insights\n")
    L.append("*To be filled after reviewing results.*\n")
    L.append("1. **Does standard SAE's SAE→KL dominance survive cosine ablation?**\n")
    L.append("2. **Does norm→KL drop under cosine ablation?** (validates the method)\n")
    L.append("3. **Do cosine/adaptive SAEs gain under cosine ablation?** (confirms bias)\n")
    L.append("4. **Does the variant ranking change?** (the headline result)\n")
    L.append("5. **What does this mean for exp2-12's ablation results?**\n")

    L.append("## Caveats\n")
    L.append(f"- {N_EVAL_TOKENS:,} eval tokens per layer")
    L.append(f"- {N_ABLATION_FEATURES} features × {N_ABLATION_SAMPLES} samples — noisy per-feature estimates")
    L.append("- SAE checkpoints trained on only 5M tokens (may be undertrained)")
    L.append("- Each variant's top-30 features are different — cross-variant ranking "
             "confounds feature quality with feature selection")
    L.append("- Cosine ablation constant C = median(||x||) is one reasonable choice; "
             "other choices (mean, fixed) would give different KL scales but same correlations")
    L.append("- Single model (Qwen3-8B with RMSNorm)")
    L.append("")

    with open(ANALYSIS_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nAnalysis written to {ANALYSIS_PATH}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 13: Cosine-Based Ablation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Variants: {VARIANT_NAMES}")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print(f"Ablation: {N_ABLATION_FEATURES} features × {N_ABLATION_SAMPLES} samples, "
          f"dual mode (inner_product + cosine)")

    # ---- Load model ----
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

    # ---- Collect texts once ----
    print("\nCollecting FineWeb texts...")
    texts = collect_texts(N_EVAL_TOKENS)

    # ---- Load or create results ----
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results")
    else:
        all_results = {
            "config": {
                "model_name": MODEL_NAME,
                "layers": LAYERS,
                "d_model": D_MODEL,
                "d_sae": D_SAE,
                "k": K,
                "n_eval_tokens": N_EVAL_TOKENS,
                "n_ablation_features": N_ABLATION_FEATURES,
                "n_ablation_samples": N_ABLATION_SAMPLES,
                "seed": SEED,
                "checkpoint_dir": CHECKPOINT_DIR,
            },
            "layers": {},
        }

    # ---- Run each layer ----
    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Collect eval activations
        eval_data = texts_to_activations(model, tokenizer, texts, layer_idx, N_EVAL_TOKENS)

        # Compute median norm for cosine ablation constant
        norms = eval_data.float().norm(dim=-1)
        median_norm = float(norms.median().item())
        print(f"  Median ||x|| = {median_norm:.1f} (used as cosine ablation constant C)")

        layer_results = {"median_norm": median_norm}

        # Run dual ablation for each variant
        for variant in VARIANT_NAMES:
            sae = load_sae(variant, layer_idx, CHECKPOINT_DIR)
            if sae is None:
                print(f"    Skipping {variant}/L{layer_idx} (no checkpoint)")
                continue

            ablation = evaluate_ablation_dual(
                variant, model, sae, eval_data, layer_idx, median_norm
            )
            layer_results[variant] = {"ablation": ablation}

            # Free SAE memory between variants
            del sae
            gc.collect()
            torch.cuda.empty_cache()

        # Save layer results
        all_results["layers"][str(layer_idx)] = layer_results
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to {RESULTS_PATH}")

        # Free eval data
        del eval_data
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Generate analysis ----
    write_analysis(all_results)

    # ---- Print cross-layer summary ----
    print(f"\n{'='*70}")
    print("  CROSS-LAYER SUMMARY: SAE→KL under both ablation modes")
    print(f"{'='*70}")

    for li in LAYERS:
        lr = all_results["layers"].get(str(li), {})
        print(f"\n  Layer {li} (median_norm={lr.get('median_norm', 0):.1f}):")
        print(f"  {'Variant':<20s} | {'IP SAE→KL':>10s} | {'COS SAE→KL':>11s} | {'Delta':>8s} | "
              f"{'IP SAE>inn':>10s} | {'COS SAE>inn':>11s}")
        print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}-+-{'-'*10}-+-{'-'*11}")
        for v in VARIANT_NAMES:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                n = a["n_features"]
                print(f"  {v:<20s} | {a['ip_sae_kl_mean']:>10.4f} | {a['cos_sae_kl_mean']:>11.4f} | "
                      f"{a['sae_kl_change']:>+8.4f} | "
                      f"{a['ip_sae_wins_inner']:>4d}/{n:<4d}  | "
                      f"{a['cos_sae_wins_inner']:>5d}/{n:<4d}")
            else:
                print(f"  {v:<20s} | {'—':>10s} | {'—':>11s} | {'—':>8s} | {'—':>10s} | {'—':>11s}")

    # Validation: norm→KL should drop under cosine ablation
    print(f"\n{'='*70}")
    print("  VALIDATION: norm→KL (should drop under cosine ablation)")
    print(f"{'='*70}")
    for li in LAYERS:
        lr = all_results["layers"].get(str(li), {})
        print(f"\n  Layer {li}:")
        for v in VARIANT_NAMES:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                print(f"    {v:<20s}: IP norm→KL={a['ip_norm_kl_mean']:+.4f}  "
                      f"COS norm→KL={a['cos_norm_kl_mean']:+.4f}  "
                      f"delta={a['norm_kl_change']:+.4f}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Analysis: {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
