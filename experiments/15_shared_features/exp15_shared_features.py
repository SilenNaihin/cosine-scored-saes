"""
Experiment 15: Shared Feature Evaluation
=========================================

The problem: Every ablation comparison between SAE architectures (exp10, exp12)
is confounded by the fact that each SAE learns different features. When we report
"standard SAE cos→KL=0.368, cosine SAE cos→KL=0.308 at L27," those are 30
different features from each SAE. The gap might be entirely explained by which
features each SAE discovered, not how well the architecture encodes feature strength.

The fix: Find features that are shared across SAE architectures (by decoder cosine
similarity > 0.9), then compare how well each SAE's activation predicts the ablation
effect for those same features. This is the first head-to-head comparison on
identical feature directions.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python experiments/exp15_shared_features.py
"""

import json
import gc
import math
import os
import time
from pathlib import Path
from itertools import combinations

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
D_MODEL = 4096

# --- SAE architecture (must match exp10/12 training) ---
D_SAE = 16384
K = 80

# --- Shared feature matching ---
COS_SIM_THRESHOLD = 0.9       # decoder cosine sim to count as "same feature"
MIN_ACTIVATION_FREQ = 0.01    # feature must be active on >1% of tokens in BOTH SAEs
CHUNK_SIZE = 2048              # chunk decoder similarity to avoid OOM (2048 x 16384 x 4096)

# --- Data ---
N_EVAL_TOKENS = 500_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Ablation ---
N_ABLATION_SAMPLES = 100      # more samples than exp10 (50) since we have fewer features
BATCH_SIZE = 4096              # for encoding activations

# --- Layers ---
PRIMARY_LAYER = 18             # best layer for cosine SAEs
SECONDARY_LAYERS = [9, 27]    # expand if enough shared features at L18

# --- Paths ---
CHECKPOINT_DIR = Path("checkpoints/exp10")
RESULTS_PATH = "experiments/exp15_results.json"
ANALYSIS_PATH = "experiments/exp15_analysis.md"

SEED = 42

# --- SAE variants (checkpoint filename prefix → display name) ---
VARIANT_NAMES = {
    "standard":         "Standard (inner product)",
    "cosine_l2":        "Cosine + L2 loss",
    "cosine_cosloss":   "Cosine + cosine loss",
    "adaptive_l2":      "Adaptive + L2 loss",
    "adaptive_cosloss": "Adaptive + cosine loss",
}


# =============================================================================
# SAE Architectures (identical to exp10_cosine_sae.py)
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        scale = self.log_scale.exp()
        pre_acts = scale * (x_unit @ w_unit.T) + self.b_enc
        acts = F.relu(pre_acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f


# Map variant name → SAE class
VARIANT_CLASSES = {
    "standard":         BatchTopKSAE,
    "cosine_l2":        CosineBatchTopKSAE,
    "cosine_cosloss":   CosineBatchTopKSAE,
    "adaptive_l2":      AdaptiveCosineBatchTopKSAE,
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
# Load SAE Checkpoints
# =============================================================================

def load_sae(variant_name, layer_idx):
    """Load a trained SAE checkpoint. Returns SAE in eval mode on DEVICE."""
    cls = VARIANT_CLASSES[variant_name]
    sae = cls(D_MODEL, D_SAE, K)
    ckpt_path = CHECKPOINT_DIR / f"{variant_name}_L{layer_idx}.pt"
    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint not found: {ckpt_path}")
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae.load_state_dict(state)
    sae.eval()
    return sae  # stays on CPU; callers move to GPU as needed


def load_all_saes(layer_idx):
    """Load all 5 SAE variants for a layer. Returns dict {name: sae}."""
    saes = {}
    for vname in VARIANT_CLASSES:
        sae = load_sae(vname, layer_idx)
        if sae is not None:
            saes[vname] = sae
            print(f"  Loaded {vname} at L{layer_idx}")
    print(f"  {len(saes)} SAEs loaded for layer {layer_idx}")
    return saes


# =============================================================================
# Shared Feature Discovery
# =============================================================================

def compute_alive_features(sae, eval_data, min_freq=MIN_ACTIVATION_FREQ):
    """Find features that activate on >= min_freq fraction of tokens.

    Returns boolean mask [d_sae] and frequency array [d_sae].
    """
    n = eval_data.shape[0]
    counts = torch.zeros(D_SAE, device="cpu")

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        with torch.no_grad():
            _, feats = sae(batch)
        counts += (feats > 0).float().sum(dim=0).cpu()
        del batch, feats

    freq = counts / n
    alive = freq >= min_freq
    return alive, freq


def find_shared_features(sae_a, sae_b, alive_a, alive_b):
    """Find feature pairs between two SAEs with decoder cosine similarity > threshold.

    Only considers features that are alive in both SAEs.
    Computes in chunks to avoid OOM on the [d_sae, d_sae] similarity matrix.

    Returns list of dicts: {idx_a, idx_b, cos_sim}
    """
    # Get decoder weights for alive features only
    alive_idx_a = torch.where(alive_a)[0]
    alive_idx_b = torch.where(alive_b)[0]

    if len(alive_idx_a) == 0 or len(alive_idx_b) == 0:
        return []

    # Normalize decoder rows
    dec_a = sae_a.W_dec[alive_idx_a].float()
    dec_a = F.normalize(dec_a, dim=-1)  # [n_alive_a, d_model]
    dec_b = sae_b.W_dec[alive_idx_b].float()
    dec_b = F.normalize(dec_b, dim=-1)  # [n_alive_b, d_model]

    print(f"    Matching {len(alive_idx_a)} x {len(alive_idx_b)} alive features...")

    matches = []
    # Track best match per feature in B to enforce 1-to-1 matching
    best_for_b = {}  # idx_b -> (idx_a, cos_sim)

    # Process in chunks over A
    for chunk_start in range(0, len(alive_idx_a), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(alive_idx_a))
        chunk_a = dec_a[chunk_start:chunk_end].to(DEVICE)  # [chunk, d_model]

        # Compute similarity with all of B in sub-chunks
        for sub_start in range(0, len(alive_idx_b), CHUNK_SIZE):
            sub_end = min(sub_start + CHUNK_SIZE, len(alive_idx_b))
            chunk_b = dec_b[sub_start:sub_end].to(DEVICE)  # [sub_chunk, d_model]

            # [chunk, sub_chunk]
            sim = chunk_a @ chunk_b.T

            # Find pairs above threshold
            above = torch.where(sim > COS_SIM_THRESHOLD)
            for i, j in zip(above[0].tolist(), above[1].tolist()):
                real_a = alive_idx_a[chunk_start + i].item()
                real_b = alive_idx_b[sub_start + j].item()
                s = sim[i, j].item()

                # Greedy 1-to-1: keep highest sim for each B feature
                if real_b not in best_for_b or s > best_for_b[real_b][1]:
                    best_for_b[real_b] = (real_a, s)

            del chunk_b, sim
        del chunk_a

    # Also enforce 1-to-1 from A side: each A feature maps to at most one B
    best_for_a = {}
    for real_b, (real_a, s) in best_for_b.items():
        if real_a not in best_for_a or s > best_for_a[real_a][1]:
            best_for_a[real_a] = (real_b, s)

    for real_a, (real_b, s) in best_for_a.items():
        matches.append({"idx_a": real_a, "idx_b": real_b, "cos_sim": s})

    matches.sort(key=lambda x: -x["cos_sim"])
    return matches


# =============================================================================
# Ablation (reused from exp10, with canonical direction support)
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


# =============================================================================
# Head-to-Head Evaluation on Shared Features
# =============================================================================

def evaluate_shared_features(model, sae_a, sae_b, name_a, name_b,
                             matches, eval_data, layer_idx):
    """For each shared feature pair, compare which SAE's activation better predicts KL.

    Uses canonical direction (average of both decoders, normalized) for ablation.
    Collects activations from both SAEs on the same tokens, correlates with KL.

    Returns detailed results per feature and aggregate comparison.
    """
    tag = f"{name_a} vs {name_b} @ L{layer_idx}"
    print(f"\n  Evaluating {len(matches)} shared features: {tag}")

    if not matches:
        return {"n_shared": 0, "matches": [], "aggregate": {}}

    # Pre-encode eval data one SAE at a time to avoid OOM
    # (model ~16GB + SAE ~270MB + activations — can't fit two SAEs encoding simultaneously)
    ENCODE_BATCH = 2048  # smaller than BATCH_SIZE to limit peak VRAM
    print(f"    Encoding eval data with both SAEs (sequential)...")
    t0 = time.time()
    n = eval_data.shape[0]

    # Only keep re<author>ant feature columns (matched indices) to save RAM
    matched_idx_a = sorted(set(m["idx_a"] for m in matches))
    matched_idx_b = sorted(set(m["idx_b"] for m in matches))

    def encode_one_sae(sae, matched_indices):
        sae.to(DEVICE)
        all_feats = []
        for i in range(0, n, ENCODE_BATCH):
            batch = eval_data[i:i+ENCODE_BATCH].to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                _, f = sae(batch)
            # Only keep matched feature columns
            all_feats.append(f[:, matched_indices].cpu())
            del batch, f
        sae.cpu()
        torch.cuda.empty_cache()
        return torch.cat(all_feats, dim=0)

    feats_a_all = encode_one_sae(sae_a, matched_idx_a)
    feats_b_all = encode_one_sae(sae_b, matched_idx_b)
    print(f"    Encoded in {time.time()-t0:.1f}s")

    # Build index maps: original feature idx -> column in feats_*_all
    idx_a_to_col = {idx: col for col, idx in enumerate(matched_idx_a)}
    idx_b_to_col = {idx: col for col, idx in enumerate(matched_idx_b)}

    feature_results = []

    for mi, match in enumerate(matches):
        idx_a = match["idx_a"]
        idx_b = match["idx_b"]
        dec_sim = match["cos_sim"]

        # Canonical direction: average of both decoder vectors, normalized
        dir_a = sae_a.W_dec[idx_a].float().cpu()
        dir_b = sae_b.W_dec[idx_b].float().cpu()
        canonical_dir = F.normalize(dir_a + dir_b, dim=-1)

        # Get activations for this feature from both SAEs (using column index)
        acts_a = feats_a_all[:, idx_a_to_col[idx_a]]
        acts_b = feats_b_all[:, idx_b_to_col[idx_b]]

        # Find tokens where EITHER SAE activates (union of active tokens)
        active_either = torch.where((acts_a > 0) | (acts_b > 0))[0]
        if len(active_either) < 20:
            continue

        # Sample tokens for ablation
        n_sample = min(N_ABLATION_SAMPLES, len(active_either))
        torch.manual_seed(SEED + mi)
        chosen = active_either[torch.randperm(len(active_either))[:n_sample]]

        # Collect metrics
        cos_v, norm_v, inner_v = [], [], []
        act_a_v, act_b_v, kl_v = [], [], []

        for idx in chosen:
            x = eval_data[idx].to(DEVICE, dtype=torch.float32)
            kl = ablate_feature_kl(model, x, canonical_dir.to(DEVICE), layer_idx)
            if kl is None:
                continue

            cos_v.append(F.cosine_similarity(
                x.unsqueeze(0), canonical_dir.unsqueeze(0).to(DEVICE)
            ).item())
            norm_v.append(x.norm().item())
            inner_v.append((x @ canonical_dir.to(DEVICE)).item())
            act_a_v.append(acts_a[idx].item())
            act_b_v.append(acts_b[idx].item())
            kl_v.append(kl)

        if len(kl_v) < 10:
            continue

        kl_arr = np.array(kl_v)
        if kl_arr.std() < 1e-10:
            continue

        cos_arr = np.array(cos_v)
        norm_arr = np.array(norm_v)
        inner_arr = np.array(inner_v)
        act_a_arr = np.array(act_a_v)
        act_b_arr = np.array(act_b_v)

        # Correlations with KL
        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]

        # Handle case where one SAE has zero variance (all zeros or all same value)
        if act_a_arr.std() > 1e-10:
            corr_a = np.corrcoef(act_a_arr, kl_arr)[0, 1]
        else:
            corr_a = 0.0

        if act_b_arr.std() > 1e-10:
            corr_b = np.corrcoef(act_b_arr, kl_arr)[0, 1]
        else:
            corr_b = 0.0

        result = {
            "idx_a": idx_a, "idx_b": idx_b,
            "decoder_cos_sim": float(dec_sim),
            "n_ablated": len(kl_v),
            "n_active_a": int((act_a_arr > 0).sum()),
            "n_active_b": int((act_b_arr > 0).sum()),
            "corr_cos_kl": float(corr_cos),
            "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner),
            f"corr_{name_a}_kl": float(corr_a),
            f"corr_{name_b}_kl": float(corr_b),
            "a_wins": bool(abs(corr_a) > abs(corr_b)),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
        }
        feature_results.append(result)

        if mi < 10 or mi % 20 == 0:
            winner = name_a if result["a_wins"] else name_b
            print(f"      [{mi:>3d}] feat {idx_a}↔{idx_b} (sim={dec_sim:.3f}) | "
                  f"n={len(kl_v)} | {name_a}→KL={corr_a:.3f} | "
                  f"{name_b}→KL={corr_b:.3f} | cos→KL={corr_cos:.3f} | "
                  f"winner={winner}")

    if not feature_results:
        print(f"    No features with enough ablation data")
        return {"n_shared": 0, "matches": [], "aggregate": {}}

    # Aggregate
    n = len(feature_results)
    a_wins = sum(r["a_wins"] for r in feature_results)
    cos_wins = sum(r["cos_wins_inner"] for r in feature_results)
    mean_corr_a = float(np.mean([r[f"corr_{name_a}_kl"] for r in feature_results]))
    mean_corr_b = float(np.mean([r[f"corr_{name_b}_kl"] for r in feature_results]))
    mean_cos_kl = float(np.mean([r["corr_cos_kl"] for r in feature_results]))
    mean_inner_kl = float(np.mean([r["corr_inner_kl"] for r in feature_results]))

    aggregate = {
        "n_features": n,
        "a_wins": a_wins,
        "b_wins": n - a_wins,
        "a_win_rate": float(a_wins / n),
        f"mean_corr_{name_a}_kl": mean_corr_a,
        f"mean_corr_{name_b}_kl": mean_corr_b,
        "mean_corr_cos_kl": mean_cos_kl,
        "mean_corr_inner_kl": mean_inner_kl,
        "cos_wins_inner": cos_wins,
        "cos_win_rate": float(cos_wins / n),
    }

    print(f"\n    Summary ({n} shared features): "
          f"{name_a}→KL={mean_corr_a:.4f} vs {name_b}→KL={mean_corr_b:.4f} | "
          f"{name_a} wins {a_wins}/{n} ({100*a_wins/n:.0f}%) | "
          f"cos→KL={mean_cos_kl:.4f} | cos>inner: {cos_wins}/{n}")

    return {
        "n_shared": len(matches),
        "n_evaluated": n,
        "matches": [m for m in matches],  # all matches (including unevaluated)
        "features": feature_results,
        "aggregate": aggregate,
    }


# =============================================================================
# Per-Layer Runner
# =============================================================================

def run_layer(model, tokenizer, texts, layer_idx):
    """Run shared feature evaluation at one layer."""
    print(f"\n{'='*70}")
    print(f"  LAYER {layer_idx} — Shared Feature Evaluation")
    print(f"{'='*70}")

    # Collect eval activations
    eval_data = texts_to_activations(model, tokenizer, texts, layer_idx, N_EVAL_TOKENS)
    print(f"  Eval data: {eval_data.shape[0]:,} tokens")

    # Load all SAEs
    saes = load_all_saes(layer_idx)
    if len(saes) < 2:
        print(f"  Need at least 2 SAEs, found {len(saes)}. Skipping layer.")
        return {}

    # Compute alive features one SAE at a time (move to GPU, compute, move back)
    print(f"\n  Computing alive features (freq >= {MIN_ACTIVATION_FREQ})...")
    alive_masks = {}
    freq_arrays = {}
    for vname, sae in saes.items():
        sae.to(DEVICE)
        alive, freq = compute_alive_features(sae, eval_data)
        sae.cpu()
        torch.cuda.empty_cache()
        alive_masks[vname] = alive
        freq_arrays[vname] = freq
        n_alive = alive.sum().item()
        print(f"    {vname}: {n_alive} alive ({100*n_alive/D_SAE:.1f}%)")

    # Find shared features for all pairs
    print(f"\n  Finding shared features (cos_sim > {COS_SIM_THRESHOLD})...")
    pair_matches = {}
    variant_names = list(saes.keys())

    for va, vb in combinations(variant_names, 2):
        matches = find_shared_features(
            saes[va], saes[vb], alive_masks[va], alive_masks[vb]
        )
        pair_key = f"{va}_vs_{vb}"
        pair_matches[pair_key] = matches
        print(f"    {va} ↔ {vb}: {len(matches)} shared features")

    # Report match counts
    total_matches = sum(len(m) for m in pair_matches.values())
    print(f"\n  Total shared feature pairs across all SAE combinations: {total_matches}")

    if total_matches == 0:
        print("  No shared features found. This is informative — SAEs learn very different features.")
        return {
            "n_eval_tokens": eval_data.shape[0],
            "alive_counts": {v: int(alive_masks[v].sum()) for v in variant_names},
            "pair_matches": {k: len(v) for k, v in pair_matches.items()},
            "comparisons": {},
        }

    # Move all SAEs to CPU before ablation phase — evaluate_shared_features
    # will move each pair to GPU one at a time during encoding
    print(f"\n  Moving SAEs to CPU for ablation phase...")
    for vname, sae in saes.items():
        sae.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # Head-to-head evaluation on shared features
    print(f"\n  Running head-to-head ablation comparisons...")
    comparisons = {}

    for va, vb in combinations(variant_names, 2):
        pair_key = f"{va}_vs_{vb}"
        matches = pair_matches[pair_key]
        if not matches:
            comparisons[pair_key] = {"n_shared": 0}
            continue

        result = evaluate_shared_features(
            model, saes[va], saes[vb], va, vb,
            matches, eval_data, layer_idx
        )
        comparisons[pair_key] = result

        gc.collect()
        torch.cuda.empty_cache()

    # Cleanup
    del saes, eval_data
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "n_eval_tokens": N_EVAL_TOKENS,
        "alive_counts": {v: int(alive_masks[v].sum()) for v in variant_names},
        "pair_match_counts": {k: len(v) for k, v in pair_matches.items()},
        "comparisons": comparisons,
    }


# =============================================================================
# Analysis Generation
# =============================================================================

def write_analysis(results):
    """Generate analysis markdown from results."""
    L = []
    L.append("# Experiment 15: Shared Feature Evaluation — Results\n")

    L.append("## Why this experiment\n")
    L.append("Every ablation comparison between SAE architectures in exp10/12 is confounded by "
             "the fact that each SAE learns different features. When we report correlation gaps "
             "between architectures, those are measured on different feature sets. The gap might "
             "be entirely explained by which features each SAE discovered, not how well the "
             "architecture encodes feature strength.\n")
    L.append("This experiment finds features shared across SAE architectures (decoder cosine "
             f"similarity > {COS_SIM_THRESHOLD}) and compares activation predictions head-to-head "
             "on the same feature directions.\n")

    L.append("## Setup\n")
    L.append(f"| Dimension | Value |")
    L.append(f"|---|---|")
    L.append(f"| Model | {MODEL_NAME} |")
    L.append(f"| SAE checkpoints | exp10/12 (5M token, d_sae={D_SAE}, k={K}) |")
    L.append(f"| Matching threshold | decoder cos_sim > {COS_SIM_THRESHOLD} |")
    L.append(f"| Min activation freq | > {MIN_ACTIVATION_FREQ} in both SAEs |")
    L.append(f"| Ablation samples | {N_ABLATION_SAMPLES} per feature |")
    L.append(f"| Eval tokens | {N_EVAL_TOKENS:,} |")
    L.append(f"| Ablation direction | canonical (mean of both decoders, normalized) |")
    L.append("")

    layers = results.get("layers", {})

    for layer_str, layer_data in sorted(layers.items()):
        layer_idx = int(layer_str)
        L.append(f"\n## Layer {layer_idx}\n")

        # Alive feature counts
        alive = layer_data.get("alive_counts", {})
        if alive:
            L.append("### Alive features\n")
            L.append("| Variant | Alive | % |")
            L.append("|---|---|---|")
            for v, count in sorted(alive.items()):
                L.append(f"| {v} | {count} | {100*count/D_SAE:.1f}% |")
            L.append("")

        # Match counts
        match_counts = layer_data.get("pair_match_counts", {})
        if match_counts:
            L.append("### Shared feature counts\n")
            L.append("| SAE Pair | Shared Features |")
            L.append("|---|---|")
            for pair, count in sorted(match_counts.items()):
                L.append(f"| {pair.replace('_vs_', ' ↔ ')} | {count} |")
            L.append("")

        # Head-to-head results
        comparisons = layer_data.get("comparisons", {})
        if comparisons:
            L.append("### Head-to-head results\n")
            for pair_key, comp in sorted(comparisons.items()):
                agg = comp.get("aggregate", {})
                if not agg or agg.get("n_features", 0) == 0:
                    continue

                names = pair_key.split("_vs_")
                if len(names) != 2:
                    continue
                name_a, name_b = names[0], names[1]

                n = agg["n_features"]
                a_wins = agg.get("a_wins", 0)
                L.append(f"#### {name_a} vs {name_b}\n")
                L.append(f"- **Shared features evaluated:** {n}")
                L.append(f"- **{name_a} wins:** {a_wins}/{n} ({100*a_wins/n:.0f}%)")
                L.append(f"- **{name_b} wins:** {n-a_wins}/{n} ({100*(n-a_wins)/n:.0f}%)")

                key_a = f"mean_corr_{name_a}_kl"
                key_b = f"mean_corr_{name_b}_kl"
                if key_a in agg:
                    L.append(f"- **Mean {name_a}→KL:** {agg[key_a]:.4f}")
                if key_b in agg:
                    L.append(f"- **Mean {name_b}→KL:** {agg[key_b]:.4f}")
                L.append(f"- **Mean cos→KL:** {agg.get('mean_corr_cos_kl', 0):.4f}")
                L.append(f"- **Mean inner→KL:** {agg.get('mean_corr_inner_kl', 0):.4f}")
                cos_wins = agg.get("cos_wins_inner", 0)
                L.append(f"- **cos > inner:** {cos_wins}/{n} ({100*cos_wins/n:.0f}%)")
                L.append("")

    L.append("\n## Key Insights\n")
    L.append("*(To be written after results are available)*\n")

    L.append("\n## Caveats\n")
    L.append(f"- SAEs trained on only 5M tokens with d_sae={D_SAE} — 75-92% dead features")
    L.append(f"- Matching threshold ({COS_SIM_THRESHOLD}) is arbitrary — "
             "lower threshold = more matches but less confidence they're the 'same' feature")
    L.append("- Canonical direction (average of decoders) is a compromise — "
             "neither SAE's exact direction, which may understate both")
    L.append("- Greedy 1-to-1 matching may miss better global assignments (Hungarian algorithm)")
    L.append("- If few shared features exist, statistical power is low")
    L.append("- The ablation metric itself still uses inner-product projection (see Exp13)")
    L.append("")

    with open(ANALYSIS_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nAnalysis written to {ANALYSIS_PATH}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 15: Shared Feature Evaluation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Primary layer: {PRIMARY_LAYER}")
    print(f"Matching: decoder cos_sim > {COS_SIM_THRESHOLD}, alive freq > {MIN_ACTIVATION_FREQ}")
    print(f"Eval tokens: {N_EVAL_TOKENS:,}")
    print(f"Ablation samples per feature: {N_ABLATION_SAMPLES}")

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

    # Collect texts
    print("\nCollecting FineWeb texts...")
    texts = collect_texts(N_EVAL_TOKENS)

    # Verify checkpoints exist
    print(f"\nChecking checkpoints in {CHECKPOINT_DIR}...")
    available_layers = []
    for layer in [PRIMARY_LAYER] + SECONDARY_LAYERS:
        ckpts = list(CHECKPOINT_DIR.glob(f"*_L{layer}.pt"))
        if len(ckpts) >= 2:
            available_layers.append(layer)
            print(f"  L{layer}: {len(ckpts)} checkpoints found")
        else:
            print(f"  L{layer}: only {len(ckpts)} checkpoints — skipping")

    if not available_layers:
        print("\nERROR: No layers with >= 2 SAE checkpoints. Run exp10/12 first.")
        return

    # Run evaluation
    all_results = {
        "config": {
            "model_name": MODEL_NAME,
            "d_sae": D_SAE,
            "k": K,
            "cos_sim_threshold": COS_SIM_THRESHOLD,
            "min_activation_freq": MIN_ACTIVATION_FREQ,
            "n_eval_tokens": N_EVAL_TOKENS,
            "n_ablation_samples": N_ABLATION_SAMPLES,
            "seed": SEED,
        },
        "layers": {},
    }

    # Always run primary layer first
    run_order = [l for l in available_layers if l == PRIMARY_LAYER] + \
                [l for l in available_layers if l != PRIMARY_LAYER]

    for layer_idx in run_order:
        layer_result = run_layer(model, tokenizer, texts, layer_idx)
        all_results["layers"][str(layer_idx)] = layer_result

        # Save after each layer
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to {RESULTS_PATH}")

        # Check if primary layer had enough shared features to justify expanding
        if layer_idx == PRIMARY_LAYER:
            total = sum(
                layer_result.get("pair_match_counts", {}).values()
            )
            if total < 5:
                print(f"\n  Only {total} shared features at L{PRIMARY_LAYER}. "
                      f"Skipping secondary layers — not enough statistical power.")
                break
            else:
                print(f"\n  {total} shared features at L{PRIMARY_LAYER}. "
                      f"Proceeding to secondary layers.")

    # Generate analysis
    write_analysis(all_results)

    # Final summary
    print(f"\n{'='*70}")
    print("  EXPERIMENT 15 SUMMARY")
    print(f"{'='*70}")

    for layer_str, layer_data in sorted(all_results["layers"].items()):
        layer_idx = int(layer_str)
        print(f"\n  Layer {layer_idx}:")
        match_counts = layer_data.get("pair_match_counts", {})
        print(f"    Shared features per pair: {match_counts}")

        comparisons = layer_data.get("comparisons", {})
        for pair_key, comp in sorted(comparisons.items()):
            agg = comp.get("aggregate", {})
            if not agg or agg.get("n_features", 0) == 0:
                continue
            names = pair_key.split("_vs_")
            if len(names) == 2:
                n = agg["n_features"]
                a_wins = agg.get("a_wins", 0)
                print(f"    {pair_key}: {names[0]} wins {a_wins}/{n} ({100*a_wins/n:.0f}%)")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Analysis: {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
