"""
Experiment 37 Part A: Goldilocks Analysis — Why adaptive_l2 beats perfeature_l2
================================================================================

Motivation:
  Exp35 found a paradox: adaptive_l2 (1 global scale_a) outperforms perfeature_l2
  (9216 per-feature scale_a params) on downstream task evals (sparse probing, RAVEL)
  despite perfeature_l2 achieving better FVE. More parameters should help, not hurt.

  This experiment investigates WHY by analyzing the learned scale_a distributions
  and testing whether the problem is overfitting, parameter compression (norm-adaptive
  init artifact from exp34), or something else entirely.

Hypotheses:
  H1: Per-feature scale_a values cluster near zero (norm-adaptive init suppresses
      learning) — the extra parameters don't actually learn anything useful.
  H2: High-scale_a features are the ones that matter for downstream tasks, and
      there aren't enough of them to justify per-feature parameterization.
  H3: Per-feature scale_a introduces noise that hurts the encoder's ability to
      produce clean feature activations for probing.

Sub-experiments:
  A1: Per-feature scale_a distribution analysis (no GPU needed)
  A2: Correlate scale_a with feature utility (needs GPU + model)
  A3: Clamp-and-evaluate causal test (needs GPU + model + SAEBench)
  A4: Cross-reference with Qwen3-8B exp17 checkpoints

Analyzes existing exp35 checkpoints (Gemma-2-2b, trained at 50M tokens).

Usage:
    python experiments/exp37_goldilocks.py --part A1
    python experiments/exp37_goldilocks.py --part A2
    python experiments/exp37_goldilocks.py --part A3
    python experiments/exp37_goldilocks.py --part A4
    python experiments/exp37_goldilocks.py --part all
    python experiments/exp37_goldilocks.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

# --- Model (from exp35) ---
MODEL_NAME = "google/gemma-2-2b"
RAVEL_MODEL_NAME = "gemma-2-2b"
LAYERS = [6, 13, 20]
D_MODEL = 2304
D_SAE = 9216
K = 80

# --- Paths ---
EXP35_SAVE_DIR = Path("checkpoints/exp35")
EXP35_RESULTS_PATH = "experiments/exp35_results.json"
EXP17_SAVE_DIR = Path("checkpoints/exp17")

PLOT_DIR = Path("experiments/exp37_plots")
RESULTS_PATH = "experiments/exp37_results.json"

# --- A2 activation collection ---
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 32
OUTLIER_MULTIPLIER = 10.0
FREQ_TOKENS = 500_000
FREQ_BATCH_SIZE = 4096

# --- A3 clamp-and-eval ---
CLAMP_LAYER = 13
KEEP_TOP_K_VALUES = [10, 50, 100, 500]


# =============================================================================
# SAE Architectures (copied from exp35 for self-contained checkpoint loading)
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

    def _batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
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
    """BatchTopK SAE with per-token adaptive-scale cosine encoder.

    scale(x) = exp(scale_a * log(||x - b_dec||) + scale_b)
      - scale_a=0: global scale (norm-invariant)
      - scale_a=1: scale proportional to ||x|| (inner-product-like)

    Uses norm-adaptive init (exp27): scale_b = log(mean(||x_train||))
    instead of log(sqrt(d_model)) to avoid the Mistral-style init mismatch.
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80,
                 init_norm: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        scale_init = math.log(init_norm) if init_norm else math.log(math.sqrt(d_model))
        self.scale_b = nn.Parameter(torch.tensor(scale_init))
        self.register_buffer("threshold", torch.zeros(()))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))
            self.W_enc.copy_(self.W_dec * 0.1)

    def _batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
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

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


class PerFeatureAdaptiveCosineSAE(nn.Module):
    """BatchTopK SAE with per-feature adaptive-scale cosine encoder.

    scale_i(x) = exp(a_i * log(||x - b_dec||) + b_i)
    Each of d_sae features learns its own magnitude sensitivity a_i.

    Uses norm-adaptive init (exp27).
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80,
                 init_norm: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.zeros(d_sae))
        scale_init = math.log(init_norm) if init_norm else math.log(math.sqrt(d_model))
        self.scale_b = nn.Parameter(torch.full((d_sae,), scale_init))
        self.register_buffer("threshold", torch.zeros(()))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))
            self.W_enc.copy_(self.W_dec * 0.1)

    def _batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
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


# =============================================================================
# Checkpoint loading helpers
# =============================================================================

VARIANT_CLASSES = {
    "standard": BatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
    "perfeature_l2": PerFeatureAdaptiveCosineSAE,
}


def load_sae(variant: str, layer: int, save_dir: Path = EXP35_SAVE_DIR,
             device: str = "cpu") -> nn.Module:
    """Load a trained SAE checkpoint."""
    cls = VARIANT_CLASSES[variant]
    sae = cls(D_MODEL, D_SAE, K)
    path = save_dir / f"{variant}_L{layer}_final.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    sae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    sae = sae.to(device=device).eval()
    return sae


def load_exp35_results() -> dict:
    """Load exp35 results JSON."""
    with open(EXP35_RESULTS_PATH) as f:
        return json.load(f)


# =============================================================================
# Streaming activation collection (same pattern as exp35)
# =============================================================================

class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
    """Capture residual stream activations at a Gemma-2 layer via forward hook."""
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


class ActivationStream:
    """Streams activations from FineWeb, yielding shuffled batches."""

    def __init__(self, model, tokenizer, layer_idx, buffer_tokens=FREQ_TOKENS):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.buffer_tokens = buffer_tokens
        self.buffer = None
        self._text_iter = None
        self._init_dataset()

    def _init_dataset(self):
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT",
            split="train", streaming=True,
        )
        self._text_iter = iter(ds)

    def fill_buffer(self):
        all_acts = []
        tokens_collected = 0

        while tokens_collected < self.buffer_tokens:
            batch_texts = []
            for _ in range(COLLECTION_BATCH_SIZE):
                try:
                    row = next(self._text_iter)
                    if len(row["text"]) > 50:
                        batch_texts.append(row["text"][:2048])
                except StopIteration:
                    self._init_dataset()
                    row = next(self._text_iter)
                    if len(row["text"]) > 50:
                        batch_texts.append(row["text"][:2048])

            if not batch_texts:
                continue

            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=CTX_LEN,
            ).to(DEVICE)

            acts = _collect_layer_acts(self.model, self.layer_idx, inputs)
            flat = acts[inputs["attention_mask"].bool()]

            norms = flat.float().norm(dim=-1)
            median = norms.median()
            if median > 0:
                flat = flat[norms < median * OUTLIER_MULTIPLIER]

            all_acts.append(flat.to("cpu", dtype=DTYPE))
            tokens_collected += flat.shape[0]

        self.buffer = torch.cat(all_acts, dim=0)[:self.buffer_tokens]
        perm = torch.randperm(self.buffer.shape[0])
        self.buffer = self.buffer[perm]
        return self.buffer.shape[0]

    def get_batch(self, batch_idx):
        start = (batch_idx * FREQ_BATCH_SIZE) % self.buffer.shape[0]
        end = start + FREQ_BATCH_SIZE
        if end > self.buffer.shape[0]:
            idx = torch.cat([
                torch.arange(start, self.buffer.shape[0]),
                torch.arange(0, end - self.buffer.shape[0]),
            ])
        else:
            idx = torch.arange(start, end)
        return self.buffer[idx].to(DEVICE, dtype=torch.float32)


# =============================================================================
# A1: Per-feature scale_a distribution analysis
# =============================================================================

def run_a1(results: dict) -> dict:
    """Analyze per-feature scale_a distributions across layers."""
    print("\n" + "=" * 70)
    print("A1: Per-feature scale_a distribution analysis")
    print("=" * 70)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    a1_results = {}

    # Load global scale_a from adaptive_l2 as reference
    global_scale_a = {}
    for layer in LAYERS:
        try:
            sae = load_sae("adaptive_l2", layer)
            global_scale_a[layer] = sae.scale_a.item()
            print(f"  adaptive_l2 L{layer}: global scale_a = {global_scale_a[layer]:.6f}")
            del sae
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            global_scale_a[layer] = None

    # Analyze perfeature_l2 distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, layer in enumerate(LAYERS):
        try:
            sae = load_sae("perfeature_l2", layer)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        scale_a = sae.scale_a.detach().cpu().numpy()
        del sae

        mean_a = float(np.mean(scale_a))
        median_a = float(np.median(scale_a))
        std_a = float(np.std(scale_a))
        min_a = float(np.min(scale_a))
        max_a = float(np.max(scale_a))

        abs_a = np.abs(scale_a)
        near_zero_frac = float(np.mean(abs_a < 0.01))
        moderate_frac = float(np.mean((abs_a >= 0.01) & (abs_a < 0.1)))
        high_frac = float(np.mean(abs_a >= 0.1))

        layer_stats = {
            "mean": mean_a, "median": median_a, "std": std_a,
            "min": min_a, "max": max_a,
            "pct_near_zero_abs_lt_0.01": near_zero_frac,
            "pct_moderate_0.01_to_0.1": moderate_frac,
            "pct_high_gt_0.1": high_frac,
            "global_scale_a_ref": global_scale_a.get(layer),
        }
        a1_results[f"L{layer}"] = layer_stats

        print(f"\n  perfeature_l2 L{layer}:")
        print(f"    mean={mean_a:.6f}, median={median_a:.6f}, std={std_a:.6f}")
        print(f"    range=[{min_a:.6f}, {max_a:.6f}]")
        print(f"    near-zero (|a|<0.01): {near_zero_frac:.1%}")
        print(f"    moderate  (0.01-0.1): {moderate_frac:.1%}")
        print(f"    high      (|a|>0.1):  {high_frac:.1%}")
        if global_scale_a.get(layer) is not None:
            print(f"    global adaptive_l2 ref: {global_scale_a[layer]:.6f}")

        # Histogram
        ax = axes[idx]
        ax.hist(scale_a, bins=100, alpha=0.7, color="steelblue", edgecolor="none")
        if global_scale_a.get(layer) is not None:
            ax.axvline(global_scale_a[layer], color="red", linestyle="--",
                       linewidth=2, label=f"adaptive_l2 global: {global_scale_a[layer]:.4f}")
        ax.axvline(median_a, color="orange", linestyle="-",
                   linewidth=1.5, label=f"median: {median_a:.4f}")
        ax.set_title(f"Layer {layer} scale_a distribution\n"
                     f"mean={mean_a:.4f}, std={std_a:.4f}")
        ax.set_xlabel("scale_a value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    # Check for norm-adaptive init artifact
    all_means = [a1_results[f"L{l}"]["mean"] for l in LAYERS if f"L{l}" in a1_results]
    if all_means and all(abs(m) < 0.01 for m in all_means):
        warning = ("NORM-ADAPTIVE INIT ARTIFACT DETECTED: mean scale_a << 0.01 across "
                    "all layers. Exp34 found that scale_b = log(mean_norm) suppresses "
                    "scale_a learning. The per-feature params may not have learned "
                    "anything meaningful.")
        a1_results["init_artifact_warning"] = warning
        print(f"\n  WARNING: {warning}")
    else:
        a1_results["init_artifact_warning"] = None
        print("\n  No norm-adaptive init artifact detected (scale_a values are not "
              "uniformly suppressed).")

    plt.tight_layout()
    hist_path = PLOT_DIR / "a1_scale_a_distributions.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\n  Saved histogram: {hist_path}")

    # Cross-layer comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in LAYERS:
        try:
            sae = load_sae("perfeature_l2", layer)
            scale_a = sae.scale_a.detach().cpu().numpy()
            del sae
            ax.hist(scale_a, bins=100, alpha=0.5, label=f"Layer {layer}")
        except FileNotFoundError:
            continue
    ax.set_title("Per-feature scale_a: cross-layer comparison")
    ax.set_xlabel("scale_a value")
    ax.set_ylabel("count")
    ax.legend()
    cross_path = PLOT_DIR / "a1_scale_a_cross_layer.png"
    plt.savefig(cross_path, dpi=150)
    plt.close()
    print(f"  Saved cross-layer comparison: {cross_path}")

    return a1_results


# =============================================================================
# A2: Correlate scale_a with feature utility
# =============================================================================

def run_a2(results: dict, device: str = DEVICE) -> dict:
    """Correlate per-feature scale_a with ablation KL, decoder norm, firing rate."""
    from scipy.stats import spearmanr

    print("\n" + "=" * 70)
    print("A2: Correlate scale_a with feature utility")
    print("=" * 70)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load exp35 results for ablation data
    exp35 = load_exp35_results()

    # Load model for activation frequency
    print("\n  Loading model for activation frequency measurement...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    a2_results = {}

    for layer in LAYERS:
        print(f"\n  --- Layer {layer} ---")

        try:
            sae = load_sae("perfeature_l2", layer, device=device)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        scale_a = sae.scale_a.detach().cpu().numpy()
        decoder_norms = sae.W_dec.detach().float().norm(dim=1).cpu().numpy()

        # Compute per-feature firing rate
        print(f"  Collecting activations for firing rate ({FREQ_TOKENS:,} tokens)...")
        stream = ActivationStream(model, tokenizer, layer, buffer_tokens=FREQ_TOKENS)
        stream.fill_buffer()

        fire_counts = torch.zeros(D_SAE)
        total_tokens = 0
        n_batches = stream.buffer.shape[0] // FREQ_BATCH_SIZE

        for b in range(n_batches):
            batch = stream.get_batch(b)
            with torch.no_grad():
                _, feats = sae(batch)
            fire_counts += (feats > 0).float().sum(dim=0).cpu()
            total_tokens += batch.shape[0]

        firing_rate = (fire_counts / max(total_tokens, 1)).numpy()
        del stream
        gc.collect()
        torch.cuda.empty_cache()

        # Get ablation KL data from exp35 results
        key = f"perfeature_l2_L{layer}"
        abl_features = exp35["results"][key]["ablation"].get("features", [])

        # Build per-feature ablation map: feature_idx -> metrics
        abl_map = {}
        for feat in abl_features:
            abl_map[feat["feature_idx"]] = feat

        # Correlations over all features (scale_a vs decoder_norm, firing_rate)
        alive_mask = firing_rate > 0
        n_alive = int(alive_mask.sum())
        print(f"  {n_alive} alive features (of {D_SAE})")

        layer_results = {
            "n_alive": n_alive,
            "correlations": {},
        }

        # scale_a vs decoder_norm (all features)
        rho_dec, p_dec = spearmanr(scale_a, decoder_norms)
        layer_results["correlations"]["scale_a_vs_decoder_norm"] = {
            "spearman_rho": float(rho_dec), "p_value": float(p_dec),
        }
        print(f"  scale_a vs decoder_norm: rho={rho_dec:.4f} (p={p_dec:.2e})")

        # scale_a vs firing_rate (alive features only)
        if n_alive > 10:
            rho_fire, p_fire = spearmanr(scale_a[alive_mask], firing_rate[alive_mask])
            layer_results["correlations"]["scale_a_vs_firing_rate"] = {
                "spearman_rho": float(rho_fire), "p_value": float(p_fire),
            }
            print(f"  scale_a vs firing_rate: rho={rho_fire:.4f} (p={p_fire:.2e})")

        # scale_a vs ablation KL (only for features with ablation data)
        if abl_map:
            abl_indices = sorted(abl_map.keys())
            abl_scale_a = np.array([scale_a[i] for i in abl_indices])
            abl_cos_kl = np.array([abl_map[i]["corr_cos_kl"] for i in abl_indices])
            abl_inner_kl = np.array([abl_map[i]["corr_inner_kl"] for i in abl_indices])
            abl_sae_kl = np.array([abl_map[i]["corr_sae_kl"] for i in abl_indices])

            rho_cos, p_cos = spearmanr(abl_scale_a, abl_cos_kl)
            rho_inner, p_inner = spearmanr(abl_scale_a, abl_inner_kl)
            rho_sae, p_sae = spearmanr(abl_scale_a, abl_sae_kl)

            layer_results["correlations"]["scale_a_vs_corr_cos_kl"] = {
                "spearman_rho": float(rho_cos), "p_value": float(p_cos),
            }
            layer_results["correlations"]["scale_a_vs_corr_inner_kl"] = {
                "spearman_rho": float(rho_inner), "p_value": float(p_inner),
            }
            layer_results["correlations"]["scale_a_vs_corr_sae_kl"] = {
                "spearman_rho": float(rho_sae), "p_value": float(p_sae),
            }
            print(f"  scale_a vs corr_cos_kl:   rho={rho_cos:.4f} (p={p_cos:.2e})")
            print(f"  scale_a vs corr_inner_kl: rho={rho_inner:.4f} (p={p_inner:.2e})")
            print(f"  scale_a vs corr_sae_kl:   rho={rho_sae:.4f} (p={p_sae:.2e})")

        # --- Plots ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # scale_a vs decoder_norm
        ax = axes[0, 0]
        ax.scatter(scale_a, decoder_norms, alpha=0.1, s=2, color="steelblue")
        ax.set_xlabel("scale_a")
        ax.set_ylabel("||W_dec[i]||")
        ax.set_title(f"L{layer}: scale_a vs decoder norm (rho={rho_dec:.3f})")

        # scale_a vs firing_rate (alive only)
        ax = axes[0, 1]
        if n_alive > 10:
            ax.scatter(scale_a[alive_mask], firing_rate[alive_mask],
                       alpha=0.1, s=2, color="forestgreen")
            ax.set_xlabel("scale_a")
            ax.set_ylabel("firing rate")
            ax.set_title(f"L{layer}: scale_a vs firing rate (rho={rho_fire:.3f})")
        else:
            ax.text(0.5, 0.5, "Too few alive features", ha="center", va="center",
                    transform=ax.transAxes)

        # scale_a vs ablation corr_cos_kl
        ax = axes[1, 0]
        if abl_map:
            ax.scatter(abl_scale_a, abl_cos_kl, alpha=0.5, s=10, color="coral")
            ax.set_xlabel("scale_a")
            ax.set_ylabel("corr(cos_sim, ablation KL)")
            ax.set_title(f"L{layer}: scale_a vs cos-KL corr (rho={rho_cos:.3f})")
        else:
            ax.text(0.5, 0.5, "No ablation data", ha="center", va="center",
                    transform=ax.transAxes)

        # scale_a vs ablation corr_sae_kl
        ax = axes[1, 1]
        if abl_map:
            ax.scatter(abl_scale_a, abl_sae_kl, alpha=0.5, s=10, color="purple")
            ax.set_xlabel("scale_a")
            ax.set_ylabel("corr(sae_act, ablation KL)")
            ax.set_title(f"L{layer}: scale_a vs SAE-KL corr (rho={rho_sae:.3f})")
        else:
            ax.text(0.5, 0.5, "No ablation data", ha="center", va="center",
                    transform=ax.transAxes)

        plt.suptitle(f"Experiment 37 A2: Layer {layer} scale_a correlations", fontsize=14)
        plt.tight_layout()
        plot_path = PLOT_DIR / f"a2_correlations_L{layer}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {plot_path}")

        a2_results[f"L{layer}"] = layer_results
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return a2_results


# =============================================================================
# A3: Clamp-and-evaluate (causal test)
# =============================================================================

def make_bench_sae(sae: nn.Module, layer: int, device: str) -> object:
    """Wrap an SAE as a BenchSAE for SAEBench evaluation."""
    from benchmarks.adapter import BenchSAE

    _sae = sae
    def _make_fns(s):
        return lambda x: s.encode(x), lambda f: s.decode(f)
    enc_fn, dec_fn = _make_fns(_sae)

    W_enc = sae.W_enc.detach().T
    W_dec = F.normalize(sae.W_dec.detach(), dim=1)
    b_enc = sae.b_enc.detach()
    b_dec = sae.b_dec.detach()

    bench_sae = BenchSAE(
        W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec,
        encode_fn=enc_fn, decode_fn=dec_fn,
        model_name=MODEL_NAME,
        hook_layer=layer,
        device=device, dtype=torch.bfloat16,
    )
    return bench_sae


def run_saebench_on_sae(name: str, sae: nn.Module, layer: int,
                        device: str, output_dir: str,
                        force_rerun: bool = True) -> dict:
    """Run sparse_probing and ravel on a single SAE."""
    import sae_bench.evals.sparse_probing.main as sp_eval
    import sae_bench.evals.ravel.main as ravel_eval

    bench_sae = make_bench_sae(sae, layer, device)
    eval_results = {}

    # Sparse probing
    print(f"    Running sparse_probing for {name}...")
    sp_output = os.path.join(output_dir, "sparse_probing")
    os.makedirs(sp_output, exist_ok=True)
    sp_config = sp_eval.SparseProbingEvalConfig(
        model_name=MODEL_NAME,
        llm_batch_size=16,
        llm_dtype="bfloat16",
    )
    sp_eval.run_eval(sp_config, [(name, bench_sae)], device, sp_output,
                     force_rerun=force_rerun, clean_up_activations=True,
                     save_activations=False)
    eval_results["sparse_probing"] = _load_eval_result(sp_output, name)

    # RAVEL
    print(f"    Running ravel for {name}...")
    ravel_output = os.path.join(output_dir, "ravel")
    os.makedirs(ravel_output, exist_ok=True)
    ravel_config = ravel_eval.RAVE<author>alConfig(
        model_name=RAVEL_MODEL_NAME,
        llm_batch_size=16,
        llm_dtype="bfloat16",
    )
    ravel_eval.run_eval(ravel_config, [(name, bench_sae)], device, ravel_output,
                        force_rerun=force_rerun)
    eval_results["ravel"] = _load_eval_result(ravel_output, name)

    return eval_results


def _load_eval_result(output_dir: str, sae_name: str) -> dict:
    """Load the most recent eval result for a given SAE name."""
    for p in Path(output_dir).glob("*.json"):
        if sae_name in p.stem:
            with open(p) as f:
                return json.load(f)
    jsons = sorted(Path(output_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    if jsons:
        with open(jsons[-1]) as f:
            return json.load(f)
    return {"error": "result file not found"}


def run_a3(results: dict, device: str = DEVICE) -> dict:
    """Clamp-and-evaluate: test whether clamping scale_a changes downstream performance."""
    print("\n" + "=" * 70)
    print(f"A3: Clamp-and-evaluate (Layer {CLAMP_LAYER})")
    print("=" * 70)

    layer = CLAMP_LAYER
    output_dir = str(PLOT_DIR / "a3_saebench")

    # Load the original perfeature_l2 checkpoint
    sae_orig = load_sae("perfeature_l2", layer, device=device)
    sae_orig = sae_orig.to(dtype=torch.bfloat16)
    scale_a_orig = sae_orig.scale_a.detach().clone()
    median_a = scale_a_orig.median().item()

    print(f"  Original scale_a: mean={scale_a_orig.mean().item():.6f}, "
          f"median={median_a:.6f}, std={scale_a_orig.std().item():.6f}")

    a3_results = {
        "layer": layer,
        "original_scale_a_mean": float(scale_a_orig.mean().item()),
        "original_scale_a_median": float(median_a),
        "original_scale_a_std": float(scale_a_orig.std().item()),
        "conditions": {},
    }

    # Define clamping conditions
    conditions = {}

    # Original (baseline)
    conditions["original"] = scale_a_orig.clone()

    # Clamp to global (median)
    conditions["clamp_to_median"] = torch.full_like(scale_a_orig, median_a)

    # Clamp to zero (pure cosine)
    conditions["clamp_to_zero"] = torch.zeros_like(scale_a_orig)

    # Keep top-K by |scale_a|
    for top_k in KEEP_TOP_K_VALUES:
        mask = torch.zeros_like(scale_a_orig)
        _, top_indices = scale_a_orig.abs().topk(min(top_k, D_SAE))
        mask[top_indices] = scale_a_orig[top_indices]
        conditions[f"keep_top_{top_k}"] = mask

    # Run each condition
    for cond_name, clamped_a in conditions.items():
        print(f"\n  --- Condition: {cond_name} ---")

        nonzero_count = (clamped_a != 0).sum().item()
        print(f"    Non-zero scale_a: {nonzero_count}/{D_SAE}")

        # Create clamped SAE (deep copy + overwrite scale_a)
        sae_clamped = copy.deepcopy(sae_orig)
        with torch.no_grad():
            sae_clamped.scale_a.copy_(clamped_a)
        sae_clamped.eval()

        sae_name = f"exp37-perfeature-{cond_name}-L{layer}"

        try:
            eval_results = run_saebench_on_sae(
                sae_name, sae_clamped, layer, device, output_dir,
            )
            a3_results["conditions"][cond_name] = {
                "nonzero_scale_a": int(nonzero_count),
                "eval_results": eval_results,
            }
            print(f"    Results: {json.dumps(eval_results, indent=2, default=str)[:500]}")
        except Exception as e:
            print(f"    ERROR running eval for {cond_name}: {e}")
            a3_results["conditions"][cond_name] = {"error": str(e)}

        del sae_clamped
        gc.collect()
        torch.cuda.empty_cache()

    del sae_orig
    gc.collect()
    torch.cuda.empty_cache()

    return a3_results


# =============================================================================
# A4: Cross-reference with Qwen3-8B (exp17)
# =============================================================================

def run_a4(results: dict) -> dict:
    """Compare scale_a distributions with Qwen3-8B exp17 checkpoints."""
    print("\n" + "=" * 70)
    print("A4: Cross-reference with Qwen3-8B (exp17)")
    print("=" * 70)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    a4_results = {}

    if not EXP17_SAVE_DIR.exists():
        msg = (f"Exp17 checkpoint directory not found: {EXP17_SAVE_DIR}. "
               f"Skipping cross-reference.")
        print(f"  {msg}")
        return {"skipped": True, "reason": msg}

    # Try to find perfeature_l2 checkpoints
    exp17_ckpts = sorted(EXP17_SAVE_DIR.glob("perfeature_l2_L*_final.pt"))
    if not exp17_ckpts:
        msg = "No perfeature_l2 checkpoints found in exp17 directory."
        print(f"  {msg}")
        return {"skipped": True, "reason": msg}

    print(f"  Found {len(exp17_ckpts)} exp17 perfeature_l2 checkpoints")

    # Qwen3-8B has different dimensions
    # exp17 used d_model=4096, d_sae=16384 (4x expansion)
    QWEN_D_MODEL = 4096
    QWEN_D_SAE = 16384

    fig, axes = plt.subplots(1, max(len(exp17_ckpts), 1), figsize=(6 * len(exp17_ckpts), 5))
    if len(exp17_ckpts) == 1:
        axes = [axes]

    for idx, ckpt_path in enumerate(exp17_ckpts):
        layer_str = ckpt_path.stem.split("_L")[1].split("_")[0]
        print(f"\n  Loading exp17 perfeature_l2 L{layer_str}...")

        try:
            sae = PerFeatureAdaptiveCosineSAE(QWEN_D_MODEL, QWEN_D_SAE, K)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            sae.load_state_dict(state)
            scale_a = sae.scale_a.detach().cpu().numpy()
            del sae

            mean_a = float(np.mean(scale_a))
            std_a = float(np.std(scale_a))
            near_zero = float(np.mean(np.abs(scale_a) < 0.01))

            a4_results[f"qwen_L{layer_str}"] = {
                "mean": mean_a, "std": std_a, "near_zero_frac": near_zero,
                "init_type": "sqrt(d)",
            }
            print(f"    mean={mean_a:.6f}, std={std_a:.6f}, near-zero={near_zero:.1%}")
            print(f"    NOTE: Qwen3-8B used sqrt(d) init -> scale_a should be less "
                  f"compressed than Gemma-2-2b (norm-adaptive init)")

            ax = axes[idx]
            ax.hist(scale_a, bins=100, alpha=0.7, color="darkorange", edgecolor="none")
            ax.set_title(f"Qwen3-8B L{layer_str}\nmean={mean_a:.4f}, std={std_a:.4f}")
            ax.set_xlabel("scale_a value")
            ax.set_ylabel("count")

        except Exception as e:
            print(f"    ERROR loading checkpoint: {e}")
            a4_results[f"qwen_L{layer_str}"] = {"error": str(e)}

    plt.suptitle("Exp17 (Qwen3-8B) per-feature scale_a distributions", fontsize=13)
    plt.tight_layout()
    plot_path = PLOT_DIR / "a4_qwen_scale_a.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {plot_path}")

    return a4_results


# =============================================================================
# Result I/O
# =============================================================================

def load_existing_results() -> dict:
    """Load existing results file, or return empty dict."""
    if Path(RESULTS_PATH).exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    """Save results to JSON."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


# =============================================================================
# Dry run
# =============================================================================

def dry_run() -> None:
    """Load checkpoints, print stats, no eval."""
    print("\n" + "=" * 70)
    print("DRY RUN: Load checkpoints and print stats")
    print("=" * 70)

    for variant in ["adaptive_l2", "perfeature_l2"]:
        for layer in LAYERS:
            try:
                sae = load_sae(variant, layer)
                print(f"\n  {variant} L{layer}: loaded OK")

                if hasattr(sae, "scale_a"):
                    if sae.scale_a.dim() == 0:
                        print(f"    scale_a = {sae.scale_a.item():.6f} (global)")
                        print(f"    scale_b = {sae.scale_b.item():.6f} "
                              f"(exp = {sae.scale_b.exp().item():.2f})")
                    else:
                        a = sae.scale_a.detach()
                        print(f"    scale_a: mean={a.mean():.6f}, median={a.median():.6f}, "
                              f"std={a.std():.6f}")
                        print(f"    scale_a: min={a.min():.6f}, max={a.max():.6f}")
                        print(f"    scale_a |a|<0.01: {(a.abs() < 0.01).float().mean():.1%}")
                        b = sae.scale_b.detach()
                        print(f"    scale_b: mean={b.mean():.6f}, std={b.std():.6f}")

                dec_norms = sae.W_dec.detach().float().norm(dim=1)
                print(f"    decoder norms: mean={dec_norms.mean():.4f}, "
                      f"std={dec_norms.std():.4f}")

                del sae
            except FileNotFoundError as e:
                print(f"\n  {variant} L{layer}: NOT FOUND ({e})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp37 Part A: Goldilocks Analysis — why adaptive_l2 > perfeature_l2"
    )
    parser.add_argument("--part", default="all",
                        choices=["A1", "A2", "A3", "A4", "all"],
                        help="Which sub-experiment to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load checkpoints, print stats, no eval")
    parser.add_argument("--device", default="cuda",
                        help="Device for GPU-requiring parts (A2, A3)")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 37 Part A: Goldilocks Analysis")
    print("=" * 70)
    print(f"Model:    {MODEL_NAME}")
    print(f"Layers:   {LAYERS}")
    print(f"Exp35 checkpoints: {EXP35_SAVE_DIR}")
    print(f"Output:   {RESULTS_PATH}, {PLOT_DIR}/")
    print()

    if args.dry_run:
        dry_run()
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_existing_results()
    parts_to_run = ["A1", "A2", "A3", "A4"] if args.part == "all" else [args.part]

    t0 = time.time()

    if "A1" in parts_to_run:
        results["A1"] = run_a1(results)
        save_results(results)

    if "A2" in parts_to_run:
        results["A2"] = run_a2(results, device=args.device)
        save_results(results)

    if "A3" in parts_to_run:
        results["A3"] = run_a3(results, device=args.device)
        save_results(results)

    if "A4" in parts_to_run:
        results["A4"] = run_a4(results)
        save_results(results)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"Results: {RESULTS_PATH}")
    print(f"Plots:   {PLOT_DIR}/")


if __name__ == "__main__":
    main()
