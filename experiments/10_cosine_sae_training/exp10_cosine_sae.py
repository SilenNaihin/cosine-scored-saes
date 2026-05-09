"""
Experiment 10: Cosine-Normalized SAE Training (Multi-Layer)
===========================================================
V4: Adaptive per-token scaling.

The global scale in cosine_l2 breaks at L27 (norm~407) because a single scalar
can't track the 7x norm variation across tokens. Per-token adaptive scaling
learns scale = exp(a * log(||x||) + b), interpolating between:
  - a=0: global scale (current cosine SAE, norm-invariant)
  - a=1: scale ∝ ||x|| (inner product with normalized weights)

The learned value of `a` reveals how much magnitude the model actually uses.

This run trains two new variants at all 3 layers, appending to existing results:
  4. Adaptive+L2  — adaptive-cosine encoder, L2 loss
  5. Adaptive+cos — adaptive-cosine encoder, cosine loss (control: a should stay ~0)

Usage:
    ssh <server>     cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python experiments/exp10_cosine_sae.py
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
LAYERS = [9, 18, 27]        # All three layers for adaptive variants
D_MODEL = 4096

# --- SAE architecture ---
D_SAE = 16384       # 4x expansion (production: 65536)
K = 80              # BatchTopK sparsity, matches existing SAEs

# --- Data ---
N_TRAIN_TOKENS = 5_000_000   # Per layer.  Fast iteration (production: 50_000_000)
N_EVAL_TOKENS = 500_000      # Per layer
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR = 3e-4
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 50

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 50

# --- Output ---
SAVE_DIR = "checkpoints/exp10"
RESULTS_PATH = "experiments/exp10_results.json"
ANALYSIS_PATH = "experiments/exp10_analysis.md"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)


def get_config_dict():
    """Capture all config constants for reproducibility."""
    return {
        "model_name": MODEL_NAME, "layers": LAYERS, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
        "n_steps": N_STEPS,
    }


# =============================================================================
# SAE Architectures
# =============================================================================

class BatchTopKSAE(nn.Module):
    """Standard BatchTopK SAE with inner-product encoder.

    encode: x -> BatchTopK(ReLU(W_enc @ (x - b_dec) + b_enc))
    decode: f -> f @ W_dec + b_dec

    W_enc: (d_sae, d_model) — each row is a feature detector
    W_dec: (d_sae, d_model) — each row is a feature direction
    """

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


class CosineBatchTopKSAE(nn.Module):
    """BatchTopK SAE with scaled-cosine-similarity encoder.

    Identical to BatchTopKSAE except the encoder normalizes both the centered
    input and the encoder weight rows before the matmul, then multiplies by a
    learnable scale factor.  This makes feature *detection* magnitude-invariant
    (scaling x doesn't change which features fire) while letting the activation
    *values* be large enough for the decoder to reconstruct real activations.

    V1 (naive, no scale) failed: cosine outputs are bounded in [-1,1], but
    the decoder needs to reconstruct activations with ||x|| ~ 100-400.
    Result: FVE ≈ 0, 97% dead features.

    encode: x -> BatchTopK(ReLU(scale * cos_sim(x - b_dec, W_enc_rows) + b_enc))
    decode: f -> f @ W_dec + b_dec       [unchanged]
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # Learnable scale: initialized to sqrt(d_model) ≈ 64 for d=4096.
        # This roughly matches the magnitude of standard encoder outputs
        # (which scale with ||x|| ≈ 60-400 depending on layer).
        self.log_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
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
        # Normalize both sides for cosine similarity, then scale
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        scale = self.log_scale.exp()
        pre_acts = scale * (x_unit @ w_unit.T) + self.b_enc
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

    Instead of a single global scale, uses:
        scale(x) = exp(scale_a * log(||x - b_dec||) + scale_b)

    This is a log-linear function of input norm, interpolating between:
      - scale_a=0: global scale (identical to CosineBatchTopKSAE)
      - scale_a=1: scale ∝ ||x||, so activation = (x @ w_unit) / ||w|| (inner-product-like)
    The optimizer learns how much norm-dependence helps reconstruction.

    Initialized with scale_a=0 (starts as global-scale cosine SAE).
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # Adaptive scale: scale = exp(scale_a * log(||x||) + scale_b)
        self.scale_a = nn.Parameter(torch.tensor(0.0))   # norm exponent (0=global, 1=linear)
        self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))  # base scale
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
        # Cosine similarity
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T  # [batch, d_sae]
        # Per-token adaptive scale
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, 1]
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)  # [batch, 1]
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
# Data Collection
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


def collect_texts(n_total_tokens):
    """Download FineWeb texts once (lightweight strings, reused for all layers).

    Over-fetches slightly because we estimate ~200 usable tokens per doc.
    """
    # Rough estimate: request 1.5x docs to account for short docs + filtering
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
    """Convert texts to layer activations. Returns CPU bf16 tensor."""
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

        # Filter attention sinks
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
# Training
# =============================================================================

def lr_schedule(step):
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_sae(name, sae, train_data, layer_idx, loss_type="l2"):
    """Train an SAE on pre-collected activations. Returns list of metric dicts.

    loss_type: "l2" for standard MSE, "cosine" for 1 - cos_sim(x, x_hat).
    """
    n_tokens = train_data.shape[0]
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, loss={loss_type}, "
          f"{n_tokens:,} tokens, {N_STEPS} steps")

    # Deterministic shuffle — same order for all SAEs at the same layer
    torch.manual_seed(SEED + 100 + layer_idx)
    perm = torch.randperm(n_tokens)

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    sae.train()
    log = []
    t0 = time.time()

    for step in range(N_STEPS):
        start = (step * BATCH_SIZE) % n_tokens
        end = start + BATCH_SIZE
        if end > n_tokens:
            idx = torch.cat([perm[start:], perm[:end - n_tokens]])
        else:
            idx = perm[start:end]

        batch = train_data[idx].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)

        # Loss: TopK enforces sparsity, loss drives reconstruction
        if loss_type == "cosine":
            recon_loss = (1 - F.cosine_similarity(batch, x_hat, dim=-1)).mean()
        else:
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

        optimizer.zero_grad(set_to_none=True)
        recon_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                l0 = (features != 0).float().sum(dim=-1).mean().item()
                total_var = torch.var(batch, dim=0, unbiased=False).sum()
                resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                dead = (features.sum(dim=0) == 0).float().mean().item()

            # Log scale parameters
            scale_val = None
            scale_a_val = None
            if hasattr(sae, "log_scale"):
                scale_val = sae.log_scale.exp().item()
            if hasattr(sae, "scale_a"):
                scale_a_val = sae.scale_a.item()
                scale_val = sae.scale_b.exp().item()  # base scale for logging

            entry = {
                "step": step, "recon_loss": recon_loss.item(),
                "l0": l0, "fve": fve, "cos_recon": cos_r,
                "dead_frac": dead, "lr": scheduler.get_last_lr()[0],
            }
            if scale_val is not None:
                entry["scale"] = scale_val
            if scale_a_val is not None:
                entry["scale_a"] = scale_a_val
            log.append(entry)
            scale_str = ""
            if scale_a_val is not None:
                scale_str = f" | a={scale_a_val:.3f} b={scale_val:.1f}"
            elif scale_val is not None:
                scale_str = f" | scale={scale_val:.1f}"
            print(f"    [{tag:>12s}] step {step:>5d}/{N_STEPS} | "
                  f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                  f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f}{scale_str} | "
                  f"{time.time()-t0:.0f}s")

    sae.eval()
    print(f"    [{tag}] Done in {time.time()-t0:.1f}s")
    return log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data, layer_idx):
    """Reconstruction metrics on held-out data."""
    tag = f"{name}/L{layer_idx}"
    sae.eval()
    n = eval_data.shape[0]
    recon_losses, cos_sims, l0s = [], [], []
    total_var_sum, resid_var_sum = 0.0, 0.0

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        recon_losses.append((batch - x_hat).pow(2).sum(dim=-1).mean().item())
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()

    results = {
        "recon_loss_l2": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
    }
    print(f"    [{tag}] L2={results['recon_loss_l2']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"FVE={results['fve']:.4f}")
    return results


@torch.no_grad()
def test_norm_invariance(name, sae, eval_data, layer_idx, scales=(0.5, 2.0, 5.0)):
    """Test whether SAE activations change when input is scaled."""
    tag = f"{name}/L{layer_idx}"
    sae.eval()
    sample = eval_data[:BATCH_SIZE].to(DEVICE, dtype=torch.float32)
    base_feats = sae.encode(sample)

    results = {}
    for scale in scales:
        scaled_feats = sae.encode(sample * scale)

        both_on = (base_feats > 0) & (scaled_feats > 0)
        if both_on.any():
            ratios = scaled_feats[both_on] / base_feats[both_on]
            mean_ratio = ratios.mean().item()
            std_ratio = ratios.std().item()
        else:
            mean_ratio = std_ratio = float("nan")

        agreement = ((base_feats > 0) == (scaled_feats > 0)).float().mean().item()
        cos = F.cosine_similarity(
            base_feats.float(), scaled_feats.float(), dim=-1
        ).mean().item()

        results[f"scale_{scale}"] = {
            "mean_ratio": mean_ratio, "std_ratio": std_ratio,
            "feature_agreement": agreement, "activation_cosine": cos,
        }
        print(f"    [{tag}] scale={scale}: ratio={mean_ratio:.3f} "
              f"(std~{scale:.1f}/cos~1.0) | agree={agreement:.3f} | cos={cos:.4f}")
    return results


def ablate_feature_kl(model, activation, feature_dir, layer_idx):
    """Ablate a feature direction from the residual stream, measure KL at logits."""
    projection = (activation @ feature_dir) * feature_dir
    # Cast to model dtype (bf16) to avoid dtype mismatch in subsequent layers
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


def evaluate_ablation(name, model, sae, eval_data, layer_idx):
    """For top features, correlate SAE activation with ablation KL divergence."""
    tag = f"{name}/L{layer_idx}"
    print(f"\n    Ablation [{tag}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    n_probe = min(50_000, eval_data.shape[0])
    probe = eval_data[:n_probe]  # keep on CPU, move batches to GPU as needed
    all_feats = []
    for i in range(0, n_probe, BATCH_SIZE):
        batch = probe[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        _, f = sae(batch)
        all_feats.append(f.detach().cpu())  # keep features on CPU to avoid OOM
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
        chosen = active[torch.randperm(len(active))[:n_sample]]

        cos_v, norm_v, inner_v, sae_v, kl_v = [], [], [], [], []
        for idx in chosen:
            x = probe[idx].to(DEVICE, dtype=torch.float32)
            kl = ablate_feature_kl(model, x, feat_dir, layer_idx)
            if kl is None:
                continue
            cos_v.append(F.cosine_similarity(x.unsqueeze(0), feat_dir.unsqueeze(0)).item())
            norm_v.append(x.norm().item())
            inner_v.append((x @ feat_dir).item())
            sae_v.append(feat_acts[idx].item())
            kl_v.append(kl)

        if len(kl_v) < 10:
            continue
        kl_arr = np.array(kl_v)
        if kl_arr.std() < 1e-10:
            continue

        cos_arr, norm_arr = np.array(cos_v), np.array(norm_v)
        inner_arr, sae_arr = np.array(inner_v), np.array(sae_v)

        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]

        feature_results.append({
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "cos_wins_sae": bool(abs(corr_cos) > abs(corr_sae)),
            "sae_wins_inner": bool(abs(corr_sae) > abs(corr_inner)),
        })

        if rank < 5 or rank % 10 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f} | norm→KL={corr_norm:.3f}")

    if not feature_results:
        print(f"    [{tag}] No features with enough data for ablation")
        return {"n_features": 0}

    n = len(feature_results)
    agg = {
        "n_features": n,
        "cos_kl_mean": float(np.mean([r["corr_cos_kl"] for r in feature_results])),
        "inner_kl_mean": float(np.mean([r["corr_inner_kl"] for r in feature_results])),
        "sae_kl_mean": float(np.mean([r["corr_sae_kl"] for r in feature_results])),
        "norm_kl_mean": float(np.mean([r["corr_norm_kl"] for r in feature_results])),
        "cos_wins_inner": sum(r["cos_wins_inner"] for r in feature_results),
        "cos_wins_sae": sum(r["cos_wins_sae"] for r in feature_results),
        "sae_wins_inner": sum(r["sae_wins_inner"] for r in feature_results),
    }

    print(f"    [{tag}] Summary ({n} features): "
          f"cos→KL={agg['cos_kl_mean']:.4f} | SAE→KL={agg['sae_kl_mean']:.4f} | "
          f"SAE>inner: {agg['sae_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Per-Layer Runner
# =============================================================================

def run_layer(model, tokenizer, texts, layer_idx, save_dir):
    """Train + evaluate all 3 SAEs at one layer. Returns results dict."""
    print(f"\n{'='*70}")
    print(f"  LAYER {layer_idx}")
    print(f"{'='*70}")

    # ---- Activations ----
    n_total = N_TRAIN_TOKENS + N_EVAL_TOKENS
    all_acts = texts_to_activations(model, tokenizer, texts, layer_idx, n_total)
    train_data = all_acts[:N_TRAIN_TOKENS]
    eval_data = all_acts[N_TRAIN_TOKENS:N_TRAIN_TOKENS + N_EVAL_TOKENS]
    print(f"  Split: train={train_data.shape[0]:,}, eval={eval_data.shape[0]:,}")
    del all_acts

    # ---- 3 SAE variants ----
    # All share the same init seed, data order, and hyperparams.
    # Only the encoder architecture and loss function differ.
    variants = [
        ("adaptive_l2",      AdaptiveCosineBatchTopKSAE, "l2"),
        ("adaptive_cosloss", AdaptiveCosineBatchTopKSAE, "cosine"),
    ]

    saes = {}
    logs = {}
    for vname, cls, loss_type in variants:
        torch.manual_seed(SEED)
        sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
        logs[vname] = train_sae(vname, sae, train_data, layer_idx, loss_type=loss_type)
        saes[vname] = sae
        # Free optimizer memory between trainings
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Evaluate (free training data first) ----
    del train_data
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Evaluation — Layer {layer_idx}")

    results = {}
    for vname, sae in saes.items():
        recon = evaluate_reconstruction(vname, sae, eval_data, layer_idx)
        inv = test_norm_invariance(vname, sae, eval_data, layer_idx)
        abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)
        torch.save(sae.state_dict(), save_dir / f"{vname}_L{layer_idx}.pt")
        results[vname] = {
            "training": logs[vname],
            "reconstruction": recon,
            "norm_invariance": inv,
            "ablation": abl,
        }

    # ---- Cleanup ----
    del saes, eval_data
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# Analysis
# =============================================================================

VARIANT_NAMES = ["standard", "cosine_l2", "cosine_cosloss", "adaptive_l2", "adaptive_cosloss"]
VARIANT_SHORT = {
    "standard": "Std", "cosine_l2": "CosL2", "cosine_cosloss": "CosCos",
    "adaptive_l2": "AdpL2", "adaptive_cosloss": "AdpCos",
}


def write_analysis(results):
    """Generate analysis markdown from multi-layer results."""
    cfg = results["config"]
    layers = results["layers"]
    L = []
    vnames = VARIANT_NAMES

    L.append("# Experiment 10: Cosine-Normalized SAE Training — Auto-Generated Results\n")
    L.append("## Setup\n")
    L.append("| Dimension | standard | cosine_l2 | cosine_cosloss |")
    L.append("|---|---|---|---|")
    L.append(f"| Model | {cfg['model_name']} | same | same |")
    L.append(f"| Layers | {cfg['layers']} | same | same |")
    L.append(f"| d_sae / k | {cfg['d_sae']} / {cfg['k']} | same | same |")
    L.append(f"| Tokens/layer | {cfg['n_train_tokens']:,} | same | same |")
    L.append(f"| Encoder | inner product | scaled cosine | scaled cosine |")
    L.append(f"| Loss | L2 | L2 | **cosine** |")
    L.append(f"| LR / seed | {cfg['lr']} / {cfg['seed']} | same | same |\n")

    # Per-layer reconstruction
    L.append("\n## Reconstruction Quality\n")
    hdr = "| Layer |" + "".join(f" {VARIANT_SHORT[v]} L2 | {VARIANT_SHORT[v]} FVE | {VARIANT_SHORT[v]} cos |" for v in vnames)
    L.append(hdr)
    L.append("|---|" + "---|---|---|" * len(vnames))
    for li in cfg["layers"]:
        lr = layers.get(str(li), {})
        cells = [f" {li} "]
        for v in vnames:
            r = lr.get(v, {}).get("reconstruction", {})
            if r:
                cells.append(f" {r['recon_loss_l2']:.1f} | {r['fve']:.4f} | {r['cos_recon']:.4f} ")
            else:
                cells.append(" — | — | — ")
        L.append("|" + "|".join(cells) + "|")
    L.append("")

    # Per-layer norm invariance at 2x
    L.append("\n## Norm Invariance (2x scale)\n")
    hdr = "| Layer |" + "".join(f" {VARIANT_SHORT[v]} ratio | {VARIANT_SHORT[v]} agree |" for v in vnames)
    L.append(hdr)
    L.append("|---|" + "---|---|" * len(vnames))
    for li in cfg["layers"]:
        lr = layers.get(str(li), {})
        cells = [f" {li} "]
        for v in vnames:
            inv = lr.get(v, {}).get("norm_invariance", {}).get("scale_2.0", {})
            if inv:
                cells.append(f" {inv['mean_ratio']:.3f} | {inv['feature_agreement']:.3f} ")
            else:
                cells.append(" — | — ")
        L.append("|" + "|".join(cells) + "|")
    L.append("")

    # Per-layer ablation
    L.append("\n## Ablation Quality\n")
    hdr = "| Layer |" + "".join(f" {VARIANT_SHORT[v]} SAE→KL | {VARIANT_SHORT[v]} cos→KL | {VARIANT_SHORT[v]} SAE>inn |" for v in vnames)
    L.append(hdr)
    L.append("|---|" + "---|---|---|" * len(vnames))
    for li in cfg["layers"]:
        lr = layers.get(str(li), {})
        cells = [f" {li} "]
        for v in vnames:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                cells.append(
                    f" {a['sae_kl_mean']:.4f} | {a['cos_kl_mean']:.4f} | "
                    f"{a['sae_wins_inner']}/{a['n_features']} "
                )
            else:
                cells.append(" — | — | — ")
        L.append("|" + "|".join(cells) + "|")
    L.append("")

    L.append("\n## Caveats\n")
    L.append(f"- {cfg['n_train_tokens']:,} training tokens per layer (production SAEs use 500M+)")
    L.append(f"- d_sae={cfg['d_sae']} (production: 65k)")
    L.append("- Each SAE learns different features — ablation compares aggregate quality")
    L.append("- Single model (Qwen3-8B with RMSNorm); LayerNorm models may differ per exp7")

    with open(ANALYSIS_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nAnalysis written to {ANALYSIS_PATH}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 10: Cosine-Normalized SAE Training (Multi-Layer)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Per layer: {N_TRAIN_TOKENS:,} train tokens ({N_STEPS} steps), "
          f"{N_EVAL_TOKENS:,} eval tokens")
    print(f"Batch: {BATCH_SIZE}, Warmup: {WARMUP_STEPS} steps")

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

    # ---- Collect texts once (reused for all layers) ----
    print("\nCollecting FineWeb texts...")
    n_needed = N_TRAIN_TOKENS + N_EVAL_TOKENS
    texts = collect_texts(n_needed)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing results (preserve completed layers) ----
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        existing = list(all_results.get("layers", {}).keys())
        print(f"  Loaded existing results for layers: {existing}")
    else:
        all_results = {"config": get_config_dict(), "layers": {}}

    # ---- Run each layer ----

    for layer_idx in LAYERS:
        layer_result = run_layer(model, tokenizer, texts, layer_idx, save_dir)
        # Merge new variant results into existing layer data (preserve prior variants)
        if str(layer_idx) not in all_results["layers"]:
            all_results["layers"][str(layer_idx)] = {}
        all_results["layers"][str(layer_idx)].update(layer_result)

        # Save after each layer (survive crashes)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Intermediate results saved to {RESULTS_PATH}")

    # ---- Generate analysis ----
    write_analysis(all_results)

    # ---- Cross-layer summary ----
    print(f"\n{'='*70}")
    print("  CROSS-LAYER SUMMARY")
    print(f"{'='*70}")
    vnames = VARIANT_NAMES
    vshort = VARIANT_SHORT

    print(f"\n  Reconstruction (FVE / cos):")
    hdr = f"  {'Layer':>6s} |"
    for v in vnames:
        hdr += f" {vshort[v]+' FVE':>10s} {vshort[v]+' cos':>10s} |"
    print(hdr)
    for li in LAYERS:
        lr = all_results["layers"][str(li)]
        row = f"  {li:>6d} |"
        for v in vnames:
            r = lr.get(v, {}).get("reconstruction", {})
            if r:
                row += f" {r['fve']:>10.4f} {r['cos_recon']:>10.4f} |"
            else:
                row += f" {'—':>10s} {'—':>10s} |"
        print(row)

    print(f"\n  Norm Invariance (2x scale — expect std~2.0, cos~1.0):")
    for li in LAYERS:
        lr = all_results["layers"][str(li)]
        parts = []
        for v in vnames:
            inv = lr.get(v, {}).get("norm_invariance", {}).get("scale_2.0", {})
            if inv:
                parts.append(f"{vshort[v]}={inv['mean_ratio']:.3f}")
        print(f"    L{li}: {'  '.join(parts)}")

    print(f"\n  Ablation (SAE act → KL / SAE>inner):")
    hdr = f"  {'Layer':>6s} |"
    for v in vnames:
        hdr += f" {vshort[v]+' SAE→KL':>14s} {vshort[v]+' >inn':>10s} |"
    print(hdr)
    for li in LAYERS:
        lr = all_results["layers"][str(li)]
        row = f"  {li:>6d} |"
        for v in vnames:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                row += f" {a['sae_kl_mean']:>14.4f} {a['sae_wins_inner']:>4d}/{a['n_features']:<4d} |"
            else:
                row += f" {'—':>14s} {'—':>10s} |"
        print(row)

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Analysis: {ANALYSIS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
