"""
Experiment 21: Composite Loss (Post-RMSNorm + L2)
====================================================

Exp18 showed that post-norm loss and adaptive scaling don't compose: post-norm
loss is scale-invariant, so `a` gets zero gradient and freezes. But post-norm
loss produces the best causal features at L27 (SAE->KL=0.252 > standard's 0.198).

The idea: use a composite loss that provides both signals:
  loss = alpha * ||RMSNorm(x) - RMSNorm(x_hat)||^2 + (1-alpha) * ||x - x_hat||^2

The L2 term gives `a` the gradient it needs. The post-norm term steers the decoder
toward gain-weighted directions (source of the L27 SAE->KL breakthrough).

This is a gating experiment at 5M tokens, L27 only (where both effects are
strongest). Sweep alpha in {0.5, 0.8, 0.9, 0.95, 0.99}.

Success criteria: some alpha where `a` learns a positive value (L2 term working)
AND SAE->KL stays above standard's 0.198 (post-norm term working).

Usage:
    ssh <gpu-host>
    cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python3 experiments/exp21_composite_loss.py
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
LAYERS = [27]  # L27 only for alpha sweep (strongest effects)
D_MODEL = 4096
N_LAYERS_TOTAL = 36
RMS_NORM_EPS = 1e-6

# --- SAE architecture ---
D_SAE = 16384
K = 80

# --- Data ---
N_TRAIN_TOKENS = 5_000_000
N_EVAL_TOKENS = 500_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR = 3e-4
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 50

# --- Alpha sweep ---
ALPHAS = [0.5, 0.8, 0.9, 0.95, 0.99]

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 50

# --- Output ---
SAVE_DIR = "checkpoints/exp21"
RESULTS_PATH = "experiments/exp21_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)


def get_config_dict():
    return {
        "model_name": MODEL_NAME, "layers": LAYERS, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
        "n_steps": N_STEPS, "alphas": ALPHAS,
        "rms_norm_eps": RMS_NORM_EPS,
    }


# =============================================================================
# SAE Architecture: AdaptiveCosineBatchTopKSAE (from exp12/18)
# =============================================================================

class AdaptiveCosineBatchTopKSAE(nn.Module):
    """BatchTopK SAE with per-token adaptive-scale cosine encoder.

    scale(x) = exp(scale_a * log(||x - b_dec||) + scale_b)

    Interpolates between:
      - scale_a=0: global scale (identical to CosineBatchTopKSAE)
      - scale_a=1: scale proportional to ||x|| (inner-product-like)
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
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
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


# =============================================================================
# Standard BatchTopK (baseline)
# =============================================================================

class BatchTopKSAE(nn.Module):
    """Standard inner-product BatchTopK SAE (exp17 baseline)."""

    def __init__(self, d_model: int, d_sae: int, k: int = 80):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.zeros(()))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=1, keepdim=True)
            )
            self.encoder.weight.copy_(self.decoder.weight.T * 0.1)
            nn.init.zeros_(self.encoder.bias)

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
        pre_acts = self.encoder(x - self.b_dec)
        acts = F.relu(pre_acts)
        if self.training:
            return self._batch_topk(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f) + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


# =============================================================================
# RMSNorm Loss
# =============================================================================

def get_rmsnorm_for_layer(model, layer_idx):
    if layer_idx + 1 < N_LAYERS_TOTAL:
        return model.model.layers[layer_idx + 1].input_layernorm
    else:
        return model.model.norm


def apply_rmsnorm_f32(x, rmsnorm_weight, eps=RMS_NORM_EPS):
    """RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight, all in float32."""
    weight = rmsnorm_weight.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


# =============================================================================
# Data Collection
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
# Training
# =============================================================================

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_sae(name, sae, train_data, layer_idx, loss_type="l2",
              rmsnorm_weight=None, alpha=None):
    """Train an SAE.

    loss_type: "l2", "cosine", "postnorm", or "composite"
    alpha: mixing weight for composite loss (1.0 = pure postnorm, 0.0 = pure L2)
    """
    n_tokens = train_data.shape[0]
    tag = f"{name}/L{layer_idx}"
    loss_desc = loss_type
    if loss_type == "composite":
        loss_desc = f"composite(α={alpha})"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, loss={loss_desc}, "
          f"{n_tokens:,} tokens, {N_STEPS} steps")

    rmsnorm_w_f32 = None
    if loss_type in ("postnorm", "composite"):
        assert rmsnorm_weight is not None
        rmsnorm_w_f32 = rmsnorm_weight.float().to(DEVICE)
        gain = rmsnorm_w_f32.detach()
        print(f"    RMSNorm gain: mean={gain.mean():.4f}, std={gain.std():.4f}, "
              f"min={gain.min():.4f}, max={gain.max():.4f}")

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

        if loss_type == "cosine":
            recon_loss = (1 - F.cosine_similarity(batch, x_hat, dim=-1)).mean()
        elif loss_type == "postnorm":
            x_normed = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
            xhat_normed = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
            recon_loss = (x_normed - xhat_normed).pow(2).sum(dim=-1).mean()
        elif loss_type == "composite":
            l2_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()
            x_normed = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
            xhat_normed = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
            pn_loss = (x_normed - xhat_normed).pow(2).sum(dim=-1).mean()
            recon_loss = alpha * pn_loss + (1 - alpha) * l2_loss
        else:  # l2
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

                postnorm_fve = None
                if rmsnorm_w_f32 is not None:
                    x_n = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
                    xh_n = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
                    pn_total = torch.var(x_n, dim=0, unbiased=False).sum()
                    pn_resid = torch.var(x_n - xh_n, dim=0, unbiased=False).sum()
                    postnorm_fve = (1 - pn_resid / pn_total).item() if pn_total > 0 else 0

            entry = {
                "step": step, "recon_loss": recon_loss.item(),
                "l0": l0, "fve": fve, "cos_recon": cos_r,
                "dead_frac": dead, "lr": scheduler.get_last_lr()[0],
            }
            if postnorm_fve is not None:
                entry["postnorm_fve"] = postnorm_fve

            # Track scale params for adaptive SAEs
            if hasattr(sae, "scale_a"):
                entry["scale_a"] = sae.scale_a.item()
                entry["scale_b"] = sae.scale_b.exp().item()

            log.append(entry)

            scale_str = ""
            if hasattr(sae, "scale_a"):
                scale_str = f" | a={sae.scale_a.item():.4f} b={sae.scale_b.exp().item():.1f}"
            pn_str = f" | pnFVE={postnorm_fve:.4f}" if postnorm_fve is not None else ""
            print(f"    [{tag:>30s}] step {step:>5d}/{N_STEPS} | "
                  f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                  f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f}"
                  f"{scale_str}{pn_str} | {time.time()-t0:.0f}s")

    sae.eval()
    print(f"    [{tag}] Done in {time.time()-t0:.1f}s")
    return log


# =============================================================================
# Evaluation (same as exp18)
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data, layer_idx, rmsnorm_weight=None):
    tag = f"{name}/L{layer_idx}"
    sae.eval()
    n = eval_data.shape[0]
    recon_losses, cos_sims, l0s = [], [], []
    total_var_sum, resid_var_sum = 0.0, 0.0
    pn_total_var_sum, pn_resid_var_sum = 0.0, 0.0
    postnorm_losses = []

    rmsnorm_w_f32 = rmsnorm_weight.float().to(DEVICE) if rmsnorm_weight is not None else None

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        recon_losses.append((batch - x_hat).pow(2).sum(dim=-1).mean().item())
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()

        if rmsnorm_w_f32 is not None:
            x_n = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
            xh_n = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
            postnorm_losses.append((x_n - xh_n).pow(2).sum(dim=-1).mean().item())
            pn_total_var_sum += torch.var(x_n, dim=0, unbiased=False).sum().item()
            pn_resid_var_sum += torch.var(x_n - xh_n, dim=0, unbiased=False).sum().item()

    results = {
        "recon_loss_l2": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
    }
    if postnorm_losses:
        results["postnorm_loss"] = float(np.mean(postnorm_losses))
        results["postnorm_fve"] = float(1 - pn_resid_var_sum / pn_total_var_sum) if pn_total_var_sum > 0 else 0

    pn_str = ""
    if "postnorm_loss" in results:
        pn_str = f" | pnL2={results['postnorm_loss']:.4f} | pnFVE={results['postnorm_fve']:.4f}"
    print(f"    [{tag}] L2={results['recon_loss_l2']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"FVE={results['fve']:.4f}{pn_str}")
    return results


@torch.no_grad()
def test_norm_invariance(name, sae, eval_data, layer_idx, scales=(0.5, 2.0, 5.0)):
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
        else:
            mean_ratio = float("nan")
        agreement = ((base_feats > 0) == (scaled_feats > 0)).float().mean().item()
        cos = F.cosine_similarity(
            base_feats.float(), scaled_feats.float(), dim=-1
        ).mean().item()
        results[f"scale_{scale}"] = {
            "mean_ratio": mean_ratio, "feature_agreement": agreement,
            "activation_cosine": cos,
        }
        print(f"    [{tag}] scale={scale}: ratio={mean_ratio:.3f} "
              f"| agree={agreement:.3f} | cos={cos:.4f}")
    return results


def ablate_feature_kl(model, activation, feature_dir, layer_idx):
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


def evaluate_ablation(name, model, sae, eval_data, layer_idx):
    tag = f"{name}/L{layer_idx}"
    print(f"\n    Ablation [{tag}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    # Get decoder weights — handle both architectures
    # W_dec must be (d_sae, d_model) so W_dec[fi] gives the feature direction
    if hasattr(sae, "W_dec"):
        W_dec = sae.W_dec  # already (d_sae, d_model)
    elif hasattr(sae, "decoder"):
        W_dec = sae.decoder.weight.T  # nn.Linear weight is (d_model, d_sae), transpose
    else:
        raise ValueError(f"Can't find decoder weights on {type(sae)}")

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
        feat_dir = W_dec[fi].float()
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

        cos_arr = np.array(cos_v)
        norm_arr = np.array(norm_v)
        inner_arr = np.array(inner_v)
        sae_arr = np.array(sae_v)

        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]

        feature_results.append({
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "sae_wins_inner": bool(abs(corr_sae) > abs(corr_inner)),
        })

        if rank < 5 or rank % 10 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos->KL={corr_cos:.3f} | inner->KL={corr_inner:.3f} | "
                  f"SAE->KL={corr_sae:.3f} | norm->KL={corr_norm:.3f}")

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
        "sae_wins_inner": sum(r["sae_wins_inner"] for r in feature_results),
    }

    print(f"    [{tag}] Summary ({n} features): "
          f"cos->KL={agg['cos_kl_mean']:.4f} | SAE->KL={agg['sae_kl_mean']:.4f} | "
          f"cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 21: Composite Loss (Post-RMSNorm + L2)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Alpha sweep: {ALPHAS}")
    print(f"Per variant: {N_TRAIN_TOKENS:,} train tokens ({N_STEPS} steps), "
          f"{N_EVAL_TOKENS:,} eval tokens")
    print(f"Batch: {BATCH_SIZE}, Warmup: {WARMUP_STEPS} steps")

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

    # RMSNorm gain stats
    gain_stats = {}
    for li in LAYERS:
        rmsnorm = get_rmsnorm_for_layer(model, li)
        w = rmsnorm.weight.float()
        stats = {
            "mean": w.mean().item(), "std": w.std().item(),
            "min": w.min().item(), "max": w.max().item(),
            "cv": (w.std() / w.mean()).item(),
        }
        gain_stats[str(li)] = stats
        print(f"  L{li} RMSNorm gain: mean={stats['mean']:.4f}, CV={stats['cv']:.1%}, "
              f"max={stats['max']:.4f}")

    print("\nCollecting FineWeb texts...")
    texts = collect_texts(N_TRAIN_TOKENS + N_EVAL_TOKENS)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results")
    else:
        all_results = {
            "config": get_config_dict(),
            "rmsnorm_gain_stats": gain_stats,
            "layers": {},
        }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        rmsnorm = get_rmsnorm_for_layer(model, layer_idx)
        rmsnorm_weight = rmsnorm.weight.detach()

        n_total = N_TRAIN_TOKENS + N_EVAL_TOKENS
        all_acts = texts_to_activations(model, tokenizer, texts, layer_idx, n_total)
        train_data = all_acts[:N_TRAIN_TOKENS]
        eval_data = all_acts[N_TRAIN_TOKENS:N_TRAIN_TOKENS + N_EVAL_TOKENS]
        print(f"  Split: train={train_data.shape[0]:,}, eval={eval_data.shape[0]:,}")
        del all_acts

        li = str(layer_idx)
        if li not in all_results["layers"]:
            all_results["layers"][li] = {}

        # --- Baselines ---
        baselines = [
            ("standard",       BatchTopKSAE,              "l2",       None),
            ("adaptive_l2",    AdaptiveCosineBatchTopKSAE, "l2",       None),
            ("adaptive_postnorm", AdaptiveCosineBatchTopKSAE, "postnorm", None),
        ]

        # --- Alpha sweep ---
        alpha_variants = [
            (f"composite_a{alpha:.2f}", AdaptiveCosineBatchTopKSAE, "composite", alpha)
            for alpha in ALPHAS
        ]

        all_variants = baselines + alpha_variants

        for vname, sae_cls, loss_type, alpha in all_variants:
            # Skip if already computed
            if vname in all_results["layers"][li]:
                print(f"\n  Skipping {vname}/L{layer_idx} (already computed)")
                continue

            torch.manual_seed(SEED)
            if sae_cls == BatchTopKSAE:
                sae = BatchTopKSAE(D_MODEL, D_SAE, K).to(DEVICE)
            else:
                sae = AdaptiveCosineBatchTopKSAE(D_MODEL, D_SAE, K).to(DEVICE)

            train_log = train_sae(
                vname, sae, train_data, layer_idx,
                loss_type=loss_type,
                rmsnorm_weight=rmsnorm_weight if loss_type in ("postnorm", "composite") else None,
                alpha=alpha,
            )

            recon = evaluate_reconstruction(
                vname, sae, eval_data, layer_idx,
                rmsnorm_weight=rmsnorm_weight,
            )
            inv = test_norm_invariance(vname, sae, eval_data, layer_idx)
            abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)

            torch.save(sae.state_dict(), save_dir / f"{vname}_L{layer_idx}.pt")

            result = {
                "training": train_log,
                "reconstruction": recon,
                "norm_invariance": inv,
                "ablation": abl,
                "loss_type": loss_type,
            }
            if alpha is not None:
                result["alpha"] = alpha
            if hasattr(sae, "scale_a"):
                result["scale_params"] = {
                    "scale_a_final": sae.scale_a.item(),
                    "scale_b_final": sae.scale_b.exp().item(),
                }

            all_results["layers"][li][vname] = result

            # Save after each variant
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Saved {vname}/L{layer_idx} to {RESULTS_PATH}")

            del sae
            gc.collect()
            torch.cuda.empty_cache()

        del train_data, eval_data
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  SUMMARY — Composite Loss Alpha Sweep at L27")
    print(f"{'='*70}")

    li = str(LAYERS[0])
    layer_data = all_results.get("layers", {}).get(li, {})

    print(f"\n  {'Variant':>25s} | {'α':>5s} | {'scale_a':>8s} | "
          f"{'FVE':>7s} | {'pnFVE':>7s} | {'dead':>6s} | "
          f"{'cos→KL':>7s} | {'SAE→KL':>7s} | {'cos>inn':>8s}")
    print(f"  {'-'*25}-+-{'-'*5}-+-{'-'*8}-+-"
          f"{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-"
          f"{'-'*7}-+-{'-'*7}-+-{'-'*8}")

    for vname in ["standard", "adaptive_l2", "adaptive_postnorm"] + \
                 [f"composite_a{a:.2f}" for a in ALPHAS]:
        v = layer_data.get(vname, {})
        if not v:
            continue

        alpha_str = str(v.get("alpha", "-"))
        a_val = v.get("scale_params", {}).get("scale_a_final", "-")
        a_str = f"{a_val:.4f}" if isinstance(a_val, float) else str(a_val)

        r = v.get("reconstruction", {})
        fve = f"{r.get('fve', 0):.4f}" if r else "-"
        pnfve = f"{r.get('postnorm_fve', 0):.4f}" if r.get("postnorm_fve") else "-"

        # Dead features from last training log entry
        tlog = v.get("training", [])
        dead = f"{tlog[-1].get('dead_frac', 0):.3f}" if tlog else "-"

        abl = v.get("ablation", {}).get("aggregate", {})
        cos_kl = f"{abl.get('cos_kl_mean', 0):.4f}" if abl.get("n_features") else "-"
        sae_kl = f"{abl.get('sae_kl_mean', 0):.4f}" if abl.get("n_features") else "-"
        n_feat = abl.get("n_features", 0)
        cos_wins = abl.get("cos_wins_inner", 0)
        cos_inn = f"{cos_wins}/{n_feat}" if n_feat else "-"

        print(f"  {vname:>25s} | {alpha_str:>5s} | {a_str:>8s} | "
              f"{fve:>7s} | {pnfve:>7s} | {dead:>6s} | "
              f"{cos_kl:>7s} | {sae_kl:>7s} | {cos_inn:>8s}")

    print(f"\n  Key question: is there an alpha where scale_a > 0 AND SAE->KL > 0.198?")
    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
