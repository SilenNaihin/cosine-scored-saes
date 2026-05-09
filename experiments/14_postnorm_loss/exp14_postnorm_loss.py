"""
Experiment 14: Post-RMSNorm Loss
================================

If the RNH is correct, SAE reconstruction quality should be measured after
normalization -- how similar are RMSNorm(x) and RMSNorm(x_hat) as seen by
the next layer?

Mathematically, ||RMSNorm(x) - RMSNorm(x_hat)||^2 ~ 2*(1 - cos(x, x_hat))
for *pure* RMSNorm (no gain). But the actual module has a learned per-dimension
gain parameter (weight), so some dimensions matter more than others. Post-norm
loss respects this weighting; cosine loss treats all dimensions equally.

This trains a single new variant at layers [9, 18, 27]:
  - cosine_postnorm: CosineBatchTopKSAE encoder, post-RMSNorm L2 loss

Results are saved to exp14_results.json and also merged into exp10_results.json
for cross-variant comparison.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python experiments/exp14_postnorm_loss.py
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
N_LAYERS_TOTAL = 36  # Qwen3-8B has 36 transformer layers
RMS_NORM_EPS = 1e-6

# --- SAE architecture ---
D_SAE = 16384       # 4x expansion (production: 65536)
K = 80              # BatchTopK sparsity, matches existing SAEs

# --- Data ---
N_TRAIN_TOKENS = 5_000_000   # Per layer
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
SAVE_DIR = "checkpoints/exp14"
RESULTS_PATH = "experiments/exp14_results.json"
EXP10_RESULTS_PATH = "experiments/exp10_results.json"

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
        "rms_norm_eps": RMS_NORM_EPS,
    }


# =============================================================================
# SAE Architecture (CosineBatchTopKSAE from exp10)
# =============================================================================

class CosineBatchTopKSAE(nn.Module):
    """BatchTopK SAE with scaled-cosine-similarity encoder.

    encode: x -> BatchTopK(ReLU(scale * cos_sim(x - b_dec, W_enc_rows) + b_enc))
    decode: f -> f @ W_dec + b_dec
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


# =============================================================================
# RMSNorm Loss
# =============================================================================

def get_rmsnorm_for_layer(model, layer_idx):
    """Get the RMSNorm module that processes the output of `layer_idx`.

    The output of layer `i` goes into layer `i+1`, whose input_layernorm
    is the RMSNorm that processes it. For the final layer, the model's
    final norm (model.model.norm) is used instead.
    """
    if layer_idx + 1 < N_LAYERS_TOTAL:
        return model.model.layers[layer_idx + 1].input_layernorm
    else:
        return model.model.norm


def apply_rmsnorm_f32(x, rmsnorm_weight, eps=RMS_NORM_EPS):
    """Apply RMSNorm in float32 with explicit gain weight.

    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight

    The rmsnorm_weight is cast to float32 for the computation.
    x is expected to already be float32.
    """
    weight = rmsnorm_weight.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


# =============================================================================
# Data Collection (identical to exp10)
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
    """Download FineWeb texts once (lightweight strings, reused for all layers)."""
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


def train_sae(name, sae, train_data, layer_idx, loss_type="l2",
              rmsnorm_weight=None):
    """Train an SAE on pre-collected activations. Returns list of metric dicts.

    loss_type:
      - "l2": standard MSE ||x - x_hat||^2
      - "cosine": 1 - cos_sim(x, x_hat)
      - "postnorm": ||rmsnorm(x) - rmsnorm(x_hat)||^2  (requires rmsnorm_weight)
    """
    n_tokens = train_data.shape[0]
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, loss={loss_type}, "
          f"{n_tokens:,} tokens, {N_STEPS} steps")

    if loss_type == "postnorm":
        assert rmsnorm_weight is not None, "postnorm loss requires rmsnorm_weight"
        # Pre-cast to float32 once (avoid repeated casting per step)
        rmsnorm_w_f32 = rmsnorm_weight.float().to(DEVICE)
        # Log the gain statistics
        gain = rmsnorm_w_f32.detach()
        print(f"    RMSNorm gain: mean={gain.mean():.4f}, std={gain.std():.4f}, "
              f"min={gain.min():.4f}, max={gain.max():.4f}")

    # Deterministic shuffle
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

        # Loss selection
        if loss_type == "cosine":
            recon_loss = (1 - F.cosine_similarity(batch, x_hat, dim=-1)).mean()
        elif loss_type == "postnorm":
            # Apply RMSNorm to both original and reconstruction in float32
            x_normed = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
            xhat_normed = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
            recon_loss = (x_normed - xhat_normed).pow(2).sum(dim=-1).mean()
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

                # Also compute post-norm FVE if postnorm loss
                postnorm_fve = None
                if loss_type == "postnorm":
                    x_n = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
                    xh_n = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
                    pn_total = torch.var(x_n, dim=0, unbiased=False).sum()
                    pn_resid = torch.var(x_n - xh_n, dim=0, unbiased=False).sum()
                    postnorm_fve = (1 - pn_resid / pn_total).item() if pn_total > 0 else 0

            scale_val = None
            if hasattr(sae, "log_scale"):
                scale_val = sae.log_scale.exp().item()

            entry = {
                "step": step, "recon_loss": recon_loss.item(),
                "l0": l0, "fve": fve, "cos_recon": cos_r,
                "dead_frac": dead, "lr": scheduler.get_last_lr()[0],
            }
            if scale_val is not None:
                entry["scale"] = scale_val
            if postnorm_fve is not None:
                entry["postnorm_fve"] = postnorm_fve
            log.append(entry)

            scale_str = f" | scale={scale_val:.1f}" if scale_val else ""
            pn_str = f" | pnFVE={postnorm_fve:.4f}" if postnorm_fve is not None else ""
            print(f"    [{tag:>16s}] step {step:>5d}/{N_STEPS} | "
                  f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                  f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f}"
                  f"{scale_str}{pn_str} | {time.time()-t0:.0f}s")

    sae.eval()
    print(f"    [{tag}] Done in {time.time()-t0:.1f}s")
    return log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data, layer_idx, rmsnorm_weight=None):
    """Reconstruction metrics on held-out data."""
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

        # Post-norm metrics
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
              f"| agree={agreement:.3f} | cos={cos:.4f}")
    return results


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


def evaluate_ablation(name, model, sae, eval_data, layer_idx):
    """For top features, correlate SAE activation with ablation KL divergence."""
    tag = f"{name}/L{layer_idx}"
    print(f"\n    Ablation [{tag}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

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
        "cos_wins_sae": sum(r["cos_wins_sae"] for r in feature_results),
        "sae_wins_inner": sum(r["sae_wins_inner"] for r in feature_results),
    }

    print(f"    [{tag}] Summary ({n} features): "
          f"cos->KL={agg['cos_kl_mean']:.4f} | SAE->KL={agg['sae_kl_mean']:.4f} | "
          f"SAE>inner: {agg['sae_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# RMSNorm Gain Analysis
# =============================================================================

@torch.no_grad()
def analyze_rmsnorm_gain(model, layers):
    """Analyze the learned RMSNorm gain parameters across layers."""
    print("\n  RMSNorm Gain Analysis")
    print("  " + "-" * 50)
    gain_stats = {}
    for layer_idx in layers:
        rmsnorm = get_rmsnorm_for_layer(model, layer_idx)
        w = rmsnorm.weight.float()
        stats = {
            "mean": w.mean().item(),
            "std": w.std().item(),
            "min": w.min().item(),
            "max": w.max().item(),
            "range": (w.max() - w.min()).item(),
            "cv": (w.std() / w.mean()).item(),  # coefficient of variation
        }
        gain_stats[str(layer_idx)] = stats
        print(f"    L{layer_idx} -> L{layer_idx+1} norm: "
              f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"range=[{stats['min']:.4f}, {stats['max']:.4f}], "
              f"CV={stats['cv']:.4f}")

        # How non-uniform is the gain? If CV is small, gain ~ constant, and
        # postnorm loss ~ cosine loss. If CV is large, dimensions are weighted
        # very differently.
        if stats['cv'] < 0.05:
            print(f"      -> Gain is nearly uniform (CV < 5%). "
                  f"Postnorm ~ cosine loss.")
        elif stats['cv'] < 0.2:
            print(f"      -> Gain has moderate variation (CV={stats['cv']:.1%}). "
                  f"Postnorm may differ from cosine.")
        else:
            print(f"      -> Gain is highly non-uniform (CV={stats['cv']:.1%}). "
                  f"Postnorm should differ meaningfully from cosine.")

    return gain_stats


# =============================================================================
# Per-Layer Runner
# =============================================================================

def run_layer(model, tokenizer, texts, layer_idx, save_dir):
    """Train + evaluate cosine_postnorm SAE at one layer."""
    print(f"\n{'='*70}")
    print(f"  LAYER {layer_idx}")
    print(f"{'='*70}")

    # ---- Get RMSNorm weight ----
    rmsnorm = get_rmsnorm_for_layer(model, layer_idx)
    rmsnorm_weight = rmsnorm.weight.detach()

    # ---- Activations ----
    n_total = N_TRAIN_TOKENS + N_EVAL_TOKENS
    all_acts = texts_to_activations(model, tokenizer, texts, layer_idx, n_total)
    train_data = all_acts[:N_TRAIN_TOKENS]
    eval_data = all_acts[N_TRAIN_TOKENS:N_TRAIN_TOKENS + N_EVAL_TOKENS]
    print(f"  Split: train={train_data.shape[0]:,}, eval={eval_data.shape[0]:,}")
    del all_acts

    # ---- Train cosine_postnorm ----
    torch.manual_seed(SEED)
    sae = CosineBatchTopKSAE(D_MODEL, D_SAE, K).to(DEVICE)
    training_log = train_sae(
        "cosine_postnorm", sae, train_data, layer_idx,
        loss_type="postnorm", rmsnorm_weight=rmsnorm_weight,
    )

    # ---- Evaluate ----
    del train_data
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Evaluation -- Layer {layer_idx}")

    recon = evaluate_reconstruction(
        "cosine_postnorm", sae, eval_data, layer_idx,
        rmsnorm_weight=rmsnorm_weight,
    )
    inv = test_norm_invariance("cosine_postnorm", sae, eval_data, layer_idx)
    abl = evaluate_ablation("cosine_postnorm", model, sae, eval_data, layer_idx)
    torch.save(sae.state_dict(), save_dir / f"cosine_postnorm_L{layer_idx}.pt")

    result = {
        "cosine_postnorm": {
            "training": training_log,
            "reconstruction": recon,
            "norm_invariance": inv,
            "ablation": abl,
        }
    }

    # ---- Cleanup ----
    del sae, eval_data
    gc.collect()
    torch.cuda.empty_cache()

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 14: Post-RMSNorm Loss")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Per layer: {N_TRAIN_TOKENS:,} train tokens ({N_STEPS} steps), "
          f"{N_EVAL_TOKENS:,} eval tokens")
    print(f"Batch: {BATCH_SIZE}, Warmup: {WARMUP_STEPS} steps")
    print(f"Loss: postnorm (||RMSNorm(x) - RMSNorm(x_hat)||^2)")

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

    # ---- Analyze RMSNorm gain parameters ----
    gain_stats = analyze_rmsnorm_gain(model, LAYERS)

    # ---- Collect texts once ----
    print("\nCollecting FineWeb texts...")
    n_needed = N_TRAIN_TOKENS + N_EVAL_TOKENS
    texts = collect_texts(n_needed)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing exp14 results ----
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        existing = list(all_results.get("layers", {}).keys())
        print(f"  Loaded existing exp14 results for layers: {existing}")
    else:
        all_results = {
            "config": get_config_dict(),
            "rmsnorm_gain_stats": gain_stats,
            "layers": {},
        }

    # ---- Run each layer ----
    for layer_idx in LAYERS:
        layer_result = run_layer(model, tokenizer, texts, layer_idx, save_dir)

        if str(layer_idx) not in all_results["layers"]:
            all_results["layers"][str(layer_idx)] = {}
        all_results["layers"][str(layer_idx)].update(layer_result)

        # Save after each layer
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Intermediate results saved to {RESULTS_PATH}")

    # ---- Also merge into exp10_results.json for cross-variant comparison ----
    if os.path.exists(EXP10_RESULTS_PATH):
        with open(EXP10_RESULTS_PATH) as f:
            exp10_results = json.load(f)
        for layer_idx in LAYERS:
            li = str(layer_idx)
            if li in all_results["layers"] and "cosine_postnorm" in all_results["layers"][li]:
                if li not in exp10_results["layers"]:
                    exp10_results["layers"][li] = {}
                exp10_results["layers"][li]["cosine_postnorm"] = \
                    all_results["layers"][li]["cosine_postnorm"]
        with open(EXP10_RESULTS_PATH, "w") as f:
            json.dump(exp10_results, f, indent=2, default=str)
        print(f"  Results also merged into {EXP10_RESULTS_PATH}")

    # ---- Print summary ----
    print(f"\n{'='*70}")
    print("  SUMMARY: cosine_postnorm vs existing variants")
    print(f"{'='*70}")

    # Load exp10 results for comparison
    exp10 = {}
    if os.path.exists(EXP10_RESULTS_PATH):
        with open(EXP10_RESULTS_PATH) as f:
            exp10 = json.load(f)

    compare_variants = [
        "standard", "cosine_l2", "cosine_cosloss",
        "adaptive_l2", "cosine_postnorm",
    ]
    short = {
        "standard": "Std", "cosine_l2": "CosL2", "cosine_cosloss": "CosCos",
        "adaptive_l2": "AdpL2", "cosine_postnorm": "PostN",
    }

    print(f"\n  Reconstruction (FVE / cos):")
    hdr = f"  {'Layer':>6s} |"
    for v in compare_variants:
        hdr += f" {short[v]+' FVE':>10s} {short[v]+' cos':>10s} |"
    print(hdr)
    for li in LAYERS:
        row = f"  {li:>6d} |"
        for v in compare_variants:
            # Check exp14 results first, then exp10
            r = None
            if v == "cosine_postnorm":
                r = all_results.get("layers", {}).get(str(li), {}).get(v, {}).get("reconstruction", {})
            else:
                r = exp10.get("layers", {}).get(str(li), {}).get(v, {}).get("reconstruction", {})
            if r:
                row += f" {r['fve']:>10.4f} {r['cos_recon']:>10.4f} |"
            else:
                row += f" {'--':>10s} {'--':>10s} |"
        print(row)

    print(f"\n  Ablation (cos->KL / SAE->KL):")
    hdr = f"  {'Layer':>6s} |"
    for v in compare_variants:
        hdr += f" {short[v]+' cos':>10s} {short[v]+' SAE':>10s} |"
    print(hdr)
    for li in LAYERS:
        row = f"  {li:>6d} |"
        for v in compare_variants:
            a = None
            if v == "cosine_postnorm":
                a = all_results.get("layers", {}).get(str(li), {}).get(v, {}).get("ablation", {}).get("aggregate", {})
            else:
                a = exp10.get("layers", {}).get(str(li), {}).get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                row += f" {a['cos_kl_mean']:>10.4f} {a['sae_kl_mean']:>10.4f} |"
            else:
                row += f" {'--':>10s} {'--':>10s} |"
        print(row)

    print(f"\n  RMSNorm Gain Statistics:")
    for li in LAYERS:
        s = gain_stats.get(str(li), {})
        if s:
            print(f"    L{li}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                  f"CV={s['cv']:.4f}, range=[{s['min']:.4f}, {s['max']:.4f}]")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
