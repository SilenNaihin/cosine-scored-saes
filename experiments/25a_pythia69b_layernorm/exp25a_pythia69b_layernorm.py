"""
Experiment 25a: Pythia-6.9B LayerNorm Control
==============================================

Part of the 2x2 normalization matrix (exp25a/b/c).

Tests whether the cosine SAE advantage is specific to RMSNorm models.

On RMSNorm models (Qwen3-8B), cosine SAEs consistently beat standard:
+8 FVE at L27, 3.3x more alive features (exp17), cos>inner 70-90%.

The RNH claims cosine is better BECAUSE RMSNorm erases magnitude.
If the cosine advantage disappears on a LayerNorm model, that's the
strongest evidence the advantage IS about normalization. If it persists,
the advantage is something else (regularization?) and the RMSNorm
narrative is weaker.

3 variants x 3 layers (8, 16, 24) at 5M tokens:
  1. standard     — inner-product encoder
  2. cosine       — full cosine encoder
  3. adaptive_l2  — cosine + adaptive per-token scale

IMPORTANT: Run on GPU 1 only (GPU 0 is running exp22).
    CUDA_VISIBLE_DEVICES=1 python -u experiments/exp25a_pythia69b_layernorm.py

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/exp25a_pythia69b_layernorm.py > experiments/exp25a_output.log 2>&1 &
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
MODEL_DTYPE = torch.float16   # 6.9B is too large for float32 to be comfortable
STORAGE_DTYPE = torch.float16  # Save CPU memory during collection

# --- Model ---
MODEL_NAME = "EleutherAI/pythia-6.9b-deduped"
LAYERS = [8, 16, 24]  # 25%, 50%, 75% of 32 layers
D_MODEL = 4096

# --- SAE architecture ---
D_SAE = 16384  # 4x d_model
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
LOG_EVERY = 100

# --- Ablation ---
N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 50

# --- Output ---
SAVE_DIR = "checkpoints/exp25a"
RESULTS_PATH = "experiments/exp25a_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)


# =============================================================================
# SAE Architectures (from exp23, unchanged)
# =============================================================================

class BatchTopKSAE(nn.Module):
    """Standard BatchTopK SAE with inner-product encoder."""

    def __init__(self, d_model: int, d_sae: int, k: int = 50):
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
    """Full cosine encoder — normalize BOTH input AND weights."""

    def __init__(self, d_model: int, d_sae: int, k: int = 50):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
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
        scale = torch.exp(self.scale_b)
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


class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Cosine encoder with adaptive per-token scale.

    scale(x) = exp(scale_a * log(||x - b_dec||) + scale_b)
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 50):
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


VARIANTS = [
    ("standard",    BatchTopKSAE),
    ("cosine",      CosineBatchTopKSAE),
    ("adaptive_l2", AdaptiveCosineBatchTopKSAE),
]


# =============================================================================
# Activation Collection (Pythia-specific hooks)
# =============================================================================

class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
    """Capture residual stream activations at a Pythia layer via forward hook."""
    captured = {}

    def hook(module, inp, out):
        captured["act"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        raise _EarlyStop

    layer = model.gpt_neox.layers[layer_idx]
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    except _EarlyStop:
        pass
    handle.remove()
    return captured["act"]


def collect_activations(model, tokenizer, layer_idx, n_tokens, skip_docs=0):
    """Pre-collect activations from FineWeb for a given layer."""
    label = "eval" if skip_docs > 0 else "train"
    print(f"  Collecting {label} activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)

    # Skip docs to avoid train/eval overlap
    if skip_docs > 0:
        for i, _ in enumerate(text_iter):
            if i >= skip_docs:
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

        # Filter attention sinks (high-norm outliers)
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            flat = flat[norms < median * OUTLIER_MULTIPLIER]

        all_acts.append(flat.to("cpu", dtype=STORAGE_DTYPE))
        tokens_collected += flat.shape[0]

    result = torch.cat(all_acts, dim=0)[:n_tokens]
    norms = result.float().norm(dim=-1)
    print(f"  Layer {layer_idx}: {result.shape[0]:,} {label} tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={norms.mean():.1f}, std={norms.std():.1f})")
    return result


# =============================================================================
# Training
# =============================================================================

def lr_schedule(step):
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_sae(name, sae, train_data, layer_idx):
    """Train an SAE on pre-collected activations."""
    print(f"\n  Training {name} | L{layer_idx} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    perm = torch.randperm(train_data.shape[0])
    train_shuffled = train_data[perm]

    sae.train()
    log = []
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        start = ((step - 1) * BATCH_SIZE) % train_shuffled.shape[0]
        end = start + BATCH_SIZE
        if end > train_shuffled.shape[0]:
            idx = torch.cat([
                torch.arange(start, train_shuffled.shape[0]),
                torch.arange(0, end - train_shuffled.shape[0]),
            ])
        else:
            idx = torch.arange(start, end)
        batch = train_shuffled[idx].to(DEVICE, dtype=torch.float32)

        x_hat, features = sae(batch)
        recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

        optimizer.zero_grad(set_to_none=True)
        recon_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == N_STEPS:
            with torch.no_grad():
                l0 = (features != 0).float().sum(dim=-1).mean().item()
                total_var = torch.var(batch, dim=0, unbiased=False).sum()
                resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                dead = (features.sum(dim=0) == 0).float().mean().item()

            tokens_seen = step * BATCH_SIZE
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
            eta_sec = (N_STEPS - step) * (elapsed / step) if step > 0 else 0

            entry = {
                "step": step, "recon_loss": recon_loss.item(),
                "l0": l0, "fve": fve, "cos_recon": cos_r, "dead_frac": dead,
                "lr": scheduler.get_last_lr()[0],
            }

            scale_str = ""
            if hasattr(sae, "scale_a"):
                entry["scale_a"] = sae.scale_a.item()
                entry["scale_b_exp"] = sae.scale_b.exp().item()
                scale_str = f" | a={sae.scale_a.item():.4f} b={sae.scale_b.exp().item():.1f}"
            elif hasattr(sae, "scale_b"):
                entry["scale_b_exp"] = sae.scale_b.exp().item()
                scale_str = f" | scale={sae.scale_b.exp().item():.1f}"

            log.append(entry)
            print(f"    [{name:>12s}] step {step:>5d}/{N_STEPS} | "
                  f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                  f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f} | "
                  f"tok={tokens_seen/1e6:.1f}M | {tok_per_sec/1e3:.1f}K/s | "
                  f"ETA={eta_sec/60:.0f}m{scale_str}")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{name}] Done in {elapsed:.1f}s")
    return log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data):
    """Reconstruction metrics on held-out data."""
    sae.eval()
    n = eval_data.shape[0]
    recon_losses, cos_sims, l0s = [], [], []
    total_var_sum, resid_var_sum = 0.0, 0.0
    dead_counts = None

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        recon_losses.append((batch - x_hat).pow(2).sum(dim=-1).mean().item())
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        alive = (features > 0).any(dim=0)
        if dead_counts is None:
            dead_counts = ~alive
        else:
            dead_counts &= ~alive

    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0

    results = {
        "recon_loss_l2": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
        "dead_frac": dead_frac,
    }
    print(f"    [{name}] L2={results['recon_loss_l2']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"FVE={results['fve']:.4f} | dead={dead_frac:.3f}")
    return results


@torch.no_grad()
def test_norm_invariance(name, sae, eval_data, scales=(0.5, 2.0, 5.0)):
    """Test whether SAE activations change when input is scaled."""
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
            "mean_ratio": mean_ratio,
            "feature_agreement": agreement,
            "activation_cosine": cos,
        }
        print(f"    [{name}] scale={scale}: ratio={mean_ratio:.3f} | "
              f"agree={agreement:.3f} | cos={cos:.4f}")
    return results


def ablate_feature_kl(model, activation, feature_dir, layer_idx):
    """Ablate a feature direction from the residual stream, measure KL at logits."""
    projection = (activation @ feature_dir) * feature_dir
    model_dtype = next(model.parameters()).dtype
    x = activation.unsqueeze(0).unsqueeze(0).to(model_dtype)
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0).to(model_dtype)

    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

    def make_hook(replacement):
        def hook(module, inputs, outputs):
            return (replacement,) + outputs[1:] if isinstance(outputs, tuple) else replacement
        return hook

    h = model.gpt_neox.layers[layer_idx].register_forward_hook(make_hook(x))
    with torch.no_grad():
        try:
            orig_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            h.remove()
            return None
    h.remove()

    h = model.gpt_neox.layers[layer_idx].register_forward_hook(make_hook(x_ablated))
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
    """Ablation evaluation: 30 features x 50 samples."""
    print(f"\n    Ablation [{name}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    # Probe for active features
    n_probe = min(100_000, eval_data.shape[0])
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
    print(f"    [{name}] {n_alive} alive features (of {D_SAE})")

    n_to_select = min(N_ABLATION_FEATURES, n_alive)
    if n_to_select == 0:
        print(f"    [{name}] No alive features — skipping ablation")
        return {"features": [], "aggregate": {"n_features": 0}}

    top_idx = freq.topk(n_to_select).indices
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

        result = {
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "cos_wins_sae": bool(abs(corr_cos) > abs(corr_sae)),
            "sae_wins_inner": bool(abs(corr_sae) > abs(corr_inner)),
        }
        feature_results.append(result)

        if rank < 5 or rank % 10 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f} | norm→KL={corr_norm:.3f}")

    if not feature_results:
        print(f"    [{name}] No features with enough data")
        return {"features": [], "aggregate": {"n_features": 0}}

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

    print(f"    [{name}] Summary ({n} features): "
          f"cos→KL={agg['cos_kl_mean']:.4f} | inner→KL={agg['inner_kl_mean']:.4f} | "
          f"SAE→KL={agg['sae_kl_mean']:.4f} | cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 25a: Pythia-6.9B LayerNorm Control")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_model: {D_MODEL}, d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Tokens: {N_TRAIN_TOKENS:,} train, {N_EVAL_TOKENS:,} eval")
    print(f"Steps: {N_STEPS}, Warmup: {WARMUP_STEPS}")
    print(f"Ablation: {N_ABLATION_FEATURES} features x {N_ABLATION_SAMPLES} samples")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Device: {DEVICE}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=MODEL_DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "config": {
            "model_name": MODEL_NAME,
            "normalization": "LayerNorm",
            "layers": LAYERS,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "k": K,
            "n_train_tokens": N_TRAIN_TOKENS,
            "n_eval_tokens": N_EVAL_TOKENS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "n_steps": N_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "n_ablation_features": N_ABLATION_FEATURES,
            "n_ablation_samples": N_ABLATION_SAMPLES,
        },
        "layers": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx} (of 32)")
        print(f"{'='*70}")

        # Collect activations for this layer
        train_data = collect_activations(model, tokenizer, layer_idx, N_TRAIN_TOKENS)
        eval_data = collect_activations(
            model, tokenizer, layer_idx, N_EVAL_TOKENS, skip_docs=200_000
        )

        layer_results = {}

        for vname, cls in VARIANTS:
            print(f"\n  --- VARIANT: {vname} (L{layer_idx}) ---")

            torch.manual_seed(SEED)
            sae = cls(D_MODEL, D_SAE, K).to(DEVICE)

            # Train
            train_log = train_sae(vname, sae, train_data, layer_idx)

            # Evaluate
            print(f"\n  Reconstruction — {vname}")
            recon = evaluate_reconstruction(vname, sae, eval_data)

            print(f"\n  Norm Invariance — {vname}")
            inv = test_norm_invariance(vname, sae, eval_data)

            abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)

            result = {
                "training": train_log,
                "reconstruction": recon,
                "norm_invariance": inv,
                "ablation": abl,
            }

            # Log scale params
            if hasattr(sae, "scale_a"):
                result["scale_a"] = sae.scale_a.item()
                result["scale_b_exp"] = sae.scale_b.exp().item()
            elif hasattr(sae, "scale_b"):
                result["scale_b_exp"] = sae.scale_b.exp().item()

            layer_results[vname] = result

            # Save checkpoint
            torch.save(sae.state_dict(), save_dir / f"{vname}_L{layer_idx}_final.pt")

            # Save results incrementally after each variant
            all_results["layers"][str(layer_idx)] = layer_results
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            del sae
            gc.collect()
            torch.cuda.empty_cache()

        # Free layer data before moving to next
        del train_data, eval_data
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary Table
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE — Pythia-6.9B (LayerNorm), 5M tokens")
    print(f"{'='*70}")
    print(f"\n  {'Layer':>5s}  {'Variant':<14s} {'FVE':>7s} {'Dead%':>7s} "
          f"{'cos→KL':>8s} {'SAE→KL':>8s} {'cos>inn':>8s} {'2x ratio':>8s}")
    print(f"  {'-'*5}  {'-'*14} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        for vname, _ in VARIANTS:
            r = lr.get(vname, {})
            recon = r.get("reconstruction", {})
            abl_agg = r.get("ablation", {}).get("aggregate", {})
            inv = r.get("norm_invariance", {}).get("scale_2.0", {})

            fve = recon.get("fve", 0)
            dead = recon.get("dead_frac", 1)
            cos_kl = abl_agg.get("cos_kl_mean", 0)
            sae_kl = abl_agg.get("sae_kl_mean", 0)
            cos_wins = abl_agg.get("cos_wins_inner", 0)
            n_feats = abl_agg.get("n_features", 0)
            ratio_2x = inv.get("mean_ratio", float("nan"))

            cos_win_str = f"{cos_wins}/{n_feats}" if n_feats > 0 else "N/A"
            print(f"  {layer_idx:>5d}  {vname:<14s} {fve:>7.4f} {dead*100:>6.1f}% "
                  f"{cos_kl:>8.4f} {sae_kl:>8.4f} {cos_win_str:>8s} {ratio_2x:>8.3f}")

    # Key comparison: cosine/adaptive vs standard per layer
    print(f"\n  KEY COMPARISON vs Qwen3-8B (RMSNorm):")
    print(f"  On Qwen3-8B: cosine SAEs get +8 FVE, 3.3x alive, cos>inner 70-90%")
    print(f"  On Pythia-6.9B (LayerNorm):")
    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        std = lr.get("standard", {}).get("reconstruction", {})
        cos = lr.get("cosine", {}).get("reconstruction", {})
        ada = lr.get("adaptive_l2", {}).get("reconstruction", {})

        std_fve = std.get("fve", 0)
        cos_fve = cos.get("fve", 0)
        ada_fve = ada.get("fve", 0)

        std_dead = std.get("dead_frac", 1)
        cos_dead = cos.get("dead_frac", 1)
        ada_dead = ada.get("dead_frac", 1)

        std_alive = int((1 - std_dead) * D_SAE)
        cos_alive = int((1 - cos_dead) * D_SAE)
        ada_alive = int((1 - ada_dead) * D_SAE)

        std_abl = lr.get("standard", {}).get("ablation", {}).get("aggregate", {})
        cos_abl = lr.get("cosine", {}).get("ablation", {}).get("aggregate", {})
        ada_abl = lr.get("adaptive_l2", {}).get("ablation", {}).get("aggregate", {})

        print(f"\n    L{layer_idx}:")
        print(f"      FVE:   std={std_fve:.4f}  cos={cos_fve:.4f} ({cos_fve-std_fve:+.4f})  "
              f"ada={ada_fve:.4f} ({ada_fve-std_fve:+.4f})")
        print(f"      Alive: std={std_alive}  cos={cos_alive} "
              f"({cos_alive/max(std_alive,1):.1f}x)  "
              f"ada={ada_alive} ({ada_alive/max(std_alive,1):.1f}x)")
        print(f"      cos>inner: std={std_abl.get('cos_wins_inner',0)}"
              f"/{std_abl.get('n_features',0)}  "
              f"cos={cos_abl.get('cos_wins_inner',0)}"
              f"/{cos_abl.get('n_features',0)}  "
              f"ada={ada_abl.get('cos_wins_inner',0)}"
              f"/{ada_abl.get('n_features',0)}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
