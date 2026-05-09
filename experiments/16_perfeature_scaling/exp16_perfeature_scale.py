"""
Experiment 16: Per-Feature Adaptive Scaling
============================================
Exp12 learned a single global a=0.1 — every feature gets the same magnitude
sensitivity. But this is an average: a keyword detector shouldn't care about
norm, while a norm-outlier detector genuinely needs it.

Per-feature scaling lets each feature i learn its own a_i:
    scale_i(x) = exp(a_i * log(||x||) + b_i)

The distribution of learned a_i values tells us:
  - What fraction of features are purely directional (a_i ≈ 0)?
  - Which features use magnitude, and what do they detect?
  - Is there a bimodal split or a smooth distribution?

Two variants:
  1. perfeature_l2     — L2 loss (compare to adaptive_l2 from exp12)
  2. perfeature_cosloss — cosine loss (control: a_i should all stay ~0)

Usage:
    ssh <server>     cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python experiments/exp16_perfeature_scale.py
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

MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]
D_MODEL = 4096

D_SAE = 16384
K = 80

N_TRAIN_TOKENS = 5_000_000
N_EVAL_TOKENS = 500_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

LR = 3e-4
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 50

N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 50

SAVE_DIR = "checkpoints/exp16"
RESULTS_PATH = "experiments/exp16_results.json"

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
        "n_steps": N_STEPS,
    }


# =============================================================================
# SAE Architectures
# =============================================================================

class BatchTopKSAE(nn.Module):
    """Standard BatchTopK SAE (baseline, for comparison loading)."""

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


class PerFeatureAdaptiveCosineSAE(nn.Module):
    """BatchTopK SAE with per-feature adaptive-scale cosine encoder.

    Each feature i has its own magnitude sensitivity:
        scale_i(x) = exp(a_i * log(||x - b_dec||) + b_i)
        pre_act_i = scale_i * cos_sim(x, w_enc_i) + b_enc_i

    This is a strict generalization of exp12's global adaptive SAE
    (setting all a_i = constant recovers it).

    The distribution of learned a_i values reveals which features
    are purely directional (a_i ≈ 0) vs magnitude-sensitive (a_i >> 0).
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

        # Per-feature adaptive scale: scale_i = exp(a_i * log(||x||) + b_i)
        self.scale_a = nn.Parameter(torch.zeros(d_sae))           # [d_sae], init 0 = global cosine
        self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))  # [d_sae]

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
        # Cosine similarity: [batch, d_sae]
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T

        # Per-feature adaptive scale: [batch, d_sae]
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, 1]
        log_norm = torch.log(input_norm)  # [batch, 1]
        # scale_a: [d_sae], log_norm: [batch, 1] -> broadcast to [batch, d_sae]
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)  # [batch, d_sae]

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


# =============================================================================
# Data Collection (same as exp10)
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


def train_sae(name, sae, train_data, layer_idx, loss_type="l2"):
    n_tokens = train_data.shape[0]
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, loss={loss_type}, "
          f"{n_tokens:,} tokens, {N_STEPS} steps")

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

            # Per-feature scale stats
            a_vals = sae.scale_a.detach()
            a_mean = a_vals.mean().item()
            a_std = a_vals.std().item()
            a_min = a_vals.min().item()
            a_max = a_vals.max().item()
            b_mean = sae.scale_b.exp().detach().mean().item()

            entry = {
                "step": step, "recon_loss": recon_loss.item(),
                "l0": l0, "fve": fve, "cos_recon": cos_r,
                "dead_frac": dead, "lr": scheduler.get_last_lr()[0],
                "a_mean": a_mean, "a_std": a_std,
                "a_min": a_min, "a_max": a_max, "b_mean": b_mean,
            }
            log.append(entry)

            print(f"    [{tag:>20s}] step {step:>5d}/{N_STEPS} | "
                  f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                  f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f} | "
                  f"a={a_mean:.3f}+/-{a_std:.3f} [{a_min:.3f},{a_max:.3f}] "
                  f"b={b_mean:.1f} | {time.time()-t0:.0f}s")

    sae.eval()
    print(f"    [{tag}] Done in {time.time()-t0:.1f}s")
    return log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data, layer_idx):
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
        print(f"    [{tag}] scale={scale}: ratio={mean_ratio:.3f} | "
              f"agree={agreement:.3f} | cos={cos:.4f}")
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

        # Also record this feature's learned a_i value
        a_i = sae.scale_a[fi].item() if hasattr(sae, 'scale_a') and sae.scale_a.dim() > 0 else None

        feature_results.append({
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "cos_wins_sae": bool(abs(corr_cos) > abs(corr_sae)),
            "sae_wins_inner": bool(abs(corr_sae) > abs(corr_inner)),
            "scale_a_i": a_i,
        })

        if rank < 5 or rank % 10 == 0:
            a_str = f" | a_i={a_i:.3f}" if a_i is not None else ""
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f} | norm→KL={corr_norm:.3f}{a_str}")

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


@torch.no_grad()
def analyze_scale_distribution(name, sae, layer_idx):
    """Analyze the distribution of per-feature scale_a values."""
    tag = f"{name}/L{layer_idx}"
    a_vals = sae.scale_a.detach().cpu().numpy()
    b_vals = sae.scale_b.exp().detach().cpu().numpy()

    # Compute distribution stats
    stats = {
        "a_mean": float(a_vals.mean()),
        "a_std": float(a_vals.std()),
        "a_median": float(np.median(a_vals)),
        "a_min": float(a_vals.min()),
        "a_max": float(a_vals.max()),
        "a_q25": float(np.percentile(a_vals, 25)),
        "a_q75": float(np.percentile(a_vals, 75)),
        "a_q05": float(np.percentile(a_vals, 5)),
        "a_q95": float(np.percentile(a_vals, 95)),
        "b_mean": float(b_vals.mean()),
        "b_std": float(b_vals.std()),
        # Fraction of features in different a ranges
        "frac_near_zero": float((np.abs(a_vals) < 0.05).mean()),     # |a| < 0.05
        "frac_low": float(((a_vals >= 0.05) & (a_vals < 0.2)).mean()),  # 0.05-0.2
        "frac_medium": float(((a_vals >= 0.2) & (a_vals < 0.5)).mean()),  # 0.2-0.5
        "frac_high": float((a_vals >= 0.5).mean()),                    # > 0.5
        "frac_negative": float((a_vals < -0.05).mean()),               # < -0.05
        # Full distribution as histogram for analysis
        "a_histogram": {
            "bins": [float(x) for x in np.linspace(a_vals.min() - 0.01, a_vals.max() + 0.01, 51)],
            "counts": [int(x) for x in np.histogram(a_vals, bins=50)[0]],
        },
        # All a values (for detailed analysis)
        "a_values": [float(x) for x in a_vals],
    }

    print(f"    [{tag}] scale_a distribution:")
    print(f"      mean={stats['a_mean']:.4f} +/- {stats['a_std']:.4f}")
    print(f"      median={stats['a_median']:.4f} [{stats['a_q05']:.4f}, {stats['a_q95']:.4f}] (5-95%)")
    print(f"      near-zero (|a|<0.05): {stats['frac_near_zero']*100:.1f}%")
    print(f"      low (0.05-0.2):       {stats['frac_low']*100:.1f}%")
    print(f"      medium (0.2-0.5):     {stats['frac_medium']*100:.1f}%")
    print(f"      high (>0.5):          {stats['frac_high']*100:.1f}%")
    print(f"      negative (<-0.05):    {stats['frac_negative']*100:.1f}%")

    return stats


# =============================================================================
# Per-Layer Runner
# =============================================================================

def run_layer(model, tokenizer, texts, layer_idx, save_dir):
    print(f"\n{'='*70}")
    print(f"  LAYER {layer_idx}")
    print(f"{'='*70}")

    n_total = N_TRAIN_TOKENS + N_EVAL_TOKENS
    all_acts = texts_to_activations(model, tokenizer, texts, layer_idx, n_total)
    train_data = all_acts[:N_TRAIN_TOKENS]
    eval_data = all_acts[N_TRAIN_TOKENS:N_TRAIN_TOKENS + N_EVAL_TOKENS]
    print(f"  Split: train={train_data.shape[0]:,}, eval={eval_data.shape[0]:,}")
    del all_acts

    variants = [
        ("perfeature_l2",      PerFeatureAdaptiveCosineSAE, "l2"),
        ("perfeature_cosloss", PerFeatureAdaptiveCosineSAE, "cosine"),
    ]

    saes = {}
    logs = {}
    for vname, cls, loss_type in variants:
        torch.manual_seed(SEED)
        sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
        logs[vname] = train_sae(vname, sae, train_data, layer_idx, loss_type=loss_type)
        saes[vname] = sae
        gc.collect()
        torch.cuda.empty_cache()

    del train_data
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Evaluation — Layer {layer_idx}")

    results = {}
    for vname, sae in saes.items():
        recon = evaluate_reconstruction(vname, sae, eval_data, layer_idx)
        inv = test_norm_invariance(vname, sae, eval_data, layer_idx)
        abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)
        scale_dist = analyze_scale_distribution(vname, sae, layer_idx)
        torch.save(sae.state_dict(), save_dir / f"{vname}_L{layer_idx}.pt")
        results[vname] = {
            "training": logs[vname],
            "reconstruction": recon,
            "norm_invariance": inv,
            "ablation": abl,
            "scale_distribution": scale_dist,
        }

    del saes, eval_data
    gc.collect()
    torch.cuda.empty_cache()
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 16: Per-Feature Adaptive Scaling")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Per layer: {N_TRAIN_TOKENS:,} train tokens ({N_STEPS} steps), "
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

    print("\nCollecting FineWeb texts...")
    n_needed = N_TRAIN_TOKENS + N_EVAL_TOKENS
    texts = collect_texts(n_needed)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        existing = list(all_results.get("layers", {}).keys())
        print(f"  Loaded existing results for layers: {existing}")
    else:
        all_results = {"config": get_config_dict(), "layers": {}}

    for layer_idx in LAYERS:
        layer_result = run_layer(model, tokenizer, texts, layer_idx, save_dir)
        if str(layer_idx) not in all_results["layers"]:
            all_results["layers"][str(layer_idx)] = {}
        all_results["layers"][str(layer_idx)].update(layer_result)

        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to {RESULTS_PATH}")

    # Cross-layer summary
    print(f"\n{'='*70}")
    print("  CROSS-LAYER SUMMARY")
    print(f"{'='*70}")

    for vname in ["perfeature_l2", "perfeature_cosloss"]:
        print(f"\n  {vname}:")
        for li in LAYERS:
            lr = all_results["layers"].get(str(li), {}).get(vname, {})
            r = lr.get("reconstruction", {})
            sd = lr.get("scale_distribution", {})
            a = lr.get("ablation", {}).get("aggregate", {})
            if r and sd:
                print(f"    L{li}: FVE={r.get('fve',0):.4f} cos={r.get('cos_recon',0):.4f} | "
                      f"a_mean={sd.get('a_mean',0):.4f}+/-{sd.get('a_std',0):.4f} | "
                      f"near_zero={sd.get('frac_near_zero',0)*100:.0f}% high={sd.get('frac_high',0)*100:.0f}% | "
                      f"cos→KL={a.get('cos_kl_mean',0):.4f}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
