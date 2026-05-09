"""
Experiment 31: Norm Injection — Direct RNH Test
=================================================

If magnitude is truly noise for downstream computation, injecting random
magnitude variation during training should hurt standard SAE more than cosine.

Design: 4 conditions per layer: {standard, adaptive_l2} x {clean, noised}.
Noise = per-token random scaling by Uniform(0.5, 2.0). Preserves direction,
corrupts magnitude. Evaluate ALL variants on CLEAN held-out data.

Prediction:
  - standard_noised degrades significantly vs standard_clean
    (inner-product encoder tries to use magnitude info that's now random)
  - cosine_noised barely degrades vs cosine_clean
    (cosine encoder is norm-invariant; adaptive scale can compensate)

Run on <gpu-server> GPU 1.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/exp31_norm_injection.py > experiments/exp31_output.log 2>&1 &
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
MODEL_DTYPE = torch.bfloat16
STORAGE_DTYPE = torch.float16

# --- Model ---
MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]
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

# --- Noise ---
NOISE_LOW = 0.5
NOISE_HIGH = 2.0

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
SAVE_DIR = "checkpoints/exp31"
RESULTS_PATH = "experiments/exp31_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)


# =============================================================================
# SAE Architectures
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


class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Cosine encoder with adaptive per-token scale.

    scale(x) = exp(scale_a * log(||x - b_dec||) + scale_b)
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 50, init_norm: float = None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        scale_init = math.log(init_norm) if init_norm is not None else math.log(math.sqrt(d_model))
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


# (name, class, apply_noise)
VARIANTS = [
    ("standard_clean",  BatchTopKSAE,                 False),
    ("standard_noised", BatchTopKSAE,                 True),
    ("cosine_clean",    AdaptiveCosineBatchTopKSAE,   False),
    ("cosine_noised",   AdaptiveCosineBatchTopKSAE,   True),
]


# =============================================================================
# Activation Collection
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


def collect_activations(model, tokenizer, layer_idx, n_tokens, skip_docs=0):
    label = "eval" if skip_docs > 0 else "train"
    print(f"  Collecting {label} activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)

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
# Training (with optional norm noise injection)
# =============================================================================

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_sae(name, sae, train_data, layer_idx, apply_noise=False):
    """Train an SAE. If apply_noise=True, each token gets scaled by Uniform(0.5, 2.0)."""
    noise_str = f" [NOISE: U({NOISE_LOW},{NOISE_HIGH})]" if apply_noise else " [CLEAN]"
    print(f"\n  Training {name} | L{layer_idx} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps{noise_str}")

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

        # --- Norm injection: per-token random scaling ---
        if apply_noise:
            scales = torch.empty(batch.shape[0], 1, device=DEVICE).uniform_(NOISE_LOW, NOISE_HIGH)
            batch = batch * scales

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
            if step % (LOG_EVERY * 5) == 0 or step == N_STEPS:
                print(f"    [{name:>16s}] step {step:>5d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f} | "
                      f"tok={tokens_seen/1e6:.1f}M{scale_str}")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{name}] Done in {elapsed:.1f}s")
    return log


# =============================================================================
# Evaluation (always on CLEAN data)
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data):
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
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0

    results = {
        "recon_loss_l2": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
        "dead_frac": dead_frac,
        "alive_count": alive_count,
    }
    print(f"    [{name:>16s}] FVE={results['fve']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"dead={dead_frac:.3f} ({dead_frac*100:.1f}%) | alive={alive_count}")
    return results


@torch.no_grad()
def evaluate_norm_robustness(name, sae, eval_data, scales=(0.5, 1.0, 2.0, 5.0)):
    """Evaluate how much performance changes when eval data is scaled."""
    sae.eval()
    results = {}
    for scale in scales:
        n = eval_data.shape[0]
        total_var_sum, resid_var_sum = 0.0, 0.0
        cos_sims = []
        for i in range(0, n, BATCH_SIZE):
            batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32) * scale
            x_hat, features = sae(batch)
            cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
            total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
            resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        fve = float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0
        cos_r = float(np.mean(cos_sims))
        results[f"scale_{scale}"] = {"fve": fve, "cos_recon": cos_r}
        print(f"    [{name:>16s}] eval_scale={scale}: FVE={fve:.4f} | cos={cos_r:.4f}")
    return results


def ablate_feature_kl(model, activation, feature_dir, layer_idx):
    projection = (activation @ feature_dir) * feature_dir
    model_dtype = next(model.parameters()).dtype
    x = activation.unsqueeze(0).unsqueeze(0).to(model_dtype)
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0).to(model_dtype)

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
    print(f"\n    Ablation [{name}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

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
        }
        feature_results.append(result)

        if rank < 5 or rank % 10 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos->KL={corr_cos:.3f} | inner->KL={corr_inner:.3f} | "
                  f"SAE->KL={corr_sae:.3f} | norm->KL={corr_norm:.3f}")

    if not feature_results:
        return {"features": [], "aggregate": {"n_features": 0}}

    n = len(feature_results)
    agg = {
        "n_features": n,
        "cos_kl_mean": float(np.mean([r["corr_cos_kl"] for r in feature_results])),
        "inner_kl_mean": float(np.mean([r["corr_inner_kl"] for r in feature_results])),
        "sae_kl_mean": float(np.mean([r["corr_sae_kl"] for r in feature_results])),
        "norm_kl_mean": float(np.mean([r["corr_norm_kl"] for r in feature_results])),
        "cos_wins_inner": sum(r["cos_wins_inner"] for r in feature_results),
    }

    print(f"    [{name}] Summary ({n} features): "
          f"cos->KL={agg['cos_kl_mean']:.4f} | inner->KL={agg['inner_kl_mean']:.4f} | "
          f"cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Feature overlap between clean and noised variants
# =============================================================================

@torch.no_grad()
def feature_overlap(sae_a, sae_b, eval_data, name_a, name_b):
    """Compare alive feature sets and decoder directions between two SAEs."""
    print(f"\n    Feature overlap: {name_a} vs {name_b}")

    n_probe = min(100_000, eval_data.shape[0])
    probe = eval_data[:n_probe]

    for sae_name, sae, label in [(name_a, sae_a, "A"), (name_b, sae_b, "B")]:
        sae.eval()

    alive_a = torch.zeros(D_SAE, dtype=torch.bool)
    alive_b = torch.zeros(D_SAE, dtype=torch.bool)

    for i in range(0, n_probe, BATCH_SIZE):
        batch = probe[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        _, fa = sae_a(batch)
        _, fb = sae_b(batch)
        alive_a |= (fa > 0).any(dim=0).cpu()
        alive_b |= (fb > 0).any(dim=0).cpu()

    n_a = alive_a.sum().item()
    n_b = alive_b.sum().item()
    both = (alive_a & alive_b).sum().item()
    union = (alive_a | alive_b).sum().item()
    jaccard = both / union if union > 0 else 0

    # Decoder direction similarity for shared alive features
    shared_idx = torch.where(alive_a & alive_b)[0]
    if len(shared_idx) > 0:
        dec_a = F.normalize(sae_a.W_dec[shared_idx].float(), dim=-1)
        dec_b = F.normalize(sae_b.W_dec[shared_idx].float(), dim=-1)
        cos_sims = (dec_a * dec_b).sum(dim=-1)
        dec_cos_mean = cos_sims.mean().item()
        dec_cos_median = cos_sims.median().item()
    else:
        dec_cos_mean = float("nan")
        dec_cos_median = float("nan")

    result = {
        "alive_a": n_a, "alive_b": n_b,
        "both_alive": both, "jaccard": jaccard,
        "dec_cos_mean": dec_cos_mean, "dec_cos_median": dec_cos_median,
    }

    print(f"      {name_a}: {n_a} alive | {name_b}: {n_b} alive | "
          f"Both: {both} | Jaccard: {jaccard:.3f} | "
          f"Dec cos: mean={dec_cos_mean:.3f}, median={dec_cos_median:.3f}")
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 31: Norm Injection — Direct RNH Test")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Noise range: Uniform({NOISE_LOW}, {NOISE_HIGH})")
    print(f"d_model: {D_MODEL}, d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Tokens: {N_TRAIN_TOKENS:,} train, {N_EVAL_TOKENS:,} eval")
    print(f"Steps: {N_STEPS}, Warmup: {WARMUP_STEPS}")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Prediction: standard degrades under noise, cosine does not")

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
            "experiment": "norm_injection",
            "layers": LAYERS,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "k": K,
            "noise_low": NOISE_LOW,
            "noise_high": NOISE_HIGH,
            "n_train_tokens": N_TRAIN_TOKENS,
            "n_eval_tokens": N_EVAL_TOKENS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        "layers": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Collect activations (once per layer — shared across all variants)
        train_data = collect_activations(model, tokenizer, layer_idx, N_TRAIN_TOKENS)
        eval_data = collect_activations(
            model, tokenizer, layer_idx, N_EVAL_TOKENS, skip_docs=200_000
        )

        mean_norm = train_data.float().norm(dim=-1).mean().item()
        layer_results = {"mean_norm": mean_norm}
        print(f"  Mean activation norm: {mean_norm:.2f}")

        trained_saes = {}  # Store for overlap analysis

        for vname, cls, apply_noise in VARIANTS:
            print(f"\n  --- VARIANT: {vname} (L{layer_idx}) ---")

            torch.manual_seed(SEED)
            if cls == AdaptiveCosineBatchTopKSAE:
                sae = cls(D_MODEL, D_SAE, K, init_norm=mean_norm).to(DEVICE)
                print(f"    scale_b init: log({mean_norm:.2f}) = {math.log(mean_norm):.4f}")
            else:
                sae = cls(D_MODEL, D_SAE, K).to(DEVICE)

            # Train (noise applied inside train_sae if apply_noise=True)
            train_log = train_sae(vname, sae, train_data, layer_idx, apply_noise=apply_noise)

            # Evaluate on CLEAN data (always)
            print(f"\n  Reconstruction on CLEAN eval data -- {vname}")
            recon = evaluate_reconstruction(vname, sae, eval_data)

            # Norm robustness: how does performance change at different eval scales?
            print(f"\n  Norm robustness -- {vname}")
            robustness = evaluate_norm_robustness(vname, sae, eval_data)

            # Ablation
            abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)

            result = {
                "training": train_log,
                "reconstruction_clean": recon,
                "norm_robustness": robustness,
                "ablation": abl,
                "apply_noise": apply_noise,
            }

            if hasattr(sae, "scale_a"):
                result["scale_a"] = sae.scale_a.item()
                result["scale_b_exp"] = sae.scale_b.exp().item()

            layer_results[vname] = result
            trained_saes[vname] = sae

            # Save checkpoint
            torch.save(sae.state_dict(), save_dir / f"{vname}_L{layer_idx}_final.pt")

            # Save results incrementally
            all_results["layers"][str(layer_idx)] = layer_results
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        # Feature overlap: clean vs noised for each architecture
        print(f"\n  --- FEATURE OVERLAP (L{layer_idx}) ---")
        if "standard_clean" in trained_saes and "standard_noised" in trained_saes:
            layer_results["overlap_standard"] = feature_overlap(
                trained_saes["standard_clean"], trained_saes["standard_noised"],
                eval_data, "standard_clean", "standard_noised"
            )
        if "cosine_clean" in trained_saes and "cosine_noised" in trained_saes:
            layer_results["overlap_cosine"] = feature_overlap(
                trained_saes["cosine_clean"], trained_saes["cosine_noised"],
                eval_data, "cosine_clean", "cosine_noised"
            )
        # Also compare standard_clean vs cosine_clean
        if "standard_clean" in trained_saes and "cosine_clean" in trained_saes:
            layer_results["overlap_std_vs_cos"] = feature_overlap(
                trained_saes["standard_clean"], trained_saes["cosine_clean"],
                eval_data, "standard_clean", "cosine_clean"
            )

        # Save final overlap results
        all_results["layers"][str(layer_idx)] = layer_results
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Clean up
        for sae in trained_saes.values():
            del sae
        del trained_saes, train_data, eval_data
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary Table
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Norm Injection Impact")
    print(f"{'='*70}")

    print(f"\n  {'Layer':>5s}  {'Variant':<18s} {'FVE':>7s} {'Dead%':>7s} {'Alive':>6s} "
          f"{'cos->KL':>8s} {'cos>inn':>8s}")
    print(f"  {'-'*5}  {'-'*18} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*8}")

    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        for vname, _, _ in VARIANTS:
            r = lr.get(vname, {})
            recon = r.get("reconstruction_clean", {})
            abl_agg = r.get("ablation", {}).get("aggregate", {})

            fve = recon.get("fve", 0)
            dead = recon.get("dead_frac", 1)
            alive = recon.get("alive_count", 0)
            cos_kl = abl_agg.get("cos_kl_mean", 0)
            cos_wins = abl_agg.get("cos_wins_inner", 0)
            n_feats = abl_agg.get("n_features", 0)
            cos_win_str = f"{cos_wins}/{n_feats}" if n_feats > 0 else "N/A"

            print(f"  {layer_idx:>5d}  {vname:<18s} {fve:>7.4f} {dead*100:>6.1f}% "
                  f"{alive:>6d} {cos_kl:>8.4f} {cos_win_str:>8s}")

    # Degradation summary
    print(f"\n  DEGRADATION: noised vs clean (on clean eval data)")
    print(f"  {'Layer':>5s}  {'Architecture':<12s} {'FVE_clean':>10s} {'FVE_noised':>11s} "
          f"{'Delta':>7s} {'Dead_clean':>10s} {'Dead_noised':>11s} {'Delta':>7s}")
    print(f"  {'-'*5}  {'-'*12} {'-'*10} {'-'*11} {'-'*7} {'-'*10} {'-'*11} {'-'*7}")

    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        for arch in ["standard", "cosine"]:
            clean = lr.get(f"{arch}_clean", {}).get("reconstruction_clean", {})
            noised = lr.get(f"{arch}_noised", {}).get("reconstruction_clean", {})
            fve_c = clean.get("fve", 0)
            fve_n = noised.get("fve", 0)
            dead_c = clean.get("dead_frac", 1)
            dead_n = noised.get("dead_frac", 1)
            delta_fve = fve_n - fve_c
            delta_dead = (dead_n - dead_c) * 100

            print(f"  {layer_idx:>5d}  {arch:<12s} {fve_c:>10.4f} {fve_n:>11.4f} "
                  f"{delta_fve:>+7.4f} {dead_c*100:>9.1f}% {dead_n*100:>10.1f}% "
                  f"{delta_dead:>+6.1f}pp")

    # Feature overlap summary
    print(f"\n  FEATURE OVERLAP: clean vs noised variants")
    print(f"  {'Layer':>5s}  {'Comparison':<28s} {'Jaccard':>8s} {'Dec_cos':>8s}")
    print(f"  {'-'*5}  {'-'*28} {'-'*8} {'-'*8}")

    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        for key, label in [
            ("overlap_standard", "standard: clean vs noised"),
            ("overlap_cosine",   "cosine: clean vs noised"),
            ("overlap_std_vs_cos", "standard vs cosine (clean)"),
        ]:
            ov = lr.get(key, {})
            if ov:
                print(f"  {layer_idx:>5d}  {label:<28s} {ov.get('jaccard',0):>8.3f} "
                      f"{ov.get('dec_cos_mean',0):>8.3f}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
