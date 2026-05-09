"""
Experiment 34b: Multi-Seed Cosine Variants with sqrt(d) Init (50M tokens)
==========================================================================

Correction to exp34: norm-adaptive init (scale_b = log(407)) hurts at 50M
tokens — scale_a gets stuck at 0.01 instead of learning 0.21. The optimizer
needs to climb from sqrt(d) init to discover the a≈0.2 regime.

Standard already has 3 solid seeds from exp34. This runs only the cosine
variants with sqrt(d) init (matching exp17's configuration).

2 cosine variants × 3 seeds = 6 runs at 50M tokens.
Standard seeds reused from exp34 results.

Estimated runtime: ~6 hours on H100 (6 × ~1 hour per variant).

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 experiments/exp34b_multi_seed_sqrtd.py 2>&1 | tee experiments/exp34b_output.log
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
LAYER_IDX = 27
D_MODEL = 4096

# --- SAE ---
D_SAE = 16384
K = 80

# --- Data ---
N_TRAIN_TOKENS = 50_000_000
N_EVAL_TOKENS = 1_000_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR = 3e-4
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
LOG_EVERY = 200

# --- Seeds ---
SEEDS = [42, 123, 456]

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

# --- Output ---
SAVE_DIR = "checkpoints/exp34b_sqrtd"
RESULTS_PATH = "experiments/exp34b_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)

# --- Streaming buffer ---
BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE


def get_config_dict():
    return {
        "model_name": MODEL_NAME, "layer": LAYER_IDX, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seeds": SEEDS,
        "n_steps": N_STEPS, "warmup_steps": WARMUP_STEPS,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
    }


# =============================================================================
# SAE Architectures
# =============================================================================

class BatchTopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int = 80, init_norm: float = None):
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
        x_hat = self.decode(f)
        return x_hat, f


class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Cosine encoder with adaptive per-token scale + norm-adaptive init."""

    def __init__(self, d_model: int, d_sae: int, k: int = 80, init_norm: float = None):
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
        x_hat = self.decode(f)
        return x_hat, f


class PerFeatureAdaptiveCosineSAE(nn.Module):
    """Per-feature adaptive-scale cosine encoder + norm-adaptive init."""

    def __init__(self, d_model: int, d_sae: int, k: int = 80, init_norm: float = None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        scale_init = math.log(init_norm) if init_norm is not None else math.log(math.sqrt(d_model))
        self.scale_a = nn.Parameter(torch.zeros(d_sae))
        self.scale_b = nn.Parameter(torch.full((d_sae,), scale_init))
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
        log_norm = torch.log(input_norm)
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
            return self._batch_topk(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


# =============================================================================
# Streaming Activation Collection (from exp30)
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


class ActivationStream:
    def __init__(self, model, tokenizer, layer_idx, seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.seed = seed
        self.buffer = None
        self._text_iter = None
        self._init_dataset()

    def _init_dataset(self):
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT",
            split="train", streaming=True,
        )
        # Shuffle with seed for reproducibility across runs with same seed
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        self._text_iter = iter(ds)

    def fill_buffer(self):
        all_acts = []
        tokens_collected = 0
        while tokens_collected < BUFFER_TOKENS:
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
        self.buffer = torch.cat(all_acts, dim=0)[:BUFFER_TOKENS]
        perm = torch.randperm(self.buffer.shape[0])
        self.buffer = self.buffer[perm]
        return self.buffer.shape[0]

    def get_batch(self, batch_idx):
        start = (batch_idx * BATCH_SIZE) % self.buffer.shape[0]
        end = start + BATCH_SIZE
        if end > self.buffer.shape[0]:
            idx = torch.cat([
                torch.arange(start, self.buffer.shape[0]),
                torch.arange(0, end - self.buffer.shape[0]),
            ])
        else:
            idx = torch.arange(start, end)
        return self.buffer[idx].to(DEVICE, dtype=torch.float32)


def collect_eval_data(model, tokenizer, layer_idx, n_tokens):
    print(f"  Collecting eval activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= 500_000:
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
    return result, norms.mean().item()


# =============================================================================
# Training
# =============================================================================

def make_lr_schedule(n_steps, warmup_steps):
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule


def train_sae_streaming(name, sae, stream, save_dir):
    tag = f"{name}/L{LAYER_IDX}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    schedule_fn = make_lr_schedule(N_STEPS, WARMUP_STEPS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    sae.train()
    log = []
    t0 = time.time()
    global_step = 0

    while global_step < N_STEPS:
        stream.fill_buffer()
        steps_in_buffer = min(BUFFER_BATCHES, N_STEPS - global_step)

        for buf_step in range(steps_in_buffer):
            batch = stream.get_batch(buf_step)
            x_hat, features = sae(batch)
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            optimizer.zero_grad(set_to_none=True)
            recon_loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % LOG_EVERY == 0 or global_step == N_STEPS:
                with torch.no_grad():
                    l0 = (features != 0).float().sum(dim=-1).mean().item()
                    total_var = torch.var(batch, dim=0, unbiased=False).sum()
                    resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                    fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                    cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                    dead = (features.sum(dim=0) == 0).float().mean().item()

                entry = {
                    "step": global_step, "recon_loss": recon_loss.item(),
                    "l0": l0, "fve": fve, "cos_recon": cos_r,
                    "dead_frac": dead, "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": global_step * BATCH_SIZE,
                }

                if hasattr(sae, "scale_a"):
                    if sae.scale_a.dim() == 0:
                        entry["scale_a"] = sae.scale_a.item()
                        entry["scale_b"] = sae.scale_b.exp().item()
                        scale_str = f" | a={sae.scale_a.item():.4f}"
                    else:
                        entry["scale_a_mean"] = sae.scale_a.mean().item()
                        entry["scale_a_median"] = sae.scale_a.median().item()
                        near_zero = (sae.scale_a.abs() < 0.05).float().mean().item()
                        entry["near_zero_frac"] = near_zero
                        scale_str = f" | a_mean={sae.scale_a.mean().item():.4f}"
                else:
                    scale_str = ""

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag:>30s}] step {global_step:>5d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.1f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | dead={dead:.3f}{scale_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | ETA {eta_sec/60:.0f}m")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    torch.save(sae.state_dict(), save_dir / f"{name}_final.pt")
    return log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data):
    tag = f"{name}/L{LAYER_IDX}"
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
    print(f"    [{tag}] FVE={results['fve']:.4f} | dead={dead_frac:.3f} | "
          f"alive={alive_count}/{D_SAE} | L0={results['l0']:.0f}")
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


def evaluate_ablation(name, model, sae, eval_data):
    tag = f"{name}/L{LAYER_IDX}"
    print(f"\n    Ablation [{tag}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

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
    print(f"    [{tag}] {n_alive} alive features (of {D_SAE})")

    n_to_select = min(N_ABLATION_FEATURES, n_alive)
    if n_to_select == 0:
        return {"n_features": 0}
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
            kl = ablate_feature_kl(model, x, feat_dir, LAYER_IDX)
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
        inner_arr = np.array(inner_v)
        sae_arr = np.array(sae_v)
        norm_arr = np.array(norm_v)

        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]

        result = {
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
        }
        feature_results.append(result)

        if rank < 5 or rank % 25 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f}")

    if not feature_results:
        return {"n_features": 0}

    n = len(feature_results)
    agg = {
        "n_features": n,
        "cos_kl_mean": float(np.mean([r["corr_cos_kl"] for r in feature_results])),
        "inner_kl_mean": float(np.mean([r["corr_inner_kl"] for r in feature_results])),
        "sae_kl_mean": float(np.mean([r["corr_sae_kl"] for r in feature_results])),
        "norm_kl_mean": float(np.mean([r["corr_norm_kl"] for r in feature_results])),
        "cos_wins_inner": sum(r["cos_wins_inner"] for r in feature_results),
    }

    print(f"    [{tag}] Summary ({n} features): "
          f"cos→KL={agg['cos_kl_mean']:.4f} | SAE→KL={agg['sae_kl_mean']:.4f} | "
          f"cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Main
# =============================================================================

# (name, class, uses_norm_adaptive_init)
# sqrt(d) init for cosine variants — matching exp17 config
# Standard omitted: 3 seeds already in exp34_results.json
VARIANTS = [
    ("adaptive_l2",    AdaptiveCosineBatchTopKSAE,    False),
    ("perfeature_l2",  PerFeatureAdaptiveCosineSAE,   False),
]


def main():
    print("Experiment 34b: Multi-Seed Cosine Variants — sqrt(d) Init (50M tokens)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}, Layer: {LAYER_IDX}")
    print(f"d_sae={D_SAE}, k={K}, lr={LR}, {N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")
    print(f"Seeds: {SEEDS}")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Total runs: {len(VARIANTS) * len(SEEDS)}")
    print(f"Estimated time: ~{len(VARIANTS) * len(SEEDS)} hours on H100")

    # ---- Load model ----
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing results (resume support) ----
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        existing = list(all_results.get("runs", {}).keys())
        print(f"  Loaded existing results: {existing}")
    else:
        all_results = {"config": get_config_dict(), "runs": {}}

    # ---- Collect eval data once (seed-independent) ----
    eval_data, mean_norm = collect_eval_data(model, tokenizer, LAYER_IDX, N_EVAL_TOKENS)
    print(f"  Mean activation norm at L{LAYER_IDX}: {mean_norm:.1f}")
    print(f"  Using sqrt(d) init: scale_b = log({math.sqrt(D_MODEL):.1f}) = {math.log(math.sqrt(D_MODEL)):.4f}")
    print(f"  (NOT norm-adaptive — exp34 showed norm-adaptive hurts at 50M tokens)")
    all_results["config"]["mean_norm_L27"] = mean_norm

    # ---- Run all combinations ----
    for seed in SEEDS:
        for vname, vcls, use_norm_init in VARIANTS:
            run_name = f"{vname}_seed{seed}"

            if run_name in all_results.get("runs", {}):
                print(f"\n  {run_name} already complete, skipping")
                continue

            print(f"\n{'='*70}")
            print(f"  RUN: {run_name} (variant={vname}, seed={seed})")
            print(f"{'='*70}")

            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create SAE with norm-adaptive init if applicable
            if use_norm_init:
                sae = vcls(D_MODEL, D_SAE, K, init_norm=mean_norm).to(DEVICE)
                print(f"    Norm-adaptive init: scale_b = log({mean_norm:.1f}) = {math.log(mean_norm):.4f}")
            else:
                sae = vcls(D_MODEL, D_SAE, K).to(DEVICE)

            # Create stream with this seed
            stream = ActivationStream(model, tokenizer, LAYER_IDX, seed=seed)

            # Train
            train_log = train_sae_streaming(run_name, sae, stream, save_dir)

            # Evaluate
            print(f"\n  Evaluation — {run_name}")
            recon = evaluate_reconstruction(run_name, sae, eval_data)
            abl = evaluate_ablation(run_name, model, sae, eval_data)

            run_result = {
                "variant": vname,
                "seed": seed,
                "training": train_log,
                "reconstruction": recon,
                "ablation": abl,
            }
            if hasattr(sae, "scale_a"):
                if sae.scale_a.dim() == 0:
                    run_result["scale_a_final"] = sae.scale_a.item()
                else:
                    run_result["scale_a_mean"] = sae.scale_a.mean().item()
                    run_result["scale_a_median"] = sae.scale_a.median().item()
                    near_zero = (sae.scale_a.abs() < 0.05).float().mean().item()
                    run_result["near_zero_frac"] = near_zero

            all_results["runs"][run_name] = run_result

            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Results saved to {RESULTS_PATH}")

            del sae, stream
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Summary: mean ± std per variant ----
    # Load standard results from exp34 (already completed with 3 seeds)
    exp34_path = "experiments/exp34_results.json"
    if os.path.exists(exp34_path):
        with open(exp34_path) as f:
            exp34 = json.load(f)
        for rname, rdata in exp34.get("runs", {}).items():
            if rname.startswith("standard_"):
                all_results["runs"][rname] = rdata
        print(f"\n  Loaded standard seeds from exp34")

    print(f"\n{'='*70}")
    print("  MULTI-SEED SUMMARY (cosine: sqrt(d) init, standard: from exp34)")
    print(f"{'='*70}")

    summary = {}

    all_variant_names = [("standard", None, None)] + list(VARIANTS)
    for vname, _, _ in all_variant_names:
        fves, deads, alives, sae_kls, cos_inns = [], [], [], [], []
        for seed in SEEDS:
            run = all_results["runs"].get(f"{vname}_seed{seed}", {})
            r = run.get("reconstruction", {})
            a = run.get("ablation", {}).get("aggregate", {})
            if r:
                fves.append(r["fve"])
                deads.append(r["dead_frac"])
                alives.append(r["alive_count"])
            if a:
                sae_kls.append(a.get("sae_kl_mean", 0))
                cos_inns.append(a.get("cos_wins_inner", 0))

        if fves:
            s = {
                "fve_mean": float(np.mean(fves)), "fve_std": float(np.std(fves)),
                "dead_mean": float(np.mean(deads)), "dead_std": float(np.std(deads)),
                "alive_mean": float(np.mean(alives)), "alive_std": float(np.std(alives)),
            }
            if sae_kls:
                s["sae_kl_mean"] = float(np.mean(sae_kls))
                s["sae_kl_std"] = float(np.std(sae_kls))
            if cos_inns:
                s["cos_wins_mean"] = float(np.mean(cos_inns))
            summary[vname] = s

            print(f"\n  {vname}:")
            print(f"    FVE:   {s['fve_mean']:.4f} ± {s['fve_std']:.4f}")
            print(f"    Dead:  {s['dead_mean']*100:.1f}% ± {s['dead_std']*100:.1f}%")
            print(f"    Alive: {s['alive_mean']:.0f} ± {s['alive_std']:.0f}")
            if "sae_kl_mean" in s:
                print(f"    SAE→KL: {s['sae_kl_mean']:.4f} ± {s['sae_kl_std']:.4f}")

    # ---- Statistical significance ----
    if "standard" in summary and "adaptive_l2" in summary:
        std = summary["standard"]
        adp = summary["adaptive_l2"]
        fve_gap = adp["fve_mean"] - std["fve_mean"]
        fve_pooled_std = (std["fve_std"]**2 + adp["fve_std"]**2)**0.5
        z_fve = fve_gap / fve_pooled_std if fve_pooled_std > 0 else float("inf")

        dead_gap = std["dead_mean"] - adp["dead_mean"]
        dead_pooled_std = (std["dead_std"]**2 + adp["dead_std"]**2)**0.5
        z_dead = dead_gap / dead_pooled_std if dead_pooled_std > 0 else float("inf")

        print(f"\n  STATISTICAL SIGNIFICANCE (adaptive_l2 vs standard):")
        print(f"    FVE gap: {fve_gap:+.4f} ({z_fve:.1f}σ)")
        print(f"    Dead gap: {dead_gap:+.4f} ({z_dead:.1f}σ)")
        if abs(z_fve) > 2:
            print(f"    FVE: SIGNIFICANT (>{2}σ) — cosine advantage is real")
        else:
            print(f"    FVE: NOT significant (<2σ) — gap may be noise")
        if abs(z_dead) > 2:
            print(f"    Dead: SIGNIFICANT (>{2}σ) — alive feature advantage is real")
        else:
            print(f"    Dead: NOT significant (<2σ) — gap may be noise")

    if "standard" in summary and "perfeature_l2" in summary:
        std = summary["standard"]
        pf = summary["perfeature_l2"]
        fve_gap = pf["fve_mean"] - std["fve_mean"]
        print(f"\n  perfeature_l2 vs standard FVE gap: {fve_gap:+.4f}")

    all_results["summary"] = summary

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
