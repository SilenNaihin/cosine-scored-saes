"""
Experiment 26: Dictionary Size Control
=======================================

Critical control: does cosine SAE's advantage come from better features,
or just from keeping 3.3x more features alive? If we give the standard
SAE 3x the dictionary (49k vs 16k), does it match cosine's alive count,
FVE, and feature quality?

Variants (L27 only, 50M tokens):
  1. standard_16k  — d_sae=16384, k=80  (baseline, replicates exp17/22)
  2. adaptive_16k  — d_sae=16384, k=80  (current winner, replicates exp17/22)
  3. standard_49k  — d_sae=49152, k=80  (3x dict, same L0 per token)
  4. standard_49k_k240 — d_sae=49152, k=240 (3x dict, proportional k)

Key question: does standard_49k match adaptive_16k on alive features,
FVE, and SAEBench-style metrics? If yes → cosine is just better capacity
utilization. If no → cosine finds genuinely better features.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python experiments/exp26_dictionary_size.py 2>&1 | tee experiments/exp26_output.log
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
SEED = 42
LOG_EVERY = 200

# --- Checkpointing ---
CHECKPOINT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

# --- Output ---
SAVE_DIR = "checkpoints/exp26"
RESULTS_PATH = "experiments/exp26_results.json"

# --- Streaming activation buffer ---
BUFFER_TOKENS = 500_000

# --- Derived (per-variant, computed in main) ---
# N_STEPS, WARMUP_STEPS, etc. depend on d_sae (which affects nothing here)
# but we keep them global for simplicity since batch_size is constant
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)
CHECKPOINT_STEPS = [int(f * N_STEPS) for f in CHECKPOINT_FRACS]
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE


def get_config_dict():
    return {
        "model_name": MODEL_NAME, "layer": LAYER_IDX, "d_model": D_MODEL,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "n_steps": N_STEPS, "warmup_steps": WARMUP_STEPS,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
        "buffer_tokens": BUFFER_TOKENS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "variants": {
            "standard_16k": {"d_sae": 16384, "k": 80, "encoder": "standard"},
            "adaptive_16k": {"d_sae": 16384, "k": 80, "encoder": "adaptive_cosine"},
            "standard_49k": {"d_sae": 49152, "k": 80, "encoder": "standard"},
            "standard_49k_k240": {"d_sae": 49152, "k": 240, "encoder": "standard"},
        },
    }


# =============================================================================
# SAE Architectures
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
# Streaming Activation Collection (from exp17)
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
    def __init__(self, model, tokenizer, layer_idx, buffer_tokens=BUFFER_TOKENS):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.buffer_tokens = buffer_tokens
        self.buffer = None
        self._text_iter = None
        self._init_dataset()

    def _init_dataset(self):
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
    return result


# =============================================================================
# Training
# =============================================================================

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_sae_streaming(name, sae, stream, save_dir, d_sae, k):
    tag = f"{name}/L{LAYER_IDX}"
    print(f"\n  Training {tag} | d_sae={d_sae}, k={k}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps (streaming)")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    sae.train()
    log = []
    checkpoint_log = {}
    t0 = time.time()
    global_step = 0
    next_checkpoint_idx = 0

    while global_step < N_STEPS:
        n_filled = stream.fill_buffer()
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
                    entry["scale_a"] = sae.scale_a.item()
                    entry["scale_b"] = sae.scale_b.exp().item()
                    scale_str = f" | a={sae.scale_a.item():.4f} b={sae.scale_b.exp().item():.1f}"
                else:
                    scale_str = ""

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag:>25s}] step {global_step:>6d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f}{scale_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | {tok_per_sec/1e3:.0f}k tok/s | "
                      f"ETA {eta_sec/60:.0f}m")

            if (next_checkpoint_idx < len(CHECKPOINT_STEPS) and
                    global_step >= CHECKPOINT_STEPS[next_checkpoint_idx]):
                frac = CHECKPOINT_FRACS[next_checkpoint_idx]
                ckpt_path = save_dir / f"{name}_L{LAYER_IDX}_step{global_step}.pt"
                torch.save(sae.state_dict(), ckpt_path)
                with torch.no_grad():
                    snap_dead = dead
                    snap_fve = fve
                checkpoint_log[f"{frac:.0%}"] = {
                    "step": global_step, "tokens": global_step * BATCH_SIZE,
                    "fve": snap_fve, "dead_frac": snap_dead,
                }
                if hasattr(sae, "scale_a"):
                    checkpoint_log[f"{frac:.0%}"]["scale_a"] = sae.scale_a.item()
                print(f"    [{tag}] Checkpoint saved at {frac:.0%} ({global_step} steps)")
                next_checkpoint_idx += 1

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    torch.save(sae.state_dict(), save_dir / f"{name}_L{LAYER_IDX}_final.pt")
    return log, checkpoint_log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data):
    tag = f"{name}/L{LAYER_IDX}"
    sae.eval()
    n = eval_data.shape[0]
    d_sae = sae.d_sae
    eval_batch = min(BATCH_SIZE, 2048)  # smaller batches for 49k SAEs
    recon_losses, cos_sims, l0s = [], [], []
    total_var_sum, resid_var_sum = 0.0, 0.0
    dead_counts = None

    for i in range(0, n, eval_batch):
        batch = eval_data[i:i+eval_batch].to(DEVICE, dtype=torch.float32)
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
        del batch, x_hat, features
        torch.cuda.empty_cache()

    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0

    results = {
        "recon_loss_l2": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
        "dead_frac": dead_frac,
        "alive_count": alive_count,
        "d_sae": d_sae,
    }
    print(f"    [{tag}] L2={results['recon_loss_l2']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"FVE={results['fve']:.4f} | dead={dead_frac:.3f} | "
          f"alive={alive_count}/{d_sae}")
    return results


@torch.no_grad()
def test_norm_invariance(name, sae, eval_data, scales=(0.5, 2.0, 5.0)):
    tag = f"{name}/L{LAYER_IDX}"
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


def evaluate_ablation(name, model, sae, eval_data):
    tag = f"{name}/L{LAYER_IDX}"
    d_sae = sae.d_sae
    print(f"\n    Ablation [{tag}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    n_probe = min(200_000, eval_data.shape[0])
    probe = eval_data[:n_probe]
    eval_batch = min(BATCH_SIZE, 2048)
    all_feats = []
    for i in range(0, n_probe, eval_batch):
        batch = probe[i:i+eval_batch].to(DEVICE, dtype=torch.float32)
        _, f = sae(batch)
        all_feats.append(f.detach().cpu())
        del batch, f
        torch.cuda.empty_cache()
    all_feats = torch.cat(all_feats, dim=0)

    freq = (all_feats > 0).float().mean(dim=0)
    alive_mask = freq > 0
    n_alive = alive_mask.sum().item()
    print(f"    [{tag}] {n_alive} alive features (of {d_sae})")

    n_to_select = min(N_ABLATION_FEATURES, n_alive)
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

        if rank < 7 or rank % 20 == 0:
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
          f"cos→KL={agg['cos_kl_mean']:.4f} | inner→KL={agg['inner_kl_mean']:.4f} | "
          f"SAE→KL={agg['sae_kl_mean']:.4f} | cos>inner: {agg['cos_wins_inner']}/{n} | "
          f"SAE>inner: {agg['sae_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Main
# =============================================================================

# Variant definitions: (name, SAE class, d_sae, k)
VARIANTS = [
    ("standard_16k",      BatchTopKSAE,              16384, 80),
    ("adaptive_16k",      AdaptiveCosineBatchTopKSAE, 16384, 80),
    ("standard_49k",      BatchTopKSAE,              49152, 80),
    ("standard_49k_k240", BatchTopKSAE,              49152, 240),
]


def main():
    print("Experiment 26: Dictionary Size Control")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}, Layer: {LAYER_IDX}")
    print(f"Training: {N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps), LR={LR}")
    print(f"Eval: {N_EVAL_TOKENS:,} tokens, Ablation: {N_ABLATION_FEATURES}×{N_ABLATION_SAMPLES}")
    print(f"Variants:")
    for vname, cls, d_sae, k in VARIANTS:
        print(f"  {vname}: d_sae={d_sae}, k={k}, encoder={cls.__name__}")

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

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing results (resume after crash) ----
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        existing = list(all_results.get("variants", {}).keys())
        print(f"  Loaded existing results for variants: {existing}")
    else:
        all_results = {"config": get_config_dict(), "variants": {}}

    # ---- Collect eval data once (shared across all variants) ----
    eval_data = collect_eval_data(model, tokenizer, LAYER_IDX, N_EVAL_TOKENS)

    # ---- Train and evaluate each variant ----
    for vname, cls, d_sae, k in VARIANTS:
        if vname in all_results.get("variants", {}):
            print(f"\n  {vname} already complete, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  VARIANT: {vname} (d_sae={d_sae}, k={k})")
        print(f"{'='*70}")

        # Fresh stream per variant
        stream = ActivationStream(model, tokenizer, LAYER_IDX)

        torch.manual_seed(SEED)
        sae = cls(D_MODEL, d_sae, k).to(DEVICE)

        # Train
        train_log, ckpt_log = train_sae_streaming(
            vname, sae, stream, save_dir, d_sae, k
        )

        # Evaluate
        print(f"\n  Evaluation — {vname}")
        recon = evaluate_reconstruction(vname, sae, eval_data)
        inv = test_norm_invariance(vname, sae, eval_data)
        abl = evaluate_ablation(vname, model, sae, eval_data)

        variant_result = {
            "d_sae": d_sae,
            "k": k,
            "encoder": cls.__name__,
            "training": train_log,
            "checkpoints": ckpt_log,
            "reconstruction": recon,
            "norm_invariance": inv,
            "ablation": abl,
        }
        if hasattr(sae, "scale_a"):
            variant_result["scale_a_final"] = sae.scale_a.item()
            variant_result["scale_b_final"] = sae.scale_b.item()

        all_results["variants"][vname] = variant_result

        # Save after each variant
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to {RESULTS_PATH}")

        del sae, stream
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Variant':<22s} | {'d_sae':>6s} | {'k':>4s} | {'FVE':>6s} | "
          f"{'Dead%':>6s} | {'Alive':>6s} | {'L0':>5s} | {'cos>inn':>8s} | {'SAE→KL':>7s}")
    print(f"  {'-'*22}-+-{'-'*6}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
          f"{'-'*5}-+-{'-'*8}-+-{'-'*7}")

    for vname, _, d_sae, k in VARIANTS:
        vr = all_results["variants"].get(vname, {})
        r = vr.get("reconstruction", {})
        a = vr.get("ablation", {}).get("aggregate", {})
        if r:
            alive = r.get("alive_count", int((1 - r.get("dead_frac", 1)) * d_sae))
            cos_inn = f"{a.get('cos_wins_inner', '?')}/{a.get('n_features', '?')}" if a else "—"
            sae_kl = f"{a.get('sae_kl_mean', 0):.3f}" if a else "—"
            print(f"  {vname:<22s} | {d_sae:>6d} | {k:>4d} | "
                  f"{r['fve']:.4f} | {r['dead_frac']*100:>5.1f}% | {alive:>6d} | "
                  f"{r['l0']:>5.0f} | {cos_inn:>8s} | {sae_kl:>7s}")

    # Key comparison
    print(f"\n  Key comparison (addressing reviewer objection):")
    std_16k = all_results["variants"].get("standard_16k", {}).get("reconstruction", {})
    adp_16k = all_results["variants"].get("adaptive_16k", {}).get("reconstruction", {})
    std_49k = all_results["variants"].get("standard_49k", {}).get("reconstruction", {})
    std_49k_240 = all_results["variants"].get("standard_49k_k240", {}).get("reconstruction", {})

    if adp_16k and std_49k:
        print(f"    adaptive_16k alive: {adp_16k.get('alive_count', '?')}")
        print(f"    standard_49k alive: {std_49k.get('alive_count', '?')}")
        adp_alive = adp_16k.get('alive_count', 0)
        std3_alive = std_49k.get('alive_count', 0)
        if adp_alive > 0:
            ratio = std3_alive / adp_alive
            print(f"    Ratio: {ratio:.2f}x (>1 means 3x dict gives more alive)")
            print(f"    FVE: adaptive_16k={adp_16k['fve']:.4f} vs standard_49k={std_49k['fve']:.4f}")

    if std_49k_240:
        print(f"    standard_49k_k240 alive: {std_49k_240.get('alive_count', '?')}")
        print(f"    standard_49k_k240 FVE: {std_49k_240['fve']:.4f}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
