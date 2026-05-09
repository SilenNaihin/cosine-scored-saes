"""
Experiment 41: Optimal 500M Token SAE Training
===============================================

Exp37 found that group_G4 (4 group-<author>el scale_a params) is the best
all-rounder at 50M tokens on Gemma-2-2b — highest KL/CE/SP-all/SP-5,
strong RAVEL. This experiment validates group_G4 at production scale
(500M tokens) on the paper's primary model (Qwen3-8B).

Also provides clean adaptive_l2 baselines with correct sqrt(d) init
at all layers (exp36's L18/L27 were contaminated by norm-adaptive init).

Standard baselines already exist from exp36 — no need to retrain.

Variants (6 runs total: 2 encoders x 3 layers, sequential):
  adaptive_l2/L9, adaptive_l2/L18, adaptive_l2/L27
  group_G4/L9, group_G4/L18, group_G4/L27

All use sqrt(d) init (NOT norm-adaptive). Exp34 showed norm-adaptive
init hurts at >=50M tokens by suppressing scale_a learning.

Estimated runtime: ~50-60 hours on H100 (~8-10 hours per run).

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 experiments/exp41_optimal_500m.py 2>&1 | tee experiments/exp41_output.log
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

# --- SAE ---
D_SAE = 16384
K = 80
N_GROUPS = 4  # group_G4: 4 scale_a params for 16384 features (4096 features/group)

# --- Data ---
N_TRAIN_TOKENS = 500_000_000   # 500M per variant per layer
N_EVAL_TOKENS = 2_000_000      # 2M eval tokens per layer
CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 4
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR = 3e-4                      # Validated by exp30 for both architectures
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 500

# --- Checkpoints ---
CHECKPOINT_FRACS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

# --- Output ---
SAVE_DIR = "checkpoints/exp41"
RESULTS_PATH = "experiments/exp41_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE    # 122,070
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC) # 6,103
CHECKPOINT_STEPS = sorted(set(int(f * N_STEPS) for f in CHECKPOINT_FRACS))

# --- Streaming buffer ---
BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE


def get_config_dict():
    return {
        "experiment": 38,
        "model_name": MODEL_NAME, "layers": LAYERS, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K, "n_groups": N_GROUPS,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "n_steps": N_STEPS, "warmup_steps": WARMUP_STEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
        "init": "sqrt(d)",
        "note": "Clean adaptive_l2 + group_G4 (exp37 winner). Standard baselines from exp36.",
    }


# =============================================================================
# SAE Architectures
# =============================================================================

class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Cosine encoder with 1 global adaptive scale_a."""

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


class GroupScaleSAE(nn.Module):
    """Cosine SAE with group-wise scale_a (exp37 winner).

    Features are divided into n_groups groups that share a single scale_a
    and scale_b. G=4 for d_sae=16384 means 4 params controlling 4096
    features each — enough flexibility for feature-group-<author>el magnitude
    tuning without per-feature overfitting.
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80,
                 n_groups: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.n_groups = n_groups

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.zeros(n_groups))
        scale_init = math.log(math.sqrt(d_model))
        self.scale_b = nn.Parameter(torch.full((n_groups,), scale_init))
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

    def _expand_group_params(self):
        """Expand group-<author>el scale params to per-feature tensors."""
        group_size = self.d_sae // self.n_groups
        remainder = self.d_sae - group_size * self.n_groups
        scale_a_exp = self.scale_a.repeat_interleave(group_size)
        scale_b_exp = self.scale_b.repeat_interleave(group_size)
        if remainder > 0:
            scale_a_exp = torch.cat([scale_a_exp, self.scale_a[-1:].expand(remainder)])
            scale_b_exp = torch.cat([scale_b_exp, self.scale_b[-1:].expand(remainder)])
        return scale_a_exp, scale_b_exp

    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        log_norm = torch.log(input_norm)

        scale_a_exp, scale_b_exp = self._expand_group_params()
        scale = torch.exp(scale_a_exp * log_norm + scale_b_exp)
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
# Streaming Activation Collection
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
    """Streams activations from FineWeb through model, one layer at a time."""

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
                        batch_texts.append(row["text"][:8192])
                except StopIteration:
                    self._init_dataset()
                    row = next(self._text_iter)
                    if len(row["text"]) > 50:
                        batch_texts.append(row["text"][:8192])
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
    """Collect eval activations from a separate region of FineWeb."""
    print(f"  Collecting eval activations for L{layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    # Skip first 500K docs to avoid train/eval overlap
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
                    batch_texts.append(row["text"][:8192])
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
    mean_norm = norms.mean().item()
    print(f"    L{layer_idx}: {result.shape[0]:,} tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={mean_norm:.1f}, std={norms.std():.1f})")
    return result, mean_norm


# =============================================================================
# Training
# =============================================================================

def make_lr_schedule(n_steps, warmup_steps):
    """Cosine decay with linear warmup."""
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule


def train_sae_streaming(name, sae, stream, layer_idx, save_dir, checkpoint_steps):
    """Train SAE with streaming activations. Save mid-training checkpoints."""
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    schedule_fn = make_lr_schedule(N_STEPS, WARMUP_STEPS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    sae.train()
    log = []
    checkpoints_saved = {}
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

            # --- Logging ---
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

                # Log scale_a (works for both scalar and tensor)
                if hasattr(sae, "scale_a"):
                    sa = sae.scale_a
                    if sa.dim() == 0:
                        entry["scale_a"] = sa.item()
                        scale_str = f" | a={sa.item():.4f}"
                    else:
                        entry["scale_a_mean"] = sa.mean().item()
                        entry["scale_a_std"] = sa.std().item()
                        scale_str = (f" | a={sa.mean().item():.4f}"
                                     f"±{sa.std().item():.4f}")
                else:
                    scale_str = ""

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag}] step {global_step:>6d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.1f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | dead={dead:.3f}{scale_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | ETA {eta_sec/3600:.1f}h")

            # --- Mid-training checkpoints ---
            if global_step in checkpoint_steps:
                frac = global_step / N_STEPS
                ckpt_path = save_dir / f"{name}_L{layer_idx}_step{global_step}.pt"
                torch.save(sae.state_dict(), ckpt_path)
                checkpoints_saved[global_step] = str(ckpt_path)
                print(f"    [{tag}] Checkpoint saved at {frac:.0%} ({global_step} steps)")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

    # Save final checkpoint
    final_path = save_dir / f"{name}_L{layer_idx}_final.pt"
    torch.save(sae.state_dict(), final_path)
    checkpoints_saved["final"] = str(final_path)

    return log, checkpoints_saved


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


def evaluate_ablation(name, model, sae, eval_data, layer_idx):
    tag = f"{name}/L{layer_idx}"
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
                  f"cos->KL={corr_cos:.3f} | inner->KL={corr_inner:.3f} | "
                  f"SAE->KL={corr_sae:.3f}")

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
          f"cos->KL={agg['cos_kl_mean']:.4f} | SAE->KL={agg['sae_kl_mean']:.4f} | "
          f"cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


@torch.no_grad()
def evaluate_checkpoint_reconstruction(sae, eval_data, layer_idx, step):
    """Quick reconstruction eval for mid-training checkpoints (no ablation)."""
    sae.eval()
    n = eval_data.shape[0]
    total_var_sum, resid_var_sum = 0.0, 0.0
    dead_counts = None

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        alive = (features > 0).any(dim=0)
        if dead_counts is None:
            dead_counts = ~alive
        else:
            dead_counts &= ~alive

    fve = float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0
    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0

    return {
        "step": step,
        "frac": step / N_STEPS,
        "fve": fve,
        "dead_frac": dead_frac,
        "alive_count": alive_count,
    }


# =============================================================================
# Main
# =============================================================================

# (name, class, kwargs)
VARIANTS = [
    ("adaptive_l2", AdaptiveCosineBatchTopKSAE, {}),
    ("group_G4",    GroupScaleSAE,              {"n_groups": N_GROUPS}),
]


def main():
    print("Experiment 41: Optimal 500M Token SAE Training")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae={D_SAE}, k={K}, lr={LR}, {N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")
    print(f"Context length: {CTX_LEN}")
    print(f"Init: sqrt(d) for all variants (NOT norm-adaptive)")
    print(f"Checkpoints at: {[f'{f:.0%}' for f in CHECKPOINT_FRACS]}")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Total runs: {len(VARIANTS) * len(LAYERS)}")
    print(f"Estimated time: ~{len(VARIANTS) * len(LAYERS) * 10} hours on H100")
    print(f"\nNote: Standard baselines from exp36 (exp36_results.json)")

    # ---- Load model ----
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation="eager",
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

    # ---- Collect eval data per layer & compute mean norms ----
    eval_data_by_layer = {}
    mean_norms = {}

    for layer_idx in LAYERS:
        eval_data, mean_norm = collect_eval_data(
            model, tokenizer, layer_idx, N_EVAL_TOKENS
        )
        eval_data_by_layer[layer_idx] = eval_data
        mean_norms[layer_idx] = mean_norm
        print(f"    L{layer_idx}: sqrt(d) init: scale_b = log(sqrt({D_MODEL})) "
              f"= {math.log(math.sqrt(D_MODEL)):.4f} "
              f"(mean_norm={mean_norm:.1f} for reference)")

    all_results["config"]["mean_norms"] = {str(k): v for k, v in mean_norms.items()}

    # Save config update
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- GPU utilization check ----
    print("\n---GPU---")
    os.system("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv")

    # ---- Run all combinations ----
    for vname, vcls, vkwargs in VARIANTS:
        for layer_idx in LAYERS:
            run_name = f"{vname}_L{layer_idx}"

            if run_name in all_results.get("runs", {}):
                print(f"\n  {run_name} already complete, skipping")
                continue

            print(f"\n{'='*70}")
            print(f"  RUN: {run_name} (encoder={vname}, layer={layer_idx})")
            print(f"{'='*70}")

            # Set seed
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            # Create SAE — all use sqrt(d) init (no init_norm argument)
            sae = vcls(D_MODEL, D_SAE, K, **vkwargs).to(DEVICE)
            n_params = sum(p.numel() for p in sae.parameters())
            print(f"    sqrt(d) init: scale_b = {math.log(math.sqrt(D_MODEL)):.4f}")
            print(f"    Parameters: {n_params:,}")

            # Create stream
            stream = ActivationStream(model, tokenizer, layer_idx, seed=SEED)

            # Train with mid-training checkpoints
            train_log, ckpt_paths = train_sae_streaming(
                vname, sae, stream, layer_idx, save_dir, CHECKPOINT_STEPS
            )

            # Evaluate mid-training checkpoints (quick FVE + dead only)
            eval_data = eval_data_by_layer[layer_idx]
            checkpoint_evals = []
            for step in CHECKPOINT_STEPS[:-1]:  # Skip final (we do full eval)
                ckpt_path = ckpt_paths.get(step)
                if ckpt_path and os.path.exists(ckpt_path):
                    sae.load_state_dict(torch.load(ckpt_path, map_location=DEVICE,
                                                   weights_only=True))
                    ckpt_eval = evaluate_checkpoint_reconstruction(
                        sae, eval_data, layer_idx, step
                    )
                    checkpoint_evals.append(ckpt_eval)
                    print(f"    Checkpoint {step/N_STEPS:.0%}: "
                          f"FVE={ckpt_eval['fve']:.4f}, "
                          f"dead={ckpt_eval['dead_frac']:.3f}")

            # Load final checkpoint for full evaluation
            final_path = ckpt_paths.get("final")
            if final_path and os.path.exists(final_path):
                sae.load_state_dict(torch.load(final_path, map_location=DEVICE,
                                               weights_only=True))

            # Full evaluation (reconstruction + ablation)
            print(f"\n  Full evaluation -- {run_name}")
            recon = evaluate_reconstruction(vname, sae, eval_data, layer_idx)
            abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)

            run_result = {
                "encoder": vname,
                "layer": layer_idx,
                "training": train_log,
                "checkpoints": checkpoint_evals,
                "reconstruction": recon,
                "ablation": abl,
            }
            # Save scale_a final values
            if hasattr(sae, "scale_a"):
                sa = sae.scale_a
                if sa.dim() == 0:
                    run_result["scale_a_final"] = sa.item()
                else:
                    run_result["scale_a_final_mean"] = sa.mean().item()
                    run_result["scale_a_final_std"] = sa.std().item()
                    run_result["scale_a_final_values"] = sa.tolist()

            all_results["runs"][run_name] = run_result

            # Save incrementally
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Results saved to {RESULTS_PATH}")

            del sae, stream
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  EXP38 SUMMARY")
    print(f"{'='*70}")

    for vname, _, _ in VARIANTS:
        print(f"\n  {vname}:")
        header = f"  {'Layer':>5s} | {'FVE':>6s} | {'Dead%':>6s} | {'Alive':>6s} | {'cos>inn':>8s}"
        sep    = f"  {'-'*5:>5s}-+-{'-'*6:>6s}-+-{'-'*6:>6s}-+-{'-'*6:>6s}-+-{'-'*8:>8s}"
        print(header)
        print(sep)
        for layer_idx in LAYERS:
            run = all_results["runs"].get(f"{vname}_L{layer_idx}", {})
            r = run.get("reconstruction", {})
            a = run.get("ablation", {}).get("aggregate", {})
            if r:
                fve = r.get("fve", 0)
                dead = r.get("dead_frac", 0)
                alive = r.get("alive_count", 0)
                cos_wins = a.get("cos_wins_inner", "?")
                n_feat = a.get("n_features", "?")
                sa_mean = run.get("scale_a_final", run.get("scale_a_final_mean", ""))
                extra = f" | a={sa_mean:.3f}" if sa_mean != "" else ""
                print(f"  L{layer_idx:>4d} | {fve:.4f} | {dead*100:5.1f}% | {alive:>6d} | "
                      f"{cos_wins:>3s}/{n_feat:<4s}{extra}")

    # Cross-encoder comparison at each layer
    print(f"\n  ADAPTIVE vs GROUP_G4:")
    for layer_idx in LAYERS:
        adp_run = all_results["runs"].get(f"adaptive_l2_L{layer_idx}", {})
        grp_run = all_results["runs"].get(f"group_G4_L{layer_idx}", {})
        adp_fve = adp_run.get("reconstruction", {}).get("fve", 0)
        grp_fve = grp_run.get("reconstruction", {}).get("fve", 0)
        adp_alive = adp_run.get("reconstruction", {}).get("alive_count", 0)
        grp_alive = grp_run.get("reconstruction", {}).get("alive_count", 0)
        if adp_alive > 0:
            print(f"    L{layer_idx}: FVE gap = {grp_fve - adp_fve:+.4f} | "
                  f"Alive ratio = {grp_alive/adp_alive:.2f}x "
                  f"({grp_alive} vs {adp_alive})")

    # Convergence trajectory comparison
    print(f"\n  CONVERGENCE TRAJECTORY (FVE at each checkpoint):")
    for layer_idx in LAYERS:
        for vname, _, _ in VARIANTS:
            run = all_results["runs"].get(f"{vname}_L{layer_idx}", {})
            ckpts = run.get("checkpoints", [])
            if ckpts:
                points = " -> ".join(f"{c['frac']:.0%}:{c['fve']:.3f}" for c in ckpts)
                final_fve = run.get("reconstruction", {}).get("fve", 0)
                print(f"    {vname}/L{layer_idx}: {points} -> 100%:{final_fve:.3f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Standard baselines: experiments/exp36_results.json")


if __name__ == "__main__":
    main()
