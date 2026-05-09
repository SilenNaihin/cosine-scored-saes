"""
Experiment 17: Production-Scale Validation
==========================================

Scales up exp10/12/16 from 5M to 50M tokens to test whether cosine SAE
findings hold at production training scale.

Key changes from exp10:
  - 50M tokens (10x) — streaming activation collection (can't fit in RAM)
  - 3 variants: standard, adaptive_l2, perfeature_l2
  - Mid-training checkpoints every 20% for convergence tracking
  - Enhanced ablation: 100 features × 200 samples (was 30 × 50)
  - Per-feature scale_a distribution analysis for perfeature_l2

Estimated runtime on A100 80GB: ~11 hours (training + evaluation).

Usage:
    ssh <server>     cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python experiments/exp17_production_scale.py
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

# --- SAE architecture ---
D_SAE = 16384       # Same as toy experiments for fair comparison
K = 80              # BatchTopK sparsity

# --- Data ---
N_TRAIN_TOKENS = 50_000_000   # 10x over exp10/12/16
N_EVAL_TOKENS = 1_000_000     # 2x over exp10 (more features need more eval data)
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
# Save at 20%, 40%, 60%, 80%, 100% for convergence tracking
CHECKPOINT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 100     # Up from 30 — need statistical power
N_ABLATION_SAMPLES = 200      # Up from 50 — tighter per-feature correlations

# --- Output ---
SAVE_DIR = "checkpoints/exp17"
RESULTS_PATH = "experiments/exp17_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)
CHECKPOINT_STEPS = [int(f * N_STEPS) for f in CHECKPOINT_FRACS]

# --- Streaming activation buffer ---
# Can't fit 50M tokens in GPU memory. Stream from FineWeb in chunks.
BUFFER_TOKENS = 500_000       # Collect this many tokens at a time, train on buffer
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE  # Training steps per buffer fill


def get_config_dict():
    return {
        "model_name": MODEL_NAME, "layers": LAYERS, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "n_steps": N_STEPS, "warmup_steps": WARMUP_STEPS,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
        "buffer_tokens": BUFFER_TOKENS,
        "checkpoint_steps": CHECKPOINT_STEPS,
    }


# =============================================================================
# SAE Architectures (copied from exp10/16 for self-containment)
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
      - scale_a=1: scale ∝ ||x|| (inner-product-like)
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


class PerFeatureAdaptiveCosineSAE(nn.Module):
    """BatchTopK SAE with per-feature adaptive-scale cosine encoder.

    scale_i(x) = exp(a_i * log(||x - b_dec||) + b_i)
    Each of d_sae features learns its own magnitude sensitivity a_i.
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
        # Per-feature scale: [d_sae] params each
        self.scale_a = nn.Parameter(torch.zeros(d_sae))
        self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))
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
        cos_sim = x_unit @ w_unit.T                    # [batch, d_sae]
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, 1]
        log_norm = torch.log(input_norm)                # [batch, 1]
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)  # [batch, d_sae]
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
# Streaming Activation Collection
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


class ActivationStream:
    """Streams activations from FineWeb, yielding shuffled batches.

    Instead of collecting all 50M tokens upfront (too much RAM), we collect
    BUFFER_TOKENS at a time, shuffle within the buffer, and yield batches.
    The FineWeb dataset iterator persists across buffer fills.

    Each call to fill_buffer() collects fresh activations from the next
    chunk of FineWeb texts. Training iterates over the buffer, then refills.
    """

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
        """Collect buffer_tokens activations from the next chunk of FineWeb."""
        all_acts = []
        tokens_collected = 0

        while tokens_collected < self.buffer_tokens:
            # Collect a batch of texts
            batch_texts = []
            for _ in range(COLLECTION_BATCH_SIZE):
                try:
                    row = next(self._text_iter)
                    if len(row["text"]) > 50:
                        batch_texts.append(row["text"][:2048])
                except StopIteration:
                    # Restart dataset if exhausted (unlikely with 10BT)
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

            # Filter attention sinks
            norms = flat.float().norm(dim=-1)
            median = norms.median()
            if median > 0:
                flat = flat[norms < median * OUTLIER_MULTIPLIER]

            all_acts.append(flat.to("cpu", dtype=DTYPE))
            tokens_collected += flat.shape[0]

        self.buffer = torch.cat(all_acts, dim=0)[:self.buffer_tokens]
        # Shuffle
        perm = torch.randperm(self.buffer.shape[0])
        self.buffer = self.buffer[perm]
        return self.buffer.shape[0]

    def get_batch(self, batch_idx):
        """Get a batch from the current buffer."""
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
    """Collect evaluation activations (fits in RAM at 1M tokens)."""
    print(f"  Collecting eval activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )

    # Skip ahead to avoid overlap with training data
    # (training streams from the start; eval skips 500k docs)
    text_iter = iter(ds)
    skip_count = 500_000
    for i, _ in enumerate(text_iter):
        if i >= skip_count:
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
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_sae_streaming(name, sae, stream, layer_idx, save_dir):
    """Train an SAE with streaming activation collection.

    Instead of pre-collecting all tokens, fills a buffer, trains on it,
    then refills. Saves mid-training checkpoints for convergence tracking.
    """
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps (streaming)")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    sae.train()
    log = []
    checkpoint_log = {}
    t0 = time.time()
    global_step = 0
    buffer_step = 0
    next_checkpoint_idx = 0

    while global_step < N_STEPS:
        # Refill buffer
        n_filled = stream.fill_buffer()
        steps_in_buffer = min(BUFFER_BATCHES, N_STEPS - global_step)

        for buf_step in range(steps_in_buffer):
            batch = stream.get_batch(buf_step)
            x_hat, features = sae(batch)

            # L2 loss (all variants use L2 for production comparison)
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            optimizer.zero_grad(set_to_none=True)
            recon_loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Logging
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

                # Log scale parameters
                if hasattr(sae, "scale_a"):
                    if sae.scale_a.dim() == 0:
                        # Global adaptive
                        entry["scale_a"] = sae.scale_a.item()
                        entry["scale_b"] = sae.scale_b.exp().item()
                        scale_str = f" | a={sae.scale_a.item():.4f} b={sae.scale_b.exp().item():.1f}"
                    else:
                        # Per-feature adaptive
                        a = sae.scale_a.detach()
                        entry["scale_a_mean"] = a.mean().item()
                        entry["scale_a_std"] = a.std().item()
                        entry["scale_a_max"] = a.max().item()
                        entry["scale_a_min"] = a.min().item()
                        scale_str = (f" | a={a.mean().item():.4f}+/-{a.std().item():.4f}"
                                     f" [{a.min().item():.3f},{a.max().item():.3f}]")
                else:
                    scale_str = ""

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag:>16s}] step {global_step:>6d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f}{scale_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | {tok_per_sec/1e3:.0f}k tok/s | "
                      f"ETA {eta_sec/60:.0f}m")

            # Mid-training checkpoints
            if (next_checkpoint_idx < len(CHECKPOINT_STEPS) and
                    global_step >= CHECKPOINT_STEPS[next_checkpoint_idx]):
                frac = CHECKPOINT_FRACS[next_checkpoint_idx]
                ckpt_path = save_dir / f"{name}_L{layer_idx}_step{global_step}.pt"
                torch.save(sae.state_dict(), ckpt_path)
                # Save snapshot metrics
                with torch.no_grad():
                    snap_dead = dead  # from most recent log
                    snap_fve = fve
                checkpoint_log[f"{frac:.0%}"] = {
                    "step": global_step, "tokens": global_step * BATCH_SIZE,
                    "fve": snap_fve, "dead_frac": snap_dead,
                }
                if hasattr(sae, "scale_a") and sae.scale_a.dim() == 0:
                    checkpoint_log[f"{frac:.0%}"]["scale_a"] = sae.scale_a.item()
                elif hasattr(sae, "scale_a"):
                    a = sae.scale_a.detach()
                    checkpoint_log[f"{frac:.0%}"]["scale_a_mean"] = a.mean().item()
                    checkpoint_log[f"{frac:.0%}"]["scale_a_median"] = a.median().item()
                    nz = (a.abs() < 0.05).float().mean().item()
                    checkpoint_log[f"{frac:.0%}"]["near_zero_frac"] = nz
                print(f"    [{tag}] Checkpoint saved at {frac:.0%} ({global_step} steps)")
                next_checkpoint_idx += 1

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save final checkpoint
    torch.save(sae.state_dict(), save_dir / f"{name}_L{layer_idx}_final.pt")

    return log, checkpoint_log


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
    dead_counts = None

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        recon_losses.append((batch - x_hat).pow(2).sum(dim=-1).mean().item())
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()

        # Track dead features across full eval set
        alive = (features > 0).any(dim=0)
        if dead_counts is None:
            dead_counts = ~alive
        else:
            dead_counts &= ~alive  # Feature is dead only if never fires

    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0

    results = {
        "recon_loss_l2": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
        "dead_frac": dead_frac,
    }
    print(f"    [{tag}] L2={results['recon_loss_l2']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"FVE={results['fve']:.4f} | dead={dead_frac:.3f}")
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
    """Enhanced ablation: 100 features × 200 samples."""
    tag = f"{name}/L{layer_idx}"
    print(f"\n    Ablation [{tag}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    # Probe more tokens to find enough active features
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

    # Select top N_ABLATION_FEATURES by frequency among alive features
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

        # Get per-feature scale_a if available
        scale_a_i = None
        if hasattr(sae, "scale_a") and sae.scale_a.dim() > 0:
            scale_a_i = sae.scale_a[fi].item()

        result = {
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "cos_wins_sae": bool(abs(corr_cos) > abs(corr_sae)),
            "sae_wins_inner": bool(abs(corr_sae) > abs(corr_inner)),
        }
        if scale_a_i is not None:
            result["scale_a_i"] = scale_a_i
        feature_results.append(result)

        if rank < 7 or rank % 20 == 0:
            sa_str = f" | a_i={scale_a_i:.3f}" if scale_a_i is not None else ""
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f} | norm→KL={corr_norm:.3f}{sa_str}")

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
# Per-Feature Scale Analysis
# =============================================================================

@torch.no_grad()
def analyze_scale_distribution(name, sae, layer_idx):
    """Analyze the distribution of per-feature scale_a values."""
    if not hasattr(sae, "scale_a"):
        return None
    tag = f"{name}/L{layer_idx}"

    if sae.scale_a.dim() == 0:
        # Global adaptive — single scalar
        a = sae.scale_a.item()
        print(f"    [{tag}] Global scale_a = {a:.4f}")
        return {"type": "global", "scale_a": a, "scale_b": sae.scale_b.exp().item()}

    # Per-feature adaptive
    a = sae.scale_a.detach().cpu().numpy()
    results = {
        "type": "per_feature",
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "near_zero_frac": float((np.abs(a) < 0.05).mean()),
        "low_frac": float(((np.abs(a) >= 0.05) & (np.abs(a) < 0.2)).mean()),
        "medium_frac": float(((np.abs(a) >= 0.2) & (np.abs(a) < 0.5)).mean()),
        "high_frac": float((np.abs(a) >= 0.5).mean()),
        "negative_frac": float((a < -0.05).mean()),
        # Histogram bins for plotting
        "histogram": {
            "bins": list(np.linspace(-0.1, 0.5, 61).astype(float)),
            "counts": list(np.histogram(a, bins=np.linspace(-0.1, 0.5, 61))[0].astype(int).tolist()),
        },
    }

    print(f"    [{tag}] scale_a distribution:")
    print(f"      mean={results['mean']:.4f} +/- {results['std']:.4f}")
    print(f"      median={results['median']:.4f} [{results['p5']:.4f}, {results['p95']:.4f}] (5-95%)")
    print(f"      near-zero (|a|<0.05): {results['near_zero_frac']:.1%}")
    print(f"      low (0.05-0.2):       {results['low_frac']:.1%}")
    print(f"      medium (0.2-0.5):     {results['medium_frac']:.1%}")
    print(f"      high (>0.5):          {results['high_frac']:.1%}")
    return results


# =============================================================================
# Per-Layer Runner
# =============================================================================

VARIANTS = [
    ("standard",       BatchTopKSAE),
    ("adaptive_l2",    AdaptiveCosineBatchTopKSAE),
    ("perfeature_l2",  PerFeatureAdaptiveCosineSAE),
]


def run_layer(model, tokenizer, layer_idx, save_dir):
    """Train + evaluate all variants at one layer."""
    print(f"\n{'='*70}")
    print(f"  LAYER {layer_idx}")
    print(f"{'='*70}")

    # Collect eval data first (separate from training stream)
    eval_data = collect_eval_data(model, tokenizer, layer_idx, N_EVAL_TOKENS)

    results = {}

    for vname, cls in VARIANTS:
        # Fresh activation stream for each variant (same data distribution,
        # different random samples — acceptable for this comparison)
        stream = ActivationStream(model, tokenizer, layer_idx)

        torch.manual_seed(SEED)
        sae = cls(D_MODEL, D_SAE, K).to(DEVICE)

        # Train with streaming
        train_log, ckpt_log = train_sae_streaming(
            vname, sae, stream, layer_idx, save_dir
        )

        # Evaluate
        print(f"\n  Evaluation — {vname}/L{layer_idx}")
        recon = evaluate_reconstruction(vname, sae, eval_data, layer_idx)
        inv = test_norm_invariance(vname, sae, eval_data, layer_idx)
        scale_dist = analyze_scale_distribution(vname, sae, layer_idx)
        abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)

        results[vname] = {
            "training": train_log,
            "checkpoints": ckpt_log,
            "reconstruction": recon,
            "norm_invariance": inv,
            "ablation": abl,
        }
        if scale_dist is not None:
            results[vname]["scale_distribution"] = scale_dist

        # Free memory
        del sae, stream
        gc.collect()
        torch.cuda.empty_cache()

    del eval_data
    gc.collect()
    torch.cuda.empty_cache()

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 17: Production-Scale Validation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Per layer: {N_TRAIN_TOKENS:,} train tokens ({N_STEPS} steps), "
          f"{N_EVAL_TOKENS:,} eval tokens")
    print(f"Batch: {BATCH_SIZE}, Warmup: {WARMUP_STEPS} steps")
    print(f"Streaming buffer: {BUFFER_TOKENS:,} tokens")
    print(f"Checkpoints at steps: {CHECKPOINT_STEPS}")
    print(f"Ablation: {N_ABLATION_FEATURES} features × {N_ABLATION_SAMPLES} samples")
    print(f"Variants: {[v[0] for v in VARIANTS]}")

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
        existing = list(all_results.get("layers", {}).keys())
        print(f"  Loaded existing results for layers: {existing}")
    else:
        all_results = {"config": get_config_dict(), "layers": {}}

    # ---- Run each layer ----
    for layer_idx in LAYERS:
        layer_key = str(layer_idx)
        if layer_key in all_results["layers"]:
            # Check if all variants are done
            done = set(all_results["layers"][layer_key].keys())
            needed = {v[0] for v in VARIANTS}
            if needed.issubset(done):
                print(f"\n  Layer {layer_idx} already complete, skipping")
                continue

        layer_result = run_layer(model, tokenizer, layer_idx, save_dir)

        if layer_key not in all_results["layers"]:
            all_results["layers"][layer_key] = {}
        all_results["layers"][layer_key].update(layer_result)

        # Save after each layer
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to {RESULTS_PATH}")

    # ---- Cross-layer summary ----
    print(f"\n{'='*70}")
    print("  CROSS-LAYER SUMMARY")
    print(f"{'='*70}")

    vnames = [v[0] for v in VARIANTS]

    print(f"\n  Reconstruction:")
    for li in LAYERS:
        lr = all_results["layers"].get(str(li), {})
        parts = []
        for v in vnames:
            r = lr.get(v, {}).get("reconstruction", {})
            if r:
                parts.append(f"{v}: FVE={r['fve']:.4f} cos={r['cos_recon']:.4f} dead={r['dead_frac']:.3f}")
        print(f"    L{li}: {' | '.join(parts)}")

    print(f"\n  Ablation:")
    for li in LAYERS:
        lr = all_results["layers"].get(str(li), {})
        parts = []
        for v in vnames:
            a = lr.get(v, {}).get("ablation", {}).get("aggregate", {})
            if a and a.get("n_features"):
                parts.append(f"{v}: cos→KL={a['cos_kl_mean']:.4f} "
                             f"cos>inn={a['cos_wins_inner']}/{a['n_features']}")
        print(f"    L{li}: {' | '.join(parts)}")

    print(f"\n  Scale parameters:")
    for li in LAYERS:
        lr = all_results["layers"].get(str(li), {})
        for v in vnames:
            sd = lr.get(v, {}).get("scale_distribution", {})
            if sd:
                if sd["type"] == "global":
                    print(f"    L{li}/{v}: global a={sd['scale_a']:.4f}")
                else:
                    print(f"    L{li}/{v}: mean={sd['mean']:.4f} median={sd['median']:.4f} "
                          f"near_zero={sd['near_zero_frac']:.1%} high={sd['high_frac']:.1%}")

    print(f"\n  Convergence (checkpoints):")
    for li in LAYERS:
        lr = all_results["layers"].get(str(li), {})
        for v in vnames:
            ckpts = lr.get(v, {}).get("checkpoints", {})
            if ckpts:
                parts = []
                for frac_key in sorted(ckpts.keys()):
                    c = ckpts[frac_key]
                    parts.append(f"{frac_key}: FVE={c.get('fve', 0):.4f} dead={c.get('dead_frac', 0):.3f}")
                print(f"    L{li}/{v}: {' → '.join(parts)}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
