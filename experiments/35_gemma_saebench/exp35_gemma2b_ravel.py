"""
Experiment 35: Gemma-2-2b SAE Training + RAVEL Evaluation
==========================================================

Why this experiment exists:
  Exp20 ran 5 of 6 SAEBench evals (core, sparse probing, absorption, SCR, TPP)
  on Qwen3-8B. The 6th -- RAVEL -- is hardcoded to Gemma-2-2b and Pythia-160m.
  Qwen3-8B is fundamentally incompatible (decoder layers return bare tensors;
  RAVEL's intervention hooks assume tuples). Rather than monkey-patching a
  fragile workaround, we train production-quality SAEs directly on Gemma-2-2b --
  a natively supported RAVEL model that also uses RMSNorm, making it a clean
  RNH test.

  This completes the SAEBench evaluation matrix and provides cross-model
  validation on a second RMSNorm architecture.

Why Gemma-2-2b specifically:
  1. RAVEL natively supports it (LLM_NAME_MAP["gemma-2-2b"] -> "google/gemma-2-2b")
  2. RMSNorm -- the normalization type where cosine SAEs show the strongest
     advantage (exp24/exp25 confirmed the effect is normalization-dependent)
  3. Different model family from Qwen3-8B -- cross-architecture validation
  4. 2B params fits comfortably on a single A100 for training + eval
  5. 26 layers, d_model=2304 -- different scale from Qwen (36 layers, 4096)

What this tests:
  - Does the cosine SAE advantage (FVE, alive features, probing) replicate
    on Gemma-2-2b? (Cross-model validation of exp17/exp20)
  - RAVEL: do cosine SAE features produce better entity-attribute disentanglement?
    RAVEL trains a learned binary mask (MDBM) over SAE features to isolate
    attributes -- this is more like probing (uses all features) than ablation
    (top-N), so the concentration-vs-distribution tradeoff from SCR/TPP predicts
    cosine should win.
  - All 6 SAEBench evals on one model: a complete, single-model benchmark card.

Design decisions (informed by prior experiments):
  - 50M tokens: exp17 showed 5M→50M flips L27 from cosine's worst to best layer.
    5M results are unreliable for production conclusions.
  - Norm-adaptive init: exp27 showed sqrt(d) init fails when activation norms
    don't match sqrt(d_model). We measure mean norms and init scale_b accordingly.
  - Same 3 variants as exp17: standard, adaptive_l2, perfeature_l2.
  - 4x expansion (d_sae=9216): matches the d_sae/d_model ratio from exp17.
  - k=80: same sparsity <author>el as exp17 for fair comparison.

Estimated runtime on A100 80GB:
  - Activation collection: ~30 min per layer (Gemma-2-2b is faster than Qwen3-8B)
  - Training: ~3 hours per variant per layer (9 runs total)
  - SAEBench evals: ~8 hours total (RAVEL ~2h, others ~6h combined)
  - Total: ~40 hours

Usage:
    ssh <server>     cd ~/MechInter--RNH
    PYTHONUNBUFFERED=1 python experiments/exp35_gemma2b_ravel.py

    # Training only (skip eval):
    PYTHONUNBUFFERED=1 python experiments/exp35_gemma2b_ravel.py --train-only

    # Eval only (after training):
    PYTHONUNBUFFERED=1 python experiments/exp35_gemma2b_ravel.py --eval-only

    # Specific layers/variants:
    PYTHONUNBUFFERED=1 python experiments/exp35_gemma2b_ravel.py --layers 13 --variants standard adaptive_l2
"""

from __future__ import annotations

import argparse
import json
import math
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

# --- Model ---
MODEL_NAME = "google/gemma-2-2b"
RAVEL_MODEL_NAME = "gemma-2-2b"     # SAEBench RAVEL key (maps to google/gemma-2-2b)
LAYERS = [6, 13, 20]                # ~25%, 50%, 77% of 26 layers
D_MODEL = 2304
N_LAYERS_TOTAL = 26

# --- SAE architecture ---
D_SAE = 9216         # 4x d_model (same ratio as exp17's 16384/4096)
K = 80               # Same sparsity as exp17

# --- Data ---
N_TRAIN_TOKENS = 50_000_000
N_EVAL_TOKENS = 1_000_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 32     # Gemma-2-2b is smaller, can use larger batches
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
SAVE_DIR = Path("checkpoints/exp35")
RESULTS_PATH = "experiments/exp35_results.json"
SAEBENCH_OUTPUT = "experiments/exp35_saebench_results"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)
CHECKPOINT_STEPS = [int(f * N_STEPS) for f in CHECKPOINT_FRACS]

# --- Streaming activation buffer ---
BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE


def get_config_dict():
    return {
        "experiment": 35,
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
# SAE Architectures (same as exp17, with norm-adaptive init from exp27)
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

    Uses norm-adaptive init (exp27): scale_b = log(mean(||x_train||))
    instead of log(sqrt(d_model)) to avoid the Mistral-style init mismatch.
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80,
                 init_norm: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        # Norm-adaptive init: use observed mean norm if available
        scale_init = math.log(init_norm) if init_norm else math.log(math.sqrt(d_model))
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


class PerFeatureAdaptiveCosineSAE(nn.Module):
    """BatchTopK SAE with per-feature adaptive-scale cosine encoder.

    scale_i(x) = exp(a_i * log(||x - b_dec||) + b_i)
    Each of d_sae features learns its own magnitude sensitivity a_i.

    Uses norm-adaptive init (exp27).
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80,
                 init_norm: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.zeros(d_sae))
        scale_init = math.log(init_norm) if init_norm else math.log(math.sqrt(d_model))
        self.scale_b = nn.Parameter(torch.full((d_sae,), scale_init))
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
        log_norm = torch.log(input_norm)
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
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
    """Capture residual stream activations at a Gemma-2 layer via forward hook."""
    captured = {}

    def hook(module, inp, out):
        # Gemma-2 layers return tuples (hidden_states, ...) -- extract tensor
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

    Same design as exp17: collect BUFFER_TOKENS at a time, shuffle, yield.
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
    """Collect evaluation activations (separate from training data)."""
    print(f"  Collecting eval activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )

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


def measure_mean_norms(model, tokenizer, layers, n_tokens=100_000):
    """Measure mean activation norms per layer for norm-adaptive init (exp27)."""
    print("\n  Measuring activation norms for norm-adaptive initialization...")
    norms = {}
    for layer_idx in layers:
        data = collect_eval_data(model, tokenizer, layer_idx, n_tokens)
        mean_norm = data.float().norm(dim=-1).mean().item()
        norms[layer_idx] = mean_norm
        print(f"    Layer {layer_idx}: mean norm = {mean_norm:.1f}")
        del data
        gc.collect()
        torch.cuda.empty_cache()
    return norms


# =============================================================================
# Training
# =============================================================================

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_sae_streaming(name, sae, stream, layer_idx, save_dir):
    """Train an SAE with streaming activation collection (same as exp17)."""
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
                    if sae.scale_a.dim() == 0:
                        entry["scale_a"] = sae.scale_a.item()
                        entry["scale_b"] = sae.scale_b.exp().item()
                        scale_str = f" | a={sae.scale_a.item():.4f} b={sae.scale_b.exp().item():.1f}"
                    else:
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

            if (next_checkpoint_idx < len(CHECKPOINT_STEPS) and
                    global_step >= CHECKPOINT_STEPS[next_checkpoint_idx]):
                frac = CHECKPOINT_FRACS[next_checkpoint_idx]
                ckpt_path = save_dir / f"{name}_L{layer_idx}_step{global_step}.pt"
                torch.save(sae.state_dict(), ckpt_path)
                checkpoint_log[f"{frac:.0%}"] = {
                    "step": global_step, "tokens": global_step * BATCH_SIZE,
                    "fve": fve, "dead_frac": dead,
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

    torch.save(sae.state_dict(), save_dir / f"{name}_L{layer_idx}_final.pt")

    return log, checkpoint_log


# =============================================================================
# Evaluation (reconstruction + ablation, same as exp17)
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
# SAEBench Evaluation (all 6 evals including RAVEL)
# =============================================================================

def run_saebench_evals(
    layers: list[int],
    variants: list[str],
    eval_types: list[str],
    device: str = "cuda",
    llm_batch_size: int = 16,
    output_dir: str = SAEBENCH_OUTPUT,
    force_rerun: bool = False,
):
    """Run SAEBench evals on exp35 checkpoints.

    Wraps each checkpoint as a BenchSAE and runs requested evals.
    RAVEL natively supports gemma-2-2b -- no monkey-patching needed.
    """
    from benchmarks.adapter import BenchSAE
    import time

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for layer in layers:
        print(f"\n{'='*70}")
        print(f"SAEBench: Layer {layer}")
        print(f"{'='*70}")

        # Load all variants for this layer
        saes: list[tuple[str, BenchSAE]] = []
        for variant in variants:
            name = f"exp35-{variant}-L{layer}"
            path = SAVE_DIR / f"{variant}_L{layer}_final.pt"
            if not path.exists():
                print(f"  SKIP {name} — checkpoint not found: {path}")
                continue

            # Load SAE
            if variant == "standard":
                sae = BatchTopKSAE(D_MODEL, D_SAE, K)
            elif variant == "adaptive_l2":
                sae = AdaptiveCosineBatchTopKSAE(D_MODEL, D_SAE, K)
            elif variant == "perfeature_l2":
                sae = PerFeatureAdaptiveCosineSAE(D_MODEL, D_SAE, K)
            else:
                raise ValueError(f"Unknown variant: {variant}")

            sae.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            sae = sae.to(device=device, dtype=torch.bfloat16).eval()

            # Wrap as BenchSAE
            _sae = sae
            def _make_fns(s):
                return lambda x: s.encode(x), lambda f: s.decode(f)
            enc_fn, dec_fn = _make_fns(_sae)

            W_enc = sae.W_enc.detach().T
            W_dec = F.normalize(sae.W_dec.detach(), dim=1)
            b_enc = sae.b_enc.detach()
            b_dec = sae.b_dec.detach()

            bench_sae = BenchSAE(
                W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec,
                encode_fn=enc_fn, decode_fn=dec_fn,
                model_name=MODEL_NAME,
                hook_layer=layer,
                device=device, dtype=torch.bfloat16,
            )
            assert bench_sae.check_decoder_norms(), f"Decoder norm check failed for {name}"
            saes.append((name, bench_sae))
            print(f"  Loaded {name}")

        if not saes:
            continue

        # Run each eval
        for eval_type in eval_types:
            print(f"\n--- {eval_type} (layer {layer}) ---")
            eval_output = os.path.join(output_dir, eval_type)
            os.makedirs(eval_output, exist_ok=True)
            t0 = time.time()

            if eval_type == "core":
                import sae_bench.evals.core.main as core_eval
                result = core_eval.multiple_evals(
                    selected_saes=saes,
                    n_eval_reconstruction_batches=200,
                    n_eval_sparsity_variance_batches=2000,
                    eval_batch_size_prompts=llm_batch_size,
                    compute_featurewise_density_statistics=True,
                    compute_featurewise_weight_based_metrics=True,
                    exclude_special_tokens_from_reconstruction=True,
                    dataset="Skylion007/openwebtext",
                    context_size=128,
                    output_folder=eval_output,
                    verbose=True,
                    dtype="bfloat16",
                    device=device,
                    force_rerun=force_rerun,
                )
                for sae_result in result:
                    rname = sae_result.get("unique_id", sae_result.get("sae_set", "unknown"))
                    all_results[f"{rname}_core"] = sae_result

            elif eval_type == "sparse_probing":
                import sae_bench.evals.sparse_probing.main as sp_eval
                for sae_name, sae in saes:
                    config = sp_eval.SparseProbingEvalConfig(
                        model_name=MODEL_NAME,
                        llm_batch_size=llm_batch_size,
                        llm_dtype="bfloat16",
                    )
                    sp_eval.run_eval(config, [(sae_name, sae)], device, eval_output,
                                     force_rerun=force_rerun, clean_up_activations=True,
                                     save_activations=False)
                    all_results[f"{sae_name}_sparse_probing"] = _load_result(eval_output, sae_name)

            elif eval_type == "absorption":
                import sae_bench.evals.absorption.main as ab_eval
                for sae_name, sae in saes:
                    config = ab_eval.AbsorptionEvalConfig(
                        model_name=MODEL_NAME,
                        llm_batch_size=llm_batch_size,
                        llm_dtype="bfloat16",
                    )
                    ab_eval.run_eval(config, [(sae_name, sae)], device, eval_output,
                                     force_rerun=force_rerun)
                    all_results[f"{sae_name}_absorption"] = _load_result(eval_output, sae_name)

            elif eval_type == "scr_and_tpp":
                import sae_bench.evals.scr_and_tpp.main as scr_tpp_eval
                for sae_name, sae in saes:
                    config = scr_tpp_eval.ScrAndTppEvalConfig(
                        model_name=MODEL_NAME,
                        llm_batch_size=llm_batch_size,
                        llm_dtype="bfloat16",
                        perform_scr=True,
                    )
                    scr_tpp_eval.run_eval(config, [(sae_name, sae)], device, eval_output,
                                          force_rerun=force_rerun, clean_up_activations=True,
                                          save_activations=False)
                    all_results[f"{sae_name}_scr"] = _load_result(eval_output, sae_name)

            elif eval_type == "tpp":
                import sae_bench.evals.scr_and_tpp.main as scr_tpp_eval
                for sae_name, sae in saes:
                    config = scr_tpp_eval.ScrAndTppEvalConfig(
                        model_name=MODEL_NAME,
                        llm_batch_size=llm_batch_size,
                        llm_dtype="bfloat16",
                        perform_scr=False,
                    )
                    scr_tpp_eval.run_eval(config, [(sae_name, sae)], device, eval_output,
                                          force_rerun=force_rerun, clean_up_activations=True,
                                          save_activations=False)
                    all_results[f"{sae_name}_tpp"] = _load_result(eval_output, sae_name)

            elif eval_type == "ravel":
                import sae_bench.evals.ravel.main as ravel_eval
                for sae_name, sae in saes:
                    config = ravel_eval.RAVE<author>alConfig(
                        model_name=RAVEL_MODEL_NAME,
                        llm_batch_size=llm_batch_size,
                        llm_dtype="bfloat16",
                    )
                    ravel_eval.run_eval(config, [(sae_name, sae)], device, eval_output,
                                        force_rerun=force_rerun)
                    all_results[f"{sae_name}_ravel"] = _load_result(eval_output, sae_name)

            elapsed = time.time() - t0
            print(f"  {eval_type} completed in {elapsed:.0f}s")

    # Save combined
    combined_path = os.path.join(output_dir, "exp35_saebench_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_path}")

    return all_results


def _load_result(output_dir: str, sae_name: str) -> dict:
    for p in Path(output_dir).glob("*.json"):
        if sae_name in p.stem:
            with open(p) as f:
                return json.load(f)
    jsons = sorted(Path(output_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    if jsons:
        with open(jsons[-1]) as f:
            return json.load(f)
    return {"error": "result file not found"}


# =============================================================================
# Main
# =============================================================================

VARIANT_CLASSES = {
    "standard": (BatchTopKSAE, False),
    "adaptive_l2": (AdaptiveCosineBatchTopKSAE, True),
    "perfeature_l2": (PerFeatureAdaptiveCosineSAE, True),
}


def main():
    parser = argparse.ArgumentParser(description="Exp35: Gemma-2-2b SAEs + RAVEL")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_CLASSES.keys()))
    parser.add_argument("--train-only", action="store_true", help="Skip SAEBench eval")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, run SAEBench")
    parser.add_argument("--evals", nargs="+",
                        default=["core", "sparse_probing", "absorption", "scr_and_tpp", "tpp", "ravel"],
                        help="SAEBench evals to run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16, help="LLM batch size for SAEBench")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 35: Gemma-2-2b SAE Training + RAVEL Evaluation")
    print("=" * 70)
    print(f"Model:    {MODEL_NAME}")
    print(f"Layers:   {args.layers}")
    print(f"Variants: {args.variants}")
    print(f"Tokens:   {N_TRAIN_TOKENS:,} train, {N_EVAL_TOKENS:,} eval")
    print(f"SAE:      d_model={D_MODEL}, d_sae={D_SAE}, k={K}")
    print(f"Config:   {json.dumps(get_config_dict(), indent=2)}")
    print()

    torch.manual_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Phase 1: Training
    # =========================================================================

    if not args.eval_only:
        print("\n" + "=" * 70)
        print("PHASE 1: TRAINING")
        print("=" * 70)

        # Load model + tokenizer
        print(f"\nLoading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
            attn_implementation="eager",  # Required for Gemma-2
        )
        model.eval()
        print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

        # Measure mean activation norms for norm-adaptive init (exp27 lesson)
        layer_norms = measure_mean_norms(model, tokenizer, args.layers, n_tokens=100_000)

        # Train all variants at all layers
        all_train_results = {}
        total_t0 = time.time()

        for layer_idx in args.layers:
            print(f"\n{'='*70}")
            print(f"LAYER {layer_idx}")
            print(f"{'='*70}")

            # Collect eval data (shared across variants)
            eval_data = collect_eval_data(model, tokenizer, layer_idx, N_EVAL_TOKENS)

            for variant_name in args.variants:
                cls, needs_norm = VARIANT_CLASSES[variant_name]

                # Create SAE with norm-adaptive init if needed
                if needs_norm:
                    sae = cls(D_MODEL, D_SAE, K, init_norm=layer_norms[layer_idx])
                    print(f"\n  {variant_name}: norm-adaptive init, "
                          f"scale_b = log({layer_norms[layer_idx]:.1f}) = {math.log(layer_norms[layer_idx]):.3f}")
                else:
                    sae = cls(D_MODEL, D_SAE, K)

                sae = sae.to(device=DEVICE, dtype=torch.float32)

                # Train with streaming activations
                stream = ActivationStream(model, tokenizer, layer_idx)
                train_log, ckpt_log = train_sae_streaming(
                    variant_name, sae, stream, layer_idx, SAVE_DIR,
                )
                del stream
                gc.collect()
                torch.cuda.empty_cache()

                # Evaluate reconstruction
                print(f"\n  Evaluating reconstruction for {variant_name}/L{layer_idx}:")
                recon_results = evaluate_reconstruction(variant_name, sae, eval_data, layer_idx)

                # Evaluate ablation
                abl_results = evaluate_ablation(variant_name, model, sae, eval_data, layer_idx)

                all_train_results[f"{variant_name}_L{layer_idx}"] = {
                    "train_log": train_log,
                    "checkpoint_log": ckpt_log,
                    "reconstruction": recon_results,
                    "ablation": abl_results,
                    "init_norm": layer_norms.get(layer_idx),
                }

                del sae
                gc.collect()
                torch.cuda.empty_cache()

            del eval_data
            gc.collect()
            torch.cuda.empty_cache()

        # Save training results
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "config": get_config_dict(),
                "layer_norms": layer_norms,
                "results": all_train_results,
            }, f, indent=2, default=str)
        print(f"\nTraining results saved to {RESULTS_PATH}")

        total_elapsed = time.time() - total_t0
        print(f"\nTotal training time: {total_elapsed/3600:.1f} hours")

        # Free model memory before SAEBench
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Phase 2: SAEBench Evaluation (all 6 evals including RAVEL)
    # =========================================================================

    if not args.train_only:
        print("\n" + "=" * 70)
        print("PHASE 2: SAEBENCH EVALUATION")
        print("=" * 70)

        # Verify checkpoints exist
        missing = []
        for v in args.variants:
            for l in args.layers:
                path = SAVE_DIR / f"{v}_L{l}_final.pt"
                if not path.exists():
                    missing.append(str(path))
        if missing:
            print("ERROR: Missing checkpoints:")
            for m in missing:
                print(f"  {m}")
            print("Run training first (without --eval-only)")
            sys.exit(1)

        saebench_results = run_saebench_evals(
            layers=args.layers,
            variants=args.variants,
            eval_types=args.evals,
            device=args.device,
            llm_batch_size=args.batch_size,
            force_rerun=args.force_rerun,
        )

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"Training results:  {RESULTS_PATH}")
        print(f"SAEBench results:  {SAEBENCH_OUTPUT}/")


if __name__ == "__main__":
    main()
