"""
Experiment 40: Karvonen Training Recipe Replication
====================================================

We found 6 major differences between our SAE training and the
adamkarvonen/saprmarks/dictionary_learning recipe that achieves ~0% dead
features vs our 60-80%.  This experiment adopts ALL of their training tricks
and tests whether the cosine advantage survives:

Recipe changes (from saprmarks BatchTopKTrainer):
  1. Auxiliary k-loss (auxk_alpha=1/32) — resurrects dead features
  2. LR 5e-5 (was 3e-4) — 6x lower, matches their config
  3. Constant LR for 80%, linear decay for final 20% (was cosine from 5%)
  4. Decoder unit-norm constraint every step (gradient projection + renorm)
  5. Encoder init = decoder.T (was decoder * 0.1)
  6. b_dec init = geometric median of first batch (was zeros)
  7. Gradient clipping max_norm=1.0
  8. Adam (was AdamW)
  9. d_sae=65536 to match their actual config (was 16384)
  10. Batch size 2048, giving 244,140 steps for 500M tokens

Variants (3 runs at L18 only — middle layer, moderate norms):
  - standard:       BatchTopKSAE (inner-product encoder)
  - adaptive_l2:    AdaptiveCosineBatchTopKSAE (global adaptive cosine)
  - perfeature_l2:  PerFeatureAdaptiveCosineSAE (per-feature adaptive cosine)

After training, each SAE is evaluated on:
  - Internal metrics: FVE, dead%, alive count, cos→KL ablation
  - SAEBench: core, sparse_probing, absorption

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 experiments/exp40_karvonen_recipe.py 2>&1 | tee experiments/exp40_output.log
"""

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


# =============================================================================
# Configuration — matches adamkarvonen config.json exactly where possible
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

# --- Model ---
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 18                      # Single layer — moderate norms, middle depth
D_MODEL = 4096

# --- SAE ---
D_SAE = 65536                   # Match adamkarvonen (was 16384)
K = 80

# --- Data ---
N_TRAIN_TOKENS = 500_000_000    # 500M
N_EVAL_TOKENS = 2_000_000
CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 4
OUTLIER_MULTIPLIER = 10.0

# --- Training (saprmarks recipe) ---
LR = 5e-5                       # Match adamkarvonen (was 3e-4)
BATCH_SIZE = 2048                # Match adamkarvonen (was 4096)
WARMUP_STEPS = 1000             # Match adamkarvonen
AUXK_ALPHA = 1 / 32             # 0.03125 — auxiliary loss weight
DEAD_FEATURE_THRESHOLD = 10_000_000  # tokens since last fire to count as dead
TOP_K_AUX = D_MODEL // 2        # 2048 — heuristic from paper B.1
THRESHOLD_BETA = 0.999           # EMA for inference threshold
THRESHOLD_START_STEP = 1000
SEED = 42
LOG_EVERY = 500

# --- LR Schedule ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE   # 244,140
DECAY_START = int(0.8 * N_STEPS)         # 195,312 — constant until 80%, then linear

# --- Checkpoints ---
CHECKPOINT_FRACS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CHECKPOINT_STEPS = sorted(set(int(f * N_STEPS) for f in CHECKPOINT_FRACS))

# --- Ablation ---
N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

# --- Output ---
SAVE_DIR = "checkpoints/exp40"
RESULTS_PATH = "experiments/exp40_results.json"

# --- Streaming buffer ---
BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE


def get_config_dict():
    return {
        "experiment": "exp40_karvonen_recipe",
        "model_name": MODEL_NAME, "layer": LAYER, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS, "n_steps": N_STEPS,
        "decay_start": DECAY_START,
        "auxk_alpha": AUXK_ALPHA,
        "dead_feature_threshold": DEAD_FEATURE_THRESHOLD,
        "top_k_aux": TOP_K_AUX,
        "threshold_beta": THRESHOLD_BETA,
        "seed": SEED,
        "recipe": "saprmarks/dictionary_learning BatchTopKTrainer",
        "changes_from_exp36": [
            "aux k-loss (auxk_alpha=1/32)",
            "LR 5e-5 (was 3e-4)",
            "constant+linear LR schedule (was cosine)",
            "decoder unit-norm constraint every step",
            "encoder init = decoder.T (was decoder * 0.1)",
            "b_dec init = geometric median (was zeros)",
            "gradient clipping max_norm=1.0",
            "Adam (was AdamW)",
            "d_sae=65536 (was 16384)",
            "batch_size=2048 (was 4096)",
        ],
    }


# =============================================================================
# Geometric median (from saprmarks)
# =============================================================================

@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute geometric median of a set of points (Weiszfeld's algorithm)."""
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    for _ in range(max_iter):
        prev = guess.clone()
        dists = torch.norm(points - guess, dim=1)
        weights = 1.0 / dists.clamp(min=1e-8)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break

    return guess


# =============================================================================
# Decoder norm helpers (from saprmarks)
# =============================================================================

@torch.no_grad()
def set_decoder_norm_to_unit_norm(W_dec):
    """Normalize decoder rows to unit norm in-place."""
    norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_dec.div_(norms)
    return W_dec


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(W_dec, W_dec_grad):
    """Project out the component of decoder gradient parallel to current directions."""
    normed = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    # Dot product of gradient with normalized direction, per feature
    parallel = (W_dec_grad * normed).sum(dim=1, keepdim=True)
    W_dec_grad -= parallel * normed
    return W_dec_grad


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
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        # saprmarks init: encoder = decoder.T (not decoder * 0.1)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)
            self.b_enc.zero_()

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_active=False):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        post_relu = F.relu(pre_acts)

        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)

        if return_active:
            active_indices = encoded.sum(0) > 0
            return encoded, active_indices, post_relu
        return encoded

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x)
        return self.decode(f), f


class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Cosine encoder with global adaptive per-token scale (exp12 architecture)."""

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
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)
            self.b_enc.zero_()

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_active=False):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)

        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)

        if return_active:
            active_indices = encoded.sum(0) > 0
            return encoded, active_indices, post_relu
        return encoded

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x)
        return self.decode(f), f


class PerFeatureAdaptiveCosineSAE(nn.Module):
    """Per-feature adaptive cosine encoder (exp16 architecture).

    Each feature has its own magnitude sensitivity a_i.
    88-99% of features learn a_i ≈ 0 (pure cosine) at exp16's 5M scale.
    This is the first test at 500M with proper training recipe.
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
        self.scale_a = nn.Parameter(torch.zeros(d_sae))
        self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)
            self.b_enc.zero_()

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_active=False):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        log_norm = torch.log(input_norm)
        # Per-feature: scale_a is [d_sae], log_norm is [batch, 1] → broadcast
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)

        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)

        if return_active:
            active_indices = encoded.sum(0) > 0
            return encoded, active_indices, post_relu
        return encoded

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x)
        return self.decode(f), f


# =============================================================================
# Auxiliary k-loss (ported from saprmarks)
# =============================================================================

def get_auxiliary_loss(
    residual,           # [batch, d_model] — reconstruction error (x - x_hat)
    post_relu_acts,     # [batch, d_sae] — pre-topk activations
    num_tokens_since_fired,  # [d_sae] — token counter per feature
):
    """Auxiliary loss that trains dead features to reconstruct the residual.

    From saprmarks/dictionary_learning BatchTopKTrainer.get_auxiliary_loss().
    Dead features (not fired in DEAD_FEATURE_THRESHOLD tokens) are trained
    to reconstruct whatever the live features are missing.
    """
    dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
    n_dead = int(dead_mask.sum())

    if n_dead == 0:
        return residual.new_zeros(()), n_dead

    k_aux = min(TOP_K_AUX, n_dead)

    # Mask live features to -inf so only dead features compete
    auxk_latents = torch.where(dead_mask[None], post_relu_acts, torch.tensor(-torch.inf, device=post_relu_acts.device))

    # Top-k among dead features only
    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

    auxk_buffer = torch.zeros_like(post_relu_acts)
    auxk_acts_BF = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

    return auxk_acts_BF, n_dead


# =============================================================================
# LR Schedule (saprmarks recipe: constant + linear tail)
# =============================================================================

def make_lr_schedule(total_steps, warmup_steps, decay_start):
    """Constant LR with warmup and late linear decay."""
    def schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step >= decay_start:
            return (total_steps - step) / max(total_steps - decay_start, 1)
        return 1.0
    return schedule


# =============================================================================
# Streaming Activation Collection (from exp36)
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
    """Streams activations from FineWeb through model."""

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
# Training — full saprmarks recipe
# =============================================================================

def train_sae(name, sae, stream, save_dir, checkpoint_steps):
    """Train SAE with the full saprmarks recipe."""
    tag = f"{name}/L{LAYER}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")
    print(f"  Recipe: auxk_alpha={AUXK_ALPHA}, decay_start={DECAY_START}, "
          f"grad_clip=1.0, decoder_norm=unit")

    # Adam (not AdamW) — matches saprmarks
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    schedule_fn = make_lr_schedule(N_STEPS, WARMUP_STEPS, DECAY_START)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    # Dead feature tracking
    num_tokens_since_fired = torch.zeros(D_SAE, dtype=torch.long, device=DEVICE)

    sae.train()
    log = []
    checkpoints_saved = {}
    b_dec_initialized = False
    t0 = time.time()
    global_step = 0

    while global_step < N_STEPS:
        stream.fill_buffer()
        steps_in_buffer = min(BUFFER_BATCHES, N_STEPS - global_step)

        for buf_step in range(steps_in_buffer):
            batch = stream.get_batch(buf_step)

            # --- Step 0: initialize b_dec to geometric median ---
            if not b_dec_initialized:
                with torch.no_grad():
                    median = geometric_median(batch)
                    sae.b_dec.data.copy_(median.to(sae.b_dec.dtype))
                b_dec_initialized = True
                print(f"    [{tag}] b_dec initialized to geometric median "
                      f"(norm={median.norm():.1f})")

            # --- Forward pass with active feature tracking ---
            x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)

            # --- Reconstruction loss ---
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            # --- Update dead feature counters ---
            did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
            did_fire[active_indices] = True
            num_tokens_since_fired += batch.shape[0]
            num_tokens_since_fired[did_fire] = 0

            # --- Auxiliary k-loss for dead features ---
            residual = (batch - x_hat).detach()  # detach: aux loss doesn't backprop through main path
            auxk_acts, n_dead = get_auxiliary_loss(
                residual, post_relu_acts, num_tokens_since_fired
            )

            if n_dead > 0:
                # Reconstruct residual using only dead features
                # Use decoder directly (no b_dec) to get dead feature reconstruction
                x_reconstruct_aux = auxk_acts @ sae.W_dec  # no + b_dec (matching saprmarks)
                auxk_l2 = (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()

                # Normalize by variance of residual (from OpenAI's implementation)
                residual_mu = residual.mean(dim=0, keepdim=True)
                loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
                auxk_loss = (auxk_l2 / loss_denom.clamp(min=1e-8)).nan_to_num(0.0)
            else:
                auxk_loss = torch.tensor(0.0, device=DEVICE)

            loss = recon_loss + AUXK_ALPHA * auxk_loss

            # --- Backward + optimizer step ---
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Decoder gradient projection (remove parallel component)
            if sae.W_dec.grad is not None:
                sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                    sae.W_dec.data, sae.W_dec.grad.data
                )

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Decoder unit-norm constraint
            set_decoder_norm_to_unit_norm(sae.W_dec.data)

            # Update inference threshold (EMA of minimum active value)
            if global_step >= THRESHOLD_START_STEP:
                with torch.no_grad():
                    active_vals = features[features > 0]
                    if active_vals.numel() > 0:
                        min_active = active_vals.min().float()
                        if sae.threshold < 0:
                            sae.threshold.fill_(min_active)
                        else:
                            sae.threshold.mul_(THRESHOLD_BETA).add_(
                                (1 - THRESHOLD_BETA) * min_active
                            )

            global_step += 1

            # --- Logging ---
            if global_step % LOG_EVERY == 0 or global_step == N_STEPS:
                with torch.no_grad():
                    l0 = (features != 0).float().sum(dim=-1).mean().item()
                    total_var = torch.var(batch, dim=0, unbiased=False).sum()
                    resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                    fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                    cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                    dead_frac = (num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD).float().mean().item()

                entry = {
                    "step": global_step,
                    "recon_loss": recon_loss.item(),
                    "auxk_loss": auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                    "total_loss": loss.item(),
                    "l0": l0, "fve": fve, "cos_recon": cos_r,
                    "dead_frac": dead_frac,
                    "n_dead": n_dead,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": global_step * BATCH_SIZE,
                    "threshold": sae.threshold.item(),
                }

                if hasattr(sae, "scale_a"):
                    if sae.scale_a.dim() == 0:
                        entry["scale_a"] = sae.scale_a.item()
                        scale_str = f" | a={sae.scale_a.item():.4f}"
                    else:
                        entry["scale_a_mean"] = sae.scale_a.mean().item()
                        entry["scale_a_median"] = sae.scale_a.median().item()
                        entry["scale_a_pct_near_zero"] = (sae.scale_a.abs() < 0.05).float().mean().item()
                        scale_str = f" | a_mean={sae.scale_a.mean().item():.4f}"
                else:
                    scale_str = ""

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag}] step {global_step:>6d}/{N_STEPS} | "
                      f"loss={loss.item():.1f} | recon={recon_loss.item():.1f} | "
                      f"auxk={auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else 0:.3f} | "
                      f"L0={l0:.0f} | FVE={fve:.4f} | "
                      f"dead={dead_frac:.3f} ({n_dead:,}){scale_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | ETA {eta_sec/3600:.1f}h")

            # --- Mid-training checkpoints ---
            if global_step in checkpoint_steps:
                frac = global_step / N_STEPS
                ckpt_path = save_dir / f"{name}_L{LAYER}_step{global_step}.pt"
                torch.save({
                    "state_dict": sae.state_dict(),
                    "num_tokens_since_fired": num_tokens_since_fired,
                    "step": global_step,
                }, ckpt_path)
                checkpoints_saved[global_step] = str(ckpt_path)
                print(f"    [{tag}] Checkpoint saved at {frac:.0%} ({global_step} steps)")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

    # Save final checkpoint
    final_path = save_dir / f"{name}_L{LAYER}_final.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "num_tokens_since_fired": num_tokens_since_fired,
        "step": global_step,
    }, final_path)
    checkpoints_saved["final"] = str(final_path)

    return log, checkpoints_saved


# =============================================================================
# Evaluation — reconstruction + ablation (from exp36)
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data):
    tag = f"{name}/L{LAYER}"
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

        alive = features.sum(dim=0) != 0
        if dead_counts is None:
            dead_counts = ~alive
        else:
            dead_counts &= ~alive

    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0
    fve = 1 - resid_var_sum / total_var_sum if total_var_sum > 0 else 0

    results = {
        "fve": fve,
        "mean_recon_loss": np.mean(recon_losses),
        "cos_recon": np.mean(cos_sims),
        "mean_l0": np.mean(l0s),
        "dead_frac": dead_frac,
        "alive_count": alive_count,
    }
    print(f"    [{tag}] FVE={fve:.4f} | dead={dead_frac:.3f} | "
          f"alive={alive_count:,} | L0={np.mean(l0s):.1f} | "
          f"cos_recon={np.mean(cos_sims):.4f}")
    return results


@torch.no_grad()
def evaluate_ablation(name, model, tokenizer, sae, eval_texts):
    """Ablation evaluation using hook-based intervention on real text.

    For each feature: run text through model with and without the feature's
    projection removed from the residual stream at the target layer,
    measure KL divergence, correlate with cos/inner/SAE activation.
    """
    tag = f"{name}/L{LAYER}"
    print(f"    [{tag}] Ablation evaluation ({N_ABLATION_FEATURES} features, "
          f"{N_ABLATION_SAMPLES} samples)...")

    sae.eval()

    # Collect activations to find alive features with highest mean activation
    act_sums = torch.zeros(D_SAE, device=DEVICE)
    act_counts = torch.zeros(D_SAE, device=DEVICE)
    all_sample_acts = []

    for text in eval_texts[:200]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=CTX_LEN).to(DEVICE)
        acts = _collect_layer_acts(model, LAYER, inputs)
        flat = acts[0]  # [seq_len, d_model]
        # Filter attention sinks
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            mask = norms < median * OUTLIER_MULTIPLIER
            flat = flat[mask]
        if flat.shape[0] == 0:
            continue
        _, features = sae(flat.float())
        act_sums += features.sum(dim=0)
        act_counts += (features != 0).float().sum(dim=0)

    alive_mask = act_counts > 0
    if alive_mask.sum() == 0:
        print(f"    [{tag}] No alive features for ablation!")
        return {"error": "no_alive_features"}

    mean_acts = act_sums / act_counts.clamp(min=1)
    mean_acts[~alive_mask] = -1

    n_feat = min(N_ABLATION_FEATURES, alive_mask.sum().item())
    top_features = mean_acts.topk(n_feat).indices.cpu().tolist()

    # Ablation loop using hooks
    all_results = []
    sample_texts = eval_texts[:N_ABLATION_SAMPLES]

    for fi_idx, fi in enumerate(top_features):
        feat_dir = sae.W_dec[fi].float()
        feat_dir = feat_dir / feat_dir.norm()

        cos_vals, inner_vals, sae_vals, norm_vals, kl_vals = [], [], [], [], []

        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=CTX_LEN).to(DEVICE)
            seq_len = inputs["input_ids"].shape[1]
            if seq_len < 2:
                continue

            # Clean forward pass
            clean_out = model(**inputs)
            clean_logits = clean_out.logits[0, -1]
            clean_probs = F.softmax(clean_logits.float(), dim=-1)

            # Collect activation at target layer for this input
            layer_acts = _collect_layer_acts(model, LAYER, inputs)
            x = layer_acts[0, -1].float()  # last token position

            # Ablated forward pass: remove feature projection via hook
            def ablation_hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # Ablate at all positions (matching exp36 approach)
                proj = (h.float() @ feat_dir).unsqueeze(-1) * feat_dir.unsqueeze(0).unsqueeze(0)
                h_new = h.float() - proj
                if isinstance(out, tuple):
                    return (h_new.to(out[0].dtype),) + out[1:]
                return h_new.to(out.dtype)

            layer_module = model.model.layers[LAYER]
            handle = layer_module.register_forward_hook(ablation_hook)
            abl_out = model(**inputs)
            handle.remove()

            abl_logits = abl_out.logits[0, -1]
            abl_probs = F.softmax(abl_logits.float(), dim=-1)

            kl = F.kl_div(
                abl_probs.clamp(min=1e-10).log(), clean_probs,
                reduction="sum"
            ).item()

            # Metrics for last token
            cos_v = F.cosine_similarity(x.unsqueeze(0), feat_dir.unsqueeze(0)).item()
            inner_v = (x @ feat_dir).item()
            _, sae_acts = sae(x.unsqueeze(0))
            sae_v = sae_acts[0, fi].item()
            norm_v = x.norm().item()

            cos_vals.append(cos_v)
            inner_vals.append(inner_v)
            sae_vals.append(sae_v)
            norm_vals.append(norm_v)
            kl_vals.append(kl)

        if len(kl_vals) > 1 and np.std(kl_vals) > 0:
            all_results.append({
                "feature_idx": fi,
                "cos_kl_corr": float(np.corrcoef(cos_vals, kl_vals)[0, 1]),
                "inner_kl_corr": float(np.corrcoef(inner_vals, kl_vals)[0, 1]),
                "sae_kl_corr": float(np.corrcoef(sae_vals, kl_vals)[0, 1]),
                "norm_kl_corr": float(np.corrcoef(norm_vals, kl_vals)[0, 1]),
                "mean_kl": float(np.mean(kl_vals)),
            })

        if (fi_idx + 1) % 10 == 0:
            print(f"      Ablated {fi_idx+1}/{n_feat} features...")

    if not all_results:
        return {"error": "no_valid_features"}

    cos_wins = sum(1 for r in all_results if r["cos_kl_corr"] > r["inner_kl_corr"])
    agg = {
        "n_features": len(all_results),
        "cos_kl_mean": np.mean([r["cos_kl_corr"] for r in all_results]),
        "inner_kl_mean": np.mean([r["inner_kl_corr"] for r in all_results]),
        "sae_kl_mean": np.mean([r["sae_kl_corr"] for r in all_results]),
        "norm_kl_mean": np.mean([r["norm_kl_corr"] for r in all_results]),
        "cos_wins_inner": f"{cos_wins}/{len(all_results)}",
    }
    print(f"    [{tag}] Ablation: cos>inner {cos_wins}/{len(all_results)} | "
          f"cos→KL={agg['cos_kl_mean']:.3f} | inner→KL={agg['inner_kl_mean']:.3f} | "
          f"SAE→KL={agg['sae_kl_mean']:.3f}")

    return {"aggregate": agg, "per_feature": all_results}


# =============================================================================
# SAEBench integration
# =============================================================================

def run_saebench_eval(name, sae):
    """Wrap SAE in BenchSAE adapter and run SAEBench evaluations."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from benchmarks.adapter import BenchSAE
        from benchmarks.run_saebench import run_saebench
    except ImportError as e:
        print(f"    SAEBench import failed: {e}")
        print("    Skipping SAEBench evaluation. Install sae-bench and ensure benchmarks/ is accessible.")
        return None

    print(f"\n  SAEBench evaluation for {name}/L{LAYER}...")

    # Build encode/decode closures
    _sae = sae.eval()

    def encode_fn(x):
        return _sae.encode(x)

    def decode_fn(f):
        return _sae.decode(f)

    # Get weight tensors in SAEBench format
    W_dec = sae.W_dec.detach()  # [d_sae, d_model]
    W_dec_normed = F.normalize(W_dec, dim=1)
    W_enc = sae.W_enc.detach()  # [d_sae, d_model] → need [d_model, d_sae]

    bench_sae = BenchSAE(
        W_enc=W_enc.T,          # [d_model, d_sae]
        W_dec=W_dec_normed,     # [d_sae, d_model]
        b_enc=sae.b_enc.detach(),
        b_dec=sae.b_dec.detach(),
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name=MODEL_NAME,
        hook_layer=LAYER,
        device=DEVICE,
        dtype=DTYPE,
    )

    sae_name = f"exp40-{name}-L{LAYER}"
    results = run_saebench(
        bench_sae,
        sae_name=sae_name,
        eval_types=["core", "sparse_probing", "absorption"],
        output_dir="benchmarks/eval_results/exp40",
        llm_batch_size=4,  # Conservative for 8B model + 65k SAE
        device=DEVICE,
    )

    return results


# =============================================================================
# Main
# =============================================================================

VARIANTS = [
    ("standard",      BatchTopKSAE),
    ("adaptive_l2",   AdaptiveCosineBatchTopKSAE),
    ("perfeature_l2", PerFeatureAdaptiveCosineSAE),
]


def main():
    print("Experiment 40: Karvonen Training Recipe Replication")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"d_sae={D_SAE}, k={K}, lr={LR}, {N_TRAIN_TOKENS:,} tokens ({N_STEPS:,} steps)")
    print(f"Batch size: {BATCH_SIZE}, Decay start: {DECAY_START}")
    print(f"Aux loss: auxk_alpha={AUXK_ALPHA}, dead_threshold={DEAD_FEATURE_THRESHOLD:,}")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Estimated time: ~{len(VARIANTS) * 20} hours on A100")
    print(f"\nKey differences from exp36:")
    for change in get_config_dict()["changes_from_exp36"]:
        print(f"  + {change}")

    # ---- Load model ----
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation="eager",  # Avoid cuDNN SDPA errors on some H100 setups
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing results ----
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        existing = list(all_results.get("runs", {}).keys())
        print(f"  Loaded existing results: {existing}")
    else:
        all_results = {"config": get_config_dict(), "runs": {}}

    # ---- Collect eval data ----
    eval_data, mean_norm = collect_eval_data(model, tokenizer, LAYER, N_EVAL_TOKENS)
    all_results["config"]["mean_norm"] = mean_norm

    # Collect eval texts for ablation (separate from activation eval data)
    print("  Collecting eval texts for ablation...")
    ds_eval = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    eval_texts = []
    for i, row in enumerate(ds_eval):
        if i < 600_000:  # skip past training region
            continue
        if len(row["text"]) > 200:
            eval_texts.append(row["text"][:2048])
        if len(eval_texts) >= 500:
            break
    print(f"    Collected {len(eval_texts)} eval texts")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- Train each variant ----
    for vname, vcls in VARIANTS:
        run_name = f"{vname}_L{LAYER}"

        if run_name in all_results.get("runs", {}):
            print(f"\n  {run_name} already complete, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  RUN: {run_name}")
        print(f"{'='*70}")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        sae = vcls(D_MODEL, D_SAE, K).to(DEVICE)
        print(f"    SAE params: {sum(p.numel() for p in sae.parameters()):,}")

        stream = ActivationStream(model, tokenizer, LAYER, seed=SEED)

        # Train
        train_log, ckpt_paths = train_sae(
            vname, sae, stream, save_dir, CHECKPOINT_STEPS
        )

        # Load final checkpoint
        final_path = ckpt_paths.get("final")
        if final_path and os.path.exists(final_path):
            ckpt = torch.load(final_path, map_location=DEVICE, weights_only=False)
            sae.load_state_dict(ckpt["state_dict"])

        # Evaluate reconstruction
        print(f"\n  Evaluation — {run_name}")
        recon = evaluate_reconstruction(vname, sae, eval_data)

        # Ablation evaluation
        abl = evaluate_ablation(vname, model, tokenizer, sae, eval_texts)

        # Save results before SAEBench (in case it crashes)
        run_result = {
            "encoder": vname,
            "layer": LAYER,
            "training": train_log,
            "reconstruction": recon,
            "ablation": abl,
            "saebench": None,
        }
        if hasattr(sae, "scale_a"):
            if sae.scale_a.dim() == 0:
                run_result["scale_a_final"] = sae.scale_a.item()
            else:
                run_result["scale_a_stats"] = {
                    "mean": sae.scale_a.mean().item(),
                    "median": sae.scale_a.median().item(),
                    "std": sae.scale_a.std().item(),
                    "pct_near_zero": (sae.scale_a.abs() < 0.05).float().mean().item(),
                    "min": sae.scale_a.min().item(),
                    "max": sae.scale_a.max().item(),
                }

        all_results["runs"][run_name] = run_result

        # Save before SAEBench (crash-safe)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Core results saved to {RESULTS_PATH}")

        # SAEBench evaluation (optional — may fail on disk/memory)
        try:
            saebench_results = run_saebench_eval(vname, sae)
            run_result["saebench"] = saebench_results
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  SAEBench results saved")
        except Exception as e:
            print(f"  SAEBench failed: {e}")
            print(f"  Continuing to next variant...")

        del sae, stream
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  EXP40 SUMMARY — KARVONEN RECIPE")
    print(f"{'='*70}")

    header = f"  {'Variant':<16s} | {'FVE':>6s} | {'Dead%':>6s} | {'Alive':>7s} | {'L0':>5s} | {'cos_r':>6s}"
    print(header)
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}")

    for vname, _ in VARIANTS:
        run = all_results["runs"].get(f"{vname}_L{LAYER}", {})
        r = run.get("reconstruction", {})
        if r:
            print(f"  {vname:<16s} | {r.get('fve',0):.4f} | "
                  f"{r.get('dead_frac',0)*100:5.1f}% | "
                  f"{r.get('alive_count',0):>7,d} | "
                  f"{r.get('mean_l0',0):>5.1f} | "
                  f"{r.get('cos_recon',0):.4f}")

    # SAEBench summary
    print(f"\n  SAEBench Results:")
    for vname, _ in VARIANTS:
        run = all_results["runs"].get(f"{vname}_L{LAYER}", {})
        sb = run.get("saebench")
        if sb and isinstance(sb, dict):
            core = sb.get("core", {})
            sp = sb.get("sparse_probing", {})
            ab = sb.get("absorption", {})
            print(f"  {vname}: core={core}, sparse_probing={sp}, absorption={ab}")

    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
