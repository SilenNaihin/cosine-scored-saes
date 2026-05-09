"""
Experiment 47: Making per-feature cosine SAEs robust at L27/50M
===============================================================

Per-feature L2 is our best architecture at 500M (0.815 sparse probing top-1,
+14.9pp over standard) but catastrophically fails at L27/50M: 83% dead features
within ~1M tokens due to a winner-take-all feedback loop.

The root cause (see analysis doc §"Cross-Experiment Analysis"):
  - scale_b inits at log(sqrt(d))=log(64), but L27 norms are ~405 (6.3x gap)
  - With 65K independent a_i params and k=80 TopK, each feature fires on ~0.12%
    of tokens → ~12K gradient updates per feature in 50M tokens
  - Some features randomly get better a_i → win more TopK slots → more gradient
    → threshold rises → 43K features die in a single 500-step window (step 5000→5500)
  - Adaptive (1 global a) survives because all features share the same gradient

The init story is also messy across the project:
  - sqrt(d) works on Qwen undershoot (norms > sqrt(d)) but kills Mistral overshoot
  - norm-adaptive works at 5M but suppresses scale_a learning at 50M+ (exp34)
  - Neither init is universally correct

This experiment tests three structural fixes that aim to make per-feature robust
regardless of the init:

Variants:
  1. perfeature_original:   Control — standard per-feature (a_i=0, known 83% dead)
  2. perfeature_base_delta: a_eff = a_base (scalar, shared gradient) + a_delta (per-feature)
                            Base acts like adaptive during early training; deltas specialize later
  3. perfeature_var_reg:    Original per-feature + variance regularization on a_i
                            Decaying penalty keeps a_i values close during critical first 5K steps
  4. perfeature_gaussian:   Original per-feature but a_i ~ N(0, 0.05) init
                            Tests whether breaking a_i symmetry alone prevents the cascade
  5. adaptive_l2:           Reference — single global a (known to work: 0.05% dead)

All at L27 (norm ~405), 50M tokens, saprmarks recipe — the exact failure condition.

Run on <gpu-server>:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp47_perfeature_robustness.py \
        > experiments/exp47_output.log 2>&1 &
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN SDPA backend — broken on H100 with driver 595.58 / cuDNN 9.1
torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 27
D_MODEL = 4096

D_SAE = 65536
K = 80

N_TRAIN_TOKENS = 50_000_000
N_EVAL_TOKENS = 2_000_000
CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 4
OUTLIER_MULTIPLIER = 10.0

# saprmarks recipe
LR = 5e-5
BATCH_SIZE = 2048
WARMUP_STEPS = 1000
AUXK_ALPHA = 1 / 32
DEAD_FEATURE_THRESHOLD = 10_000_000
TOP_K_AUX = D_MODEL // 2
THRESHOLD_BETA = 0.999
THRESHOLD_START_STEP = 1000
SEED = 42
LOG_EVERY = 500
NORM_EPS = 1e-8

# Variance regularization schedule (for perfeature_var_reg)
VAR_REG_WEIGHT = 0.1
VAR_REG_DECAY_START = 2000
VAR_REG_DECAY_END = 8000

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE  # 24,414
DECAY_START = int(0.8 * N_STEPS)        # 19,531

BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE

CHECKPOINT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
CHECKPOINT_STEPS = sorted(set(int(f * N_STEPS) for f in CHECKPOINT_FRACS))

N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

SAVE_DIR = Path("/mnt/nvme0/checkpoints/exp47")
RESULTS_PATH = Path("experiments/exp47_results.json")


# =============================================================================
# Geometric median
# =============================================================================

@torch.no_grad()
def geometric_median(points, max_iter=100, tol=1e-5):
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess.clone()
        dists = torch.norm(points - guess, dim=1).clamp(min=1e-8)
        weights = 1.0 / dists
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


# =============================================================================
# Decoder norm helpers
# =============================================================================

@torch.no_grad()
def set_decoder_norm_to_unit_norm(W_dec):
    norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_dec.div_(norms)
    return W_dec

@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(W_dec, W_dec_grad):
    normed = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    parallel = (W_dec_grad * normed).sum(dim=1, keepdim=True)
    W_dec_grad -= parallel * normed
    return W_dec_grad


# =============================================================================
# SAE Architectures
# =============================================================================

class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Reference: single global scale_a. Known to work at L27/50M."""
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
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


class PerFeatureOriginalSAE(nn.Module):
    """Control: standard per-feature a_i (all init 0). Known to fail at L27/50M."""
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
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


class PerFeatureBaseDeltaSAE(nn.Module):
    """Fix 1: shared base + per-feature delta.

    a_effective = a_base (scalar) + a_delta (per-feature)

    a_base gets gradient from every feature on every batch (like adaptive),
    so it converges in ~5K steps regardless of layer norms. a_delta allows
    per-feature specialization once the base is established. This structurally
    prevents the divergence cascade because all features share the same base
    magnitude sensitivity.
    """
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a_base = nn.Parameter(torch.tensor(0.0))
        self.scale_a_delta = nn.Parameter(torch.zeros(d_sae))
        self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    @property
    def scale_a(self):
        return self.scale_a_base + self.scale_a_delta

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
        a_effective = self.scale_a_base + self.scale_a_delta
        scale = torch.exp(a_effective * log_norm + self.scale_b)
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


class PerFeatureVarRegSAE(nn.Module):
    """Fix 2: standard per-feature + variance regularization on a_i.

    Adds var(scale_a) penalty to loss, decaying over training. During early
    training (steps 0-2000), strong regularization keeps all a_i close together
    (behaves like adaptive). The penalty decays linearly to zero by step 8000,
    allowing full per-feature specialization in the second half of training.

    This is the "soft" version of base+delta: instead of structurally tying
    features, we use a loss penalty to achieve the same effect temporarily.
    """
    VAR_REG = True

    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
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

    def get_var_reg(self, step):
        if step >= VAR_REG_DECAY_END:
            return torch.tensor(0.0, device=self.scale_a.device)
        if step < VAR_REG_DECAY_START:
            weight = VAR_REG_WEIGHT
        else:
            frac = (step - VAR_REG_DECAY_START) / (VAR_REG_DECAY_END - VAR_REG_DECAY_START)
            weight = VAR_REG_WEIGHT * (1.0 - frac)
        return weight * torch.var(self.scale_a)


class PerFeatureGaussianInitSAE(nn.Module):
    """Fix 3: standard per-feature but a_i ~ N(0, 0.05).

    Tests whether breaking the symmetry of a_i initialization prevents the
    winner-take-all cascade. If all a_i start at 0, the only source of
    competitive advantage is random feature direction quality. With Gaussian
    init, features start with different magnitude sensitivities, which may
    distribute TopK wins more evenly during early training.

    Note: this does NOT fix the 6.3x scale mismatch — mean(a_i) is still ~0.
    If the cascade is driven by the mismatch rather than the symmetry, this
    variant will fail similarly to the control.
    """
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.randn(d_sae) * 0.05)
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
# Auxiliary k-loss
# =============================================================================

def get_auxiliary_loss(residual, post_relu_acts, num_tokens_since_fired):
    dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return residual.new_zeros(()), n_dead
    k_aux = min(TOP_K_AUX, n_dead)
    auxk_latents = torch.where(
        dead_mask[None], post_relu_acts,
        torch.tensor(-torch.inf, device=post_relu_acts.device)
    )
    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
    auxk_buffer = torch.zeros_like(post_relu_acts)
    auxk_acts_BF = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
    return auxk_acts_BF, n_dead


# =============================================================================
# LR Schedule
# =============================================================================

def make_lr_schedule(total_steps, warmup_steps, decay_start):
    def schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step >= decay_start:
            return (total_steps - step) / max(total_steps - decay_start, 1)
        return 1.0
    return schedule


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
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
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
# Training
# =============================================================================

def train_sae(name, sae, stream, save_dir, checkpoint_steps):
    tag = f"{name}/L{LAYER}"
    has_var_reg = getattr(sae, "VAR_REG", False)
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")
    if has_var_reg:
        print(f"    Variance reg: weight={VAR_REG_WEIGHT}, "
              f"decay {VAR_REG_DECAY_START}-{VAR_REG_DECAY_END}")

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    schedule_fn = make_lr_schedule(N_STEPS, WARMUP_STEPS, DECAY_START)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

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

            if not b_dec_initialized:
                with torch.no_grad():
                    median = geometric_median(batch)
                    sae.b_dec.data.copy_(median.to(sae.b_dec.dtype))
                b_dec_initialized = True
                print(f"    [{tag}] b_dec init (norm={median.norm():.1f})")

            x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
            did_fire[active_indices] = True
            num_tokens_since_fired += batch.shape[0]
            num_tokens_since_fired[did_fire] = 0

            residual = (batch - x_hat).detach()
            auxk_acts, n_dead = get_auxiliary_loss(
                residual, post_relu_acts, num_tokens_since_fired
            )

            if n_dead > 0:
                x_reconstruct_aux = auxk_acts @ sae.W_dec
                auxk_l2 = (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
                residual_mu = residual.mean(dim=0, keepdim=True)
                loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
                auxk_loss = (auxk_l2 / loss_denom.clamp(min=1e-8)).nan_to_num(0.0)
            else:
                auxk_loss = torch.tensor(0.0, device=DEVICE)

            loss = recon_loss + AUXK_ALPHA * auxk_loss

            if has_var_reg:
                var_reg = sae.get_var_reg(global_step)
                loss = loss + var_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if sae.W_dec.grad is not None:
                sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                    sae.W_dec.data, sae.W_dec.grad.data
                )

            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            set_decoder_norm_to_unit_norm(sae.W_dec.data)

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

            if global_step % LOG_EVERY == 0 or global_step == N_STEPS:
                with torch.no_grad():
                    l0 = (features != 0).float().sum(dim=-1).mean().item()
                    total_var = torch.var(batch, dim=0, unbiased=False).sum()
                    resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                    fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                    cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                    dead_frac = (num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD).float().mean().item()

                entry = {
                    "step": global_step, "recon_loss": recon_loss.item(),
                    "auxk_loss": auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                    "total_loss": loss.item(), "l0": l0, "fve": fve,
                    "cos_recon": cos_r, "dead_frac": dead_frac, "n_dead": n_dead,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": global_step * BATCH_SIZE,
                    "threshold": sae.threshold.item(),
                }
                # Log scale_a stats for all variants
                if hasattr(sae, "scale_a"):
                    sa = sae.scale_a
                    if sa.dim() == 0:
                        entry["scale_a"] = sa.item()
                    else:
                        entry["scale_a_mean"] = sa.mean().item()
                        entry["scale_a_std"] = sa.std().item()
                        entry["scale_a_median"] = sa.median().item()
                if hasattr(sae, "scale_a_base"):
                    entry["scale_a_base"] = sae.scale_a_base.item()
                    entry["scale_a_delta_std"] = sae.scale_a_delta.std().item()
                    entry["scale_a_delta_mean"] = sae.scale_a_delta.mean().item()
                if has_var_reg:
                    entry["var_reg"] = sae.get_var_reg(global_step).item()

                log.append(entry)
                elapsed = time.time() - t0
                tok = global_step * BATCH_SIZE
                tok_per_sec = tok / elapsed if elapsed > 0 else 0
                eta = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                scale_str = ""
                if hasattr(sae, "scale_a_base"):
                    scale_str = (f" a_base={sae.scale_a_base.item():.4f}"
                                 f" delta_std={sae.scale_a_delta.std().item():.4f}")
                elif hasattr(sae, "scale_a"):
                    sa = sae.scale_a
                    if sa.dim() == 0:
                        scale_str = f" a={sa.item():.4f}"
                    else:
                        scale_str = f" a_mean={sa.mean().item():.4f} a_std={sa.std().item():.4f}"
                vr_str = ""
                if has_var_reg:
                    vr_str = f" vreg={sae.get_var_reg(global_step).item():.6f}"
                print(f"    [{tag}] {global_step:>5d}/{N_STEPS} | "
                      f"loss={loss.item():.1f} recon={recon_loss.item():.1f} "
                      f"auxk={auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else 0:.3f} | "
                      f"FVE={fve:.4f} L0={l0:.0f} dead={dead_frac:.3f}({n_dead})"
                      f"{scale_str}{vr_str} | {tok/1e6:.1f}M ETA {eta/3600:.1f}h")

            if global_step in checkpoint_steps:
                ckpt_path = save_dir / f"{name}_L{LAYER}_step{global_step}.pt"
                torch.save({
                    "state_dict": sae.state_dict(),
                    "num_tokens_since_fired": num_tokens_since_fired,
                    "step": global_step,
                }, ckpt_path)
                print(f"    [{tag}] Checkpoint at step {global_step}")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

    final_path = save_dir / f"{name}_L{LAYER}_final.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "num_tokens_since_fired": num_tokens_since_fired,
        "step": global_step,
    }, final_path)
    checkpoints_saved["final"] = str(final_path)

    return log, checkpoints_saved


# =============================================================================
# Evaluation
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
        "fve": fve, "cos_recon": np.mean(cos_sims), "mean_l0": np.mean(l0s),
        "dead_frac": dead_frac, "alive_count": alive_count,
    }
    print(f"    [{tag}] FVE={fve:.4f} dead={dead_frac:.3f} "
          f"alive={alive_count:,} L0={np.mean(l0s):.1f}")
    return results


@torch.no_grad()
def evaluate_ablation(name, model, tokenizer, sae):
    tag = f"{name}/L{LAYER}"
    print(f"    [{tag}] Ablation ({N_ABLATION_FEATURES} features, "
          f"{N_ABLATION_SAMPLES} samples)...")

    sae.eval()

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                       split="train", streaming=True)
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= 600_000:
            break

    texts = []
    while len(texts) < N_ABLATION_SAMPLES:
        try:
            row = next(text_iter)
            if len(row["text"]) > 100:
                texts.append(row["text"][:4096])
        except StopIteration:
            break

    act_sums = torch.zeros(D_SAE, device=DEVICE)
    for text in texts[:50]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=CTX_LEN).to(DEVICE)
        acts = _collect_layer_acts(model, LAYER, inputs)
        flat = acts.reshape(-1, D_MODEL)
        features = sae.encode(flat)
        act_sums += features.sum(dim=0)

    top_features = act_sums.topk(N_ABLATION_FEATURES).indices.tolist()

    cos_wins = 0
    cos_kl_corrs, inner_kl_corrs = [], []

    for feat_idx in top_features:
        feat_dir = sae.W_dec[feat_idx]
        feat_dir_unit = feat_dir / feat_dir.norm().clamp(min=1e-8)

        cos_vals, inner_vals, kl_vals = [], [], []

        for text in texts[:N_ABLATION_SAMPLES]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=CTX_LEN).to(DEVICE)

            with torch.no_grad():
                outputs_clean = model(**inputs)
                clean_logits = outputs_clean.logits[0, -1]
                clean_probs = F.softmax(clean_logits, dim=-1)

            act = _collect_layer_acts(model, LAYER, inputs)
            act_flat = act[0, -1].float()

            cos_sim = F.cosine_similarity(act_flat.unsqueeze(0),
                                          feat_dir_unit.unsqueeze(0)).item()
            inner_prod = (act_flat * feat_dir).sum().item()

            proj = (act_flat @ feat_dir_unit) * feat_dir_unit
            ablated_act = act_flat - proj

            def ablation_hook(module, inp, out):
                result = out[0] if isinstance(out, tuple) else out
                result = result.clone()
                result[0, -1] = ablated_act.to(result.dtype)
                if isinstance(out, tuple):
                    return (result,) + out[1:]
                return result

            handle = model.model.layers[LAYER].register_forward_hook(ablation_hook)
            with torch.no_grad():
                outputs_abl = model(**inputs)
                abl_logits = outputs_abl.logits[0, -1]
                abl_probs = F.softmax(abl_logits, dim=-1)
            handle.remove()

            kl = F.kl_div(abl_probs.log(), clean_probs, reduction="sum").item()
            cos_vals.append(abs(cos_sim))
            inner_vals.append(abs(inner_prod))
            kl_vals.append(kl)

        if len(kl_vals) > 2:
            cos_arr = np.array(cos_vals)
            inner_arr = np.array(inner_vals)
            kl_arr = np.array(kl_vals)
            cos_corr = np.corrcoef(cos_arr, kl_arr)[0, 1] if cos_arr.std() > 0 else 0
            inner_corr = np.corrcoef(inner_arr, kl_arr)[0, 1] if inner_arr.std() > 0 else 0
            cos_kl_corrs.append(cos_corr)
            inner_kl_corrs.append(inner_corr)
            if cos_corr > inner_corr:
                cos_wins += 1

    results = {
        "n_features": N_ABLATION_FEATURES,
        "cos_wins_inner": f"{cos_wins}/{len(cos_kl_corrs)}",
        "cos_kl_mean": float(np.mean(cos_kl_corrs)) if cos_kl_corrs else 0,
        "inner_kl_mean": float(np.mean(inner_kl_corrs)) if inner_kl_corrs else 0,
    }
    print(f"    [{tag}] cos>inner: {cos_wins}/{len(cos_kl_corrs)}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Experiment 47: Per-feature robustness fixes at L{LAYER}")
    print(f"  {N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")
    print(f"  Variants: perfeature_original, perfeature_base_delta, "
          f"perfeature_var_reg, perfeature_gaussian, adaptive_l2")
    print()

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()

    print("Collecting eval data...")
    eval_data, mean_norm = collect_eval_data(model, tokenizer, LAYER, N_EVAL_TOKENS)
    print(f"  L{LAYER} mean activation norm: {mean_norm:.1f}")

    stream = ActivationStream(model, tokenizer, LAYER, seed=SEED)

    results = {
        "config": {
            "experiment": "exp47_perfeature_robustness",
            "model": MODEL_NAME, "layer": LAYER,
            "d_sae": D_SAE, "k": K, "lr": LR,
            "n_train_tokens": N_TRAIN_TOKENS, "n_steps": N_STEPS,
            "mean_norm": mean_norm,
            "var_reg_weight": VAR_REG_WEIGHT,
            "var_reg_decay": f"{VAR_REG_DECAY_START}-{VAR_REG_DECAY_END}",
            "gaussian_init_std": 0.05,
        },
        "runs": {},
    }

    variants = [
        ("perfeature_original", PerFeatureOriginalSAE),
        ("perfeature_base_delta", PerFeatureBaseDeltaSAE),
        ("perfeature_var_reg", PerFeatureVarRegSAE),
        ("perfeature_gaussian", PerFeatureGaussianInitSAE),
        ("adaptive_l2", AdaptiveCosineBatchTopKSAE),
    ]

    for name, cls in variants:
        print(f"\n{'='*70}")
        print(f"  {name} — L{LAYER}")
        print(f"{'='*70}")

        torch.manual_seed(SEED)
        sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
        training_log, ckpts = train_sae(name, sae, stream, SAVE_DIR, set(CHECKPOINT_STEPS))

        recon = evaluate_reconstruction(name, sae, eval_data)
        ablation = evaluate_ablation(name, model, tokenizer, sae)

        run_data = {
            "encoder": name,
            "layer": LAYER,
            "reconstruction": recon,
            "ablation": ablation,
            "checkpoints": ckpts,
            "training_log": training_log,
        }
        if hasattr(sae, "scale_a"):
            sa = sae.scale_a
            if sa.dim() == 0:
                run_data["scale_a"] = sa.item()
            else:
                run_data["scale_a_mean"] = sa.mean().item()
                run_data["scale_a_median"] = sa.median().item()
                run_data["scale_a_std"] = sa.std().item()
                run_data["scale_a_pct_near_zero"] = (sa.abs() < 0.05).float().mean().item()
        if hasattr(sae, "scale_a_base"):
            run_data["scale_a_base"] = sae.scale_a_base.item()
            run_data["scale_a_delta_mean"] = sae.scale_a_delta.mean().item()
            run_data["scale_a_delta_std"] = sae.scale_a_delta.std().item()

        results["runs"][name] = run_data

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved results for {name}")

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary — L{LAYER} at 50M tokens")
    print(f"{'='*70}")
    for name, data in results["runs"].items():
        r = data["reconstruction"]
        a = data["ablation"]
        scale = ""
        if "scale_a" in data:
            scale = f" a={data['scale_a']:.4f}"
        elif "scale_a_base" in data:
            scale = f" a_base={data['scale_a_base']:.4f} delta_std={data['scale_a_delta_std']:.4f}"
        elif "scale_a_mean" in data:
            scale = f" a_mean={data['scale_a_mean']:.4f}"
        print(f"  {name:25s} | FVE={r['fve']:.4f} | dead={r['dead_frac']:.3f} | "
              f"alive={r['alive_count']:,} | cos>inner={a['cos_wins_inner']}{scale}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
