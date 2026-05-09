"""
Experiment 48: Input Normalization Verification
================================================

Exp45 found that removing encoder normalization (F.normalize(W_enc)) from the
adaptive cosine SAE causes >90% dead features. The proposed mechanism: input
normalization removes per-token diversity in BatchTopK competition (replacing
varying ||x|| with a constant), so encoder norm divergence becomes catastrophic.
Standard SAEs survive because ||x|| variation acts as a tiebreaker.

This experiment tests 8 ablation variants at L27 only (norm ~405, hardest case):

Variants:
  1. standard_inputnorm:    Standard SAE + input normalization (the key test).
                            If this kills features, input normalization is the
                            trigger — not cosine structure.
  2. unnormed_perfeature_b: Unnormed cosine + per-feature scale_b_i.
                            Tests if per-feature bias range can rescue diverged
                            encoder norms.
  3. adaptive_l2:           Standard adaptive cosine baseline (encoder normalized).
                            Reference: 0% dead, FVE ~0.77.
  4. standard:              Standard BatchTopK SAE baseline.
                            Reference: ~0% dead with saprmarks recipe.
  5. perfeature_base_delta: Per-feature cosine with base+delta parameterization
                            (exp47 fix). a_eff = a_base (shared) + a_delta[i].
                            Tests if structural parameter coupling prevents
                            competitive exclusion without encoder normalization.
  6. noc_baseline:          Full NoC (enc norm + dec norm + norm-restoration).
                            Reference: 0% dead in exp46.
  7. noc_enc_free:          NoC without encoder norm (dec norm + norm-restoration).
                            Tests if decode-side norm restoration eliminates the
                            need for encoder normalization.
  8. perfeature_bd_no_enc_norm: Per-feature base+delta WITHOUT encoder norm,
                            WITH norm-restoration at decode. Tests if norm
                            restoration can replace encoder norm for adaptive
                            cosine per-feature architecture.

All variants log encoder weight norm stats at every log step.

L27 only, 50M tokens, saprmarks recipe, cached activation stream.

Run:
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp48_input_norm_verification.py \
        >> experiments/exp48_output.log 2>&1 &
"""

import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.activation_cache import (
    CachedActivationStream,
    build_activation_cache,
    cache_exists_and_valid,
    cache_paths,
)


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_SLUG = "qwen3_8b"
LAYERS = [27]  # L27 only — the hardest case (norm ~405)
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

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE  # 24,414
DECAY_START = int(0.8 * N_STEPS)        # 19,531

BUFFER_TOKENS = 500_000

CHECKPOINT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
CHECKPOINT_STEPS = sorted(set(int(f * N_STEPS) for f in CHECKPOINT_FRACS))

N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

SAVE_DIR = Path("/data/checkpoints/exp48")
CACHE_DIR = Path("/data/cache")
RESULTS_PATH = Path("experiments/exp48_results.json")


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
    """Variant 3 (adaptive_l2): encoder rows normalized before cosine similarity.
    This is the reference adaptive cosine SAE with global scale_a and scale_b."""
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


class AdaptiveCosineBatchTopKSAE_UnnormedPerFeatureB(nn.Module):
    """Variant 2 (unnormed_perfeature_b): encoder rows NOT normalized,
    per-feature scale_b_i instead of global scale_b. Tests if per-feature
    bias range can compensate for encoder norm divergence.

    Encode: scale = exp(scale_a * log(||x||) + scale_b_i)  [per-feature]
    pre_acts = scale * (x_unit @ W_enc.T) + b_enc
    """
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        # Per-feature scale_b: shape (d_sae,)
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
        # NO encoder normalization — W_enc norms are free parameters
        cos_like = x_unit @ self.W_enc.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # Per-feature scale: scale_b has shape (d_sae,), broadcasts over batch
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_like + self.b_enc
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


class StandardBatchTopKSAE_InputNorm(nn.Module):
    """Variant 1 (standard_inputnorm): Standard SAE architecture but normalize
    the input before the inner product. No scale parameters, no cosine structure.
    This is the KEY TEST: if input normalization alone kills features in a standard
    SAE, it confirms the mechanism.

    Encode: pre_acts = F.normalize(x - b_dec) @ W_enc.T + b_enc
    """
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # No scale_a, no scale_b — pure inner product on normalized input
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
        x_unit = F.normalize(x_centered, dim=-1)  # <-- the only change from standard
        pre_acts = x_unit @ self.W_enc.T + self.b_enc
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
    """Variant 5 (perfeature_base_delta): Per-feature cosine with base+delta
    parameterization from exp47. a_eff = a_base (shared scalar) + a_delta[i].
    Encoder is normalized. a_base gets gradient from every feature on every batch,
    preventing the winner-take-all cascade. a_delta allows per-feature specialization.

    Encode: a_eff = a_base + a_delta[i]
            scale = exp(a_eff * log(||x||) + scale_b[i])
            pre_acts = scale * cos_sim(x_unit, w_unit) + b_enc
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


class StandardBatchTopKSAE(nn.Module):
    """Variant 4 (standard): Standard BatchTopK SAE — inner product encoder,
    no normalization of anything. Reference baseline.

    Encode: pre_acts = (x - b_dec) @ W_enc.T + b_enc
    """
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
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
        pre_acts = x_centered @ self.W_enc.T + self.b_enc
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


class NoCBaselineSAE(nn.Module):
    """Variant 6 (noc_baseline): Full NoC — encoder normalized, decoder normalized,
    norm-restoration at decode. No scale parameters. This is the canonical NoC
    architecture from exp46."""
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.normalize_encoder = True
        self.normalize_decoder = True
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_active=False):
        x_c = x - self.b_dec
        self._cached_x_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_u = x_c / self._cached_x_norm
        w = F.normalize(self.W_enc, dim=-1, eps=NORM_EPS)
        post_relu = F.relu(x_u @ w.T)
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
        w = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_norm = getattr(self, "_cached_x_norm", None)
        if x_norm is not None:
            x_raw = x_raw * (x_norm / raw_norm)
        return x_raw + self.b_dec

    def forward(self, x, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def post_step(self):
        self.W_enc.div_(self.W_enc.norm(dim=1, keepdim=True).clamp(min=NORM_EPS))
        set_decoder_norm_to_unit_norm(self.W_dec.data)


class NoCEncFreeSAE(nn.Module):
    """Variant 7 (noc_enc_free): NoC without encoder normalization.
    Decoder normalized + norm-restoration. Tests if decode-side norm restoration
    alone eliminates the need for encoder normalization."""
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.normalize_encoder = False
        self.normalize_decoder = True
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_active=False):
        x_c = x - self.b_dec
        self._cached_x_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_u = x_c / self._cached_x_norm
        post_relu = F.relu(x_u @ self.W_enc.T)
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
        w = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_norm = getattr(self, "_cached_x_norm", None)
        if x_norm is not None:
            x_raw = x_raw * (x_norm / raw_norm)
        return x_raw + self.b_dec

    def forward(self, x, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def post_step(self):
        set_decoder_norm_to_unit_norm(self.W_dec.data)


class PerFeatureBaseDeltaNoEncNormSAE(nn.Module):
    """Variant 8 (perfeature_bd_no_enc_norm): Per-feature base+delta cosine SAE
    WITHOUT encoder normalization, but WITH norm-restoration at decode.
    Tests if decode-side norm restoration can replace encoder normalization
    for the per-feature architecture."""
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
        self.normalize_encoder = False
        self.normalize_decoder = False
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
        self._cached_x_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_unit = x_centered / self._cached_x_norm
        # NO encoder normalization — W_enc norms are free
        cos_like = x_unit @ self.W_enc.T
        log_norm = torch.log(self._cached_x_norm)
        a_effective = self.scale_a_base + self.scale_a_delta
        scale = torch.exp(a_effective * log_norm + self.scale_b)
        pre_acts = scale * cos_like + self.b_enc
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
        x_raw = f @ self.W_dec
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_norm = getattr(self, "_cached_x_norm", None)
        if x_norm is not None:
            x_raw = x_raw * (x_norm / raw_norm)
        return x_raw + self.b_dec

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
# Eval activation collection (streaming, separate from the training cache)
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
# Per-SAE training state (parallel-group pattern; mirrors exp46)
# =============================================================================

@dataclass
class SAEState:
    """All per-SAE state for one parallel-group training step."""
    name: str
    sae: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    num_tokens_since_fired: torch.Tensor
    save_dir: Path
    layer_idx: int
    checkpoint_steps: set
    log: list = field(default_factory=list)
    checkpoints_saved: dict = field(default_factory=dict)
    b_dec_initialized: bool = False
    t0: float = field(default_factory=time.time)


def make_sae_state(name: str, sae: nn.Module, layer_idx: int) -> SAEState:
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    schedule_fn = make_lr_schedule(N_STEPS, WARMUP_STEPS, DECAY_START)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)
    return SAEState(
        name=name,
        sae=sae,
        optimizer=optimizer,
        scheduler=scheduler,
        num_tokens_since_fired=torch.zeros(D_SAE, dtype=torch.long, device=DEVICE),
        save_dir=SAVE_DIR,
        layer_idx=layer_idx,
        checkpoint_steps=set(CHECKPOINT_STEPS),
    )


def step_one_sae(s: SAEState, batch: torch.Tensor, global_step: int):
    """Forward+backward+step for one SAE on one shared batch. Returns log entry or None."""
    sae = s.sae
    tag = f"{s.name}/L{s.layer_idx}"

    if not s.b_dec_initialized:
        with torch.no_grad():
            median = geometric_median(batch)
            sae.b_dec.data.copy_(median.to(sae.b_dec.dtype))
        s.b_dec_initialized = True
        print(f"    [{tag}] b_dec init (norm={median.norm():.1f})")

    x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)
    recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

    did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
    did_fire[active_indices] = True
    s.num_tokens_since_fired += batch.shape[0]
    s.num_tokens_since_fired[did_fire] = 0

    residual = (batch - x_hat).detach()
    auxk_acts, n_dead = get_auxiliary_loss(
        residual, post_relu_acts, s.num_tokens_since_fired
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

    s.optimizer.zero_grad(set_to_none=True)
    loss.backward()

    normalize_dec = getattr(sae, "normalize_decoder", True)
    if normalize_dec and sae.W_dec.grad is not None:
        sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
            sae.W_dec.data, sae.W_dec.grad.data
        )

    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    s.optimizer.step()
    s.scheduler.step()

    if hasattr(sae, "post_step"):
        sae.post_step()
    elif normalize_dec:
        set_decoder_norm_to_unit_norm(sae.W_dec.data)

    if global_step > THRESHOLD_START_STEP:
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

    entry = None
    if global_step % LOG_EVERY == 0 or global_step == N_STEPS:
        with torch.no_grad():
            l0 = (features != 0).float().sum(dim=-1).mean().item()
            total_var = torch.var(batch, dim=0, unbiased=False).sum()
            resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
            fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
            cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
            dead_frac = (
                s.num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
            ).float().mean().item()

        entry = {
            "step": global_step,
            "recon_loss": recon_loss.item(),
            "auxk_loss": auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
            "total_loss": loss.item(),
            "l0": l0,
            "fve": fve,
            "cos_recon": cos_r,
            "dead_frac": dead_frac,
            "n_dead": n_dead,
            "lr": s.scheduler.get_last_lr()[0],
            "tokens_seen": global_step * BATCH_SIZE,
            "threshold": sae.threshold.item(),
        }

        # Log scale_a if present
        if hasattr(sae, "scale_a_base"):
            entry["scale_a_base"] = sae.scale_a_base.item()
            a_delta = sae.scale_a_delta
            entry["scale_a_delta_mean"] = a_delta.mean().item()
            entry["scale_a_delta_std"] = a_delta.std().item()
            a_eff = sae.scale_a_base + a_delta
            entry["scale_a"] = a_eff.mean().item()
        elif hasattr(sae, "scale_a"):
            entry["scale_a"] = sae.scale_a.item()

        # Log scale_b (scalar or per-feature stats)
        if hasattr(sae, "scale_b"):
            if sae.scale_b.dim() == 0:
                # Global scalar scale_b
                entry["scale_b"] = sae.scale_b.item()
                entry["scale_b_exp"] = math.exp(sae.scale_b.item())
            else:
                # Per-feature scale_b — log stats
                entry["scale_b_mean"] = sae.scale_b.mean().item()
                entry["scale_b_std"] = sae.scale_b.std().item()
                entry["scale_b_min"] = sae.scale_b.min().item()
                entry["scale_b_max"] = sae.scale_b.max().item()
                entry["scale_b_exp_mean"] = torch.exp(sae.scale_b).mean().item()

        # ALL variants log encoder weight norm stats
        enc_norms = sae.W_enc.norm(dim=1)
        entry["enc_norm_mean"] = enc_norms.mean().item()
        entry["enc_norm_std"] = enc_norms.std().item()
        entry["enc_norm_min"] = enc_norms.min().item()
        entry["enc_norm_max"] = enc_norms.max().item()

    if global_step in s.checkpoint_steps:
        ckpt_path = s.save_dir / f"{s.name}_L{s.layer_idx}_step{global_step}.pt"
        torch.save(
            {
                "state_dict": sae.state_dict(),
                "num_tokens_since_fired": s.num_tokens_since_fired,
                "step": global_step,
            },
            ckpt_path,
        )
        s.checkpoints_saved[f"step{global_step}"] = str(ckpt_path)
        print(f"    [{tag}] Checkpoint at step {global_step}")

    return entry


def train_parallel_group(states: list, stream, layer_idx: int, n_steps: int):
    """Train all SAEs in `states` in parallel sharing one batch per step from the cache."""
    print(f"\n{'=' * 70}")
    print(f"  Parallel group of {len(states)} SAEs at L{layer_idx}, {n_steps} steps")
    print(f"  Variants: {[s.name for s in states]}")
    print(f"{'=' * 70}")
    for s in states:
        s.sae.train()
        s.t0 = time.time()

    global_step = 0
    t_group = time.time()
    while global_step < n_steps:
        stream.fill_buffer()
        steps_in_buffer = min(stream.buffer_batches, n_steps - global_step)

        for buf_step in range(steps_in_buffer):
            batch = stream.get_batch(buf_step)

            for s in states:
                entry = step_one_sae(s, batch, global_step + 1)
                if entry is not None:
                    s.log.append(entry)

            global_step += 1

            if global_step % LOG_EVERY == 0 or global_step == n_steps:
                elapsed = time.time() - t_group
                tok = global_step * BATCH_SIZE
                tok_per_sec = tok / elapsed if elapsed > 0 else 0
                eta = (n_steps - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"\n  --- step {global_step}/{n_steps} ({tok/1e6:.1f}M tok, "
                      f"{tok_per_sec/1e3:.1f}k tok/s, ETA {eta/3600:.2f}h) ---")
                for s in states:
                    if not s.log:
                        continue
                    e = s.log[-1]
                    extra = ""
                    if "scale_a_base" in e:
                        extra += (f" a_base={e['scale_a_base']:.4f}"
                                  f" a_delta={e['scale_a_delta_mean']:.4f}"
                                  f"+-{e['scale_a_delta_std']:.4f}")
                    elif "scale_a" in e:
                        extra += f" a={e['scale_a']:.4f}"
                    if "scale_b" in e:
                        extra += f" b={e['scale_b']:.3f}(exp={e['scale_b_exp']:.1f})"
                    if "scale_b_mean" in e:
                        extra += (f" b_mean={e['scale_b_mean']:.3f}"
                                  f"+-{e['scale_b_std']:.3f}")
                    extra += (f" ||w||={e['enc_norm_mean']:.3f}"
                              f"+-{e['enc_norm_std']:.3f}")
                    print(
                        f"    [{s.name}/L{s.layer_idx}] "
                        f"loss={e['total_loss']:.1f} recon={e['recon_loss']:.1f} "
                        f"auxk={e['auxk_loss']:.3f} | "
                        f"FVE={e['fve']:.4f} L0={e['l0']:.0f} "
                        f"dead={e['dead_frac']:.3f}({e['n_dead']}){extra}"
                    )

            if global_step >= n_steps:
                break

    for s in states:
        s.sae.eval()
        elapsed = time.time() - s.t0
        print(f"    [{s.name}/L{s.layer_idx}] Done in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

        final_path = s.save_dir / f"{s.name}_L{s.layer_idx}_final.pt"
        torch.save(
            {
                "state_dict": s.sae.state_dict(),
                "num_tokens_since_fired": s.num_tokens_since_fired,
                "step": global_step,
            },
            final_path,
        )
        s.checkpoints_saved["final"] = str(final_path)
        print(f"    [{s.name}/L{s.layer_idx}] Saved {final_path}")

    print(f"\n  Group done in {(time.time() - t_group) / 3600:.2f}h")


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
        alive = features.sum(dim=0) != 0
        if dead_counts is None:
            dead_counts = ~alive
        else:
            dead_counts &= ~alive

    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0
    fve = 1 - resid_var_sum / total_var_sum if total_var_sum > 0 else 0

    results = {
        "fve": fve, "cos_recon": float(np.mean(cos_sims)), "mean_l0": float(np.mean(l0s)),
        "dead_frac": dead_frac, "alive_count": alive_count,
    }
    print(f"    [{tag}] FVE={fve:.4f} dead={dead_frac:.3f} "
          f"alive={alive_count:,} L0={np.mean(l0s):.1f}")
    return results


@torch.no_grad()
def collect_ablation_corpus(model, tokenizer, layer_idx, n_samples,
                             ctx_len_eval=CTX_LEN):
    """
    Collect a small text corpus and pre-compute (inputs, clean_probs,
    last_act, full_act) per sample. Shared across all variants in the
    ablation eval to eliminate redundant clean forwards.
    """
    print(f"    [ablation] Collecting {n_samples} texts + clean forwards "
          f"(L{layer_idx})...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= 600_000:
            break

    corpus = []
    while len(corpus) < n_samples:
        try:
            row = next(text_iter)
        except StopIteration:
            break
        if len(row["text"]) <= 100:
            continue
        text = row["text"][:4096]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=ctx_len_eval,
        ).to(DEVICE)

        captured = {}

        def hook(module, inp, out):
            captured["act"] = (out[0] if isinstance(out, tuple) else out).detach()

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        try:
            outputs = model(**inputs)
        finally:
            handle.remove()
        clean_logits = outputs.logits[0, -1].float()
        clean_probs = F.softmax(clean_logits, dim=-1).clone()
        last_act = captured["act"][0, -1].float().clone()
        full_act = captured["act"][0].float().clone()

        corpus.append({
            "inputs": inputs,
            "clean_probs": clean_probs,
            "last_act": last_act,
            "full_act": full_act,
        })
    print(f"    [ablation] Collected {len(corpus)} texts")
    return corpus


@torch.no_grad()
def evaluate_ablation_shared(name, model, sae, layer_idx, corpus,
                             n_features=N_ABLATION_FEATURES):
    """RNH diagnostic: cos vs inner product as causal predictor.
    Reuses pre-computed clean forwards from `corpus`. Per-feature ablated
    forwards are unavoidably per-SAE."""
    tag = f"{name}/L{layer_idx}"
    print(f"    [{tag}] Ablation ({n_features} features, "
          f"{len(corpus)} samples)...")
    sae.eval()

    act_sums = torch.zeros(D_SAE, device=DEVICE)
    for sample in corpus:
        flat = sample["full_act"].reshape(-1, D_MODEL)
        features = sae.encode(flat)
        act_sums += features.sum(dim=0)
    top_features = act_sums.topk(n_features).indices.tolist()

    cos_wins = 0
    cos_kl_corrs, inner_kl_corrs = [], []

    for feat_idx in top_features:
        feat_dir = sae.W_dec[feat_idx]
        feat_dir_unit = feat_dir / feat_dir.norm().clamp(min=1e-8)

        cos_vals, inner_vals, kl_vals = [], [], []
        for sample in corpus:
            act_flat = sample["last_act"]

            cos_sim = F.cosine_similarity(
                act_flat.unsqueeze(0), feat_dir_unit.unsqueeze(0)
            ).item()
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

            handle = model.model.layers[layer_idx].register_forward_hook(ablation_hook)
            try:
                outputs_abl = model(**sample["inputs"])
            finally:
                handle.remove()
            abl_probs = F.softmax(outputs_abl.logits[0, -1], dim=-1)

            kl = F.kl_div(abl_probs.log(), sample["clean_probs"],
                          reduction="sum").item()
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
        "n_features": n_features,
        "cos_wins_inner": f"{cos_wins}/{len(cos_kl_corrs)}",
        "cos_kl_mean": float(np.mean(cos_kl_corrs)) if cos_kl_corrs else 0,
        "inner_kl_mean": float(np.mean(inner_kl_corrs)) if inner_kl_corrs else 0,
    }
    print(f"    [{tag}] cos>inner: {cos_wins}/{len(cos_kl_corrs)} "
          f"(cos_corr={results['cos_kl_mean']:.3f}, "
          f"inner_corr={results['inner_kl_mean']:.3f})")
    return results


# =============================================================================
# Main
# =============================================================================

VARIANTS = [
    ("standard_inputnorm", StandardBatchTopKSAE_InputNorm),
    ("unnormed_perfeature_b", AdaptiveCosineBatchTopKSAE_UnnormedPerFeatureB),
    ("adaptive_l2", AdaptiveCosineBatchTopKSAE),
    ("standard", StandardBatchTopKSAE),
    ("perfeature_base_delta", PerFeatureBaseDeltaSAE),
    ("noc_baseline", NoCBaselineSAE),
    ("noc_enc_free", NoCEncFreeSAE),
    ("perfeature_bd_no_enc_norm", PerFeatureBaseDeltaNoEncNormSAE),
]


def _persist(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _build_run_record(s: SAEState, recon: dict) -> dict:
    sae = s.sae
    run_data = {
        "encoder": s.name,
        "layer": s.layer_idx,
        "reconstruction": recon,
        "checkpoints": dict(s.checkpoints_saved),
        "training_log": list(s.log),
    }
    if hasattr(sae, "scale_a_base"):
        run_data["scale_a_base_final"] = sae.scale_a_base.item()
        a_delta = sae.scale_a_delta
        run_data["scale_a_delta_final"] = {
            "mean": a_delta.mean().item(),
            "std": a_delta.std().item(),
            "min": a_delta.min().item(),
            "max": a_delta.max().item(),
        }
        a_eff = sae.scale_a_base + a_delta
        run_data["scale_a_final"] = a_eff.mean().item()
    elif hasattr(sae, "scale_a"):
        run_data["scale_a_final"] = sae.scale_a.item()
    if hasattr(sae, "scale_b"):
        if sae.scale_b.dim() == 0:
            run_data["scale_b_final"] = sae.scale_b.item()
            run_data["scale_b_final_exp"] = math.exp(sae.scale_b.item())
        else:
            sb = sae.scale_b
            run_data["scale_b_final"] = {
                "mean": sb.mean().item(),
                "std": sb.std().item(),
                "min": sb.min().item(),
                "max": sb.max().item(),
            }
    # All variants: final encoder norm stats
    enc_norms = sae.W_enc.norm(dim=1)
    run_data["enc_norm_final"] = {
        "mean": enc_norms.mean().item(),
        "std": enc_norms.std().item(),
        "min": enc_norms.min().item(),
        "max": enc_norms.max().item(),
        "median": enc_norms.median().item(),
    }
    return run_data


def main():
    print("Experiment 48: Input Normalization Verification")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae={D_SAE}, k={K}, lr={LR}, {N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Total runs: {len(VARIANTS)} (in 1 parallel group of {len(VARIANTS)})")
    print(f"Cache dir: {CACHE_DIR}")
    print()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()

    # Load existing results for resume
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"  Loaded existing results: {list(results.get('runs', {}).keys())}")
    else:
        results = {"config": {
            "experiment": 48,
            "model": MODEL_NAME, "layers": LAYERS,
            "d_sae": D_SAE, "k": K, "lr": LR,
            "n_train_tokens": N_TRAIN_TOKENS, "n_steps": N_STEPS,
            "note": "Input normalization verification: 8 ablation variants at L27 "
                    "isolating input normalization, encoder normalization, norm-restoration, "
                    "and base+delta parameterization.",
            "variants": {
                "standard_inputnorm": "Standard SAE + input normalization (key test)",
                "unnormed_perfeature_b": "Unnormed cosine + per-feature scale_b_i",
                "adaptive_l2": "Standard adaptive cosine baseline (encoder normalized)",
                "standard": "Standard BatchTopK SAE baseline",
                "perfeature_base_delta": "Per-feature cosine with base+delta (exp47 fix)",
                "noc_baseline": "Full NoC (enc norm + dec norm + norm-restoration)",
                "noc_enc_free": "NoC without encoder norm (dec norm + norm-restoration)",
                "perfeature_bd_no_enc_norm": "Per-feature base+delta, no enc norm, with norm-restoration",
            },
            "predictions": {
                "standard_inputnorm": "Should kill features (isolates input norm as trigger)",
                "unnormed_perfeature_b": "Probably still fails (competitive advantage remains)",
                "adaptive_l2": "Reference: ~0% dead, FVE ~0.77",
                "standard": "Reference: ~0% dead",
                "perfeature_base_delta": "Should survive (~0.4% dead, matching exp47)",
                "noc_baseline": "Reference: ~0% dead",
                "noc_enc_free": "Should survive if norm-restoration is sufficient",
                "perfeature_bd_no_enc_norm": "Key test: norm-restoration as enc norm substitute for per-feature",
            },
        }, "runs": {}}

    # --- 1. Build L27 activation cache ---
    layer_idx = LAYERS[0]  # 27
    bin_path, _ = cache_paths(CACHE_DIR, MODEL_SLUG, layer_idx, N_TRAIN_TOKENS)
    if not cache_exists_and_valid(CACHE_DIR, MODEL_SLUG, layer_idx,
                                  N_TRAIN_TOKENS, D_MODEL):
        print(f"\n[cache] Building L{layer_idx} cache "
              f"({N_TRAIN_TOKENS:,} tokens, "
              f"~{N_TRAIN_TOKENS * D_MODEL * 2 / 1e9:.0f} GB bf16)...")
        bin_path = build_activation_cache(
            model, tokenizer, layer=layer_idx,
            n_tokens=N_TRAIN_TOKENS, cache_dir=CACHE_DIR,
            model_slug=MODEL_SLUG, d_model=D_MODEL,
            seed=SEED, ctx_len=CTX_LEN,
            collection_batch_size=COLLECTION_BATCH_SIZE,
            outlier_multiplier=OUTLIER_MULTIPLIER,
            chunk_tokens=BUFFER_TOKENS, text_skip=0, device=DEVICE,
        )
    else:
        print(f"[cache] Reusing L{layer_idx} cache at {bin_path}")

    # --- 2. Eval data ---
    eval_data, mean_norm = collect_eval_data(model, tokenizer, layer_idx, N_EVAL_TOKENS)
    results["config"]["mean_norms"] = {str(layer_idx): mean_norm}

    # --- 3. Parallel-train all 4 variants from cached stream ---
    print(f"\n{'#'*70}")
    print(f"  LAYER {layer_idx} (mean_norm={mean_norm:.1f})")
    print(f"{'#'*70}")

    run_keys = [f"{vname}_L{layer_idx}" for vname, _ in VARIANTS]
    if all(rk in results.get("runs", {}) for rk in run_keys):
        print(f"  L{layer_idx} already complete, skipping training")
    else:
        stream = CachedActivationStream(
            bin_path,
            batch_size=BATCH_SIZE, device=DEVICE,
            chunk_tokens=BUFFER_TOKENS, shuffle_seed=SEED,
        )
        stream._cursor = 0
        stream._chunk_idx = 0

        states = []
        for vname, vcls in VARIANTS:
            run_key = f"{vname}_L{layer_idx}"
            if run_key in results.get("runs", {}):
                print(f"  {run_key} already complete, skipping in this group")
                continue
            torch.manual_seed(SEED)
            sae = vcls(D_MODEL, D_SAE, K).to(DEVICE)
            n_params = sum(p.numel() for p in sae.parameters())
            print(f"  {vname}: {n_params:,} params", end="")
            if hasattr(sae, "scale_b"):
                if sae.scale_b.dim() == 0:
                    print(f" (scale_b init={sae.scale_b.item():.4f}, "
                          f"exp={math.exp(sae.scale_b.item()):.1f})")
                else:
                    print(f" (per-feature scale_b, init={sae.scale_b[0].item():.4f}, "
                          f"exp={math.exp(sae.scale_b[0].item()):.1f})")
            elif hasattr(sae, "scale_a"):
                print(f" (scale_a only, no scale_b)")
            else:
                print(" (no scale params)")
            states.append(make_sae_state(vname, sae, layer_idx))

        if states:
            train_parallel_group(states, stream, layer_idx, N_STEPS)

            # Per-variant FVE eval
            for s in states:
                recon = evaluate_reconstruction(s.name, s.sae, eval_data, layer_idx)
                results["runs"][f"{s.name}_L{layer_idx}"] = _build_run_record(s, recon)
                _persist(results)

            # Shared ablation eval
            corpus = collect_ablation_corpus(model, tokenizer, layer_idx,
                                             N_ABLATION_SAMPLES)
            for s in states:
                ablation = evaluate_ablation_shared(
                    s.name, model, s.sae, layer_idx, corpus,
                    n_features=N_ABLATION_FEATURES,
                )
                results["runs"][f"{s.name}_L{layer_idx}"]["ablation"] = ablation
                _persist(results)

            del states, corpus, stream
            gc.collect()
            torch.cuda.empty_cache()

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print("  EXP48 FINAL SUMMARY — Input Normalization Verification (L27)")
    print(f"{'='*70}")
    print(f"\n  Layer {layer_idx} (norm={mean_norm:.1f})")
    print(f"  {'Variant':<25s} | {'FVE':>6s} | {'Dead%':>6s} | "
          f"{'Alive':>6s} | {'cos>inn':>8s} | {'||W_enc||':>12s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*12}")

    for vn, _ in VARIANTS:
        rk = f"{vn}_L{layer_idx}"
        run = results["runs"].get(rk, {})
        r = run.get("reconstruction", {})
        a = run.get("ablation", {})
        if not r:
            print(f"  {vn:<25s} | {'N/A':>6s} | {'N/A':>6s} | {'N/A':>6s} | {'N/A':>8s} | {'N/A':>12s}")
            continue
        fve = r.get("fve", 0)
        dead = r.get("dead_frac", 0)
        alive = r.get("alive_count", 0)
        cos_w = str(a.get("cos_wins_inner", "?"))
        en = run.get("enc_norm_final", {})
        enc_str = f"{en.get('mean', 0):.3f}+-{en.get('std', 0):.3f}" if en else "N/A"
        print(f"  {vn:<25s} | {fve:.4f} | {dead*100:5.1f}% | "
              f"{alive:>6d} | {cos_w:>8s} | {enc_str:>12s}")

    # Mechanism verification summary
    print(f"\n  MECHANISM VERIFICATION:")
    r_std = results["runs"].get(f"standard_L{layer_idx}", {}).get("reconstruction", {})
    r_inp = results["runs"].get(f"standard_inputnorm_L{layer_idx}", {}).get("reconstruction", {})
    r_pfb = results["runs"].get(f"unnormed_perfeature_b_L{layer_idx}", {}).get("reconstruction", {})
    r_adp = results["runs"].get(f"adaptive_l2_L{layer_idx}", {}).get("reconstruction", {})

    if r_std and r_inp:
        dead_std = r_std.get("dead_frac", 0)
        dead_inp = r_inp.get("dead_frac", 0)
        if dead_inp > 0.5 and dead_std < 0.1:
            verdict = "CONFIRMED: input norm alone kills features"
        elif dead_inp < 0.1:
            verdict = "REFUTED: input norm does NOT kill features alone"
        else:
            verdict = f"PARTIAL: input_norm dead={dead_inp*100:.1f}% vs standard dead={dead_std*100:.1f}%"
        print(f"    Key test (standard_inputnorm): {verdict}")

    if r_pfb:
        dead_pfb = r_pfb.get("dead_frac", 0)
        if dead_pfb > 0.5:
            print(f"    Per-feature scale_b: FAILED to rescue (dead={dead_pfb*100:.1f}%)")
        else:
            print(f"    Per-feature scale_b: RESCUED features (dead={dead_pfb*100:.1f}%)")

    if r_adp:
        dead_adp = r_adp.get("dead_frac", 0)
        print(f"    Adaptive cosine reference: dead={dead_adp*100:.1f}%")

    r_bd = results["runs"].get(f"perfeature_base_delta_L{layer_idx}", {}).get("reconstruction", {})
    if r_bd:
        dead_bd = r_bd.get("dead_frac", 0)
        if dead_bd < 0.05:
            print(f"    Base+delta (exp47): CONFIRMED robust (dead={dead_bd*100:.1f}%)")
        else:
            print(f"    Base+delta (exp47): FAILED (dead={dead_bd*100:.1f}%)")

    r_noc = results["runs"].get(f"noc_baseline_L{layer_idx}", {}).get("reconstruction", {})
    r_noc_ef = results["runs"].get(f"noc_enc_free_L{layer_idx}", {}).get("reconstruction", {})
    if r_noc:
        print(f"    NoC baseline: dead={r_noc.get('dead_frac', 0)*100:.1f}%")
    if r_noc_ef:
        dead_ef = r_noc_ef.get("dead_frac", 0)
        if dead_ef < 0.05:
            print(f"    NoC enc-free: CONFIRMED norm-restoration sufficient (dead={dead_ef*100:.1f}%)")
        else:
            print(f"    NoC enc-free: norm-restoration NOT sufficient (dead={dead_ef*100:.1f}%)")

    r_bd_ne = results["runs"].get(f"perfeature_bd_no_enc_norm_L{layer_idx}", {}).get("reconstruction", {})
    if r_bd_ne:
        dead_ne = r_bd_ne.get("dead_frac", 0)
        if dead_ne < 0.05:
            print(f"    Per-feature bd+norm-restore: CONFIRMED (dead={dead_ne*100:.1f}%)")
        else:
            print(f"    Per-feature bd+norm-restore: FAILED (dead={dead_ne*100:.1f}%)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
