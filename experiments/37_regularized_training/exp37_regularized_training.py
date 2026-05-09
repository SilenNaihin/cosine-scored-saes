"""
Experiment 37: Regularized Scaling — Training + Evaluation
==========================================================

Why this experiment exists:
  Exp35 established that adaptive_l2 (1 global scale_a) outperforms
  perfeature_l2 (d_sae per-feature scale_a params) on downstream tasks
  despite worse FVE. The hypothesis: per-feature freedom causes overfitting
  to magnitude noise — the features learn to overfit individual activation
  norms rather than capturing meaningful semantic directions.

  This experiment searches for the "goldilocks" <author>el of scale
  parameterization between 1 global param and d_sae per-feature params.

What this tests:
  Three regularization strategies that constrain per-feature scale_a:

  B1 — L2 penalty: per-feature scale_a with L2 regularization pushing
        values toward zero. Lambda sweep: [0, 0.01, 0.1, 1.0, 10.0].

  B2 — Group-wise sharing: groups of features share one scale_a.
        G sweep: [1, 4, 16, 64, 256, 9216] (G=1 = adaptive, G=9216 = perfeature).

  B3 — Dropout: per-feature scale_a with dropout during training.
        p sweep: [0, 0.1, 0.3, 0.5, 0.7, 0.9].

  B4 — Controls: standard, adaptive_l2, perfeature_l2 retrained with
        sqrt(d) init for fair comparison.

  B5 — SAEBench: core, sparse_probing, ravel on B4 controls + best from
        each sweep.

Design decisions:
  - All training on Gemma-2-2b layer 13 only (same model as exp35).
  - sqrt(d) init for all cosine SAEs (init_norm=None) — NOT norm-adaptive.
  - Same hyperparameters as exp35: 50M tokens, k=80, d_sae=9216, lr=3e-4.

Usage:
    python experiments/exp37_regularized_training.py --sweep l2        # B1
    python experiments/exp37_regularized_training.py --sweep group     # B2
    python experiments/exp37_regularized_training.py --sweep dropout   # B3
    python experiments/exp37_regularized_training.py --sweep controls  # B4
    python experiments/exp37_regularized_training.py --sweep eval      # B5
    python experiments/exp37_regularized_training.py --sweep all       # Everything
    python experiments/exp37_regularized_training.py --dry-run --max-steps 100
"""

from __future__ import annotations

import argparse
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "google/gemma-2-2b"
RAVEL_MODEL_NAME = "gemma-2-2b"
D_MODEL = 2304
D_SAE = 9216
K = 80
N_TRAIN_TOKENS = 50_000_000
CTX_LEN = 256
LR = 3e-4
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LAYER = 13
LOG_EVERY = 200
COLLECTION_BATCH_SIZE = 32
OUTLIER_MULTIPLIER = 10.0
BUFFER_TOKENS = 500_000
CHECKPOINT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
SAVE_DIR = Path("checkpoints/exp37")
RESULTS_PATH = "experiments/exp37_training_results.json"
SAEBENCH_OUTPUT = "experiments/exp37_saebench_results"

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)
CHECKPOINT_STEPS = [int(f * N_STEPS) for f in CHECKPOINT_FRACS]
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE

# Sweep configurations
L2_LAMBDAS = [0, 0.01, 0.1, 1.0, 10.0]
GROUP_SIZES = [1, 4, 16, 64, 256, 9216]
DROPOUT_PS = [0, 0.1, 0.3, 0.5, 0.7, 0.9]


def get_config_dict() -> dict:
    return {
        "experiment": 37,
        "model_name": MODEL_NAME,
        "layer": LAYER,
        "d_model": D_MODEL,
        "d_sae": D_SAE,
        "k": K,
        "n_train_tokens": N_TRAIN_TOKENS,
        "ctx_len": CTX_LEN,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC,
        "seed": SEED,
        "n_steps": N_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "buffer_tokens": BUFFER_TOKENS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "l2_lambdas": L2_LAMBDAS,
        "group_sizes": GROUP_SIZES,
        "dropout_ps": DROPOUT_PS,
    }


# =============================================================================
# Base SAE Architectures
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
      - scale_a=1: scale proportional to ||x|| (inner-product-like)
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
# New Regularized Architectures
# =============================================================================

class L2RegPerFeatureSAE(PerFeatureAdaptiveCosineSAE):
    """Per-feature scale_a with L2 regularization.

    Training loop adds: loss += l2_lambda * scale_a.pow(2).sum()
    This pushes scale_a toward zero (cosine-like) while allowing features
    that genuinely benefit from norm sensitivity to resist the penalty.
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80,
                 init_norm: float | None = None, l2_lambda: float = 0.1):
        super().__init__(d_model, d_sae, k, init_norm)
        self.l2_lambda = l2_lambda


class GroupScaleSAE(nn.Module):
    """Cosine SAE with group-wise scale_a.

    Features are divided into n_groups groups that share a single scale_a
    and scale_b. Interpolates between adaptive (n_groups=1) and perfeature
    (n_groups=d_sae).
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80,
                 n_groups: int = 16, init_norm: float | None = None):
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
        scale_init = math.log(init_norm) if init_norm else math.log(math.sqrt(d_model))
        self.scale_b = nn.Parameter(torch.full((n_groups,), scale_init))
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

    def _expand_group_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand group-<author>el scale params to per-feature tensors."""
        group_size = self.d_sae // self.n_groups
        remainder = self.d_sae - group_size * self.n_groups
        scale_a_exp = self.scale_a.repeat_interleave(group_size)
        scale_b_exp = self.scale_b.repeat_interleave(group_size)
        if remainder > 0:
            scale_a_exp = torch.cat([scale_a_exp, self.scale_a[-1:].expand(remainder)])
            scale_b_exp = torch.cat([scale_b_exp, self.scale_b[-1:].expand(remainder)])
        return scale_a_exp, scale_b_exp

    def encode(self, x: torch.Tensor) -> torch.Tensor:
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

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


class DropoutScaleSAE(PerFeatureAdaptiveCosineSAE):
    """Per-feature scale_a with dropout during training.

    During training, each scale_a_i is independently zeroed with probability
    dropout_p. At eval time, scale_a is scaled by (1 - dropout_p) for
    consistent expected value. This prevents co-adaptation of scale_a values
    and encourages features to work with or without norm sensitivity.
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80,
                 init_norm: float | None = None, dropout_p: float = 0.3):
        super().__init__(d_model, d_sae, k, init_norm)
        self.dropout_p = dropout_p

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        log_norm = torch.log(input_norm)

        if self.training:
            mask = torch.bernoulli(torch.full_like(self.scale_a, 1 - self.dropout_p))
            effective_a = self.scale_a * mask
        else:
            effective_a = self.scale_a * (1 - self.dropout_p)

        scale = torch.exp(effective_a * log_norm + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
            return self._batch_topk(acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))


# =============================================================================
# Streaming Activation Collection
# =============================================================================

class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
    """Capture residual stream activations at a Gemma-2 layer via forward hook."""
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
    """Streams activations from FineWeb, yielding shuffled batches."""

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

    def fill_buffer(self) -> int:
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

    def get_batch(self, batch_idx: int) -> torch.Tensor:
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


# =============================================================================
# Training
# =============================================================================

def lr_schedule(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _format_scale_log(sae: nn.Module) -> str:
    """Format scale_a stats for logging."""
    if not hasattr(sae, "scale_a"):
        return ""

    if sae.scale_a.dim() == 0:
        # Scalar scale_a (adaptive)
        return f" | a={sae.scale_a.item():.4f} b={sae.scale_b.exp().item():.1f}"

    a = sae.scale_a.detach()

    if isinstance(sae, GroupScaleSAE):
        return (f" | G={sae.n_groups} a_grp={a.mean().item():.4f}"
                f"+/-{a.std().item():.4f}"
                f" [{a.min().item():.3f},{a.max().item():.3f}]")

    return (f" | a={a.mean().item():.4f}+/-{a.std().item():.4f}"
            f" [{a.min().item():.3f},{a.max().item():.3f}]")


def _collect_scale_entry(sae: nn.Module, entry: dict) -> None:
    """Add scale_a stats to a log entry dict."""
    if not hasattr(sae, "scale_a"):
        return

    if sae.scale_a.dim() == 0:
        entry["scale_a"] = sae.scale_a.item()
        entry["scale_b"] = sae.scale_b.exp().item()
        return

    a = sae.scale_a.detach()

    if isinstance(sae, GroupScaleSAE):
        entry["n_groups"] = sae.n_groups
        entry["scale_a_group_mean"] = a.mean().item()
        entry["scale_a_group_std"] = a.std().item()
        entry["scale_a_group_max"] = a.max().item()
        entry["scale_a_group_min"] = a.min().item()
        entry["scale_a_group_range"] = (a.max() - a.min()).item()
    else:
        entry["scale_a_mean"] = a.mean().item()
        entry["scale_a_std"] = a.std().item()
        entry["scale_a_max"] = a.max().item()
        entry["scale_a_min"] = a.min().item()


def _collect_checkpoint_scale(sae: nn.Module, ckpt_entry: dict) -> None:
    """Add scale_a stats to a checkpoint log entry."""
    if not hasattr(sae, "scale_a"):
        return

    if sae.scale_a.dim() == 0:
        ckpt_entry["scale_a"] = sae.scale_a.item()
        return

    a = sae.scale_a.detach()

    if isinstance(sae, GroupScaleSAE):
        ckpt_entry["scale_a_group_mean"] = a.mean().item()
        ckpt_entry["scale_a_group_std"] = a.std().item()
    else:
        ckpt_entry["scale_a_mean"] = a.mean().item()
        ckpt_entry["scale_a_median"] = a.median().item()
        ckpt_entry["near_zero_frac"] = (a.abs() < 0.05).float().mean().item()


def train_sae(
    name: str,
    sae: nn.Module,
    stream: ActivationStream,
    save_dir: Path,
    max_steps: int | None = None,
) -> tuple[list[dict], dict]:
    """Train an SAE with streaming activation collection.

    Supports L2 penalty on scale_a for L2RegPerFeatureSAE, and respects
    max_steps override for dry-run mode.
    """
    effective_steps = max_steps if max_steps is not None else N_STEPS
    effective_checkpoint_steps = [int(f * effective_steps) for f in CHECKPOINT_FRACS]

    tag = f"{name}/L{LAYER}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{effective_steps} steps (streaming)")
    if hasattr(sae, "l2_lambda") and sae.l2_lambda > 0:
        print(f"    L2 penalty on scale_a: lambda={sae.l2_lambda}")
    if hasattr(sae, "dropout_p") and sae.dropout_p > 0:
        print(f"    Scale_a dropout: p={sae.dropout_p}")
    if isinstance(sae, GroupScaleSAE):
        print(f"    Group-wise scale: {sae.n_groups} groups")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)

    def _lr_schedule_fn(step: int) -> float:
        warmup = int(effective_steps * WARMUP_FRAC)
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(effective_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_schedule_fn)

    sae.train()
    log = []
    checkpoint_log = {}
    t0 = time.time()
    global_step = 0
    next_checkpoint_idx = 0
    effective_buffer_batches = BUFFER_BATCHES
    fve = 0.0
    dead = 1.0

    while global_step < effective_steps:
        stream.fill_buffer()
        steps_in_buffer = min(effective_buffer_batches, effective_steps - global_step)

        for buf_step in range(steps_in_buffer):
            batch = stream.get_batch(buf_step)
            x_hat, features = sae(batch)

            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            # L2 penalty on scale_a for L2RegPerFeatureSAE
            if hasattr(sae, "l2_lambda") and sae.l2_lambda > 0:
                loss = recon_loss + sae.l2_lambda * sae.scale_a.pow(2).sum()
            else:
                loss = recon_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % LOG_EVERY == 0 or global_step == effective_steps:
                with torch.no_grad():
                    l0 = (features != 0).float().sum(dim=-1).mean().item()
                    total_var = torch.var(batch, dim=0, unbiased=False).sum()
                    resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                    fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                    cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                    dead = (features.sum(dim=0) == 0).float().mean().item()

                entry = {
                    "step": global_step,
                    "recon_loss": recon_loss.item(),
                    "total_loss": loss.item(),
                    "l0": l0,
                    "fve": fve,
                    "cos_recon": cos_r,
                    "dead_frac": dead,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": global_step * BATCH_SIZE,
                }
                _collect_scale_entry(sae, entry)
                scale_str = _format_scale_log(sae)

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (effective_steps - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag:>24s}] step {global_step:>6d}/{effective_steps} | "
                      f"loss={loss.item():.4f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f}{scale_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | {tok_per_sec/1e3:.0f}k tok/s | "
                      f"ETA {eta_sec/60:.0f}m")

            if (next_checkpoint_idx < len(effective_checkpoint_steps) and
                    global_step >= effective_checkpoint_steps[next_checkpoint_idx]):
                frac = CHECKPOINT_FRACS[next_checkpoint_idx]
                ckpt_path = save_dir / f"{name}_L{LAYER}_step{global_step}.pt"
                torch.save(sae.state_dict(), ckpt_path)
                ckpt_entry = {
                    "step": global_step,
                    "tokens": global_step * BATCH_SIZE,
                    "fve": fve,
                    "dead_frac": dead,
                }
                _collect_checkpoint_scale(sae, ckpt_entry)
                checkpoint_log[f"{frac:.0%}"] = ckpt_entry
                print(f"    [{tag}] Checkpoint saved at {frac:.0%} ({global_step} steps)")
                next_checkpoint_idx += 1

            if global_step >= effective_steps:
                break

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    final_path = save_dir / f"{name}_L{LAYER}_final.pt"
    torch.save(sae.state_dict(), final_path)

    return log, checkpoint_log


# =============================================================================
# Evaluation (reconstruction)
# =============================================================================

def collect_eval_data(model, tokenizer, layer_idx: int, n_tokens: int) -> torch.Tensor:
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


@torch.no_grad()
def evaluate_reconstruction(name: str, sae: nn.Module, eval_data: torch.Tensor) -> dict:
    """Evaluate reconstruction quality on held-out data."""
    tag = f"{name}/L{LAYER}"
    sae.eval()
    n = eval_data.shape[0]
    recon_losses, cos_sims, l0s = [], [], []
    total_var_sum, resid_var_sum = 0.0, 0.0
    dead_counts = None

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i + BATCH_SIZE].to(DEVICE, dtype=torch.float32)
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


# =============================================================================
# SAEBench Evaluation
# =============================================================================

def wrap_as_benchsae(sae: nn.Module, layer: int, device: str) -> "BenchSAE":
    """Wrap a trained SAE as a BenchSAE for SAEBench evaluation."""
    from benchmarks.adapter import BenchSAE

    _sae = sae

    def _make_fns(s):
        return lambda x: s.encode(x), lambda f: s.decode(f)

    enc_fn, dec_fn = _make_fns(_sae)
    W_enc = sae.W_enc.detach().T
    W_dec = F.normalize(sae.W_dec.detach(), dim=1)

    return BenchSAE(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=sae.b_enc.detach(),
        b_dec=sae.b_dec.detach(),
        encode_fn=enc_fn,
        decode_fn=dec_fn,
        model_name=MODEL_NAME,
        hook_layer=layer,
        device=device,
        dtype=torch.bfloat16,
    )


def _load_result(output_dir: str, sae_name: str) -> dict:
    """Load the most re<author>ant SAEBench result file."""
    for p in Path(output_dir).glob("*.json"):
        if sae_name in p.stem:
            with open(p) as f:
                return json.load(f)
    jsons = sorted(Path(output_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    if jsons:
        with open(jsons[-1]) as f:
            return json.load(f)
    return {"error": "result file not found"}


def run_saebench_evals(
    sae_specs: list[tuple[str, str]],
    eval_types: list[str],
    device: str = "cuda",
    llm_batch_size: int = 16,
    output_dir: str = SAEBENCH_OUTPUT,
    force_rerun: bool = False,
) -> dict:
    """Run SAEBench evals on exp37 checkpoints.

    Args:
        sae_specs: List of (name, checkpoint_filename) tuples.
        eval_types: Which SAEBench evals to run.
        device: Device for evaluation.
        llm_batch_size: Batch size for LLM forward passes.
        output_dir: Where to save results.
        force_rerun: Whether to rerun existing results.

    Returns:
        Combined results dict.
    """
    from benchmarks.adapter import BenchSAE

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # Load all SAEs
    saes: list[tuple[str, BenchSAE]] = []
    for name, ckpt_filename in sae_specs:
        path = SAVE_DIR / ckpt_filename
        if not path.exists():
            print(f"  SKIP {name} -- checkpoint not found: {path}")
            continue

        sae = _load_sae_from_checkpoint(name, path)
        sae = sae.to(device=device, dtype=torch.bfloat16).eval()
        bench_sae = wrap_as_benchsae(sae, LAYER, device)
        assert bench_sae.check_decoder_norms(), f"Decoder norm check failed for {name}"
        saes.append((name, bench_sae))
        print(f"  Loaded {name}")

    if not saes:
        print("  No SAEs loaded -- skipping SAEBench")
        return all_results

    # Run each eval
    for eval_type in eval_types:
        print(f"\n--- {eval_type} ---")
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
                sp_eval.run_eval(
                    config, [(sae_name, sae)], device, eval_output,
                    force_rerun=force_rerun, clean_up_activations=True,
                    save_activations=False,
                )
                all_results[f"{sae_name}_sparse_probing"] = _load_result(eval_output, sae_name)

        elif eval_type == "ravel":
            import sae_bench.evals.ravel.main as ravel_eval
            for sae_name, sae in saes:
                config = ravel_eval.RAVE<author>alConfig(
                    model_name=RAVEL_MODEL_NAME,
                    llm_batch_size=llm_batch_size,
                    llm_dtype="bfloat16",
                )
                ravel_eval.run_eval(
                    config, [(sae_name, sae)], device, eval_output,
                    force_rerun=force_rerun,
                )
                all_results[f"{sae_name}_ravel"] = _load_result(eval_output, sae_name)

        elapsed = time.time() - t0
        print(f"  {eval_type} completed in {elapsed:.0f}s")

    # Save combined results
    combined_path = os.path.join(output_dir, "exp37_saebench_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_path}")

    return all_results


def _load_sae_from_checkpoint(name: str, path: Path) -> nn.Module:
    """Instantiate the correct SAE class based on the checkpoint name and load weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)

    if name.startswith("l2_lambda"):
        lam_str = name.split("lambda")[1].split("_")[0]
        sae = L2RegPerFeatureSAE(D_MODEL, D_SAE, K, l2_lambda=float(lam_str))
    elif name.startswith("group_G"):
        g_str = name.split("G")[1].split("_")[0]
        sae = GroupScaleSAE(D_MODEL, D_SAE, K, n_groups=int(g_str))
    elif name.startswith("dropout_p"):
        p_str = name[len("dropout_p"):]
        sae = DropoutScaleSAE(D_MODEL, D_SAE, K, dropout_p=float(p_str))
    elif name == "standard":
        sae = BatchTopKSAE(D_MODEL, D_SAE, K)
    elif name == "adaptive_l2":
        sae = AdaptiveCosineBatchTopKSAE(D_MODEL, D_SAE, K)
    elif name == "perfeature_l2":
        sae = PerFeatureAdaptiveCosineSAE(D_MODEL, D_SAE, K)
    else:
        raise ValueError(f"Cannot determine SAE class from name: {name}")

    sae.load_state_dict(state)
    return sae


# =============================================================================
# Sweep Runners
# =============================================================================

def _load_model_and_tokenizer():
    """Load Gemma-2-2b model and tokenizer."""
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer


def _run_single_training(
    name: str,
    sae: nn.Module,
    model,
    tokenizer,
    eval_data: torch.Tensor,
    save_dir: Path,
    max_steps: int | None = None,
) -> dict:
    """Train one SAE and evaluate reconstruction."""
    sae = sae.to(device=DEVICE, dtype=torch.float32)
    stream = ActivationStream(model, tokenizer, LAYER)
    train_log, ckpt_log = train_sae(name, sae, stream, save_dir, max_steps)
    del stream
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  Evaluating reconstruction for {name}/L{LAYER}:")
    recon_results = evaluate_reconstruction(name, sae, eval_data)

    result = {
        "train_log": train_log,
        "checkpoint_log": ckpt_log,
        "reconstruction": recon_results,
    }

    # Capture final scale_a stats
    if hasattr(sae, "scale_a"):
        a = sae.scale_a.detach()
        if a.dim() == 0:
            result["final_scale_a"] = a.item()
        elif isinstance(sae, GroupScaleSAE):
            result["final_scale_a_groups"] = {
                "mean": a.mean().item(),
                "std": a.std().item(),
                "min": a.min().item(),
                "max": a.max().item(),
                "values": a.tolist(),
            }
        else:
            result["final_scale_a"] = {
                "mean": a.mean().item(),
                "std": a.std().item(),
                "min": a.min().item(),
                "max": a.max().item(),
                "near_zero_frac": (a.abs() < 0.05).float().mean().item(),
            }

    del sae
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_l2_sweep(
    model, tokenizer, eval_data: torch.Tensor, save_dir: Path,
    max_steps: int | None = None,
) -> dict:
    """B1: L2 penalty sweep over per-feature scale_a."""
    print("\n" + "=" * 70)
    print("SWEEP B1: L2 Penalty")
    print("=" * 70)

    results = {}
    for lam in L2_LAMBDAS:
        name = f"l2_lambda{lam}"
        print(f"\n{'─'*60}")
        print(f"  L2 lambda = {lam}")
        print(f"{'─'*60}")

        sae = L2RegPerFeatureSAE(D_MODEL, D_SAE, K, l2_lambda=lam)
        results[name] = _run_single_training(
            name, sae, model, tokenizer, eval_data, save_dir, max_steps,
        )

    return results


def run_group_sweep(
    model, tokenizer, eval_data: torch.Tensor, save_dir: Path,
    max_steps: int | None = None,
) -> dict:
    """B2: Group-wise scale_a sweep."""
    print("\n" + "=" * 70)
    print("SWEEP B2: Group-wise Scale")
    print("=" * 70)

    results = {}
    for g in GROUP_SIZES:
        name = f"group_G{g}"
        print(f"\n{'─'*60}")
        print(f"  Groups = {g} (features/group = {D_SAE // g})")
        print(f"{'─'*60}")

        sae = GroupScaleSAE(D_MODEL, D_SAE, K, n_groups=g)
        results[name] = _run_single_training(
            name, sae, model, tokenizer, eval_data, save_dir, max_steps,
        )

    return results


def run_dropout_sweep(
    model, tokenizer, eval_data: torch.Tensor, save_dir: Path,
    max_steps: int | None = None,
) -> dict:
    """B3: Dropout sweep over per-feature scale_a."""
    print("\n" + "=" * 70)
    print("SWEEP B3: Scale_a Dropout")
    print("=" * 70)

    results = {}
    for p in DROPOUT_PS:
        name = f"dropout_p{p}"
        print(f"\n{'─'*60}")
        print(f"  Dropout p = {p}")
        print(f"{'─'*60}")

        sae = DropoutScaleSAE(D_MODEL, D_SAE, K, dropout_p=p)
        results[name] = _run_single_training(
            name, sae, model, tokenizer, eval_data, save_dir, max_steps,
        )

    return results


def run_controls(
    model, tokenizer, eval_data: torch.Tensor, save_dir: Path,
    max_steps: int | None = None,
) -> dict:
    """B4: Control SAEs retrained with sqrt(d) init."""
    print("\n" + "=" * 70)
    print("SWEEP B4: Controls (sqrt(d) init)")
    print("=" * 70)

    results = {}

    controls = [
        ("standard", BatchTopKSAE(D_MODEL, D_SAE, K)),
        ("adaptive_l2", AdaptiveCosineBatchTopKSAE(D_MODEL, D_SAE, K)),
        ("perfeature_l2", PerFeatureAdaptiveCosineSAE(D_MODEL, D_SAE, K)),
    ]
    for name, sae in controls:
        print(f"\n{'─'*60}")
        print(f"  Control: {name}")
        print(f"{'─'*60}")
        results[name] = _run_single_training(
            name, sae, model, tokenizer, eval_data, save_dir, max_steps,
        )

    return results


def run_eval_sweep(
    device: str = "cuda",
    llm_batch_size: int = 16,
    force_rerun: bool = False,
) -> dict:
    """B5: SAEBench evaluation on controls + best from each sweep."""
    print("\n" + "=" * 70)
    print("SWEEP B5: SAEBench Evaluation")
    print("=" * 70)

    # Determine which SAEs to evaluate: controls + all sweep checkpoints
    sae_specs = []

    # Controls
    for name in ["standard", "adaptive_l2", "perfeature_l2"]:
        ckpt = f"{name}_L{LAYER}_final.pt"
        if (SAVE_DIR / ckpt).exists():
            sae_specs.append((name, ckpt))

    # L2 sweep
    for lam in L2_LAMBDAS:
        name = f"l2_lambda{lam}"
        ckpt = f"{name}_L{LAYER}_final.pt"
        if (SAVE_DIR / ckpt).exists():
            sae_specs.append((name, ckpt))

    # Group sweep
    for g in GROUP_SIZES:
        name = f"group_G{g}"
        ckpt = f"{name}_L{LAYER}_final.pt"
        if (SAVE_DIR / ckpt).exists():
            sae_specs.append((name, ckpt))

    # Dropout sweep
    for p in DROPOUT_PS:
        name = f"dropout_p{p}"
        ckpt = f"{name}_L{LAYER}_final.pt"
        if (SAVE_DIR / ckpt).exists():
            sae_specs.append((name, ckpt))

    if not sae_specs:
        print("  No checkpoints found -- run training sweeps first")
        return {}

    print(f"  Found {len(sae_specs)} checkpoints for evaluation:")
    for name, ckpt in sae_specs:
        print(f"    {name}: {ckpt}")

    return run_saebench_evals(
        sae_specs=sae_specs,
        eval_types=["core", "sparse_probing", "ravel"],
        device=device,
        llm_batch_size=llm_batch_size,
        force_rerun=force_rerun,
    )


# =============================================================================
# Results I/O
# =============================================================================

def _load_existing_results() -> dict:
    """Load existing results file if it exists, or return empty structure."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"config": get_config_dict(), "sweeps": {}}


def _save_results(results: dict) -> None:
    """Save results to disk."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp37: Regularized Scaling — Training + Evaluation",
    )
    parser.add_argument(
        "--sweep",
        choices=["l2", "group", "dropout", "controls", "eval", "all"],
        default="all",
        help="Which sweep to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke test")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16, help="LLM batch size for SAEBench")
    parser.add_argument("--force-rerun", action="store_true", help="Force SAEBench rerun")
    args = parser.parse_args()

    max_steps = args.max_steps
    if args.dry_run and max_steps is None:
        max_steps = 100

    print("=" * 70)
    print("Experiment 37: Regularized Scaling")
    print("=" * 70)
    print(f"Model:     {MODEL_NAME}")
    print(f"Layer:     {LAYER}")
    print(f"Sweep:     {args.sweep}")
    print(f"SAE:       d_model={D_MODEL}, d_sae={D_SAE}, k={K}")
    print(f"Training:  {N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")
    if max_steps is not None:
        print(f"Override:  max_steps={max_steps}")
    if args.dry_run:
        print("DRY RUN MODE")
    print()

    torch.manual_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = _load_existing_results()

    # Training sweeps need the model
    training_sweeps = {"l2", "group", "dropout", "controls"}
    needs_training = args.sweep == "all" or args.sweep in training_sweeps

    if needs_training:
        model, tokenizer = _load_model_and_tokenizer()

        # Collect eval data once (shared across all sweeps)
        eval_data = collect_eval_data(model, tokenizer, LAYER, 1_000_000)
        total_t0 = time.time()

        if args.sweep in ("l2", "all"):
            sweep_results = run_l2_sweep(model, tokenizer, eval_data, SAVE_DIR, max_steps)
            all_results.setdefault("sweeps", {})["l2"] = sweep_results
            _save_results(all_results)

        if args.sweep in ("group", "all"):
            sweep_results = run_group_sweep(model, tokenizer, eval_data, SAVE_DIR, max_steps)
            all_results.setdefault("sweeps", {})["group"] = sweep_results
            _save_results(all_results)

        if args.sweep in ("dropout", "all"):
            sweep_results = run_dropout_sweep(model, tokenizer, eval_data, SAVE_DIR, max_steps)
            all_results.setdefault("sweeps", {})["dropout"] = sweep_results
            _save_results(all_results)

        if args.sweep in ("controls", "all"):
            sweep_results = run_controls(model, tokenizer, eval_data, SAVE_DIR, max_steps)
            all_results.setdefault("sweeps", {})["controls"] = sweep_results
            _save_results(all_results)

        total_elapsed = time.time() - total_t0
        print(f"\nTotal training time: {total_elapsed/3600:.1f} hours")

        del model, tokenizer, eval_data
        gc.collect()
        torch.cuda.empty_cache()

    # SAEBench evaluation
    if args.sweep in ("eval", "all"):
        eval_results = run_eval_sweep(
            device=args.device,
            llm_batch_size=args.batch_size,
            force_rerun=args.force_rerun,
        )
        all_results.setdefault("sweeps", {})["saebench"] = eval_results
        _save_results(all_results)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Results:          {RESULTS_PATH}")
    print(f"Checkpoints:      {SAVE_DIR}/")
    print(f"SAEBench results: {SAEBENCH_OUTPUT}/")


if __name__ == "__main__":
    main()
