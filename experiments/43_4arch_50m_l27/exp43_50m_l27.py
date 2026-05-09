"""
Experiment 43: 50M token 4-architecture comparison at L27 (saprmarks recipe)
=============================================================================

Tests whether NoC's 4.3% dead features at 500M (exp42c) appear at 50M,
and whether all 4 architectures maintain FVE parity at L27 (higher norms ~407).

L27 is where the cosine advantage was historically largest (exp10, exp17, exp36).
50M tokens is enough to see convergence trends (exp36 showed FVE stabilizes by
~20M tokens for standard, and NoC dead features emerged at ~25M in exp42c).

Architectures:
  1. standard:       BatchTopK (inner product)
  2. adaptive_l2:    Cosine + global adaptive scale (a learns ~0.25 at L18)
  3. perfeature_l2:  Cosine + per-feature adaptive scale
  4. no_C:           NoC — pure cosine, norm-preserving decode

All use identical saprmarks recipe from exp40.

Run on <gpu-server> GPU 1:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python3 experiments/exp43_50m_l27.py \
        2>&1 | tee experiments/exp43_output.log &
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

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE  # 24,414
DECAY_START = int(0.8 * N_STEPS)        # 19,531

BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE

CHECKPOINT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
CHECKPOINT_STEPS = sorted(set(int(f * N_STEPS) for f in CHECKPOINT_FRACS))

N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

SAVE_DIR = Path("/mnt/nvme0/checkpoints/exp43")
RESULTS_PATH = Path("experiments/exp43_results.json")


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

class BatchTopKSAE(nn.Module):
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


class PerFeatureAdaptiveCosineSAE(nn.Module):
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


class NoCBatchTopKSAE(nn.Module):
    """Norm-preserving cosine SAE. Magnitude never in sparse code."""
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.tensor(-1.0))
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
        x_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_u = x_c / x_norm
        w_u = F.normalize(self.W_enc, dim=-1, eps=NORM_EPS)
        post_relu = F.relu(x_u @ w_u.T)
        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)
        self._cached_x_norm = x_norm
        if return_active:
            active_indices = encoded.sum(0) > 0
            return encoded, active_indices, post_relu
        return encoded

    def decode(self, f, x_norm=None):
        w_u = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w_u
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        if x_norm is None:
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
    # Skip past training region
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
    is_noc = isinstance(sae, NoCBatchTopKSAE)
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")

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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if sae.W_dec.grad is not None:
                sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                    sae.W_dec.data, sae.W_dec.grad.data
                )

            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Decoder unit-norm constraint
            set_decoder_norm_to_unit_norm(sae.W_dec.data)
            # NoC also normalizes encoder
            if is_noc:
                with torch.no_grad():
                    sae.W_enc.div_(sae.W_enc.norm(dim=1, keepdim=True).clamp(min=NORM_EPS))

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
                if hasattr(sae, "scale_a"):
                    if sae.scale_a.dim() == 0:
                        entry["scale_a"] = sae.scale_a.item()
                    else:
                        entry["scale_a_mean"] = sae.scale_a.mean().item()
                        entry["scale_a_median"] = sae.scale_a.median().item()

                log.append(entry)
                elapsed = time.time() - t0
                tok = global_step * BATCH_SIZE
                tok_per_sec = tok / elapsed if elapsed > 0 else 0
                eta = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                scale_str = ""
                if hasattr(sae, "scale_a"):
                    if sae.scale_a.dim() == 0:
                        scale_str = f" a={sae.scale_a.item():.4f}"
                    else:
                        scale_str = f" a_mean={sae.scale_a.mean().item():.4f}"
                print(f"    [{tag}] {global_step:>5d}/{N_STEPS} | "
                      f"loss={loss.item():.1f} recon={recon_loss.item():.1f} "
                      f"auxk={auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else 0:.3f} | "
                      f"FVE={fve:.4f} L0={l0:.0f} dead={dead_frac:.3f}({n_dead})"
                      f"{scale_str} | {tok/1e6:.1f}M ETA {eta/3600:.1f}h")

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
    """Ablation: cos vs inner product for causal prediction (RNH diagnostic)."""
    tag = f"{name}/L{LAYER}"
    print(f"    [{tag}] Ablation ({N_ABLATION_FEATURES} features, "
          f"{N_ABLATION_SAMPLES} samples)...")

    sae.eval()
    is_noc = isinstance(sae, NoCBatchTopKSAE)

    # Collect activations to find top features
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                       split="train", streaming=True)
    text_iter = iter(ds)
    # Skip to a fresh region
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

    # Find top-k most active features
    act_sums = torch.zeros(D_SAE, device=DEVICE)
    for text in texts[:50]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=CTX_LEN).to(DEVICE)
        acts = _collect_layer_acts(model, LAYER, inputs)
        flat = acts.reshape(-1, D_MODEL)
        features = sae.encode(flat)
        act_sums += features.sum(dim=0)

    top_features = act_sums.topk(N_ABLATION_FEATURES).indices.tolist()

    # For each feature, measure ablation impact
    cos_wins = 0
    cos_kl_corrs, inner_kl_corrs = [], []

    for feat_idx in top_features:
        feat_dir = sae.W_dec[feat_idx]
        feat_dir_unit = feat_dir / feat_dir.norm().clamp(min=1e-8)

        cos_vals, inner_vals, kl_vals = [], [], []

        for text in texts[:N_ABLATION_SAMPLES]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=CTX_LEN).to(DEVICE)

            # Clean run
            with torch.no_grad():
                outputs_clean = model(**inputs)
                clean_logits = outputs_clean.logits[0, -1]
                clean_probs = F.softmax(clean_logits, dim=-1)

            # Get activation at target layer
            act = _collect_layer_acts(model, LAYER, inputs)
            act_flat = act[0, -1].float()  # last token, cast to float32

            cos_sim = F.cosine_similarity(act_flat.unsqueeze(0),
                                          feat_dir_unit.unsqueeze(0)).item()
            inner_prod = (act_flat * feat_dir).sum().item()

            # Ablate: remove feature projection
            proj = (act_flat @ feat_dir_unit) * feat_dir_unit
            ablated_act = act_flat - proj

            # Run with ablated activation
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
    print(f"Experiment 43: 50M token 4-architecture comparison at L{LAYER}")
    print(f"  {N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")
    print(f"  Architectures: standard, adaptive_l2, perfeature_l2, no_C")
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
            "experiment": "exp43_50m_l27",
            "model": MODEL_NAME, "layer": LAYER,
            "d_sae": D_SAE, "k": K, "lr": LR,
            "n_train_tokens": N_TRAIN_TOKENS, "n_steps": N_STEPS,
            "mean_norm": mean_norm,
        },
        "runs": {},
    }

    variants = [
        ("standard", BatchTopKSAE),
        ("adaptive_l2", AdaptiveCosineBatchTopKSAE),
        ("perfeature_l2", PerFeatureAdaptiveCosineSAE),
        ("no_C", NoCBatchTopKSAE),
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
            if sae.scale_a.dim() == 0:
                run_data["scale_a"] = sae.scale_a.item()
            else:
                run_data["scale_a_mean"] = sae.scale_a.mean().item()
                run_data["scale_a_median"] = sae.scale_a.median().item()
                run_data["scale_a_std"] = sae.scale_a.std().item()
                run_data["scale_a_pct_near_zero"] = (sae.scale_a.abs() < 0.05).float().mean().item()

        results["runs"][name] = run_data

        # Save after each variant
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved results for {name}")

        # Clean up
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
        elif "scale_a_mean" in data:
            scale = f" a_mean={data['scale_a_mean']:.4f}"
        print(f"  {name:20s} | FVE={r['fve']:.4f} | dead={r['dead_frac']:.3f} | "
              f"alive={r['alive_count']:,} | cos>inner={a['cos_wins_inner']}{scale}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
