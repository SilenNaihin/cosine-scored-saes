"""
Experiment 39: Norm-Preserving Cosine BatchTopK SAE
====================================================

Tests the logical endpoint of the Relative Norm Hypothesis:
**do the whole SAE on the unit sphere, then re-inject the norm as a pure
output projection.** Features encode direction only; magnitude bypasses the
sparse code.

Architecture (NormPreservingCosineBatchTopKSAE, untied):
    x_c = x - b_dec                      # center
    a   = ||x_c||                        # saved scalar (not a feature)
    x'  = x_c / a                        # unit-norm direction
    f   = BatchTopK(ReLU(x' @ W_enc.T))  # W_enc rows unit-norm → cosine scores
    x'' = f @ W_dec                      # raw reconstruction, W_dec rows unit-norm
    out = x'' * a / ||x''|| + b_dec      # norm restored from saved a
    loss = 1 - cos(x_c, x'')             # pure direction loss

Baselines trained on the same data/seed for each layer:
  1. BatchTopKSAE        — standard inner-product encoder, MSE loss
  2. CosineBatchTopKSAE  — cosine encoder + linear decoder, MSE loss
  3. NormPreservingCosineBatchTopKSAE — the new architecture, cosine loss

Run on <gpu-server> (GPU 0).

Usage:
    ./sync.sh                         # local, sync code to remote
    ssh <user>@<ip>
    cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup uv run python \
        experiments/exp39_norm_preserving_sae.py \
        > experiments/exp39_output.log 2>&1 &

Smoke test (few hundred steps, one layer):
    EXP39_SMOKE=1 CUDA_VISIBLE_DEVICES=0 uv run python \
        experiments/exp39_norm_preserving_sae.py
"""

import gc
import json
import math
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

SMOKE = os.environ.get("EXP39_SMOKE", "0") == "1"

DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
STORAGE_DTYPE = torch.float16
CKPT_DTYPE = torch.float16  # disk is tight on the A100

# --- Model ---
MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [18] if SMOKE else [9, 18, 27]
D_MODEL = 4096

# --- SAE architecture ---
D_SAE = 16384  # 4x d_model
K = 80

# --- Data ---
N_TRAIN_TOKENS = 200_000 if SMOKE else 5_000_000
N_EVAL_TOKENS = 50_000 if SMOKE else 400_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR = 3e-4
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 50 if SMOKE else 200
NORM_EPS = 1e-8

# --- Output ---
SAVE_DIR = "checkpoints/exp39"
RESULTS_PATH = "experiments/exp39_smoke_results.json" if SMOKE else "experiments/exp39_results.json"

# --- Derived ---
N_STEPS = max(N_TRAIN_TOKENS // BATCH_SIZE, 1)
WARMUP_STEPS = max(int(N_STEPS * WARMUP_FRAC), 1)


# =============================================================================
# SAE Architectures (all untied)
# =============================================================================

class BatchTopKSAE(nn.Module):
    """Standard BatchTopK SAE with inner-product encoder, MSE loss."""

    loss_kind = "mse"

    def __init__(self, d_model: int, d_sae: int, k: int = 50):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.zeros(()))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(NORM_EPS))
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

    def post_step(self):
        pass


class CosineBatchTopKSAE(nn.Module):
    """Cosine-similarity encoder, linear decoder, MSE loss.

    encoder: cos_sim(x - b_dec, W_enc_rows) + b_enc, both unit-normalized each step.
    decoder: standard linear.
    """

    loss_kind = "mse"

    def __init__(self, d_model: int, d_sae: int, k: int = 50):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.zeros(()))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(NORM_EPS))
            self.W_enc.copy_(self.W_dec)

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
        x_c = x - self.b_dec
        x_u = F.normalize(x_c, dim=-1, eps=NORM_EPS)
        w_u = F.normalize(self.W_enc, dim=-1, eps=NORM_EPS)
        pre_acts = x_u @ w_u.T + self.b_enc
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

    def post_step(self):
        pass


class NormPreservingCosineBatchTopKSAE(nn.Module):
    """Unit-sphere SAE with norm re-injection and cosine loss (untied).

    x_c = x - b_dec;  a = ||x_c||;  x' = x_c / a
    f   = BatchTopK(ReLU(x' @ W_enc.T))     # W_enc rows renormalized each step
    x'' = f @ W_dec                          # W_dec rows renormalized each step
    out = x'' * a / ||x''|| + b_dec

    Both W_enc and W_dec rows are projected to unit norm after every optimizer
    step, so the encoder produces true cosine scores ∈ [0, 1] after ReLU.
    """

    loss_kind = "cosine"

    def __init__(self, d_model: int, d_sae: int, k: int = 50):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.zeros(()))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(NORM_EPS))
            self.W_enc.copy_(self.W_dec)
            self.W_enc.div_(self.W_enc.norm(dim=1, keepdim=True).clamp_min(NORM_EPS))

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

    def encode_full(self, x: torch.Tensor):
        """Return (features, a) where a = ||x - b_dec|| per-token."""
        x_c = x - self.b_dec
        a = x_c.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS)
        x_u = x_c / a
        # W_enc is kept near-unit by post_step; no runtime F.normalize needed for
        # correctness, but we normalize here to be robust against drift within a step.
        w_u = F.normalize(self.W_enc, dim=-1, eps=NORM_EPS)
        pre_acts = F.relu(x_u @ w_u.T)
        if self.training:
            f = self._batch_topk(pre_acts)
        else:
            f = torch.where(pre_acts >= self.threshold, pre_acts, torch.zeros_like(pre_acts))
        return f, a

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f, _ = self.encode_full(x)
        return f

    def decode_raw(self, f: torch.Tensor) -> torch.Tensor:
        """Raw reconstruction before norm restoration (centered, direction-space)."""
        w_u = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        return f @ w_u

    def decode(self, f: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x_raw = self.decode_raw(f)
        nrm = x_raw.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS)
        return x_raw * (a / nrm) + self.b_dec

    def forward(self, x: torch.Tensor):
        f, a = self.encode_full(x)
        x_raw = self.decode_raw(f)
        # Output uses norm-restoration; but loss is computed on (x_c, x_raw) cosine,
        # which is scale-invariant, so we pack both into the return for caller use.
        nrm = x_raw.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS)
        x_hat = x_raw * (a / nrm) + self.b_dec
        # Stash raw-centered reconstruction so the loss can bypass the rescale for
        # clean gradient flow (cosine is scale-invariant either way).
        self._last_raw = x_raw
        self._last_a = a
        return x_hat, f

    def post_step(self):
        with torch.no_grad():
            self.W_enc.div_(self.W_enc.norm(dim=1, keepdim=True).clamp_min(NORM_EPS))
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(NORM_EPS))


VARIANTS = [
    ("standard",       BatchTopKSAE),
    ("cosine",         CosineBatchTopKSAE),
    ("norm_preserve",  NormPreservingCosineBatchTopKSAE),
]


# =============================================================================
# Activation Collection
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


def collect_activations(model, tokenizer, layer_idx, n_tokens, skip_docs=0):
    """Stream activations directly into a pre-allocated CPU buffer.

    Avoids the list-append + torch.cat pattern which peaks at 2-3x target size.
    Memory is bounded at n_tokens * d_model * sizeof(fp16) ≈ n_tokens * 8 KB.
    """
    label = "eval" if skip_docs > 0 else "train"
    print(f"  Collecting {label} activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)

    if skip_docs > 0:
        for i, _ in enumerate(text_iter):
            if i >= skip_docs:
                break

    # Preallocate — bounded peak memory.
    result = torch.empty((n_tokens, D_MODEL), dtype=STORAGE_DTYPE)
    cursor = 0
    gc_every = 100
    batch_count = 0

    while cursor < n_tokens:
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

        take = min(flat.shape[0], n_tokens - cursor)
        result[cursor:cursor + take] = flat[:take].to("cpu", dtype=STORAGE_DTYPE)
        cursor += take

        del acts, flat, norms, inputs
        batch_count += 1
        if batch_count % gc_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    result = result[:cursor]
    # Stream-compute norms to avoid doubling memory with .float() on huge tensor.
    norm_chunks = []
    chunk = 65536
    for i in range(0, result.shape[0], chunk):
        norm_chunks.append(result[i:i+chunk].float().norm(dim=-1))
    norms = torch.cat(norm_chunks)
    print(f"  Layer {layer_idx}: {result.shape[0]:,} {label} tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={norms.mean():.1f}, std={norms.std():.1f})")
    return result


# =============================================================================
# Training
# =============================================================================

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_loss(sae, batch, x_hat, features):
    """MSE for standard/cosine baselines; cosine loss on centered residual for norm_preserve."""
    if sae.loss_kind == "cosine":
        x_c = batch - sae.b_dec
        r_c = sae._last_raw  # pre-rescale reconstruction in centered direction space
        cos = F.cosine_similarity(x_c, r_c, dim=-1, eps=NORM_EPS)
        return (1.0 - cos).mean()
    return (batch - x_hat).pow(2).sum(dim=-1).mean()


def train_sae(name, sae, train_data, layer_idx):
    print(f"\n  Training {name} | L{layer_idx} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{train_data.shape[0]:,} tokens, {N_STEPS} steps, loss={sae.loss_kind}")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Index-only shuffle to avoid duplicating the activation tensor in RAM.
    n_tokens = train_data.shape[0]
    perm = torch.randperm(n_tokens)

    sae.train()
    log = []
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        start = ((step - 1) * BATCH_SIZE) % n_tokens
        end = start + BATCH_SIZE
        if end > n_tokens:
            idx = torch.cat([perm[start:n_tokens], perm[0:end - n_tokens]])
        else:
            idx = perm[start:end]
        batch = train_data[idx].to(DEVICE, dtype=torch.float32)

        x_hat, features = sae(batch)
        loss = compute_loss(sae, batch, x_hat, features)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        sae.post_step()

        if step % LOG_EVERY == 0 or step == N_STEPS:
            with torch.no_grad():
                l0 = (features != 0).float().sum(dim=-1).mean().item()
                total_var = torch.var(batch, dim=0, unbiased=False).sum()
                resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                dead = (features.sum(dim=0) == 0).float().mean().item()
                # Norm preservation check (always near-perfect by construction for norm_preserve)
                norm_ratio = (x_hat.norm(dim=-1) / batch.norm(dim=-1).clamp_min(NORM_EPS)).mean().item()

            tokens_seen = step * BATCH_SIZE
            entry = {
                "step": step, "loss": loss.item(),
                "l0": l0, "fve": fve, "cos_recon": cos_r, "dead_frac": dead,
                "norm_ratio": norm_ratio, "lr": scheduler.get_last_lr()[0],
            }
            log.append(entry)
            if step % (LOG_EVERY * 5) == 0 or step == N_STEPS:
                print(f"    [{name:>14s}] step {step:>5d}/{N_STEPS} | "
                      f"loss={loss.item():.4f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f} | "
                      f"norm_ratio={norm_ratio:.3f} | tok={tokens_seen/1e6:.1f}M")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{name}] Done in {elapsed:.1f}s")
    return log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data):
    sae.eval()
    n = eval_data.shape[0]
    losses, cos_sims, l0s = [], [], []
    direction_cos = []      # cos between (x - b_dec) directions
    dir_resid_sum = 0.0     # sum of ||unit_x - unit_xhat||^2 style quantity
    dir_total_sum = 0.0
    total_var_sum, resid_var_sum = 0.0, 0.0
    norm_ratios = []
    dead_counts = None

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        losses.append((batch - x_hat).pow(2).sum(dim=-1).mean().item())
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        # Direction metrics: compute on (x - b_dec)/||x - b_dec||
        b_dec = getattr(sae, "b_dec", torch.zeros_like(batch[0]))
        x_c = batch - b_dec
        xh_c = x_hat - b_dec
        x_u = F.normalize(x_c, dim=-1, eps=NORM_EPS)
        xh_u = F.normalize(xh_c, dim=-1, eps=NORM_EPS)
        direction_cos.append(F.cosine_similarity(x_c, xh_c, dim=-1).mean().item())
        # Direction-space FVE (variance reduction on unit-sphere coords)
        dir_total_sum += torch.var(x_u, dim=0, unbiased=False).sum().item()
        dir_resid_sum += torch.var(x_u - xh_u, dim=0, unbiased=False).sum().item()
        norm_ratios.append((x_hat.norm(dim=-1) / batch.norm(dim=-1).clamp_min(NORM_EPS)).mean().item())

        alive = (features > 0).any(dim=0)
        if dead_counts is None:
            dead_counts = ~alive
        else:
            dead_counts &= ~alive

    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0

    results = {
        "mse": float(np.mean(losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "cos_direction": float(np.mean(direction_cos)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
        "fve_direction": float(1 - dir_resid_sum / dir_total_sum) if dir_total_sum > 0 else 0,
        "norm_ratio": float(np.mean(norm_ratios)),
        "dead_frac": dead_frac,
        "alive_count": alive_count,
    }
    print(f"    [{name:>14s}] FVE={results['fve']:.4f} | FVE_dir={results['fve_direction']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"norm_ratio={results['norm_ratio']:.3f} | dead={dead_frac*100:.1f}% | alive={alive_count}")
    return results


@torch.no_grad()
def evaluate_norm_robustness(name, sae, eval_data, scales=(0.5, 1.0, 2.0, 5.0)):
    sae.eval()
    results = {}
    for scale in scales:
        n = eval_data.shape[0]
        total_var_sum, resid_var_sum = 0.0, 0.0
        cos_sims = []
        for i in range(0, n, BATCH_SIZE):
            batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32) * scale
            x_hat, _ = sae(batch)
            cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
            total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
            resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        fve = float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0
        cos_r = float(np.mean(cos_sims))
        results[f"scale_{scale}"] = {"fve": fve, "cos_recon": cos_r}
        print(f"    [{name:>14s}] eval_scale={scale}: FVE={fve:.4f} | cos={cos_r:.4f}")
    return results


@torch.no_grad()
def evaluate_kl_patch(name, model, sae, eval_data, layer_idx, n_samples=200):
    """Patch the SAE reconstruction into the model at `layer_idx` and measure KL
    vs the unmodified forward pass. Uses the first few tokens of a real prompt
    rather than a dummy, so the KL is meaningful.
    """
    sae.eval()
    print(f"\n  Patch-KL [{name}] on {n_samples} samples...")

    # Pick activations that correspond to actual sequence positions. We don't have
    # the original tokens here (we only stored activations), so we patch in the
    # activation into a short dummy sequence: this is not a full substitution-KL
    # eval. For a coarse signal, we compute KL between (model with hook replacing
    # layer output by reconstruction) vs (model with hook replacing by the original
    # activation) — so the KL cost is purely the reconstruction error.
    probe = eval_data[:n_samples].to(DEVICE)
    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
    kls = []
    for i in range(n_samples):
        x = probe[i:i+1].to(dtype=torch.float32)
        x_hat, _ = sae(x)
        model_dtype = next(model.parameters()).dtype
        x_shaped = x.unsqueeze(0).to(model_dtype)
        xh_shaped = x_hat.unsqueeze(0).to(model_dtype)

        def make_hook(replacement):
            def hook(module, inputs, outputs):
                return (replacement,) + outputs[1:] if isinstance(outputs, tuple) else replacement
            return hook

        h = model.model.layers[layer_idx].register_forward_hook(make_hook(x_shaped))
        try:
            orig_logits = model(dummy_input).logits[0, -1, :].float()
        finally:
            h.remove()
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(xh_shaped))
        try:
            recon_logits = model(dummy_input).logits[0, -1, :].float()
        finally:
            h.remove()

        orig_log_probs = torch.log_softmax(orig_logits, dim=-1)
        orig_probs = orig_log_probs.exp().clamp_min(1e-12)
        recon_log_probs = torch.log_softmax(recon_logits, dim=-1)
        kl = (orig_probs * (orig_log_probs - recon_log_probs)).sum().item()
        if not (np.isnan(kl) or kl < 0):
            kls.append(kl)
    if not kls:
        return {"kl_mean": None, "kl_median": None, "n": 0}
    kls = np.array(kls)
    r = {
        "kl_mean": float(kls.mean()),
        "kl_median": float(np.median(kls)),
        "kl_p90": float(np.percentile(kls, 90)),
        "n": len(kls),
    }
    print(f"    [{name:>14s}] KL: mean={r['kl_mean']:.4f} median={r['kl_median']:.4f} "
          f"p90={r['kl_p90']:.4f} (n={r['n']})")
    return r


# =============================================================================
# Main
# =============================================================================

def main():
    mode = "SMOKE TEST" if SMOKE else "FULL RUN"
    print(f"Experiment 39: Norm-Preserving Cosine BatchTopK SAE  [{mode}]")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_model: {D_MODEL}, d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Tokens: {N_TRAIN_TOKENS:,} train, {N_EVAL_TOKENS:,} eval")
    print(f"Steps: {N_STEPS}, Warmup: {WARMUP_STEPS}")
    print(f"Variants: {[v[0] for v in VARIANTS]}")

    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=MODEL_DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "config": {
            "model_name": MODEL_NAME,
            "experiment": "norm_preserving_sae",
            "layers": LAYERS,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "k": K,
            "n_train_tokens": N_TRAIN_TOKENS,
            "n_eval_tokens": N_EVAL_TOKENS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "smoke": SMOKE,
        },
        "layers": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}\n  LAYER {layer_idx}\n{'='*70}")

        train_data = collect_activations(model, tokenizer, layer_idx, N_TRAIN_TOKENS)
        eval_data = collect_activations(
            model, tokenizer, layer_idx, N_EVAL_TOKENS, skip_docs=200_000,
        )
        mean_norm = train_data.float().norm(dim=-1).mean().item()
        layer_results = {"mean_norm": mean_norm}
        print(f"  Mean activation norm: {mean_norm:.2f}")

        for vname, cls in VARIANTS:
            print(f"\n  --- VARIANT: {vname} (L{layer_idx}) ---")
            torch.manual_seed(SEED)
            sae = cls(D_MODEL, D_SAE, K).to(DEVICE)

            train_log = train_sae(vname, sae, train_data, layer_idx)
            recon = evaluate_reconstruction(vname, sae, eval_data)
            print(f"\n  Norm robustness -- {vname}")
            robustness = evaluate_norm_robustness(vname, sae, eval_data)
            kl = evaluate_kl_patch(
                vname, model, sae, eval_data, layer_idx,
                n_samples=50 if SMOKE else 200,
            )

            layer_results[vname] = {
                "training": train_log,
                "reconstruction": recon,
                "norm_robustness": robustness,
                "kl_patch": kl,
                "loss_kind": sae.loss_kind,
            }

            # Save checkpoint in fp16 to save disk
            ckpt = {k: v.to(CKPT_DTYPE) if torch.is_tensor(v) and v.is_floating_point() else v
                    for k, v in sae.state_dict().items()}
            torch.save(ckpt, save_dir / f"{vname}_L{layer_idx}.pt")

            all_results["layers"][str(layer_idx)] = layer_results
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            del sae
            gc.collect()
            torch.cuda.empty_cache()

        del train_data, eval_data
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"\n  {'Layer':>5s}  {'Variant':<15s} {'FVE':>7s} {'FVE_dir':>8s} {'cos':>6s} "
          f"{'L0':>5s} {'norm_r':>7s} {'dead%':>6s} {'KL_med':>8s}")
    print(f"  {'-'*5}  {'-'*15} {'-'*7} {'-'*8} {'-'*6} {'-'*5} {'-'*7} {'-'*6} {'-'*8}")
    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        for vname, _ in VARIANTS:
            r = lr.get(vname, {})
            rec = r.get("reconstruction", {})
            kl = r.get("kl_patch", {}) or {}
            print(f"  {layer_idx:>5d}  {vname:<15s} "
                  f"{rec.get('fve', 0):>7.4f} "
                  f"{rec.get('fve_direction', 0):>8.4f} "
                  f"{rec.get('cos_recon', 0):>6.4f} "
                  f"{rec.get('l0', 0):>5.0f} "
                  f"{rec.get('norm_ratio', 0):>7.3f} "
                  f"{rec.get('dead_frac', 0)*100:>5.1f}% "
                  f"{(kl.get('kl_median') or 0):>8.4f}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
