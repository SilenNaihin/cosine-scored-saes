"""
Experiment 42: Mistral-7B Initialization Disambiguation
========================================================

Central contradiction in the project:
  - sqrt(d) init is BETTER at scale on Qwen (exp34/41): forces optimizer
    to learn scale_a by climbing from wrong init
  - sqrt(d) init KILLS Mistral (exp25): 100% dead features at L8 (norms=6.3
    vs init scale=64, a 10x overshoot)
  - norm-adaptive init was never tested on Mistral
  - sqrt(d) init at 50M+ was never tested on Mistral (only 5M in exp25)

The mismatch is ASYMMETRIC:
  - Qwen L27: norms=407, sqrt(d)=64 → scale too SMALL → features alive but
    suboptimal → gradients exist → optimizer can climb at 50M tokens
  - Mistral L8: norms=6.3, sqrt(d)=64 → scale too LARGE → features dead →
    NO gradient → optimizer can never escape

Hypotheses:
  H1 (Asymmetric dead zone): When init is too LARGE, features die immediately
     with no gradient to recover. Norm-adaptive is necessary for "overshoot"
     models. When init is too SMALL, the optimizer can escape given enough
     tokens. The direction of the mismatch matters.

  H2 (Scale resolves everything): Given enough tokens (50M), the optimizer
     can escape the dead zone on Mistral. The 5M exp25 failure was just
     insufficient training.

  H3 (Mistral is different): Mistral has fundamentally different activation
     geometry (exp25 showed negative scale_a at all layers). Cosine SAEs
     work differently on Mistral regardless of init.

Design:
  5 variants × 3 layers = 15 runs at 50M tokens
  - standard (baseline, no scale parameter)
  - adaptive_l2 × {sqrt(d), norm-adaptive}
  - group_G4   × {sqrt(d), norm-adaptive}

  Layer-first ordering: all 5 variants at L8, then L16, then L24.
  L8 is the critical layer (10x mismatch, 100% dead in exp25).

  Mistral-7B-v0.1 (32 layers, d_model=4096, RMSNorm)
  Layers: L8, L16, L24 (matching exp25)

Expected runtime: ~22-27 hours on H100 (~1.5h per run + eval)

How to run:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 experiments/exp42_mistral_init.py 2>&1 | tee experiments/exp42_output.log
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
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LAYERS = [8, 16, 24]  # 25%, 50%, 75% of 32 layers (matching exp25)
D_MODEL = 4096

# --- SAE ---
D_SAE = 16384          # 4x d_model (consistent with all Qwen experiments)
K = 80
N_GROUPS = 4           # for group_G4

# --- Data ---
N_TRAIN_TOKENS = 50_000_000    # 50M per variant per layer
N_EVAL_TOKENS = 2_000_000      # 2M eval tokens per layer
CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 4
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR = 3e-4                      # Validated by exp30 for both architectures
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 200                 # More frequent logging (shorter runs)

# --- Checkpoints (extra early points to see dead feature dynamics) ---
CHECKPOINT_FRACS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

# --- Output ---
SAVE_DIR = "/mnt/nvme0/checkpoints/exp42"
RESULTS_PATH = "experiments/exp42_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE    # 12,207
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC) # 610
CHECKPOINT_STEPS = sorted(set(int(f * N_STEPS) for f in CHECKPOINT_FRACS))

# --- Streaming buffer ---
BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE


def get_config_dict():
    return {
        "experiment": 39,
        "model_name": MODEL_NAME, "layers": LAYERS, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K, "n_groups": N_GROUPS,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "n_steps": N_STEPS, "warmup_steps": WARMUP_STEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
        "note": "Mistral-7B init disambiguation: sqrt(d) vs norm-adaptive. "
                "Resolves contradiction between exp34 (sqrt(d) better at scale on Qwen) "
                "and exp25 (sqrt(d) kills Mistral).",
    }


# =============================================================================
# SAE Architectures
# =============================================================================

class StandardBatchTopKSAE(nn.Module):
    """Standard inner-product encoder with BatchTopK. Baseline."""

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
        pre_acts = x_centered @ self.W_enc.T + self.b_enc
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


class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Cosine encoder with 1 global adaptive scale_a.

    Encoder computes:
        scale = exp(scale_a * log(||x||) + scale_b)
        pre_acts = scale * cos_sim(x, W_enc) + b_enc

    When scale_a=0: pure cosine (norm-invariant detection).
    When scale_a=1: equivalent to inner product.
    The optimizer learns where on this spectrum each model needs.
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
        # Default: sqrt(d_model) init. Caller overrides for norm-adaptive.
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

    Features divided into n_groups groups sharing scale_a and scale_b.
    G=4 for d_sae=16384 means 4 params controlling 4096 features each.
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
    std_norm = norms.std().item()
    print(f"    L{layer_idx}: {result.shape[0]:,} tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={mean_norm:.1f}, std={std_norm:.1f}, "
          f"sqrt(d)={math.sqrt(D_MODEL):.1f}, ratio={mean_norm/math.sqrt(D_MODEL):.2f}x)")
    return result, mean_norm


# =============================================================================
# Training
# =============================================================================

def make_lr_schedule(n_steps, warmup_steps):
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule


def train_sae_streaming(name, sae, stream, layer_idx, save_dir, checkpoint_steps):
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")

    # Log initial scale_b for cosine variants
    if hasattr(sae, "scale_b"):
        sb = sae.scale_b
        if sb.dim() == 0:
            print(f"    scale_b init = {sb.item():.4f} (exp={math.exp(sb.item()):.1f})")
        else:
            print(f"    scale_b init = {sb.mean().item():.4f} (exp={math.exp(sb.mean().item()):.1f})")

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

                if hasattr(sae, "scale_b"):
                    sb = sae.scale_b
                    if sb.dim() == 0:
                        entry["scale_b"] = sb.item()
                        scale_str += f" | b={sb.item():.3f}(exp={math.exp(sb.item()):.1f})"
                    else:
                        entry["scale_b_mean"] = sb.mean().item()
                        scale_str += f" | b={sb.mean().item():.3f}(exp={math.exp(sb.mean().item()):.1f})"

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag}] step {global_step:>5d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.1f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | dead={dead:.3f}{scale_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | ETA {eta_sec/3600:.1f}h")

            if global_step in checkpoint_steps:
                frac = global_step / N_STEPS
                ckpt_path = save_dir / f"{name}_L{layer_idx}_step{global_step}.pt"
                torch.save(sae.state_dict(), ckpt_path)
                checkpoints_saved[global_step] = str(ckpt_path)
                print(f"    [{tag}] Checkpoint saved at {frac:.0%} ({global_step} steps)")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

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
        return {"n_features": 0, "aggregate": {"n_features": 0, "cos_wins_inner": 0}}

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
        return {"n_features": 0, "aggregate": {"n_features": 0, "cos_wins_inner": 0}}

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
        "step": step, "frac": step / N_STEPS,
        "fve": fve, "dead_frac": dead_frac, "alive_count": alive_count,
    }


# =============================================================================
# Main
# =============================================================================

# (name, class, kwargs, init_mode)
# init_mode: "none" (standard), "sqrt_d" (default cosine), "norm_adaptive"
VARIANTS = [
    ("standard",        StandardBatchTopKSAE,       {},                     "none"),
    ("adaptive_sqrtd",  AdaptiveCosineBatchTopKSAE, {},                     "sqrt_d"),
    ("adaptive_norm",   AdaptiveCosineBatchTopKSAE, {},                     "norm_adaptive"),
    ("group_G4_sqrtd",  GroupScaleSAE,              {"n_groups": N_GROUPS}, "sqrt_d"),
    ("group_G4_norm",   GroupScaleSAE,              {"n_groups": N_GROUPS}, "norm_adaptive"),
]


def main():
    print("Experiment 42: Mistral-7B Initialization Disambiguation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae={D_SAE}, k={K}, lr={LR}, {N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")
    print(f"Init comparison: sqrt(d)={math.sqrt(D_MODEL):.1f} vs norm-adaptive")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Total runs: {len(VARIANTS) * len(LAYERS)} (layer-first ordering)")
    print(f"Estimated time: ~{len(VARIANTS) * len(LAYERS) * 1.5:.0f} hours on H100")

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

    all_results["config"]["mean_norms"] = {str(k): v for k, v in mean_norms.items()}

    # Print the critical init comparison
    print(f"\n  INIT COMPARISON:")
    print(f"  {'Layer':>5s} | {'mean_norm':>10s} | {'sqrt(d)':>8s} | {'ratio':>6s} | {'direction':>10s}")
    print(f"  {'-'*5:>5s}-+-{'-'*10:>10s}-+-{'-'*8:>8s}-+-{'-'*6:>6s}-+-{'-'*10:>10s}")
    sqrtd = math.sqrt(D_MODEL)
    for layer_idx in LAYERS:
        mn = mean_norms[layer_idx]
        ratio = sqrtd / mn
        direction = "OVERSHOOT" if ratio > 1.5 else ("undershoot" if ratio < 0.67 else "matched")
        print(f"  L{layer_idx:>4d} | {mn:>10.1f} | {sqrtd:>8.1f} | {ratio:>5.1f}x | {direction:>10s}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ---- GPU check ----
    print("\n---GPU---")
    os.system("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv")

    # ---- Run all combinations (LAYER-FIRST for early L8 results) ----
    for layer_idx in LAYERS:
        print(f"\n{'#'*70}")
        print(f"  LAYER {layer_idx} (mean_norm={mean_norms[layer_idx]:.1f}, "
              f"mismatch={sqrtd/mean_norms[layer_idx]:.1f}x)")
        print(f"{'#'*70}")

        for vname, vcls, vkwargs, init_mode in VARIANTS:
            run_name = f"{vname}_L{layer_idx}"

            if run_name in all_results.get("runs", {}):
                print(f"\n  {run_name} already complete, skipping")
                continue

            print(f"\n{'='*70}")
            print(f"  RUN: {run_name} (encoder={vname}, init={init_mode}, layer={layer_idx})")
            print(f"{'='*70}")

            torch.manual_seed(SEED)
            np.random.seed(SEED)

            # Create SAE
            sae = vcls(D_MODEL, D_SAE, K, **vkwargs).to(DEVICE)
            n_params = sum(p.numel() for p in sae.parameters())

            # Apply init mode
            if init_mode == "norm_adaptive" and hasattr(sae, "scale_b"):
                mn = mean_norms[layer_idx]
                with torch.no_grad():
                    sae.scale_b.fill_(math.log(mn))
                print(f"    NORM-ADAPTIVE init: scale_b = log({mn:.1f}) = {math.log(mn):.4f} "
                      f"(exp={mn:.1f})")
            elif init_mode == "sqrt_d" and hasattr(sae, "scale_b"):
                print(f"    SQRT(D) init: scale_b = log({sqrtd:.1f}) = {math.log(sqrtd):.4f} "
                      f"(exp={sqrtd:.1f}, mismatch={sqrtd/mean_norms[layer_idx]:.1f}x)")
            else:
                print(f"    Standard encoder (no scale parameter)")

            print(f"    Parameters: {n_params:,}")

            # Create stream
            stream = ActivationStream(model, tokenizer, layer_idx, seed=SEED)

            # Train
            train_log, ckpt_paths = train_sae_streaming(
                vname, sae, stream, layer_idx, save_dir, CHECKPOINT_STEPS
            )

            # Evaluate checkpoints
            eval_data = eval_data_by_layer[layer_idx]
            checkpoint_evals = []
            for step in CHECKPOINT_STEPS[:-1]:
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
                          f"dead={ckpt_eval['dead_frac']:.3f}, "
                          f"alive={ckpt_eval['alive_count']}")

            # Load final for full evaluation
            final_path = ckpt_paths.get("final")
            if final_path and os.path.exists(final_path):
                sae.load_state_dict(torch.load(final_path, map_location=DEVICE,
                                               weights_only=True))

            # Full evaluation
            print(f"\n  Full evaluation -- {run_name}")
            recon = evaluate_reconstruction(vname, sae, eval_data, layer_idx)
            abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)

            run_result = {
                "encoder": vname,
                "layer": layer_idx,
                "init_mode": init_mode,
                "training": train_log,
                "checkpoints": checkpoint_evals,
                "reconstruction": recon,
                "ablation": abl,
            }

            if hasattr(sae, "scale_a"):
                sa = sae.scale_a
                if sa.dim() == 0:
                    run_result["scale_a_final"] = sa.item()
                else:
                    run_result["scale_a_final_mean"] = sa.mean().item()
                    run_result["scale_a_final_std"] = sa.std().item()
                    run_result["scale_a_final_values"] = sa.tolist()
            if hasattr(sae, "scale_b"):
                sb = sae.scale_b
                if sb.dim() == 0:
                    run_result["scale_b_final"] = sb.item()
                    run_result["scale_b_final_exp"] = math.exp(sb.item())
                else:
                    run_result["scale_b_final_mean"] = sb.mean().item()
                    run_result["scale_b_final_exp_mean"] = math.exp(sb.mean().item())
                    run_result["scale_b_final_values"] = sb.tolist()

            all_results["runs"][run_name] = run_result

            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Results saved to {RESULTS_PATH}")

            # ---- Per-layer summary after each layer completes ----
            layer_runs = {k: v for k, v in all_results["runs"].items()
                          if k.endswith(f"_L{layer_idx}")}
            if len(layer_runs) == len(VARIANTS):
                print(f"\n  === L{layer_idx} COMPLETE ===")
                print(f"  {'Variant':<20s} | {'Init':<12s} | {'FVE':>6s} | {'Dead%':>6s} | "
                      f"{'Alive':>6s} | {'cos>inn':>8s} | {'scale_a':>8s} | {'scale_b(exp)':>12s}")
                print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-"
                      f"{'-'*8}-+-{'-'*12}")
                for vn, _, _, im in VARIANTS:
                    rn = f"{vn}_L{layer_idx}"
                    run = all_results["runs"].get(rn, {})
                    r = run.get("reconstruction", {})
                    a = run.get("ablation", {}).get("aggregate", {})
                    fve = r.get("fve", 0)
                    dead = r.get("dead_frac", 0)
                    alive = r.get("alive_count", 0)
                    cos_w = str(a.get("cos_wins_inner", "?"))
                    n_f = str(a.get("n_features", "?"))
                    sa = run.get("scale_a_final", run.get("scale_a_final_mean", ""))
                    sa_str = f"{sa:.4f}" if isinstance(sa, float) else "N/A"
                    sb_exp = run.get("scale_b_final_exp", run.get("scale_b_final_exp_mean", ""))
                    sb_str = f"{sb_exp:.1f}" if isinstance(sb_exp, float) else "N/A"
                    print(f"  {vn:<20s} | {im:<12s} | {fve:.4f} | {dead*100:5.1f}% | "
                          f"{alive:>6d} | {cos_w:>3s}/{n_f:<4s} | {sa_str:>8s} | {sb_str:>12s}")

            del sae, stream
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Final Summary ----
    print(f"\n{'='*70}")
    print("  EXP39 FINAL SUMMARY")
    print(f"{'='*70}")

    for layer_idx in LAYERS:
        print(f"\n  --- Layer {layer_idx} (norm={mean_norms[layer_idx]:.1f}, "
              f"mismatch={sqrtd/mean_norms[layer_idx]:.1f}x) ---")
        print(f"  {'Variant':<20s} | {'Init':<12s} | {'FVE':>6s} | {'Dead%':>6s} | "
              f"{'Alive':>6s} | {'cos>inn':>8s} | {'a':>6s} | {'b(exp)':>8s}")
        for vn, _, _, im in VARIANTS:
            rn = f"{vn}_L{layer_idx}"
            run = all_results["runs"].get(rn, {})
            r = run.get("reconstruction", {})
            a = run.get("ablation", {}).get("aggregate", {})
            if not r:
                continue
            fve = r.get("fve", 0)
            dead = r.get("dead_frac", 0)
            alive = r.get("alive_count", 0)
            cos_w = str(a.get("cos_wins_inner", "?"))
            n_f = str(a.get("n_features", "?"))
            sa = run.get("scale_a_final", run.get("scale_a_final_mean", ""))
            sa_str = f"{sa:.3f}" if isinstance(sa, float) else "N/A"
            sb_exp = run.get("scale_b_final_exp", run.get("scale_b_final_exp_mean", ""))
            sb_str = f"{sb_exp:.1f}" if isinstance(sb_exp, float) else "N/A"
            print(f"  {vn:<20s} | {im:<12s} | {fve:.4f} | {dead*100:5.1f}% | "
                  f"{alive:>6d} | {cos_w:>3s}/{n_f:<4s} | {sa_str:>6s} | {sb_str:>8s}")

    # ---- Init comparison ----
    print(f"\n  INIT DISAMBIGUATION (sqrt(d) vs norm-adaptive):")
    for layer_idx in LAYERS:
        for arch in ["adaptive", "group_G4"]:
            sqrtd_run = all_results["runs"].get(f"{arch}_sqrtd_L{layer_idx}", {})
            norm_run = all_results["runs"].get(f"{arch}_norm_L{layer_idx}", {})
            sqrtd_r = sqrtd_run.get("reconstruction", {})
            norm_r = norm_run.get("reconstruction", {})
            if sqrtd_r and norm_r:
                fve_diff = norm_r["fve"] - sqrtd_r["fve"]
                dead_diff = norm_r["dead_frac"] - sqrtd_r["dead_frac"]
                alive_ratio = norm_r["alive_count"] / max(sqrtd_r["alive_count"], 1)
                winner = "norm" if fve_diff > 0.005 else ("sqrt(d)" if fve_diff < -0.005 else "tie")
                print(f"    {arch}/L{layer_idx}: FVE diff={fve_diff:+.4f}, "
                      f"dead diff={dead_diff*100:+.1f}pp, alive ratio={alive_ratio:.2f}x "
                      f"→ {winner}")

    # ---- Hypothesis verdict ----
    print(f"\n  HYPOTHESIS CHECK:")
    for layer_idx in LAYERS:
        sqrtd_fve = all_results["runs"].get(f"adaptive_sqrtd_L{layer_idx}", {}).get(
            "reconstruction", {}).get("fve", -1)
        norm_fve = all_results["runs"].get(f"adaptive_norm_L{layer_idx}", {}).get(
            "reconstruction", {}).get("fve", -1)
        std_fve = all_results["runs"].get(f"standard_L{layer_idx}", {}).get(
            "reconstruction", {}).get("fve", -1)
        sqrtd_dead = all_results["runs"].get(f"adaptive_sqrtd_L{layer_idx}", {}).get(
            "reconstruction", {}).get("dead_frac", -1)

        if sqrtd_dead > 0.95 and norm_fve > 0.1:
            verdict = "H1 CONFIRMED: sqrt(d) dead, norm-adaptive works"
        elif sqrtd_fve > 0.3 and norm_fve > 0.3:
            verdict = "H2 SUPPORTED: both inits work at 50M"
        elif norm_fve < std_fve and sqrtd_fve < std_fve:
            verdict = "H3 SUPPORTED: cosine underperforms standard regardless of init"
        else:
            verdict = f"MIXED: sqrtd_fve={sqrtd_fve:.3f}, norm_fve={norm_fve:.3f}, std_fve={std_fve:.3f}"
        print(f"    L{layer_idx}: {verdict}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults: {RESULTS_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
