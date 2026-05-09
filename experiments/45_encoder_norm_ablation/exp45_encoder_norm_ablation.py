"""
Experiment 45: Encoder Normalization Ablation
=============================================

The adaptive cosine SAE normalizes both input AND encoder weights before
computing cosine similarity. Input normalization is the RNH's core claim —
feature detection should be direction-based. But encoder normalization is a
design convenience: it makes scale_b cleanly interpretable as the output
scale, and prevents W_enc norms from interacting with scale_a.

The question: does removing encoder normalization change anything? If results
match, it simplifies the architecture story. If they differ, encoder
normalization is acting as regularization.

Variants:
  1. adaptive_l2:          encoder normalized (baseline, from exp43c)
  2. adaptive_l2_unnormed: encoder NOT normalized, no scale_b. Encoder weight
                           norms are free to learn, absorbing what exp(b) did.
                           scale_a still controls input norm sensitivity.

Layers: L9, L18, L27 on Qwen3-8B (50M tokens each, saprmarks recipe)

Pipeline (mirrors exp46): one bf16 activation cache per layer is built once
on disk, then both variants are trained in parallel from the same batch
stream. Eliminates a redundant model forward pass per layer and roughly halves
training wall-time.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp45_encoder_norm_ablation.py \
        >> experiments/exp45_output.log 2>&1 &
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
LAYERS = [9, 18, 27]
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

SAVE_DIR = Path("/mnt/nvme0/checkpoints/exp45")
CACHE_DIR = Path("~/MechInter--RNH/cache").expanduser()
RESULTS_PATH = Path("experiments/exp45_results.json")


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
    """Baseline: encoder rows normalized before cosine similarity."""
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


class AdaptiveCosineBatchTopKSAE_Unnormed(nn.Module):
    """Ablation: encoder rows NOT normalized. No scale_b — encoder weight
    norms absorb the role of exp(b). Input is still normalized (RNH core).
    scale_a still controls input norm sensitivity."""
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
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
        scale = torch.exp(self.scale_a * torch.log(input_norm))
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
    is_unnormed: bool
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
        is_unnormed=isinstance(sae, AdaptiveCosineBatchTopKSAE_Unnormed),
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

    if sae.W_dec.grad is not None:
        sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
            sae.W_dec.data, sae.W_dec.grad.data
        )

    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    s.optimizer.step()
    s.scheduler.step()

    set_decoder_norm_to_unit_norm(sae.W_dec.data)

    # NOTE: `global_step` here is the post-increment value (1, 2, ..., N_STEPS).
    # The original train_sae checked `pre_step >= THRESHOLD_START_STEP` which
    # is equivalent to `post_step > THRESHOLD_START_STEP`. Using `>` matches
    # the original first-fire iteration exactly.
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
            "scale_a": sae.scale_a.item(),
        }
        if hasattr(sae, "scale_b"):
            entry["scale_b"] = sae.scale_b.item()
            entry["scale_b_exp"] = math.exp(sae.scale_b.item())

        if s.is_unnormed:
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
                    extra = f" a={e['scale_a']:.4f}"
                    if "scale_b" in e:
                        extra += f" b={e['scale_b']:.3f}(exp={e['scale_b_exp']:.1f})"
                    if "enc_norm_mean" in e:
                        extra += (f" ||w||={e['enc_norm_mean']:.3f}"
                                  f"±{e['enc_norm_std']:.3f}")
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
    last_act, full_act) per sample. Shared across both variants in the
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
    ("adaptive_l2", AdaptiveCosineBatchTopKSAE),
    ("adaptive_l2_unnormed", AdaptiveCosineBatchTopKSAE_Unnormed),
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
        "scale_a_final": sae.scale_a.item(),
    }
    if hasattr(sae, "scale_b"):
        run_data["scale_b_final"] = sae.scale_b.item()
        run_data["scale_b_final_exp"] = math.exp(sae.scale_b.item())
    if isinstance(sae, AdaptiveCosineBatchTopKSAE_Unnormed):
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
    print("Experiment 45: Encoder Normalization Ablation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"d_sae={D_SAE}, k={K}, lr={LR}, {N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")
    print(f"Variants: {[v[0] for v in VARIANTS]}")
    print(f"Total runs: {len(VARIANTS) * len(LAYERS)} (in {len(LAYERS)} parallel groups of {len(VARIANTS)})")
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
            "experiment": 45,
            "model": MODEL_NAME, "layers": LAYERS,
            "d_sae": D_SAE, "k": K, "lr": LR,
            "n_train_tokens": N_TRAIN_TOKENS, "n_steps": N_STEPS,
            "note": "Encoder normalization ablation: does removing F.normalize "
                    "on W_enc (and scale_b) change cosine SAE behavior?",
        }, "runs": {}}

    # --- 1. Build all per-layer activation caches up front ---
    cache_paths_by_layer = {}
    for layer_idx in LAYERS:
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
        cache_paths_by_layer[layer_idx] = bin_path

    # --- 2. Eval data, collected once per layer (separate from train cache) ---
    eval_data_by_layer = {}
    mean_norms = {}
    for layer_idx in LAYERS:
        eval_data, mean_norm = collect_eval_data(model, tokenizer, layer_idx, N_EVAL_TOKENS)
        eval_data_by_layer[layer_idx] = eval_data
        mean_norms[layer_idx] = mean_norm
    results["config"]["mean_norms"] = {str(k): v for k, v in mean_norms.items()}

    # --- 3. Per-layer: parallel-train both variants from cached stream ---
    for layer_idx in LAYERS:
        print(f"\n{'#'*70}")
        print(f"  LAYER {layer_idx} (mean_norm={mean_norms[layer_idx]:.1f})")
        print(f"{'#'*70}")

        run_keys = [f"{vname}_L{layer_idx}" for vname, _ in VARIANTS]
        if all(rk in results.get("runs", {}) for rk in run_keys):
            print(f"  L{layer_idx} already complete, skipping")
            continue

        stream = CachedActivationStream(
            cache_paths_by_layer[layer_idx],
            batch_size=BATCH_SIZE, device=DEVICE,
            chunk_tokens=BUFFER_TOKENS, shuffle_seed=SEED,
        )
        # Reset stream so each layer-group starts at the same activation order.
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
                print(f" (scale_b init={sae.scale_b.item():.4f}, "
                      f"exp={math.exp(sae.scale_b.item()):.1f})")
            else:
                print(" (no scale_b — encoder norms absorb scale)")
            states.append(make_sae_state(vname, sae, layer_idx))

        if not states:
            continue

        train_parallel_group(states, stream, layer_idx, N_STEPS)

        # Per-variant FVE eval (cheap, runs from cached eval_data on GPU)
        for s in states:
            recon = evaluate_reconstruction(s.name, s.sae,
                                            eval_data_by_layer[layer_idx],
                                            layer_idx)
            results["runs"][f"{s.name}_L{layer_idx}"] = _build_run_record(s, recon)
            _persist(results)

        # Shared ablation eval — clean forwards collected once for the group.
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

        # Per-layer summary
        layer_runs = {k: v for k, v in results["runs"].items()
                      if k.endswith(f"_L{layer_idx}")}
        if len(layer_runs) == len(VARIANTS):
            print(f"\n  === L{layer_idx} SUMMARY ===")
            print(f"  {'Variant':<25s} | {'FVE':>6s} | {'Dead%':>6s} | "
                  f"{'Alive':>6s} | {'cos>inn':>8s} | {'scale_a':>8s} | {'scale/norm':>12s}")
            for vn, _ in VARIANTS:
                rk = f"{vn}_L{layer_idx}"
                run = results["runs"].get(rk, {})
                r = run.get("reconstruction", {})
                a = run.get("ablation", {})
                fve = r.get("fve", 0)
                dead = r.get("dead_frac", 0)
                alive = r.get("alive_count", 0)
                cos_w = str(a.get("cos_wins_inner", "?"))
                sa = run.get("scale_a_final", 0)
                if "scale_b_final_exp" in run:
                    sn = f"b_exp={run['scale_b_final_exp']:.1f}"
                elif "enc_norm_final" in run:
                    sn = f"||w||={run['enc_norm_final']['mean']:.3f}"
                else:
                    sn = "N/A"
                print(f"  {vn:<25s} | {fve:.4f} | {dead*100:5.1f}% | "
                      f"{alive:>6d} | {cos_w:>8s} | {sa:>8.4f} | {sn:>12s}")

    # Final summary
    print(f"\n{'='*70}")
    print("  EXP45 FINAL SUMMARY")
    print(f"{'='*70}")
    for layer_idx in LAYERS:
        print(f"\n  --- Layer {layer_idx} (norm={mean_norms[layer_idx]:.1f}) ---")
        for vn, _ in VARIANTS:
            rk = f"{vn}_L{layer_idx}"
            run = results["runs"].get(rk, {})
            r = run.get("reconstruction", {})
            a = run.get("ablation", {})
            if not r:
                continue
            fve = r.get("fve", 0)
            dead = r.get("dead_frac", 0)
            alive = r.get("alive_count", 0)
            cos_w = str(a.get("cos_wins_inner", "?"))
            sa = run.get("scale_a_final", 0)
            extra = ""
            if "enc_norm_final" in run:
                en = run["enc_norm_final"]
                extra = f" ||w||={en['mean']:.3f}±{en['std']:.3f} [{en['min']:.3f},{en['max']:.3f}]"
            if "scale_b_final_exp" in run:
                extra = f" b_exp={run['scale_b_final_exp']:.1f}"
            print(f"  {vn:<25s} | FVE={fve:.4f} dead={dead*100:.1f}% "
                  f"alive={alive:,} | cos>inn={cos_w} | a={sa:.4f}{extra}")

    # Direct comparison
    print(f"\n  NORMED vs UNNORMED:")
    for layer_idx in LAYERS:
        r_n = results["runs"].get(f"adaptive_l2_L{layer_idx}", {}).get("reconstruction", {})
        r_u = results["runs"].get(f"adaptive_l2_unnormed_L{layer_idx}", {}).get("reconstruction", {})
        if r_n and r_u:
            fve_diff = r_u["fve"] - r_n["fve"]
            dead_diff = r_u["dead_frac"] - r_n["dead_frac"]
            alive_ratio = r_u["alive_count"] / max(r_n["alive_count"], 1)
            winner = "unnormed" if fve_diff > 0.005 else ("normed" if fve_diff < -0.005 else "tie")
            print(f"    L{layer_idx}: FVE diff={fve_diff:+.4f}, "
                  f"dead diff={dead_diff*100:+.1f}pp, alive ratio={alive_ratio:.2f}x "
                  f"-> {winner}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
