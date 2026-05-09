"""
Experiment 42c: NoC_SAE at 500M tokens on Qwen3-8B (saprmarks recipe)
=====================================================================

Minimum-confound comparison of the Norm-Preserving Cosine BatchTopK SAE
(NoC_SAE, from exp39d) vs the standard BatchTopK baseline in exp40.

Architecture: center -> unit-normalize -> cosine encode -> BatchTopK ->
unit-norm decode -> restore input norm.  Magnitude is NEVER in the sparse
code; reconstruction norm is restored from the input's centered norm.

Everything else is identical to exp40:
  - Qwen3-8B, Layer 18, d_sae=65536, k=80
  - 500M tokens, batch_size=2048, 244,140 steps
  - Auxiliary k-loss (auxk_alpha=1/32)
  - LR 5e-5, constant + linear decay schedule
  - Decoder unit-norm constraint every step (gradient projection + renorm)
  - Encoder init = decoder.T
  - b_dec init = geometric median of first batch
  - Gradient clipping max_norm=1.0
  - Adam optimizer

Compare results directly to exp40's standard_L18 on the same machine.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 experiments/exp42c_noc_500m_qwen.py 2>&1 | tee experiments/exp42c_noc_qwen_output.log
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
# Configuration — matches exp40 exactly
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 18
D_MODEL = 4096

D_SAE = 65536
K = 80

N_TRAIN_TOKENS = 500_000_000
N_EVAL_TOKENS = 2_000_000
CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 4
OUTLIER_MULTIPLIER = 10.0

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

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE  # 244,140
DECAY_START = int(0.8 * N_STEPS)

CHECKPOINT_FRACS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CHECKPOINT_STEPS = sorted(set(int(f * N_STEPS) for f in CHECKPOINT_FRACS))

N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

SAVE_DIR = "/mnt/nvme0/checkpoints/exp42c"
RESULTS_PATH = "experiments/exp42c_noc_qwen_results.json"

BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE

NORM_EPS = 1e-8


def get_config_dict():
    return {
        "experiment": "exp42c_noc_500m_qwen",
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
        "recipe": "saprmarks/dictionary_learning BatchTopKTrainer (same as exp40)",
        "architecture": "NoC_SAE (Norm-Preserving Cosine BatchTopK)",
        "architecture_details": [
            "Cosine encoder: input centered, unit-normalized, W_enc unit-normalized",
            "No encoder bias (b_enc) — pure cosine similarity",
            "BatchTopK sparsity on cosine activations",
            "Unit-norm decoder (W_dec normalized in post_step + gradient projection)",
            "Norm restoration: output norm = input centered norm",
            "W_enc also kept unit-norm in post_step",
        ],
    }


# =============================================================================
# Geometric median (from saprmarks)
# =============================================================================

@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
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
# NoC_SAE — Norm-Preserving Cosine BatchTopK (from exp39d, adapted for
#            saprmarks training loop: return_active, inference threshold)
# =============================================================================

class NoCBatchTopKSAE(nn.Module):
    """Norm-Preserving Cosine BatchTopK SAE.

    Architecture identical to exp39d NoC_SAE:
      - Encoder: cosine similarity (input + W_enc both unit-normalized)
      - No encoder bias (pure cosine)
      - BatchTopK sparsity on cosine activations
      - Decoder: unit-norm W_dec (post_step constraint)
      - Norm restoration: output norm = input centered norm

    Adapted for saprmarks training recipe (return_active, threshold EMA).
    """

    def __init__(self, d_model: int, d_sae: int, k: int = 80):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # No b_enc — cosine encoder doesn't use encoder bias
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        # saprmarks init: encoder = decoder.T
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

        # Cache x_norm for decode's norm restoration
        self._cached_x_norm = x_norm

        if return_active:
            active_indices = encoded.sum(0) > 0
            return encoded, active_indices, post_relu
        return encoded

    def decode(self, f, x_norm=None):
        w_u = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w_u
        # Norm restoration: scale output to match input centered norm
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

    def post_step(self):
        """Normalize both W_enc and W_dec rows to unit norm."""
        with torch.no_grad():
            self.W_enc.div_(self.W_enc.norm(dim=1, keepdim=True).clamp(min=NORM_EPS))
            # W_dec normalization handled by set_decoder_norm_to_unit_norm in training loop


# =============================================================================
# Auxiliary k-loss (from saprmarks, identical to exp40)
# =============================================================================

def get_auxiliary_loss(residual, post_relu_acts, num_tokens_since_fired):
    dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return residual.new_zeros(()), n_dead
    k_aux = min(TOP_K_AUX, n_dead)
    auxk_latents = torch.where(
        dead_mask[None], post_relu_acts,
        torch.tensor(-torch.inf, device=post_relu_acts.device),
    )
    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
    auxk_buffer = torch.zeros_like(post_relu_acts)
    auxk_acts_BF = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
    return auxk_acts_BF, n_dead


# =============================================================================
# LR Schedule (saprmarks: constant + linear tail)
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
# Streaming Activation Collection (identical to exp40)
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
# Training — full saprmarks recipe (identical to exp40 training loop)
# =============================================================================

def train_sae(name, sae, stream, save_dir, checkpoint_steps):
    tag = f"{name}/L{LAYER}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")
    print(f"  Recipe: auxk_alpha={AUXK_ALPHA}, decay_start={DECAY_START}, "
          f"grad_clip=1.0, decoder_norm=unit")

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

            # Initialize b_dec to geometric median
            if not b_dec_initialized:
                with torch.no_grad():
                    median = geometric_median(batch)
                    sae.b_dec.data.copy_(median.to(sae.b_dec.dtype))
                b_dec_initialized = True
                print(f"    [{tag}] b_dec initialized to geometric median "
                      f"(norm={median.norm():.1f})")

            # Forward pass with active feature tracking
            x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)

            # Reconstruction loss
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            # Update dead feature counters
            did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
            did_fire[active_indices] = True
            num_tokens_since_fired += batch.shape[0]
            num_tokens_since_fired[did_fire] = 0

            # Auxiliary k-loss for dead features
            residual = (batch - x_hat).detach()
            auxk_acts, n_dead = get_auxiliary_loss(
                residual, post_relu_acts, num_tokens_since_fired
            )

            if n_dead > 0:
                # Reconstruct residual using only dead features (no b_dec)
                x_reconstruct_aux = auxk_acts @ sae.W_dec
                auxk_l2 = (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
                residual_mu = residual.mean(dim=0, keepdim=True)
                loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
                auxk_loss = (auxk_l2 / loss_denom.clamp(min=1e-8)).nan_to_num(0.0)
            else:
                auxk_loss = torch.tensor(0.0, device=DEVICE)

            loss = recon_loss + AUXK_ALPHA * auxk_loss

            # Backward + optimizer step
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
            # Encoder unit-norm constraint (NoC-specific)
            sae.post_step()

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

            # Logging
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
                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag}] step {global_step:>6d}/{N_STEPS} | "
                      f"loss={loss.item():.1f} | recon={recon_loss.item():.1f} | "
                      f"auxk={auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else 0:.3f} | "
                      f"L0={l0:.0f} | FVE={fve:.4f} | "
                      f"dead={dead_frac:.3f} ({n_dead:,}) | "
                      f"{tokens_seen/1e6:.1f}M tok | ETA {eta_sec/3600:.1f}h")

            # Mid-training checkpoints
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
        "fve": fve,
        "mean_recon_loss": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "mean_l0": float(np.mean(l0s)),
        "dead_frac": dead_frac,
        "alive_count": alive_count,
    }
    print(f"    [{tag}] FVE={fve:.4f} | dead={dead_frac:.3f} | "
          f"alive={alive_count:,} | L0={np.mean(l0s):.1f} | "
          f"cos_recon={np.mean(cos_sims):.4f}")
    return results


@torch.no_grad()
def evaluate_ablation(name, model, tokenizer, sae, eval_texts):
    tag = f"{name}/L{LAYER}"
    print(f"    [{tag}] Ablation evaluation ({N_ABLATION_FEATURES} features, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    act_sums = torch.zeros(D_SAE, device=DEVICE)
    act_counts = torch.zeros(D_SAE, device=DEVICE)

    for text in eval_texts[:200]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=CTX_LEN).to(DEVICE)
        acts = _collect_layer_acts(model, LAYER, inputs)
        flat = acts[0]
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

            clean_out = model(**inputs)
            clean_logits = clean_out.logits[0, -1]
            clean_probs = F.softmax(clean_logits.float(), dim=-1)

            layer_acts = _collect_layer_acts(model, LAYER, inputs)
            x = layer_acts[0, -1].float()

            def ablation_hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
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
        "cos_kl_mean": float(np.mean([r["cos_kl_corr"] for r in all_results])),
        "inner_kl_mean": float(np.mean([r["inner_kl_corr"] for r in all_results])),
        "sae_kl_mean": float(np.mean([r["sae_kl_corr"] for r in all_results])),
        "norm_kl_mean": float(np.mean([r["norm_kl_corr"] for r in all_results])),
        "cos_wins_inner": f"{cos_wins}/{len(all_results)}",
    }
    print(f"    [{tag}] Ablation: cos>inner {cos_wins}/{len(all_results)} | "
          f"cos->KL={agg['cos_kl_mean']:.3f} | inner->KL={agg['inner_kl_mean']:.3f} | "
          f"SAE->KL={agg['sae_kl_mean']:.3f}")

    return {"aggregate": agg, "per_feature": all_results}


# =============================================================================
# SAEBench integration
# =============================================================================

def run_saebench_eval(name, sae):
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from benchmarks.adapter import BenchSAE
        from benchmarks.run_saebench import run_saebench
    except ImportError as e:
        print(f"    SAEBench import failed: {e}")
        return None

    print(f"\n  SAEBench evaluation for {name}/L{LAYER}...")

    _sae = sae.eval()

    def encode_fn(x):
        return _sae.encode(x)

    def decode_fn(f):
        return _sae.decode(f)

    W_dec = sae.W_dec.detach()
    W_dec_normed = F.normalize(W_dec, dim=1)
    W_enc = sae.W_enc.detach()

    bench_sae = BenchSAE(
        W_enc=W_enc.T,
        W_dec=W_dec_normed,
        b_enc=torch.zeros(D_SAE, device=DEVICE, dtype=W_enc.dtype),  # NoC has no b_enc
        b_dec=sae.b_dec.detach(),
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name=MODEL_NAME,
        hook_layer=LAYER,
        device=DEVICE,
        dtype=DTYPE,
    )

    sae_name = f"exp42c-no_C-L{LAYER}"
    results = run_saebench(
        bench_sae,
        sae_name=sae_name,
        eval_types=["core", "sparse_probing", "absorption"],
        output_dir="/mnt/nvme0/saebench_results/exp42c",
        llm_batch_size=4,
        device=DEVICE,
    )
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 42c: NoC_SAE at 500M tokens (Qwen3-8B, saprmarks recipe)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"d_sae={D_SAE}, k={K}, lr={LR}, {N_TRAIN_TOKENS:,} tokens ({N_STEPS:,} steps)")
    print(f"Batch size: {BATCH_SIZE}, Decay start: {DECAY_START}")
    print(f"Aux loss: auxk_alpha={AUXK_ALPHA}, dead_threshold={DEAD_FEATURE_THRESHOLD:,}")
    print(f"Architecture: NoC_SAE (Norm-Preserving Cosine BatchTopK)")
    print(f"Checkpoints: {SAVE_DIR}")
    print(f"\nMinimum-confound comparison to exp40 standard_L18")
    print(f"Only difference: SAE architecture (NoC vs inner-product encoder)")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation="eager",  # Avoid cuDNN SDPA errors on H100
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results")
    else:
        all_results = {"config": get_config_dict()}

    # Collect eval data
    eval_data, mean_norm = collect_eval_data(model, tokenizer, LAYER, N_EVAL_TOKENS)
    all_results["config"]["mean_norm"] = mean_norm

    # Collect eval texts for ablation
    print("  Collecting eval texts for ablation...")
    ds_eval = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    eval_texts = []
    for i, row in enumerate(ds_eval):
        if i < 600_000:
            continue
        if len(row["text"]) > 200:
            eval_texts.append(row["text"][:2048])
        if len(eval_texts) >= 500:
            break
    print(f"    Collected {len(eval_texts)} eval texts")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Train NoC_SAE
    name = "no_C"
    run_name = f"{name}_L{LAYER}"

    if run_name in all_results and "reconstruction" in all_results.get(run_name, {}):
        print(f"\n  {run_name} already complete, skipping training")
    else:
        print(f"\n{'='*70}")
        print(f"  RUN: {run_name}")
        print(f"{'='*70}")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        sae = NoCBatchTopKSAE(D_MODEL, D_SAE, K).to(DEVICE)
        print(f"    SAE params: {sum(p.numel() for p in sae.parameters()):,}")

        stream = ActivationStream(model, tokenizer, LAYER, seed=SEED)

        train_log, ckpt_paths = train_sae(name, sae, stream, save_dir, CHECKPOINT_STEPS)

        # Load final checkpoint
        final_path = ckpt_paths.get("final")
        if final_path and os.path.exists(final_path):
            ckpt = torch.load(final_path, map_location=DEVICE, weights_only=False)
            sae.load_state_dict(ckpt["state_dict"])

        # Evaluate reconstruction
        print(f"\n  Evaluation - {run_name}")
        recon = evaluate_reconstruction(name, sae, eval_data)

        # Ablation evaluation
        abl = evaluate_ablation(name, model, tokenizer, sae, eval_texts)

        # Save results before SAEBench
        run_result = {
            "encoder": name,
            "layer": LAYER,
            "training": train_log,
            "reconstruction": recon,
            "ablation": abl,
            "saebench": None,
        }
        all_results[run_name] = run_result
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Core results saved to {RESULTS_PATH}")

        # SAEBench evaluation
        try:
            saebench_results = run_saebench_eval(name, sae)
            run_result["saebench"] = saebench_results
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  SAEBench results saved")
        except Exception as e:
            print(f"  SAEBench failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("  EXP42C SUMMARY")
    print(f"{'='*70}")
    run = all_results.get(run_name, {})
    r = run.get("reconstruction", {})
    if r:
        print(f"  NoC_SAE L{LAYER}: FVE={r.get('fve',0):.4f} | "
              f"dead={r.get('dead_frac',0)*100:.1f}% | "
              f"alive={r.get('alive_count',0):,} | "
              f"L0={r.get('mean_l0',0):.1f} | "
              f"cos_recon={r.get('cos_recon',0):.4f}")
    a = run.get("ablation", {}).get("aggregate", {})
    if a:
        print(f"  Ablation: cos>inner {a.get('cos_wins_inner','-')} | "
              f"cos->KL={a.get('cos_kl_mean',0):.3f} | "
              f"inner->KL={a.get('inner_kl_mean',0):.3f}")
    print(f"\n  Results: {RESULTS_PATH}")
    print(f"  Checkpoints: {SAVE_DIR}/")
    print(f"\n  Compare to exp40 standard_L18 for minimum-confound architecture comparison.")


if __name__ == "__main__":
    main()
