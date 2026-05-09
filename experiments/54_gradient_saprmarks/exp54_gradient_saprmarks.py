"""
Experiment 54: Gradient Analysis with saprmarks Recipe
======================================================

Does gradient equalization (exp28) persist under the saprmarks training recipe
(aux-k loss, LR=5e-5, batch=2048)?

Exp28 showed: standard encoder gradients are Q4-dominated (Q4/Q1=1.55x, 35% of
features >2x), cosine is balanced (1.03x, 13.5%). But that was without aux-k.
Aux-k adds gradient signal to dead features — does it compensate for standard's
Q4 bias, or does the main-loss gradient carry the bias regardless?

Critical addition: separate main-loss vs aux-k gradients. For each logging step:
  1. Forward pass → main recon loss → backward → save W_enc.grad as main_grad
  2. Zero grad → compute aux-k loss → backward → save W_enc.grad as aux_grad
  3. Combine: W_enc.grad = main_grad + AUXK_ALPHA * aux_grad → optimizer step
  4. Log quartile stats for main_grad and aux_grad separately

6 runs: {standard, adaptive_l2} × {L9, L18, L27}, 10M tokens each.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp54_gradient_saprmarks.py \
        > experiments/exp54_output.log 2>&1 &
"""

import json
import math
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL = 4096
D_SAE = 65536
K = 80

N_TRAIN_TOKENS = 10_000_000
LR = 5e-5
BATCH_SIZE = 2048
WARMUP_STEPS = 1000
AUXK_ALPHA = 1 / 32
DEAD_FEATURE_THRESHOLD = 1_000_000  # 1M tokens — reached at step 488, so aux-k active for ~80% of training
TOP_K_AUX = D_MODEL // 2  # 2048
THRESHOLD_BETA = 0.999
THRESHOLD_START_STEP = 1000

LAYERS = [9, 18, 27]
VARIANTS = ["standard", "adaptive_l2"]

LOG_GRAD_EVERY = 50
N_QUARTILES = 4
SEED = 42
OUTLIER_MULTIPLIER = 10.0
CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 4
BUFFER_TOKENS = 500_000

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
DECAY_START = int(0.8 * N_STEPS)
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE

SAVE_DIR = "/data/checkpoints/exp54b"
RESULTS_PATH = "experiments/exp54b_results.json"


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
# SAE Architectures (from exp51)
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


SAE_CLASSES = {
    "standard": BatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
}


# =============================================================================
# Aux-k loss (from exp40/saprmarks)
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
# Activation Streaming
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
        self.buffer_norms = self.buffer.float().norm(dim=-1)
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
        return (
            self.buffer[idx].to(DEVICE, dtype=torch.float32),
            self.buffer_norms[idx].to(DEVICE),
        )


# =============================================================================
# Gradient quartile analysis
# =============================================================================

def compute_quartile_grads_for_grad(grad_tensor, batch_norms):
    """Given a W_enc gradient tensor and per-token norm assignments,
    this is called AFTER backward — we already have the full gradient.

    But we need per-quartile gradients: run separate forward+backward
    on each quartile subset. This function is a helper that computes
    quartile boundaries and returns labels.
    """
    quartile_boundaries = torch.quantile(
        batch_norms, torch.tensor([0.25, 0.5, 0.75], device=batch_norms.device)
    )
    q_labels = torch.zeros_like(batch_norms, dtype=torch.long)
    q_labels[batch_norms >= quartile_boundaries[0]] = 1
    q_labels[batch_norms >= quartile_boundaries[1]] = 2
    q_labels[batch_norms >= quartile_boundaries[2]] = 3
    return q_labels, quartile_boundaries


def compute_quartile_grads_recon(sae, batch, batch_norms):
    """Per-quartile gradient norms for reconstruction loss only."""
    q_labels, q_bounds = compute_quartile_grads_for_grad(None, batch_norms)
    quartile_grad_norms = {}

    for q in range(N_QUARTILES):
        mask = q_labels == q
        if mask.sum() < 10:
            continue
        subset = batch[mask]
        sae.zero_grad(set_to_none=True)
        x_hat, features = sae(subset)
        loss = (subset - x_hat).pow(2).sum(dim=-1).mean()
        loss.backward()
        grad = sae.W_enc.grad
        if grad is not None:
            quartile_grad_norms[q] = grad.norm(dim=1).detach().cpu()

    return quartile_grad_norms, q_bounds


def compute_quartile_grads_auxk(sae, batch, batch_norms, num_tokens_since_fired):
    """Per-quartile gradient norms for aux-k loss only."""
    q_labels, _ = compute_quartile_grads_for_grad(None, batch_norms)
    quartile_grad_norms = {}

    for q in range(N_QUARTILES):
        mask = q_labels == q
        if mask.sum() < 10:
            continue
        subset = batch[mask]
        sae.zero_grad(set_to_none=True)

        x_hat, features, active_indices, post_relu_acts = sae(subset, return_active=True)
        residual = (subset - x_hat).detach()
        auxk_acts, n_dead = get_auxiliary_loss(
            residual, post_relu_acts, num_tokens_since_fired,
        )

        if n_dead > 0:
            x_reconstruct_aux = auxk_acts @ sae.W_dec
            auxk_l2 = (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            residual_mu = residual.mean(dim=0, keepdim=True)
            loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            auxk_loss = (auxk_l2 / loss_denom.clamp(min=1e-8)).nan_to_num(0.0)
            auxk_loss.backward()

        grad = sae.W_enc.grad
        if grad is not None:
            quartile_grad_norms[q] = grad.norm(dim=1).detach().cpu()
        else:
            quartile_grad_norms[q] = torch.zeros(sae.d_sae)

    return quartile_grad_norms


def summarize_quartile_grads(quartile_grad_norms, prefix=""):
    """Compute summary statistics from per-quartile gradient norms."""
    stats = {}
    q_means = {}
    for q in range(N_QUARTILES):
        if q in quartile_grad_norms:
            gn = quartile_grad_norms[q]
            q_means[q] = gn.mean().item()
            stats[f"{prefix}Q{q+1}_mean_grad"] = gn.mean().item()
            stats[f"{prefix}Q{q+1}_median_grad"] = gn.median().item()

    if 0 in q_means and 3 in q_means and q_means[0] > 1e-12:
        stats[f"{prefix}Q4_Q1_ratio_mean"] = q_means[3] / q_means[0]
    if 0 in q_means and 3 in q_means:
        stats[f"{prefix}Q4_Q1_ratio_raw"] = q_means.get(3, 0)
        stats[f"{prefix}Q1_raw"] = q_means.get(0, 0)

    if 0 in quartile_grad_norms and 3 in quartile_grad_norms:
        q1_gn = quartile_grad_norms[0]
        q4_gn = quartile_grad_norms[3]
        active = (q1_gn > 1e-10) | (q4_gn > 1e-10)
        n_active = active.sum().item()
        stats[f"{prefix}n_active_features"] = int(n_active)
        if n_active > 100:
            q1_a = q1_gn[active]
            q4_a = q4_gn[active]
            safe_q1 = q1_a.clamp(min=1e-12)
            per_feat_ratio = q4_a / safe_q1
            stats[f"{prefix}Q4_Q1_ratio_median_pf"] = per_feat_ratio.median().item()
            stats[f"{prefix}frac_gt2x_snapshot"] = (per_feat_ratio > 2.0).float().mean().item()
            stats[f"{prefix}n_Q1_spec_snapshot"] = int((q1_a > q4_a).sum().item())
        else:
            stats[f"{prefix}Q4_Q1_ratio_median_pf"] = 0
            stats[f"{prefix}frac_gt2x_snapshot"] = 0
            stats[f"{prefix}n_Q1_spec_snapshot"] = 0

    return stats


# =============================================================================
# Training with separated gradient logging
# =============================================================================

def train_with_gradient_analysis(name, sae, stream, layer_idx):
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    schedule_fn = make_lr_schedule(N_STEPS, WARMUP_STEPS, DECAY_START)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    num_tokens_since_fired = torch.zeros(D_SAE, dtype=torch.long, device=DEVICE)

    sae.train()
    b_dec_initialized = False
    t0 = time.time()
    global_step = 0

    grad_log = []
    agg_main_q = {q: [] for q in range(N_QUARTILES)}
    agg_auxk_q = {q: [] for q in range(N_QUARTILES)}

    while global_step < N_STEPS:
        stream.fill_buffer()
        steps_in_buffer = min(BUFFER_BATCHES, N_STEPS - global_step)

        for buf_step in range(steps_in_buffer):
            batch, batch_norms = stream.get_batch(buf_step)

            if not b_dec_initialized:
                with torch.no_grad():
                    median = geometric_median(batch)
                    sae.b_dec.data.copy_(median.to(sae.b_dec.dtype))
                b_dec_initialized = True
                print(f"    [{tag}] b_dec initialized (norm={median.norm():.1f})")

            # === Forward pass ===
            x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)

            # === Main reconstruction loss ===
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            # === Dead feature tracking ===
            did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
            did_fire[active_indices] = True
            num_tokens_since_fired += batch.shape[0]
            num_tokens_since_fired[did_fire] = 0

            # === Aux-k loss ===
            residual = (batch - x_hat).detach()
            auxk_acts, n_dead = get_auxiliary_loss(
                residual, post_relu_acts, num_tokens_since_fired,
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

            # === Backward + step ===
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

            # === Gradient logging ===
            if global_step % LOG_GRAD_EVERY == 0 or global_step == 1 or global_step == N_STEPS:
                sae.train()

                main_q_grads, q_bounds = compute_quartile_grads_recon(sae, batch, batch_norms)
                auxk_q_grads = compute_quartile_grads_auxk(
                    sae, batch, batch_norms, num_tokens_since_fired,
                )

                main_stats = summarize_quartile_grads(main_q_grads, prefix="main_")
                auxk_stats = summarize_quartile_grads(auxk_q_grads, prefix="auxk_")

                for q in range(N_QUARTILES):
                    if q in main_q_grads:
                        agg_main_q[q].append(main_q_grads[q])
                    if q in auxk_q_grads:
                        agg_auxk_q[q].append(auxk_q_grads[q])

                snapshot = {"step": global_step}
                snapshot.update(main_stats)
                snapshot.update(auxk_stats)

                snapshot["norm_Q1_upper"] = q_bounds[0].item()
                snapshot["norm_Q2_upper"] = q_bounds[1].item()
                snapshot["norm_Q3_upper"] = q_bounds[2].item()
                norm_mean = batch_norms.mean().item()
                norm_std = batch_norms.std().item()
                snapshot["norm_cv"] = norm_std / norm_mean if norm_mean > 0 else 0

                with torch.no_grad():
                    l0 = (features != 0).float().sum(dim=-1).mean().item()
                    total_var = torch.var(batch, dim=0, unbiased=False).sum()
                    resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                    fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                    dead_frac = (num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD).float().mean().item()

                snapshot["recon_loss"] = recon_loss.item()
                snapshot["auxk_loss"] = auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else 0
                snapshot["l0"] = l0
                snapshot["fve"] = fve
                snapshot["dead_frac"] = dead_frac
                snapshot["n_dead"] = n_dead

                grad_log.append(snapshot)

                main_ratio = main_stats.get("main_Q4_Q1_ratio_median_pf", 0)
                main_gt2x = main_stats.get("main_frac_gt2x_snapshot", 0) * 100
                elapsed = time.time() - t0
                tok_seen = global_step * BATCH_SIZE
                tok_per_sec = tok_seen / elapsed if elapsed > 0 else 0
                eta = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0

                print(f"    [{tag}] step {global_step:>5d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.1f} | FVE={fve:.4f} | "
                      f"main Q4/Q1={main_ratio:.2f}x (>{main_gt2x:.0f}% >2x) | "
                      f"norm_cv={snapshot['norm_cv']:.3f} | "
                      f"ETA {eta/60:.0f}m")

            elif global_step % 200 == 0:
                with torch.no_grad():
                    l0 = (features != 0).float().sum(dim=-1).mean().item()
                    total_var = torch.var(batch, dim=0, unbiased=False).sum()
                    resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                    fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                    dead_frac = (num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD).float().mean().item()
                elapsed = time.time() - t0
                tok_seen = global_step * BATCH_SIZE
                tok_per_sec = tok_seen / elapsed if elapsed > 0 else 0
                eta = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag}] step {global_step:>5d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.1f} | FVE={fve:.4f} | "
                      f"dead={dead_frac:.3f} | ETA {eta/60:.0f}m")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed/60:.1f}m")

    # === Aggregate analysis ===
    agg_results = {}
    for prefix, agg_q in [("main_", agg_main_q), ("auxk_", agg_auxk_q)]:
        for q in range(N_QUARTILES):
            if agg_q[q]:
                stacked = torch.stack(agg_q[q])
                mean_over_time = stacked.mean(dim=0)
                agg_results[f"{prefix}Q{q+1}_per_feature_mean"] = mean_over_time.mean().item()
                agg_results[f"{prefix}Q{q+1}_per_feature_median"] = mean_over_time.median().item()

        if agg_q[0] and agg_q[3]:
            q1_mean = torch.stack(agg_q[0]).mean(dim=0)
            q4_mean = torch.stack(agg_q[3]).mean(dim=0)

            active = (q1_mean > 1e-10) | (q4_mean > 1e-10)
            n_active = active.sum().item()
            agg_results[f"{prefix}n_active_features"] = int(n_active)

            q1_a = q1_mean[active]
            q4_a = q4_mean[active]
            safe_q1 = q1_a.clamp(min=1e-12)
            ratio = q4_a / safe_q1

            agg_results[f"{prefix}per_feature_Q4_Q1_ratio_mean"] = ratio.mean().item()
            agg_results[f"{prefix}per_feature_Q4_Q1_ratio_median"] = ratio.median().item()
            agg_results[f"{prefix}frac_features_Q4_dominates_2x"] = (ratio > 2.0).float().mean().item()
            agg_results[f"{prefix}frac_features_Q4_dominates_5x"] = (ratio > 5.0).float().mean().item()

            q1_specialized = (q1_a > q4_a).sum().item()
            agg_results[f"{prefix}n_Q1_specialized"] = int(q1_specialized)

    print(f"\n  Aggregate gradient analysis for {tag}:")
    for prefix, label in [("main_", "Main loss"), ("auxk_", "Aux-k loss")]:
        ratio_mean = agg_results.get(f"{prefix}per_feature_Q4_Q1_ratio_mean", 0)
        ratio_med = agg_results.get(f"{prefix}per_feature_Q4_Q1_ratio_median", 0)
        gt2x = agg_results.get(f"{prefix}frac_features_Q4_dominates_2x", 0) * 100
        n_q1 = agg_results.get(f"{prefix}n_Q1_specialized", 0)
        print(f"    {label}: Q4/Q1 mean={ratio_mean:.3f}x median={ratio_med:.3f}x "
              f">2x={gt2x:.1f}% Q1-specialized={n_q1}")

    return {
        "gradient_snapshots": grad_log,
        "aggregate": agg_results,
    }


# =============================================================================
# Dead feature quartile analysis
# =============================================================================

@torch.no_grad()
def analyze_dead_features(name, sae, stream, layer_idx):
    tag = f"{name}/L{layer_idx}"
    print(f"\n  Dead feature analysis for {tag}...")
    sae.eval()

    stream.fill_buffer()
    data = stream.buffer
    norms = stream.buffer_norms

    quartile_boundaries = torch.quantile(norms, torch.tensor([0.25, 0.5, 0.75]))
    q_labels = torch.zeros_like(norms, dtype=torch.long)
    q_labels[norms >= quartile_boundaries[0]] = 1
    q_labels[norms >= quartile_boundaries[1]] = 2
    q_labels[norms >= quartile_boundaries[2]] = 3

    q_alive = {}
    for q in range(N_QUARTILES):
        mask = q_labels == q
        if mask.sum() < 100:
            continue
        subset = data[mask]
        all_feats = []
        for i in range(0, subset.shape[0], BATCH_SIZE):
            batch = subset[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
            _, f = sae(batch)
            all_feats.append((f > 0).any(dim=0).cpu())
        alive_mask = torch.stack(all_feats).any(dim=0)
        q_alive[q] = alive_mask
        n_alive = alive_mask.sum().item()
        print(f"    Q{q+1}: {n_alive} features alive ({n_alive/D_SAE*100:.1f}%)")

    result = {f"Q{q+1}_alive": q_alive[q].sum().item() for q in q_alive}

    if 0 in q_alive and 3 in q_alive:
        q4_only = q_alive[3] & ~q_alive[0]
        q1_only = q_alive[0] & ~q_alive[3]
        both = q_alive[0] & q_alive[3]
        result["Q4_only"] = q4_only.sum().item()
        result["Q1_only"] = q1_only.sum().item()
        result["both_Q1_Q4"] = both.sum().item()
        print(f"    Both Q1+Q4: {both.sum().item()} | Q4-only: {q4_only.sum().item()} | "
              f"Q1-only: {q1_only.sum().item()}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 54: Gradient Analysis with saprmarks Recipe")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"d_model={D_MODEL}, d_sae={D_SAE}, k={K}, lr={LR}")
    print(f"Tokens per run: {N_TRAIN_TOKENS:,}, Steps: {N_STEPS}")
    print(f"Layers: {LAYERS}, Variants: {VARIANTS}")
    print(f"Gradient logging every {LOG_GRAD_EVERY} steps")
    print(f"Recipe: saprmarks (aux-k α={AUXK_ALPHA}, batch={BATCH_SIZE}, "
          f"warmup={WARMUP_STEPS}, decay@{DECAY_START})")
    print()

    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "config": {
            "model_name": MODEL_NAME,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "k": K,
            "n_train_tokens": N_TRAIN_TOKENS,
            "n_steps": N_STEPS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "auxk_alpha": AUXK_ALPHA,
            "log_grad_every": LOG_GRAD_EVERY,
            "layers": LAYERS,
            "variants": VARIANTS,
            "recipe": "saprmarks (aux-k, constant+linear LR, decoder norm, enc=dec.T init)",
        },
        "runs": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*80}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*80}")

        stream = ActivationStream(model, tokenizer, layer_idx, seed=SEED)

        for variant_name in VARIANTS:
            print(f"\n{'-'*70}")
            print(f"  {variant_name} / L{layer_idx}")
            print(f"{'-'*70}")

            torch.manual_seed(SEED)
            cls = SAE_CLASSES[variant_name]
            sae = cls(D_MODEL, D_SAE, K).to(DEVICE)

            stream._init_dataset()

            grad_results = train_with_gradient_analysis(
                variant_name, sae, stream, layer_idx,
            )

            dead_analysis = analyze_dead_features(
                variant_name, sae, stream, layer_idx,
            )

            run_key = f"{variant_name}_L{layer_idx}"
            run_result = {
                "variant": variant_name,
                "layer": layer_idx,
                "gradient_analysis": grad_results["aggregate"],
                "gradient_snapshots": grad_results["gradient_snapshots"],
                "dead_feature_analysis": dead_analysis,
            }
            if hasattr(sae, "scale_a"):
                run_result["scale_a"] = sae.scale_a.item()
                run_result["scale_b_exp"] = sae.scale_b.exp().item()

            all_results["runs"][run_key] = run_result

            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Results saved to {RESULTS_PATH}")

            ckpt_path = save_dir / f"{variant_name}_L{layer_idx}_final.pt"
            torch.save(sae.state_dict(), ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

            del sae
            gc.collect()
            torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================

    print(f"\n{'='*80}")
    print(f"  SUMMARY — Gradient Domination: saprmarks Recipe")
    print(f"{'='*80}")

    header = (f"  {'Variant':<15s} {'Layer':>5s} | "
              f"{'Main Q4/Q1':>10s} {'Main >2x':>8s} {'Main Q1-spec':>12s} | "
              f"{'Auxk Q4/Q1':>10s} {'Auxk >2x':>8s} {'Auxk Q1-spec':>12s}")
    print(header)
    print(f"  {'-'*15} {'-'*5}-+-{'-'*10}-{'-'*8}-{'-'*12}-+-{'-'*10}-{'-'*8}-{'-'*12}")

    for layer_idx in LAYERS:
        for variant_name in VARIANTS:
            key = f"{variant_name}_L{layer_idx}"
            if key not in all_results["runs"]:
                continue
            agg = all_results["runs"][key]["gradient_analysis"]

            m_ratio = agg.get("main_per_feature_Q4_Q1_ratio_median", 0)
            m_gt2x = agg.get("main_frac_features_Q4_dominates_2x", 0) * 100
            m_q1 = agg.get("main_n_Q1_specialized", 0)

            a_ratio = agg.get("auxk_per_feature_Q4_Q1_ratio_median", 0)
            a_gt2x = agg.get("auxk_frac_features_Q4_dominates_2x", 0) * 100
            a_q1 = agg.get("auxk_n_Q1_specialized", 0)

            print(f"  {variant_name:<15s} L{layer_idx:>3d} | "
                  f"{m_ratio:>9.2f}x {m_gt2x:>7.1f}% {m_q1:>12,d} | "
                  f"{a_ratio:>9.2f}x {a_gt2x:>7.1f}% {a_q1:>12,d}")

    print(f"\n  Comparison with exp28 (no aux-k, LR=3e-4, L27 only, d_sae=16384):")
    print(f"    standard:        Q4/Q1=1.55x, >2x=35.0%, Q1-spec=18")
    print(f"    cosine_adaptive: Q4/Q1=1.03x, >2x=13.5%, Q1-spec=562")

    print(f"\n  Key question: Does main-loss Q4/Q1 for standard remain e<author>ated?")
    for layer_idx in LAYERS:
        std_key = f"standard_L{layer_idx}"
        cos_key = f"adaptive_l2_L{layer_idx}"
        if std_key in all_results["runs"] and cos_key in all_results["runs"]:
            std_r = all_results["runs"][std_key]["gradient_analysis"].get(
                "main_per_feature_Q4_Q1_ratio_median", 0)
            cos_r = all_results["runs"][cos_key]["gradient_analysis"].get(
                "main_per_feature_Q4_Q1_ratio_median", 0)
            gap = std_r - cos_r
            print(f"    L{layer_idx}: standard={std_r:.2f}x, cosine={cos_r:.2f}x, "
                  f"gap={gap:+.2f}x {'✓ CONFIRMED' if gap > 0.1 else '✗ DISAPPEARED'}")

    print(f"\n  Depth trend (per-snapshot median Q4/Q1, last 10 logging steps):")
    for layer_idx in LAYERS:
        for variant_name in VARIANTS:
            key = f"{variant_name}_L{layer_idx}"
            if key not in all_results["runs"]:
                continue
            snaps = all_results["runs"][key]["gradient_snapshots"]
            last10 = snaps[-10:]
            ratios = [s.get("main_Q4_Q1_ratio_median_pf", 0) for s in last10]
            gt2x = [s.get("main_frac_gt2x_snapshot", 0) * 100 for s in last10]
            cvs = [s.get("norm_cv", 0) for s in last10]
            avg_r = sum(ratios) / len(ratios) if ratios else 0
            avg_gt2x = sum(gt2x) / len(gt2x) if gt2x else 0
            avg_cv = sum(cvs) / len(cvs) if cvs else 0
            print(f"    {variant_name:<15s} L{layer_idx}: "
                  f"Q4/Q1={avg_r:.2f}x  >2x={avg_gt2x:.1f}%  norm_cv={avg_cv:.3f}")

    print(f"\n  Results: {RESULTS_PATH}")
    print("  Done.")


if __name__ == "__main__":
    main()
