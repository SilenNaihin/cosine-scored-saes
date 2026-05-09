"""
Experiment 22: Production-Scale Post-Norm Loss
===============================================

Exp14 showed cosine_postnorm beats standard on SAE→KL at L27 (0.252 vs 0.198)
at 5M tokens. Exp17 showed 5M results flip at 50M (L27 FVE went from -3 to +8).
Exp21 closed the composite loss approach. The open question: does postnorm's
SAE→KL advantage scale?

Trains 4 variants at 50M tokens, L27 only:
  - standard: inner-product encoder + L2 loss (exp17 baseline)
  - adaptive_l2: cosine encoder + L2 loss (exp17 best FVE)
  - cosine_postnorm: cosine encoder + post-norm loss (exp14 best causal)
  - adaptive_postnorm: adaptive cosine encoder + post-norm loss (exp18 control)

Key question: Does postnorm SAE→KL advantage grow or shrink with 10x data?

If SAE→KL advantage grows → gain weighting is a genuine insight, worth combining
with the training-dynamics advantage of adaptive_l2 via new approaches.
If SAE→KL advantage shrinks → adaptive_l2 is the clear winner at scale.

Estimated runtime on H100: ~8-12 hours (4 variants × training + eval).

Usage:
    ssh <server>     cd ~/MechInter--RNH
    nohup .venv/bin/python -u experiments/exp22_postnorm_scale.py > experiments/exp22_output.log 2>&1 &
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
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 27              # L27 only (strongest postnorm effects)
D_MODEL = 4096
N_LAYERS_TOTAL = 36
RMS_NORM_EPS = 1e-6

# --- SAE architecture ---
D_SAE = 16384
K = 80

# --- Data ---
N_TRAIN_TOKENS = 50_000_000
N_EVAL_TOKENS = 1_000_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR = 3e-4
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 200

# --- Checkpointing ---
CHECKPOINT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

# --- Output ---
SAVE_DIR = "checkpoints/exp22"
RESULTS_PATH = "experiments/exp22_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)
CHECKPOINT_STEPS = [int(f * N_STEPS) for f in CHECKPOINT_FRACS]

# --- Streaming ---
BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE


def get_config_dict():
    return {
        "model_name": MODEL_NAME, "layer": LAYER, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr": LR, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "n_steps": N_STEPS, "warmup_steps": WARMUP_STEPS,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
        "buffer_tokens": BUFFER_TOKENS,
        "checkpoint_steps": CHECKPOINT_STEPS,
    }


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
    """BatchTopK SAE with per-token adaptive-scale cosine encoder."""

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


# =============================================================================
# RMSNorm Utilities
# =============================================================================

def get_rmsnorm_for_layer(model, layer_idx):
    """Get the RMSNorm module that normalizes the output of this layer.

    Layer k's output goes through layers[k+1].input_layernorm before
    the next attention block. For the final layer, use model.model.norm.
    """
    if layer_idx + 1 < N_LAYERS_TOTAL:
        return model.model.layers[layer_idx + 1].input_layernorm
    else:
        return model.model.norm


def apply_rmsnorm_f32(x, rmsnorm_weight, eps=RMS_NORM_EPS):
    x_f32 = x.float()
    weight = rmsnorm_weight.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 / rms) * weight


# =============================================================================
# Streaming Activation Collection (from exp17)
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
    """Streams activations from FineWeb with shuffle buffer."""

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

    def fill_buffer(self):
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
    print(f"  Collecting eval activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    # Skip 500k docs to avoid overlap with training data
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


# =============================================================================
# Training
# =============================================================================

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_sae_streaming(name, sae, stream, loss_type, rmsnorm_weight=None):
    """Train SAE with streaming activations.

    loss_type: "l2" or "postnorm"
    """
    tag = f"{name}/L{LAYER}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={LR}, loss={loss_type}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps (streaming)")

    rmsnorm_w_f32 = None
    if loss_type == "postnorm":
        assert rmsnorm_weight is not None
        rmsnorm_w_f32 = rmsnorm_weight.float().to(DEVICE)
        gain = rmsnorm_w_f32.detach()
        print(f"    RMSNorm gain: mean={gain.mean():.4f}, std={gain.std():.4f}, "
              f"min={gain.min():.4f}, max={gain.max():.4f}")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    sae.train()
    log = []
    checkpoint_log = {}
    t0 = time.time()
    global_step = 0
    next_checkpoint_idx = 0

    save_dir = Path(SAVE_DIR)

    while global_step < N_STEPS:
        n_filled = stream.fill_buffer()
        steps_in_buffer = min(BUFFER_BATCHES, N_STEPS - global_step)

        for buf_step in range(steps_in_buffer):
            batch = stream.get_batch(buf_step)
            x_hat, features = sae(batch)

            if loss_type == "l2":
                recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()
            elif loss_type == "postnorm":
                x_normed = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
                xhat_normed = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
                recon_loss = (x_normed - xhat_normed).pow(2).sum(dim=-1).mean()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

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

                    # Post-norm FVE
                    pnfve_str = ""
                    if rmsnorm_w_f32 is not None:
                        x_n = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
                        xh_n = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
                        pn_tv = torch.var(x_n, dim=0, unbiased=False).sum()
                        pn_rv = torch.var(x_n - xh_n, dim=0, unbiased=False).sum()
                        pnfve = (1 - pn_rv / pn_tv).item() if pn_tv > 0 else 0
                        pnfve_str = f" | pnFVE={pnfve:.4f}"

                entry = {
                    "step": global_step, "recon_loss": recon_loss.item(),
                    "l0": l0, "fve": fve, "cos_recon": cos_r,
                    "dead_frac": dead, "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": global_step * BATCH_SIZE,
                }

                scale_str = ""
                if hasattr(sae, "scale_a"):
                    entry["scale_a"] = sae.scale_a.item()
                    entry["scale_b"] = sae.scale_b.exp().item()
                    scale_str = f" | a={sae.scale_a.item():.4f} b={sae.scale_b.exp().item():.1f}"

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag:>24s}] step {global_step:>6d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.4f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | cos={cos_r:.4f} | dead={dead:.3f}"
                      f"{scale_str}{pnfve_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | {tok_per_sec/1e3:.0f}k tok/s | "
                      f"ETA {eta_sec/60:.0f}m")

            # Mid-training checkpoints
            if (next_checkpoint_idx < len(CHECKPOINT_STEPS) and
                    global_step >= CHECKPOINT_STEPS[next_checkpoint_idx]):
                frac = CHECKPOINT_FRACS[next_checkpoint_idx]
                ckpt_path = save_dir / f"{name}_L{LAYER}_step{global_step}.pt"
                torch.save(sae.state_dict(), ckpt_path)
                with torch.no_grad():
                    snap_dead = dead
                    snap_fve = fve
                checkpoint_log[f"{frac:.0%}"] = {
                    "step": global_step, "tokens": global_step * BATCH_SIZE,
                    "fve": snap_fve, "dead_frac": snap_dead,
                }
                if hasattr(sae, "scale_a"):
                    checkpoint_log[f"{frac:.0%}"]["scale_a"] = sae.scale_a.item()
                print(f"    [{tag}] Checkpoint saved at {frac:.0%} ({global_step} steps)")
                next_checkpoint_idx += 1

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    torch.save(sae.state_dict(), save_dir / f"{name}_L{LAYER}_final.pt")
    return log, checkpoint_log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data, rmsnorm_weight=None):
    tag = f"{name}/L{LAYER}"
    sae.eval()
    n = eval_data.shape[0]
    recon_losses, cos_sims, l0s = [], [], []
    total_var_sum, resid_var_sum = 0.0, 0.0
    pn_total_var_sum, pn_resid_var_sum = 0.0, 0.0
    dead_counts = None

    rmsnorm_w_f32 = rmsnorm_weight.float().to(DEVICE) if rmsnorm_weight is not None else None

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        recon_losses.append((batch - x_hat).pow(2).sum(dim=-1).mean().item())
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()

        if rmsnorm_w_f32 is not None:
            x_n = apply_rmsnorm_f32(batch, rmsnorm_w_f32)
            xh_n = apply_rmsnorm_f32(x_hat, rmsnorm_w_f32)
            pn_total_var_sum += torch.var(x_n, dim=0, unbiased=False).sum().item()
            pn_resid_var_sum += torch.var(x_n - xh_n, dim=0, unbiased=False).sum().item()

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
    if rmsnorm_w_f32 is not None:
        results["pnfve"] = float(1 - pn_resid_var_sum / pn_total_var_sum) if pn_total_var_sum > 0 else 0
        results["pn_recon_loss"] = float(pn_resid_var_sum / max(1, n // BATCH_SIZE))

    pnfve_str = f" | pnFVE={results['pnfve']:.4f}" if 'pnfve' in results else ''
    print(f"    [{tag}] L2={results['recon_loss_l2']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"FVE={results['fve']:.4f} | dead={dead_frac:.3f}{pnfve_str}")
    return results


@torch.no_grad()
def test_norm_invariance(name, sae, eval_data, scales=(0.5, 2.0, 5.0)):
    tag = f"{name}/L{LAYER}"
    sae.eval()
    sample = eval_data[:BATCH_SIZE].to(DEVICE, dtype=torch.float32)
    base_feats = sae.encode(sample)

    results = {}
    for scale in scales:
        scaled_feats = sae.encode(sample * scale)
        both_on = (base_feats > 0) & (scaled_feats > 0)
        if both_on.any():
            ratios = scaled_feats[both_on] / base_feats[both_on]
            mean_ratio = ratios.mean().item()
        else:
            mean_ratio = float("nan")
        agreement = ((base_feats > 0) == (scaled_feats > 0)).float().mean().item()
        cos = F.cosine_similarity(
            base_feats.float(), scaled_feats.float(), dim=-1
        ).mean().item()
        results[f"scale_{scale}"] = {
            "mean_ratio": mean_ratio,
            "feature_agreement": agreement, "activation_cosine": cos,
        }
        print(f"    [{tag}] scale={scale}: ratio={mean_ratio:.3f} | "
              f"agree={agreement:.3f} | cos={cos:.4f}")
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


def evaluate_ablation(name, model, sae, eval_data):
    tag = f"{name}/L{LAYER}"
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
    top_idx = freq.topk(n_to_select).indices

    # Get decoder directions
    if hasattr(sae, "W_dec"):
        W_dec = sae.W_dec  # (d_sae, d_model) for adaptive cosine SAEs
    elif hasattr(sae, "decoder"):
        W_dec = sae.decoder.weight.T  # nn.Linear weight is (d_model, d_sae)
    else:
        raise ValueError("Cannot find decoder weights")

    feature_results = []
    for rank, fi in enumerate(top_idx):
        fi = fi.item()
        feat_dir = W_dec[fi].float()
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
            kl = ablate_feature_kl(model, x, feat_dir, LAYER)
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

        cos_arr, norm_arr = np.array(cos_v), np.array(norm_v)
        inner_arr, sae_arr = np.array(inner_v), np.array(sae_v)

        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]

        result = {
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
        }
        feature_results.append(result)

        if rank < 7 or rank % 20 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f} | norm→KL={corr_norm:.3f}")

    if not feature_results:
        print(f"    [{tag}] No features with enough data for ablation")
        return {"n_features": 0}

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
          f"cos→KL={agg['cos_kl_mean']:.4f} | inner→KL={agg['inner_kl_mean']:.4f} | "
          f"SAE→KL={agg['sae_kl_mean']:.4f} | cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Variant Definitions
# =============================================================================

# (name, SAE class, loss_type)
VARIANTS = [
    ("standard",           BatchTopKSAE,                "l2"),
    ("adaptive_l2",        AdaptiveCosineBatchTopKSAE,  "l2"),
    ("cosine_postnorm",    AdaptiveCosineBatchTopKSAE,  "postnorm"),  # a=0 expected (postnorm is scale-invariant)
    ("adaptive_postnorm",  AdaptiveCosineBatchTopKSAE,  "postnorm"),  # same, confirming exp18
]


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 22: Production-Scale Post-Norm Loss")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Per variant: {N_TRAIN_TOKENS:,} train tokens ({N_STEPS} steps), "
          f"{N_EVAL_TOKENS:,} eval tokens")
    print(f"Batch: {BATCH_SIZE}, Warmup: {WARMUP_STEPS} steps")
    print(f"Streaming buffer: {BUFFER_TOKENS:,} tokens")
    print(f"Checkpoints at steps: {CHECKPOINT_STEPS}")
    print(f"Ablation: {N_ABLATION_FEATURES} features × {N_ABLATION_SAMPLES} samples")
    print(f"Variants: {[v[0] for v in VARIANTS]}")

    # ---- Load model ----
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Get RMSNorm weight for post-norm loss
    rmsnorm = get_rmsnorm_for_layer(model, LAYER)
    rmsnorm_weight = rmsnorm.weight.detach()
    gain = rmsnorm_weight.float()
    print(f"  L{LAYER} RMSNorm gain: mean={gain.mean():.4f}, CV={gain.std()/gain.mean()*100:.1f}%, "
          f"max={gain.max():.4f}")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing results (resume support) ----
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        existing = list(all_results.get("variants", {}).keys())
        print(f"  Loaded existing results for variants: {existing}")
    else:
        all_results = {"config": get_config_dict(), "variants": {}}

    # ---- Collect eval data ----
    eval_data = collect_eval_data(model, tokenizer, LAYER, N_EVAL_TOKENS)

    # ---- Run each variant ----
    for vname, cls, loss_type in VARIANTS:
        if vname in all_results["variants"]:
            print(f"\n  {vname} already complete, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  VARIANT: {vname} (encoder={cls.__name__}, loss={loss_type})")
        print(f"{'='*70}")

        stream = ActivationStream(model, tokenizer, LAYER)
        torch.manual_seed(SEED)
        sae = cls(D_MODEL, D_SAE, K).to(DEVICE)

        train_log, ckpt_log = train_sae_streaming(
            vname, sae, stream, loss_type,
            rmsnorm_weight=rmsnorm_weight if loss_type == "postnorm" else None,
        )

        print(f"\n  Evaluation — {vname}")
        recon = evaluate_reconstruction(vname, sae, eval_data, rmsnorm_weight=rmsnorm_weight)
        inv = test_norm_invariance(vname, sae, eval_data)
        abl = evaluate_ablation(vname, model, sae, eval_data)

        result = {
            "training": train_log,
            "checkpoints": ckpt_log,
            "reconstruction": recon,
            "norm_invariance": inv,
            "ablation": abl,
        }
        if hasattr(sae, "scale_a"):
            result["scale_a_final"] = sae.scale_a.item()
            result["scale_b_final"] = sae.scale_b.exp().item()

        all_results["variants"][vname] = result

        # Save after each variant (resume support)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved {vname} to {RESULTS_PATH}")

        del sae, stream
        gc.collect()
        torch.cuda.empty_cache()

    del eval_data
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Production-Scale Post-Norm Loss at L{LAYER}")
    print(f"{'='*70}")

    header = f"{'Variant':>24s} | {'loss':>8s} | {'scale_a':>8s} | {'FVE':>7s} | {'pnFVE':>7s} | {'dead':>6s} | {'cos→KL':>7s} | {'SAE→KL':>7s} | {'cos>inn':>8s}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for vname, cls, loss_type in VARIANTS:
        v = all_results["variants"].get(vname, {})
        r = v.get("reconstruction", {})
        a = v.get("ablation", {}).get("aggregate", {})
        sa = v.get("scale_a_final", None)

        fve = r.get("fve", 0)
        pnfve = r.get("pnfve", 0)
        dead = r.get("dead_frac", 1)
        cos_kl = a.get("cos_kl_mean", 0)
        sae_kl = a.get("sae_kl_mean", 0)
        cw = a.get("cos_wins_inner", 0)
        nf = a.get("n_features", 0)

        sa_str = f"{sa:.4f}" if sa is not None else "-"
        print(f"  {vname:>24s} | {loss_type:>8s} | {sa_str:>8s} | {fve:>7.4f} | "
              f"{pnfve:>7.4f} | {dead:>6.3f} | {cos_kl:>7.4f} | {sae_kl:>7.4f} | "
              f"{cw:>4d}/{nf}")

    print(f"\n  Key question: Does postnorm SAE→KL advantage scale from 5M→50M?")
    print(f"  (5M reference: postnorm SAE→KL=0.252 > standard 0.198)")
    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
