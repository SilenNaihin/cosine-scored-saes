"""
Experiment 29: Learning Rate Sweep
====================================

Fairness control: all prior comparisons use LR=3e-4. If the cosine encoder
has a different optimal LR than the standard encoder, the comparison is
unfair. Sweep LR in {1e-4, 3e-4, 1e-3} for standard and adaptive_l2 at
L27, 5M tokens.

If both prefer the same LR → comparison is fair, proceed to multi-seed.
If not → the 50M comparison should use per-variant best LR.

Variants (6 total, trained sequentially):
  standard_lr1e4, standard_lr3e4, standard_lr1e3
  adaptive_lr1e4, adaptive_lr3e4, adaptive_lr1e3

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 experiments/exp29_lr_sweep.py 2>&1 | tee experiments/exp29_output.log
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
LAYER_IDX = 27
D_MODEL = 4096

# --- SAE ---
D_SAE = 16384
K = 80

# --- Data ---
N_TRAIN_TOKENS = 5_000_000
N_EVAL_TOKENS = 1_000_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR_SWEEP = [1e-4, 3e-4, 1e-3]
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 100

# --- Ablation evaluation ---
N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

# --- Output ---
SAVE_DIR = "checkpoints/exp29"
RESULTS_PATH = "experiments/exp29_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)

# --- Streaming buffer ---
BUFFER_TOKENS = 500_000
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE


def get_config_dict():
    return {
        "model_name": MODEL_NAME, "layer": LAYER_IDX, "d_model": D_MODEL,
        "d_sae": D_SAE, "k": K,
        "n_train_tokens": N_TRAIN_TOKENS, "n_eval_tokens": N_EVAL_TOKENS,
        "ctx_len": CTX_LEN, "lr_sweep": LR_SWEEP, "batch_size": BATCH_SIZE,
        "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "n_steps": N_STEPS, "warmup_steps": WARMUP_STEPS,
        "n_ablation_features": N_ABLATION_FEATURES,
        "n_ablation_samples": N_ABLATION_SAMPLES,
    }


# =============================================================================
# SAE Architectures
# =============================================================================

class BatchTopKSAE(nn.Module):
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
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
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

def make_lr_schedule(lr, n_steps, warmup_steps):
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule


def train_sae_streaming(name, sae, stream, lr, save_dir):
    tag = f"{name}/L{LAYER_IDX}"
    print(f"\n  Training {tag} | d_sae={D_SAE}, k={K}, lr={lr}, "
          f"{N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)
    schedule_fn = make_lr_schedule(lr, N_STEPS, WARMUP_STEPS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    sae.train()
    log = []
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
                    entry["scale_a"] = sae.scale_a.item()
                    entry["scale_b"] = sae.scale_b.exp().item()
                    scale_str = f" | a={sae.scale_a.item():.4f}"
                else:
                    scale_str = ""

                log.append(entry)
                elapsed = time.time() - t0
                tokens_seen = global_step * BATCH_SIZE
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                eta_sec = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"    [{tag:>20s}] step {global_step:>5d}/{N_STEPS} | "
                      f"loss={recon_loss.item():.1f} | L0={l0:.0f} | "
                      f"FVE={fve:.4f} | dead={dead:.3f}{scale_str} | "
                      f"{tokens_seen/1e6:.1f}M tok | ETA {eta_sec/60:.0f}m")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    torch.save(sae.state_dict(), save_dir / f"{name}_final.pt")
    return log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data):
    tag = f"{name}/L{LAYER_IDX}"
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


def evaluate_ablation(name, model, sae, eval_data):
    tag = f"{name}/L{LAYER_IDX}"
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
        return {"n_features": 0}
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
            kl = ablate_feature_kl(model, x, feat_dir, LAYER_IDX)
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
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f}")

    if not feature_results:
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
          f"cos→KL={agg['cos_kl_mean']:.4f} | SAE→KL={agg['sae_kl_mean']:.4f} | "
          f"cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Main
# =============================================================================

ENCODER_TYPES = [
    ("standard", BatchTopKSAE),
    ("adaptive", AdaptiveCosineBatchTopKSAE),
]


def main():
    print("Experiment 29: Learning Rate Sweep")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}, Layer: {LAYER_IDX}")
    print(f"d_sae={D_SAE}, k={K}, {N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")
    print(f"LR sweep: {LR_SWEEP}")
    print(f"Encoders: {[e[0] for e in ENCODER_TYPES]}")
    print(f"Total runs: {len(ENCODER_TYPES) * len(LR_SWEEP)}")

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

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing results ----
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        existing = list(all_results.get("runs", {}).keys())
        print(f"  Loaded existing results: {existing}")
    else:
        all_results = {"config": get_config_dict(), "runs": {}}

    # ---- Collect eval data once ----
    eval_data = collect_eval_data(model, tokenizer, LAYER_IDX, N_EVAL_TOKENS)

    # ---- Run all combinations ----
    for enc_name, enc_cls in ENCODER_TYPES:
        for lr in LR_SWEEP:
            run_name = f"{enc_name}_lr{lr:.0e}"
            if run_name in all_results.get("runs", {}):
                print(f"\n  {run_name} already complete, skipping")
                continue

            print(f"\n{'='*70}")
            print(f"  RUN: {run_name} (encoder={enc_name}, lr={lr})")
            print(f"{'='*70}")

            stream = ActivationStream(model, tokenizer, LAYER_IDX)

            torch.manual_seed(SEED)
            sae = enc_cls(D_MODEL, D_SAE, K).to(DEVICE)

            # Train
            train_log = train_sae_streaming(run_name, sae, stream, lr, save_dir)

            # Evaluate
            print(f"\n  Evaluation — {run_name}")
            recon = evaluate_reconstruction(run_name, sae, eval_data)
            abl = evaluate_ablation(run_name, model, sae, eval_data)

            run_result = {
                "encoder": enc_name,
                "lr": lr,
                "training": train_log,
                "reconstruction": recon,
                "ablation": abl,
            }
            if hasattr(sae, "scale_a"):
                run_result["scale_a_final"] = sae.scale_a.item()

            all_results["runs"][run_name] = run_result

            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Results saved to {RESULTS_PATH}")

            del sae, stream
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  LR SWEEP SUMMARY")
    print(f"{'='*70}")

    # Group by encoder
    for enc_name, _ in ENCODER_TYPES:
        print(f"\n  {enc_name}:")
        print(f"  {'LR':>8s} | {'FVE':>6s} | {'Dead%':>6s} | {'Alive':>6s} | "
              f"{'cos>inn':>8s} | {'SAE→KL':>7s}", end="")
        if enc_name == "adaptive":
            print(f" | {'a':>6s}", end="")
        print()
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}", end="")
        if enc_name == "adaptive":
            print(f"-+-{'-'*6}", end="")
        print()

        best_fve = -1
        best_lr = None
        for lr in LR_SWEEP:
            run_name = f"{enc_name}_lr{lr:.0e}"
            run = all_results["runs"].get(run_name, {})
            r = run.get("reconstruction", {})
            a = run.get("ablation", {}).get("aggregate", {})
            if r:
                fve = r["fve"]
                if fve > best_fve:
                    best_fve = fve
                    best_lr = lr
                cos_inn = f"{a.get('cos_wins_inner', '?')}/{a.get('n_features', '?')}" if a else "—"
                sae_kl = f"{a.get('sae_kl_mean', 0):.3f}" if a else "—"
                line = (f"  {lr:>8.0e} | {fve:.4f} | {r['dead_frac']*100:>5.1f}% | "
                        f"{r['alive_count']:>6d} | {cos_inn:>8s} | {sae_kl:>7s}")
                if enc_name == "adaptive" and "scale_a_final" in run:
                    line += f" | {run['scale_a_final']:>6.3f}"
                print(line)
        if best_lr is not None:
            print(f"  Best LR: {best_lr:.0e} (FVE={best_fve:.4f})")

    # Cross-encoder comparison at best LR
    print(f"\n  Cross-encoder at best LR:")
    for enc_name, _ in ENCODER_TYPES:
        best_fve = -1
        best_lr = None
        for lr in LR_SWEEP:
            run_name = f"{enc_name}_lr{lr:.0e}"
            run = all_results["runs"].get(run_name, {})
            r = run.get("reconstruction", {})
            if r and r["fve"] > best_fve:
                best_fve = r["fve"]
                best_lr = lr
        if best_lr:
            print(f"    {enc_name}: best LR={best_lr:.0e}, FVE={best_fve:.4f}")

    # Fairness verdict
    std_best = None
    adp_best = None
    for lr in LR_SWEEP:
        r = all_results["runs"].get(f"standard_lr{lr:.0e}", {}).get("reconstruction", {})
        if r and (std_best is None or r["fve"] > std_best[1]):
            std_best = (lr, r["fve"])
        r = all_results["runs"].get(f"adaptive_lr{lr:.0e}", {}).get("reconstruction", {})
        if r and (adp_best is None or r["fve"] > adp_best[1]):
            adp_best = (lr, r["fve"])

    if std_best and adp_best:
        print(f"\n  FAIRNESS VERDICT:")
        if std_best[0] == adp_best[0]:
            print(f"    FAIR — both prefer LR={std_best[0]:.0e}")
            print(f"    The 50M comparison at LR=3e-4 is valid.")
        else:
            print(f"    UNFAIR — standard prefers {std_best[0]:.0e}, adaptive prefers {adp_best[0]:.0e}")
            print(f"    The 50M comparison should use per-variant best LR.")
            print(f"    Standard best FVE: {std_best[1]:.4f} at LR={std_best[0]:.0e}")
            print(f"    Adaptive best FVE: {adp_best[1]:.4f} at LR={adp_best[0]:.0e}")
            gap = adp_best[1] - std_best[1]
            print(f"    Gap at best-LR: {gap:+.4f} (adaptive - standard)")

    print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
