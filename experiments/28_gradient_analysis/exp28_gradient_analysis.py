"""
Experiment 28: Gradient Analysis — Why Does Cosine Keep More Features Alive?
=============================================================================

Exp19 showed the cosine SAE advantage comes from training dynamics, not
norm-specific correction. But it didn't explain WHY. The likely mechanism:

  Standard encoder:  pre_acts = x @ W_enc.T
    → gradient dL/dW_enc[i] ∝ ||x||  (high-norm tokens dominate)

  Cosine encoder:    pre_acts = scale * cos_sim(x, W_enc)
    → gradient dL/dW_enc[i] independent of ||x||  (all tokens equal)

If standard gradients are dominated by high-norm tokens, low-frequency features
that only activate on lower-norm tokens get starved of gradient signal and die.
Cosine normalizes this away, keeping more features alive.

This experiment proves or disproves this by logging:
  1. Per-feature W_enc gradient norms stratified by input norm quartile
  2. Gradient variance across quartiles (standard should show ~10x, cosine ~1x)
  3. Per-feature "gradient domination ratio" (Q4 grad / Q1 grad)

2M tokens on Qwen3-8B L27, 2 variants (standard, cosine_adaptive).

Run on <gpu-server> GPU 1.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/exp28_gradient_analysis.py > experiments/exp28_output.log 2>&1 &
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
STORAGE_DTYPE = torch.float16

# --- Model ---
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER_IDX = 27  # The key layer — largest norms, strongest cosine effects
D_MODEL = 4096

# --- SAE architecture ---
D_SAE = 16384  # 4x d_model
K = 80

# --- Data ---
N_TRAIN_TOKENS = 2_000_000  # Short run — enough to see gradient patterns
N_EVAL_TOKENS = 500_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Training ---
LR = 3e-4
BATCH_SIZE = 4096
WARMUP_FRAC = 0.05
SEED = 42

# --- Gradient logging ---
LOG_GRAD_EVERY = 10  # Log detailed gradients every N steps
N_QUARTILES = 4

# --- Output ---
SAVE_DIR = "checkpoints/exp28"
RESULTS_PATH = "experiments/exp28_results.json"

# --- Derived ---
N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
WARMUP_STEPS = int(N_STEPS * WARMUP_FRAC)


# =============================================================================
# SAE Architectures
# =============================================================================

class BatchTopKSAE(nn.Module):
    """Standard BatchTopK SAE with inner-product encoder."""

    def __init__(self, d_model: int, d_sae: int, k: int = 50):
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


class CosineBatchTopKSAE(nn.Module):
    """Full cosine encoder with norm-adaptive init."""

    def __init__(self, d_model: int, d_sae: int, k: int = 50, init_norm: float = None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        scale_init = math.log(init_norm) if init_norm is not None else math.log(math.sqrt(d_model))
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
        scale = torch.exp(self.scale_b)
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


VARIANTS = [
    ("standard",        BatchTopKSAE,      False),
    ("cosine_adaptive", CosineBatchTopKSAE, True),
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

        all_acts.append(flat.to("cpu", dtype=STORAGE_DTYPE))
        tokens_collected += flat.shape[0]

    result = torch.cat(all_acts, dim=0)[:n_tokens]
    norms = result.float().norm(dim=-1)
    print(f"  Layer {layer_idx}: {result.shape[0]:,} {label} tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={norms.mean():.1f}, std={norms.std():.1f})")
    return result


# =============================================================================
# Gradient Analysis Training Loop
# =============================================================================

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return (step + 1) / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_quartile_grads(sae, batch, batch_norms):
    """Compute per-feature W_enc gradient norms, stratified by input norm quartile.

    For each quartile Q1-Q4:
      1. Select tokens in that quartile
      2. Forward + backward on just those tokens
      3. Record the per-feature gradient norm of W_enc

    Returns dict with per-quartile gradient stats.
    """
    quartile_boundaries = torch.quantile(
        batch_norms, torch.tensor([0.25, 0.5, 0.75], device=batch_norms.device)
    )
    q_labels = torch.zeros_like(batch_norms, dtype=torch.long)
    q_labels[batch_norms >= quartile_boundaries[0]] = 1
    q_labels[batch_norms >= quartile_boundaries[1]] = 2
    q_labels[batch_norms >= quartile_boundaries[2]] = 3

    quartile_grad_norms = {}  # q_idx -> [d_sae] per-feature grad norms

    for q in range(N_QUARTILES):
        mask = q_labels == q
        if mask.sum() < 10:
            continue
        subset = batch[mask]

        sae.zero_grad(set_to_none=True)
        x_hat, features = sae(subset)
        loss = (subset - x_hat).pow(2).sum(dim=-1).mean()
        loss.backward()

        # Per-feature gradient norm: ||dL/dW_enc[i, :]|| for each feature i
        grad = sae.W_enc.grad  # [d_sae, d_model]
        if grad is not None:
            per_feat_grad_norm = grad.norm(dim=1)  # [d_sae]
            quartile_grad_norms[q] = per_feat_grad_norm.detach().cpu()

    return quartile_grad_norms


def train_with_gradient_logging(name, sae, train_data, mean_norm):
    """Train SAE with detailed gradient logging."""
    print(f"\n  Training {name} | L{LAYER_IDX} | {N_TRAIN_TOKENS:,} tokens, {N_STEPS} steps")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    perm = torch.randperm(train_data.shape[0])
    train_shuffled = train_data[perm]

    # Pre-compute norms for quartile stratification
    all_norms = train_shuffled.float().norm(dim=-1)

    sae.train()
    t0 = time.time()

    # Gradient log: list of per-step snapshots
    grad_log = []
    # Aggregate stats across training
    agg_q_grad_norms = {q: [] for q in range(N_QUARTILES)}  # q -> list of [d_sae] tensors

    # Also track per-step W_enc global gradient norm
    global_grad_log = []

    for step in range(1, N_STEPS + 1):
        start = ((step - 1) * BATCH_SIZE) % train_shuffled.shape[0]
        end = start + BATCH_SIZE
        if end > train_shuffled.shape[0]:
            idx = torch.cat([
                torch.arange(start, train_shuffled.shape[0]),
                torch.arange(0, end - train_shuffled.shape[0]),
            ])
        else:
            idx = torch.arange(start, end)
        batch = train_shuffled[idx].to(DEVICE, dtype=torch.float32)
        batch_norms = all_norms[idx].to(DEVICE)

        # === Standard training step ===
        x_hat, features = sae(batch)
        recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

        optimizer.zero_grad(set_to_none=True)
        recon_loss.backward()

        # Log global W_enc gradient norm every step
        if sae.W_enc.grad is not None:
            global_grad_norm = sae.W_enc.grad.norm().item()
        else:
            global_grad_norm = 0.0

        optimizer.step()
        scheduler.step()

        # === Detailed gradient logging every LOG_GRAD_EVERY steps ===
        if step % LOG_GRAD_EVERY == 0 or step == 1 or step == N_STEPS:
            sae.train()  # Ensure training mode
            quartile_grads = compute_quartile_grads(sae, batch, batch_norms)

            # Compute summary stats
            snapshot = {"step": step}
            q_means = {}
            q_medians = {}
            for q in range(N_QUARTILES):
                if q in quartile_grads:
                    gn = quartile_grads[q]
                    agg_q_grad_norms[q].append(gn)
                    q_means[q] = gn.mean().item()
                    q_medians[q] = gn.median().item()
                    snapshot[f"Q{q+1}_mean_grad"] = gn.mean().item()
                    snapshot[f"Q{q+1}_median_grad"] = gn.median().item()
                    snapshot[f"Q{q+1}_max_grad"] = gn.max().item()
                    snapshot[f"Q{q+1}_std_grad"] = gn.std().item()

            # Domination ratio: Q4 / Q1 (high-norm grad / low-norm grad)
            if 0 in q_means and 3 in q_means and q_means[0] > 1e-12:
                snapshot["Q4_Q1_ratio_mean"] = q_means[3] / q_means[0]
            if 0 in q_medians and 3 in q_medians and q_medians[0] > 1e-12:
                snapshot["Q4_Q1_ratio_median"] = q_medians[3] / q_medians[0]

            # Norm quartile boundaries (for reference)
            quartile_boundaries = torch.quantile(
                batch_norms, torch.tensor([0.25, 0.5, 0.75], device=batch_norms.device)
            )
            snapshot["norm_Q1_upper"] = quartile_boundaries[0].item()
            snapshot["norm_Q2_upper"] = quartile_boundaries[1].item()
            snapshot["norm_Q3_upper"] = quartile_boundaries[2].item()
            snapshot["norm_mean"] = batch_norms.mean().item()

            # Training metrics
            with torch.no_grad():
                l0 = (features != 0).float().sum(dim=-1).mean().item()
                total_var = torch.var(batch, dim=0, unbiased=False).sum()
                resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                dead = (features.sum(dim=0) == 0).float().mean().item()

            snapshot["recon_loss"] = recon_loss.item()
            snapshot["l0"] = l0
            snapshot["fve"] = fve
            snapshot["dead_frac"] = dead
            snapshot["global_grad_norm"] = global_grad_norm

            grad_log.append(snapshot)

            ratio_str = ""
            if "Q4_Q1_ratio_mean" in snapshot:
                ratio_str = f" | Q4/Q1={snapshot['Q4_Q1_ratio_mean']:.2f}x"

            print(f"    [{name:>16s}] step {step:>4d}/{N_STEPS} | "
                  f"loss={recon_loss.item():.2f} | FVE={fve:.4f} | dead={dead:.3f} | "
                  f"Q1_grad={snapshot.get('Q1_mean_grad', 0):.6f} | "
                  f"Q4_grad={snapshot.get('Q4_mean_grad', 0):.6f}{ratio_str}")

        elif step % 50 == 0:
            with torch.no_grad():
                l0 = (features != 0).float().sum(dim=-1).mean().item()
                total_var = torch.var(batch, dim=0, unbiased=False).sum()
                resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
            print(f"    [{name:>16s}] step {step:>4d}/{N_STEPS} | "
                  f"loss={recon_loss.item():.2f} | FVE={fve:.4f} | grad_norm={global_grad_norm:.4f}")

        global_grad_log.append({"step": step, "global_grad_norm": global_grad_norm})

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{name}] Done in {elapsed:.1f}s")

    # === Aggregate analysis: per-feature gradient domination across training ===
    print(f"\n  Aggregate gradient analysis for {name}:")
    agg_results = {}
    for q in range(N_QUARTILES):
        if agg_q_grad_norms[q]:
            stacked = torch.stack(agg_q_grad_norms[q])  # [n_snapshots, d_sae]
            mean_across_time = stacked.mean(dim=0)  # [d_sae] — avg grad norm per feature
            agg_results[f"Q{q+1}_per_feature_mean"] = mean_across_time.mean().item()
            agg_results[f"Q{q+1}_per_feature_std"] = mean_across_time.std().item()
            agg_results[f"Q{q+1}_per_feature_median"] = mean_across_time.median().item()

    # Per-feature domination ratio: for each feature, Q4_grad / Q1_grad averaged over training
    if agg_q_grad_norms[0] and agg_q_grad_norms[3]:
        q1_stacked = torch.stack(agg_q_grad_norms[0]).mean(dim=0)  # [d_sae]
        q4_stacked = torch.stack(agg_q_grad_norms[3]).mean(dim=0)  # [d_sae]
        safe_q1 = q1_stacked.clamp(min=1e-12)
        per_feature_ratio = q4_stacked / safe_q1
        agg_results["per_feature_Q4_Q1_ratio_mean"] = per_feature_ratio.mean().item()
        agg_results["per_feature_Q4_Q1_ratio_median"] = per_feature_ratio.median().item()
        agg_results["per_feature_Q4_Q1_ratio_std"] = per_feature_ratio.std().item()
        # What fraction of features have Q4 > 2x Q1?
        agg_results["frac_features_Q4_dominates_2x"] = (per_feature_ratio > 2.0).float().mean().item()
        agg_results["frac_features_Q4_dominates_5x"] = (per_feature_ratio > 5.0).float().mean().item()
        agg_results["frac_features_Q4_dominates_10x"] = (per_feature_ratio > 10.0).float().mean().item()

        print(f"    Per-feature Q4/Q1 gradient domination ratio:")
        print(f"      mean={agg_results['per_feature_Q4_Q1_ratio_mean']:.3f} "
              f"median={agg_results['per_feature_Q4_Q1_ratio_median']:.3f} "
              f"std={agg_results['per_feature_Q4_Q1_ratio_std']:.3f}")
        print(f"      >2x: {agg_results['frac_features_Q4_dominates_2x']*100:.1f}% of features")
        print(f"      >5x: {agg_results['frac_features_Q4_dominates_5x']*100:.1f}% of features")
        print(f"      >10x: {agg_results['frac_features_Q4_dominates_10x']*100:.1f}% of features")

    for q in range(N_QUARTILES):
        v = agg_results.get(f"Q{q+1}_per_feature_mean", 0)
        print(f"    Q{q+1} avg per-feature grad norm: {v:.6f}")

    return {
        "gradient_snapshots": grad_log,
        "global_grad_log": global_grad_log,
        "aggregate": agg_results,
    }


# =============================================================================
# Dead Feature Analysis
# =============================================================================

@torch.no_grad()
def analyze_dead_features(name, sae, train_data, all_norms):
    """Analyze which norm quartile's tokens contribute to keeping features alive."""
    print(f"\n  Dead feature analysis for {name}...")
    sae.eval()

    quartile_boundaries = torch.quantile(
        all_norms, torch.tensor([0.25, 0.5, 0.75])
    )
    q_labels = torch.zeros_like(all_norms, dtype=torch.long)
    q_labels[all_norms >= quartile_boundaries[0]] = 1
    q_labels[all_norms >= quartile_boundaries[1]] = 2
    q_labels[all_norms >= quartile_boundaries[2]] = 3

    # For each quartile, compute which features activate
    q_alive = {}
    n_probe = min(500_000, train_data.shape[0])
    for q in range(N_QUARTILES):
        mask = (q_labels[:n_probe] == q)
        if mask.sum() < 100:
            continue
        subset = train_data[:n_probe][mask]
        all_feats = []
        for i in range(0, subset.shape[0], BATCH_SIZE):
            batch = subset[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
            _, f = sae(batch)
            all_feats.append((f > 0).any(dim=0).cpu())
        alive_mask = torch.stack(all_feats).any(dim=0)
        q_alive[q] = alive_mask
        n_alive = alive_mask.sum().item()
        print(f"    Q{q+1} ({mask.sum().item()} tokens): {n_alive} features activate "
              f"({n_alive/D_SAE*100:.1f}%)")

    # Features alive in Q4 but dead in Q1
    if 0 in q_alive and 3 in q_alive:
        q4_only = q_alive[3] & ~q_alive[0]
        q1_only = q_alive[0] & ~q_alive[3]
        both = q_alive[0] & q_alive[3]
        neither = ~q_alive[0] & ~q_alive[3]
        print(f"    Both Q1+Q4: {both.sum().item()} | Q4-only: {q4_only.sum().item()} | "
              f"Q1-only: {q1_only.sum().item()} | Neither: {neither.sum().item()}")

    return {f"Q{q+1}_alive": q_alive[q].sum().item() for q in q_alive}


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 28: Gradient Analysis — Why Does Cosine Keep More Features Alive?")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}, Layer: {LAYER_IDX}")
    print(f"d_model: {D_MODEL}, d_sae: {D_SAE}, k: {K}, lr: {LR}")
    print(f"Tokens: {N_TRAIN_TOKENS:,} train, Steps: {N_STEPS}")
    print(f"Gradient logging every {LOG_GRAD_EVERY} steps")
    print(f"Variants: {[v[0] for v in VARIANTS]}")

    # Load model
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

    # Collect activations
    train_data = collect_activations(model, tokenizer, LAYER_IDX, N_TRAIN_TOKENS)
    all_norms = train_data.float().norm(dim=-1)
    mean_norm = all_norms.mean().item()

    print(f"\n  Activation norm stats for L{LAYER_IDX}:")
    print(f"    mean={mean_norm:.1f}, std={all_norms.std().item():.1f}")
    q_bounds = torch.quantile(all_norms, torch.tensor([0.25, 0.5, 0.75]))
    print(f"    Q1=[..{q_bounds[0]:.1f}], Q2=[{q_bounds[0]:.1f}..{q_bounds[1]:.1f}], "
          f"Q3=[{q_bounds[1]:.1f}..{q_bounds[2]:.1f}], Q4=[{q_bounds[2]:.1f}..]")
    print(f"    Q4/Q1 norm ratio: {q_bounds[2].item()/(q_bounds[0].item()+1e-8):.2f}x")

    all_results = {
        "config": {
            "model_name": MODEL_NAME,
            "layer": LAYER_IDX,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "k": K,
            "n_train_tokens": N_TRAIN_TOKENS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "n_steps": N_STEPS,
            "log_grad_every": LOG_GRAD_EVERY,
            "mean_norm": mean_norm,
            "norm_quartiles": [q_bounds[i].item() for i in range(3)],
        },
        "variants": {},
    }

    for vname, cls, use_norm_adaptive in VARIANTS:
        print(f"\n{'='*70}")
        print(f"  VARIANT: {vname}")
        print(f"{'='*70}")

        torch.manual_seed(SEED)
        if use_norm_adaptive:
            sae = cls(D_MODEL, D_SAE, K, init_norm=mean_norm).to(DEVICE)
            print(f"  scale_b init: log({mean_norm:.1f}) = {math.log(mean_norm):.4f}")
        else:
            sae = cls(D_MODEL, D_SAE, K).to(DEVICE)

        # Train with gradient logging
        grad_results = train_with_gradient_logging(vname, sae, train_data, mean_norm)

        # Dead feature quartile analysis
        dead_analysis = analyze_dead_features(vname, sae, train_data, all_norms)

        all_results["variants"][vname] = {
            "gradient_analysis": grad_results["aggregate"],
            "gradient_snapshots": grad_results["gradient_snapshots"],
            "dead_feature_analysis": dead_analysis,
        }

        if hasattr(sae, "scale_b"):
            all_results["variants"][vname]["scale_b_exp"] = sae.scale_b.exp().item()

        # Save checkpoint
        torch.save(sae.state_dict(), save_dir / f"{vname}_L{LAYER_IDX}_final.pt")

        # Save results incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        del sae
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  SUMMARY — Gradient Domination Analysis")
    print(f"{'='*80}")

    print(f"\n  Input norm quartile boundaries:")
    print(f"    Q1: [..{q_bounds[0]:.1f}]  Q2: [{q_bounds[0]:.1f}..{q_bounds[1]:.1f}]  "
          f"Q3: [{q_bounds[1]:.1f}..{q_bounds[2]:.1f}]  Q4: [{q_bounds[2]:.1f}..]")

    print(f"\n  {'Variant':<18s} {'Q1 grad':>10s} {'Q4 grad':>10s} {'Q4/Q1':>8s} "
          f"{'% >2x':>7s} {'% >5x':>7s} {'% >10x':>7s}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    for vname, _, _ in VARIANTS:
        agg = all_results["variants"][vname]["gradient_analysis"]
        q1 = agg.get("Q1_per_feature_mean", 0)
        q4 = agg.get("Q4_per_feature_mean", 0)
        ratio = agg.get("per_feature_Q4_Q1_ratio_mean", 0)
        gt2x = agg.get("frac_features_Q4_dominates_2x", 0) * 100
        gt5x = agg.get("frac_features_Q4_dominates_5x", 0) * 100
        gt10x = agg.get("frac_features_Q4_dominates_10x", 0) * 100
        print(f"  {vname:<18s} {q1:>10.6f} {q4:>10.6f} {ratio:>7.2f}x "
              f"{gt2x:>6.1f}% {gt5x:>6.1f}% {gt10x:>6.1f}%")

    print(f"\n  Dead feature analysis:")
    for vname, _, _ in VARIANTS:
        da = all_results["variants"][vname]["dead_feature_analysis"]
        print(f"    {vname}: " + " | ".join(
            f"Q{q+1}={da.get(f'Q{q+1}_alive', 0)}" for q in range(N_QUARTILES)
        ))

    print(f"\n  Hypothesis test:")
    std_agg = all_results["variants"].get("standard", {}).get("gradient_analysis", {})
    cos_agg = all_results["variants"].get("cosine_adaptive", {}).get("gradient_analysis", {})
    std_ratio = std_agg.get("per_feature_Q4_Q1_ratio_mean", 0)
    cos_ratio = cos_agg.get("per_feature_Q4_Q1_ratio_mean", 0)
    if std_ratio > 0 and cos_ratio > 0:
        print(f"    Standard Q4/Q1 domination: {std_ratio:.2f}x")
        print(f"    Cosine Q4/Q1 domination:   {cos_ratio:.2f}x")
        print(f"    Ratio of ratios:            {std_ratio/cos_ratio:.2f}x")
        if std_ratio > 2 * cos_ratio:
            print(f"    CONFIRMED: Standard gradients are {std_ratio/cos_ratio:.1f}x more "
                  f"norm-dominated than cosine")
        elif std_ratio > 1.5 * cos_ratio:
            print(f"    PARTIAL: Standard is {std_ratio/cos_ratio:.1f}x more dominated "
                  f"(weaker than predicted)")
        else:
            print(f"    REJECTED: Both variants show similar gradient domination patterns")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
