"""
Experiment 43b: Why does NoC have 4.3% dead features while others have 0%?

Empirical investigation into the architectural cause of NoC's persistent dead
features despite aux k-loss. We have 4 architectures trained with identical
recipe — the ONLY difference is the encoder architecture. All share:
- Same aux k-loss (auxk_alpha=1/32)
- Same decoder constraint (unit-norm)
- Same optimizer, LR, schedule

Hypotheses to test:

H1 (Gradient magnitude): Aux loss gradients are weaker for NoC because
    activations are bounded to [0,1] (cosine similarity) while standard/adaptive
    activations scale with input norm (~100). Weaker gradients = slower resurrection.

H2 (Activation competition): NoC dead features can't compete in BatchTopK
    because their pre-topk values are bounded by 1.0, while live features
    are also bounded — so the threshold is tight and dead features can't break in.

H3 (Decoder coupling): NoC normalizes both encoder AND decoder, creating
    a double constraint that limits the directions dead features can learn.
    Standard only constrains the decoder.

H4 (Training dynamics): Dead features in NoC emerge at a specific training
    phase and then get locked. Check intermediate checkpoints.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp43b_noc_dead_investigation.py \
        2>&1 | tee experiments/exp43b_output.log &
"""

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN SDPA backend — broken on H100 driver 595.58
torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 18
D_MODEL = 4096
D_SAE = 65536
K = 80
NORM_EPS = 1e-8
BATCH_SIZE = 2048
CTX_LEN = 2048
N_ANALYSIS_TOKENS = 500_000
DEAD_FEATURE_THRESHOLD = 10_000_000
TOP_K_AUX = D_MODEL // 2
AUXK_ALPHA = 1 / 32

# Checkpoints
CKPTS = {
    "standard": ("checkpoints/exp40/standard_L18_final.pt", "standard"),
    "adaptive_l2": ("checkpoints/exp40/adaptive_l2_L18_final.pt", "adaptive"),
    "perfeature_l2": ("checkpoints/exp40/perfeature_l2_L18_final.pt", "perfeature"),
    "no_C": ("/mnt/nvme0/checkpoints/exp42c/no_C_L18_final.pt", "noc"),
}

# Intermediate NoC checkpoints for training dynamics analysis
NOC_INTERMEDIATES = [
    ("/mnt/nvme0/checkpoints/exp42c/no_C_L18_step24414.pt", 24414, "50M"),
    ("/mnt/nvme0/checkpoints/exp42c/no_C_L18_step48828.pt", 48828, "100M"),
    ("/mnt/nvme0/checkpoints/exp42c/no_C_L18_step97656.pt", 97656, "200M"),
    ("/mnt/nvme0/checkpoints/exp42c/no_C_L18_step146484.pt", 146484, "300M"),
    ("/mnt/nvme0/checkpoints/exp42c/no_C_L18_step195312.pt", 195312, "400M"),
    ("/mnt/nvme0/checkpoints/exp42c/no_C_L18_step244140.pt", 244140, "500M"),
]

STD_INTERMEDIATES = [
    ("checkpoints/exp40/standard_L18_step24414.pt", 24414, "50M"),
    ("checkpoints/exp40/standard_L18_step48828.pt", 48828, "100M"),
    ("checkpoints/exp40/standard_L18_step97656.pt", 97656, "200M"),
    ("checkpoints/exp40/standard_L18_step146484.pt", 146484, "300M"),
    ("checkpoints/exp40/standard_L18_step195312.pt", 195312, "400M"),
    ("checkpoints/exp40/standard_L18_step244140.pt", 244140, "500M"),
]

RESULTS_PATH = "experiments/exp43b_results.json"


# =============================================================================
# SAE Architectures (eval only — no training)
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

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_pre_topk=False):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.threshold < 0:
            encoded = self._batch_topk(post_relu)
        else:
            encoded = post_relu * (post_relu > self.threshold)
        if return_pre_topk:
            return encoded, post_relu
        return encoded

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
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

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_pre_topk=False):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.threshold < 0:
            encoded = self._batch_topk(post_relu)
        else:
            encoded = post_relu * (post_relu > self.threshold)
        if return_pre_topk:
            return encoded, post_relu
        return encoded

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
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

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_pre_topk=False):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        log_norm = torch.log(input_norm)
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.threshold < 0:
            encoded = self._batch_topk(post_relu)
        else:
            encoded = post_relu * (post_relu > self.threshold)
        if return_pre_topk:
            return encoded, post_relu
        return encoded

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


class NoCBatchTopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.tensor(-1.0))

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_pre_topk=False):
        x_c = x - self.b_dec
        x_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_u = x_c / x_norm
        w_u = F.normalize(self.W_enc, dim=-1, eps=NORM_EPS)
        post_relu = F.relu(x_u @ w_u.T)
        if self.threshold < 0:
            encoded = self._batch_topk(post_relu)
        else:
            encoded = post_relu * (post_relu > self.threshold)
        self._cached_x_norm = x_norm
        if return_pre_topk:
            return encoded, post_relu
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

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


SAE_CLASSES = {
    "standard": BatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
    "perfeature_l2": PerFeatureAdaptiveCosineSAE,
    "no_C": NoCBatchTopKSAE,
}


def load_sae(name, path):
    cls = SAE_CLASSES[name]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae, ckpt.get("num_tokens_since_fired")


# =============================================================================
# Activation collection
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


def collect_activations(model, tokenizer, n_tokens):
    print(f"  Collecting {n_tokens:,} activations at L{LAYER}...")
    t0 = time.time()
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                       split="train", streaming=True)
    text_iter = iter(ds)
    all_acts = []
    tokens_collected = 0
    while tokens_collected < n_tokens:
        batch_texts = []
        for _ in range(4):
            try:
                row = next(text_iter)
                if len(row["text"]) > 50:
                    batch_texts.append(row["text"][:8192])
            except StopIteration:
                break
        if not batch_texts:
            break
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=CTX_LEN).to(DEVICE)
        acts = _collect_layer_acts(model, LAYER, inputs)
        flat = acts[inputs["attention_mask"].bool()]
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            flat = flat[norms < median * 10.0]
        all_acts.append(flat.to("cpu", dtype=DTYPE))
        tokens_collected += flat.shape[0]
    result = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"    {result.shape[0]:,} tokens in {time.time()-t0:.1f}s")
    return result


# =============================================================================
# Investigation 1: Pre-TopK Activation Distributions
# =============================================================================

@torch.no_grad()
def investigate_activation_distributions(saes, activations):
    """Compare pre-topk activation distributions for live vs dead features."""
    print("\n=== H1+H2: Pre-TopK Activation Distributions ===")
    results = {}

    for name, (sae, num_tokens_since_fired) in saes.items():
        print(f"\n  {name}:")
        sae.eval()

        # Identify dead features
        if num_tokens_since_fired is not None:
            dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
        else:
            # Empirically determine from data
            ever_fired = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
            for i in range(0, activations.shape[0], BATCH_SIZE):
                batch = activations[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
                features = sae.encode(batch)
                ever_fired |= (features > 0).any(dim=0)
            dead_mask = ~ever_fired

        n_dead = int(dead_mask.sum())
        n_alive = D_SAE - n_dead
        print(f"    Dead: {n_dead} ({n_dead/D_SAE*100:.1f}%), Alive: {n_alive}")

        # Collect pre-topk activations for dead vs alive features
        all_pre_topk_dead = []
        all_pre_topk_alive = []
        all_post_topk = []

        for i in range(0, min(activations.shape[0], 50000), BATCH_SIZE):
            batch = activations[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
            encoded, pre_topk = sae.encode(batch, return_pre_topk=True)

            # Pre-topk values for dead features
            if n_dead > 0:
                dead_vals = pre_topk[:, dead_mask]
                all_pre_topk_dead.append(dead_vals.cpu())

            # Pre-topk values for alive features
            alive_vals = pre_topk[:, ~dead_mask]
            all_pre_topk_alive.append(alive_vals.cpu())

            # Post-topk (winning) values
            all_post_topk.append(encoded.cpu())

        alive_pre = torch.cat(all_pre_topk_alive, dim=0).float()
        post = torch.cat(all_post_topk, dim=0).float()

        # Compute threshold (min post-topk value across all tokens)
        active_vals = post[post > 0]
        threshold_empirical = active_vals.min().item() if active_vals.numel() > 0 else 0

        # Alive feature pre-topk statistics
        alive_max_per_feat = alive_pre.max(dim=0).values
        alive_mean_per_feat = alive_pre.mean(dim=0)

        # Sample for quantile (full tensor too large for quantile())
        SAMPLE_SIZE = 10_000_000
        alive_flat = alive_pre.reshape(-1)
        if alive_flat.numel() > SAMPLE_SIZE:
            idx = torch.randint(0, alive_flat.numel(), (SAMPLE_SIZE,))
            alive_p99 = float(alive_flat[idx].quantile(0.99))
        else:
            alive_p99 = float(alive_flat.quantile(0.99))

        entry = {
            "n_dead": n_dead,
            "n_alive": n_alive,
            "dead_pct": round(n_dead / D_SAE * 100, 2),
            "threshold_empirical": threshold_empirical,
            "alive_pre_topk_mean": float(alive_pre.mean()),
            "alive_pre_topk_max": float(alive_pre.max()),
            "alive_pre_topk_std": float(alive_pre.std()),
            "alive_pre_topk_p99": alive_p99,
        }

        if n_dead > 0:
            dead_pre = torch.cat(all_pre_topk_dead, dim=0).float()
            dead_max_per_feat = dead_pre.max(dim=0).values
            dead_mean_per_feat = dead_pre.mean(dim=0)

            dead_flat = dead_pre.reshape(-1)
            if dead_flat.numel() > SAMPLE_SIZE:
                idx = torch.randint(0, dead_flat.numel(), (SAMPLE_SIZE,))
                dead_p99 = float(dead_flat[idx].quantile(0.99))
            else:
                dead_p99 = float(dead_flat.quantile(0.99))

            entry.update({
                "dead_pre_topk_mean": float(dead_pre.mean()),
                "dead_pre_topk_max": float(dead_pre.max()),
                "dead_pre_topk_std": float(dead_pre.std()),
                "dead_pre_topk_p99": dead_p99,
                "dead_max_above_threshold": int((dead_max_per_feat > threshold_empirical).sum()),
                "dead_max_above_threshold_pct": float((dead_max_per_feat > threshold_empirical).float().mean() * 100),
                "gap_alive_dead_mean": float(alive_mean_per_feat.mean() - dead_mean_per_feat.mean()),
                "gap_alive_dead_max": float(alive_max_per_feat.mean() - dead_max_per_feat.mean()),
            })

            print(f"    Threshold: {threshold_empirical:.4f}")
            print(f"    Alive pre-topk: mean={alive_pre.mean():.4f}, max={alive_pre.max():.4f}, p99={alive_p99:.4f}")
            print(f"    Dead pre-topk:  mean={dead_pre.mean():.4f}, max={dead_pre.max():.4f}, p99={dead_p99:.4f}")
            print(f"    Dead features with max > threshold: {entry['dead_max_above_threshold']}/{n_dead} ({entry['dead_max_above_threshold_pct']:.1f}%)")
        else:
            print(f"    No dead features — skipping dead analysis")
            print(f"    Alive pre-topk: mean={alive_pre.mean():.4f}, max={alive_pre.max():.4f}")

        results[name] = entry

    return results


# =============================================================================
# Investigation 2: Aux Loss Gradient Magnitude
# =============================================================================

def investigate_aux_gradient_magnitude(saes, activations):
    """Simulate aux loss and measure gradient magnitudes for each architecture."""
    print("\n=== H1: Aux Loss Gradient Magnitude ===")
    results = {}

    for name, (sae, num_tokens_since_fired) in saes.items():
        print(f"\n  {name}:")
        sae.eval()

        if num_tokens_since_fired is None:
            print(f"    No num_tokens_since_fired — skipping")
            continue

        with torch.no_grad():
            dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
            n_dead = int(dead_mask.sum())
            if n_dead == 0:
                act_sums = torch.zeros(D_SAE, device=DEVICE)
                for i in range(0, min(activations.shape[0], 20000), BATCH_SIZE):
                    batch = activations[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
                    features = sae.encode(batch)
                    act_sums += features.sum(dim=0)
                _, least_active = act_sums.topk(2800, largest=False)
                dead_mask = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
                dead_mask[least_active] = True
                n_dead = 2800
                print(f"    No dead features — using {n_dead} least-active as proxy")

        # Compute aux loss gradients (needs grad enabled)
        sae.train()
        aux_grad_norms = []

        for i in range(0, min(activations.shape[0], 20000), BATCH_SIZE):
            batch = activations[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)

            with torch.no_grad():
                x_hat, features = sae(batch)
                residual = batch - x_hat

            # Forward with gradients for aux loss
            encoded, pre_topk = sae.encode(batch.detach(), return_pre_topk=True)

            k_aux = min(TOP_K_AUX, n_dead)
            auxk_latents = torch.where(
                dead_mask[None], pre_topk,
                torch.tensor(-torch.inf, device=DEVICE)
            )
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_buffer = torch.zeros_like(pre_topk)
            auxk_acts_BF = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            x_reconstruct_aux = auxk_acts_BF @ sae.W_dec
            auxk_l2 = (residual.detach().float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()

            residual_mu = residual.detach().mean(dim=0, keepdim=True)
            loss_denom = (residual.detach().float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            auxk_loss = (auxk_l2 / loss_denom.clamp(min=1e-8)).nan_to_num(0.0)

            sae.zero_grad()
            (AUXK_ALPHA * auxk_loss).backward()

            if sae.W_enc.grad is not None:
                dead_enc_grad = sae.W_enc.grad[dead_mask]
                grad_norm = dead_enc_grad.norm(dim=-1).mean().item()
                aux_grad_norms.append(grad_norm)

        sae.eval()

        entry = {
            "n_dead_or_proxy": n_dead,
            "mean_aux_grad_norm": float(np.mean(aux_grad_norms)) if aux_grad_norms else 0,
            "max_aux_grad_norm": float(np.max(aux_grad_norms)) if aux_grad_norms else 0,
            "aux_loss_value": auxk_loss.item() if auxk_loss is not None else 0,
        }

        print(f"    Mean aux grad norm (dead enc): {entry['mean_aux_grad_norm']:.6f}")
        print(f"    Max aux grad norm: {entry['max_aux_grad_norm']:.6f}")
        print(f"    Aux loss value: {entry['aux_loss_value']:.6f}")

        results[name] = entry

    return results


# =============================================================================
# Investigation 3: Aux Reconstruction Quality
# =============================================================================

@torch.no_grad()
def investigate_aux_reconstruction(saes, activations):
    """How well can dead features reconstruct the residual via W_dec?"""
    print("\n=== H1: Aux Reconstruction Quality ===")
    results = {}

    for name, (sae, num_tokens_since_fired) in saes.items():
        print(f"\n  {name}:")
        sae.eval()

        # Get residuals and aux reconstructions
        residual_norms = []
        aux_recon_norms = []
        aux_fves = []

        if num_tokens_since_fired is not None:
            dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
        else:
            dead_mask = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)

        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            print(f"    No dead features — measuring aux recon with proxy dead set")
            # Use least active features
            act_sums = torch.zeros(D_SAE, device=DEVICE)
            for i in range(0, min(activations.shape[0], 20000), BATCH_SIZE):
                batch = activations[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
                features = sae.encode(batch)
                act_sums += features.sum(dim=0)
            _, least_active = act_sums.topk(2800, largest=False)
            dead_mask = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
            dead_mask[least_active] = True
            n_dead = 2800

        for i in range(0, min(activations.shape[0], 20000), BATCH_SIZE):
            batch = activations[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
            x_hat, features = sae(batch)
            residual = batch - x_hat

            # Get dead feature pre-topk activations
            _, pre_topk = sae.encode(batch, return_pre_topk=True)

            k_aux = min(TOP_K_AUX, n_dead)
            auxk_latents = torch.where(
                dead_mask[None], pre_topk,
                torch.tensor(-torch.inf, device=DEVICE)
            )
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_buffer = torch.zeros_like(pre_topk)
            auxk_acts_BF = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Aux reconstruction
            x_reconstruct_aux = auxk_acts_BF @ sae.W_dec

            residual_norms.append(residual.norm(dim=-1).mean().item())
            aux_recon_norms.append(x_reconstruct_aux.norm(dim=-1).mean().item())

            # Aux FVE (how much of residual variance does aux capture?)
            total_var = torch.var(residual, dim=0, unbiased=False).sum()
            aux_resid = residual - x_reconstruct_aux
            resid_var = torch.var(aux_resid, dim=0, unbiased=False).sum()
            aux_fve = (1 - resid_var / total_var.clamp(min=1e-8)).item()
            aux_fves.append(aux_fve)

        entry = {
            "n_dead": n_dead,
            "mean_residual_norm": float(np.mean(residual_norms)),
            "mean_aux_recon_norm": float(np.mean(aux_recon_norms)),
            "aux_recon_to_residual_ratio": float(np.mean(aux_recon_norms)) / max(float(np.mean(residual_norms)), 1e-8),
            "mean_aux_fve": float(np.mean(aux_fves)),
        }

        print(f"    Residual norm: {entry['mean_residual_norm']:.2f}")
        print(f"    Aux recon norm: {entry['mean_aux_recon_norm']:.2f}")
        print(f"    Ratio (aux/residual): {entry['aux_recon_to_residual_ratio']:.4f}")
        print(f"    Aux FVE of residual: {entry['mean_aux_fve']:.4f}")

        results[name] = entry

    return results


# =============================================================================
# Investigation 4: Training Dynamics from Checkpoints
# =============================================================================

@torch.no_grad()
def investigate_training_dynamics(activations):
    """Track dead features across intermediate checkpoints for NoC vs standard."""
    print("\n=== H4: Training Dynamics from Checkpoints ===")
    results = {"no_C": [], "standard": []}

    for ckpt_path, step, label in NOC_INTERMEDIATES:
        if not os.path.exists(ckpt_path):
            print(f"    NoC {label}: checkpoint not found, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        ntsf = ckpt.get("num_tokens_since_fired")
        if ntsf is not None:
            dead = int((ntsf >= DEAD_FEATURE_THRESHOLD).sum())
            never_fired = int((ntsf >= step * BATCH_SIZE).sum())  # never fired at all
        else:
            dead = -1
            never_fired = -1
        entry = {"step": step, "label": label, "n_dead": dead, "n_never_fired": never_fired}
        results["no_C"].append(entry)
        print(f"    NoC {label}: dead={dead}, never_fired={never_fired}")

    for ckpt_path, step, label in STD_INTERMEDIATES:
        if not os.path.exists(ckpt_path):
            print(f"    Standard {label}: checkpoint not found, skipping")
            continue
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        ntsf = ckpt.get("num_tokens_since_fired")
        if ntsf is not None:
            dead = int((ntsf >= DEAD_FEATURE_THRESHOLD).sum())
            never_fired = int((ntsf >= step * BATCH_SIZE).sum())
        else:
            dead = -1
            never_fired = -1
        entry = {"step": step, "label": label, "n_dead": dead, "n_never_fired": never_fired}
        results["standard"].append(entry)
        print(f"    Standard {label}: dead={dead}, never_fired={never_fired}")

    return results


# =============================================================================
# Investigation 5: Encoder/Decoder Weight Analysis
# =============================================================================

@torch.no_grad()
def investigate_weight_structure(saes):
    """Compare encoder/decoder weight properties for dead vs alive features."""
    print("\n=== H3: Weight Structure Analysis ===")
    results = {}

    for name, (sae, num_tokens_since_fired) in saes.items():
        print(f"\n  {name}:")

        if num_tokens_since_fired is not None:
            dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
        else:
            dead_mask = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)

        n_dead = int(dead_mask.sum())

        # Encoder weight norms
        enc_norms = sae.W_enc.norm(dim=-1)
        dec_norms = sae.W_dec.norm(dim=-1)

        # Encoder-decoder alignment (how similar are enc and dec directions?)
        enc_unit = F.normalize(sae.W_enc, dim=-1)
        dec_unit = F.normalize(sae.W_dec, dim=-1)
        enc_dec_cos = (enc_unit * dec_unit).sum(dim=-1)  # per-feature cosine

        entry = {
            "enc_norm_mean": float(enc_norms.mean()),
            "enc_norm_std": float(enc_norms.std()),
            "dec_norm_mean": float(dec_norms.mean()),
            "dec_norm_std": float(dec_norms.std()),
            "enc_dec_alignment_mean": float(enc_dec_cos.mean()),
            "enc_dec_alignment_std": float(enc_dec_cos.std()),
        }

        if n_dead > 0:
            entry.update({
                "dead_enc_norm_mean": float(enc_norms[dead_mask].mean()),
                "alive_enc_norm_mean": float(enc_norms[~dead_mask].mean()),
                "dead_dec_norm_mean": float(dec_norms[dead_mask].mean()),
                "alive_dec_norm_mean": float(dec_norms[~dead_mask].mean()),
                "dead_enc_dec_alignment": float(enc_dec_cos[dead_mask].mean()),
                "alive_enc_dec_alignment": float(enc_dec_cos[~dead_mask].mean()),
            })
            print(f"    Dead enc norm: {entry['dead_enc_norm_mean']:.4f} vs alive: {entry['alive_enc_norm_mean']:.4f}")
            print(f"    Dead dec norm: {entry['dead_dec_norm_mean']:.4f} vs alive: {entry['alive_dec_norm_mean']:.4f}")
            print(f"    Dead enc-dec alignment: {entry['dead_enc_dec_alignment']:.4f} vs alive: {entry['alive_enc_dec_alignment']:.4f}")
        else:
            print(f"    Enc norm: mean={entry['enc_norm_mean']:.4f} std={entry['enc_norm_std']:.4f}")
            print(f"    Dec norm: mean={entry['dec_norm_mean']:.4f} std={entry['dec_norm_std']:.4f}")
            print(f"    Enc-dec alignment: mean={entry['enc_dec_alignment_mean']:.4f}")

        results[name] = entry

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 43b: NoC Dead Feature Investigation")
    print(f"  Model: {MODEL_NAME}, Layer {LAYER}")
    print(f"  Analyzing {N_ANALYSIS_TOKENS:,} tokens\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()

    print("Collecting activations...")
    activations = collect_activations(model, tokenizer, N_ANALYSIS_TOKENS)

    # Free model memory — we only need SAEs from here
    del model
    torch.cuda.empty_cache()

    print("\nLoading SAEs...")
    saes = {}
    for name, (path, _) in CKPTS.items():
        sae, ntsf = load_sae(name, path)
        saes[name] = (sae, ntsf)
        n_dead = int((ntsf >= DEAD_FEATURE_THRESHOLD).sum()) if ntsf is not None else "?"
        print(f"  {name}: loaded, dead={n_dead}")

    results = {}

    # Run all investigations
    results["activation_distributions"] = investigate_activation_distributions(saes, activations)
    results["aux_reconstruction"] = investigate_aux_reconstruction(saes, activations)
    results["weight_structure"] = investigate_weight_structure(saes)
    results["training_dynamics"] = investigate_training_dynamics(activations)
    results["aux_gradient_magnitude"] = investigate_aux_gradient_magnitude(saes, activations)

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Print summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    ad = results["activation_distributions"]
    print("\n  Pre-TopK Activation Range:")
    for name in ["standard", "adaptive_l2", "perfeature_l2", "no_C"]:
        d = ad[name]
        print(f"    {name:20s}: alive_mean={d['alive_pre_topk_mean']:.4f}, "
              f"alive_max={d['alive_pre_topk_max']:.4f}, "
              f"threshold={d['threshold_empirical']:.4f}")

    ar = results["aux_reconstruction"]
    print("\n  Aux Reconstruction Capacity:")
    for name in ["standard", "adaptive_l2", "perfeature_l2", "no_C"]:
        d = ar[name]
        print(f"    {name:20s}: aux_norm/residual_norm={d['aux_recon_to_residual_ratio']:.4f}, "
              f"aux_fve={d['mean_aux_fve']:.4f}")

    ag = results["aux_gradient_magnitude"]
    print("\n  Aux Gradient Magnitude (dead encoder weights):")
    for name in ["standard", "adaptive_l2", "perfeature_l2", "no_C"]:
        if name in ag:
            d = ag[name]
            print(f"    {name:20s}: mean_grad_norm={d['mean_aux_grad_norm']:.6f}")

    td = results["training_dynamics"]
    if td["no_C"]:
        print("\n  NoC Dead Feature Trajectory:")
        for entry in td["no_C"]:
            print(f"    {entry['label']:>5s}: dead={entry['n_dead']:,}")


if __name__ == "__main__":
    main()
