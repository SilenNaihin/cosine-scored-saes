"""
Experiment 29: Post-Hoc Encoder Normalization
==============================================

Does the cosine SAE advantage come from training-time dynamics or
inference-time geometry? Take a trained STANDARD SAE and normalize its
encoder weights at inference:

    pre_acts = learned_scale * cos_sim(x - b_dec, W_enc) + b_enc

If this recovers cosine-<author>el FVE/alive features, the advantage is purely
geometric and could be applied to any existing SAE. If it doesn't, then
cosine learns fundamentally different features during training (already
suggested by exp15's <1% feature overlap).

No training needed — just load exp17 standard checkpoints, apply post-hoc
normalization, and evaluate.

Run on <gpu-server> GPU 1.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/exp29_posthoc_normalization.py > experiments/exp29_output.log 2>&1 &
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

MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]
D_MODEL = 4096
D_SAE = 16384
K = 80

N_EVAL_TOKENS = 500_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0
BATCH_SIZE = 4096

N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 50

CHECKPOINT_DIR = Path("checkpoints/exp17")
RESULTS_PATH = "experiments/exp29_results.json"


# =============================================================================
# SAE Classes (must match exp17 exactly for checkpoint loading)
# =============================================================================

class BatchTopKSAE(nn.Module):
    """Standard BatchTopK SAE — loaded from exp17 checkpoints."""

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        if self.training:
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
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Adaptive cosine SAE — loaded from exp17 checkpoints as reference."""

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
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


class PostHocCosineSAE(nn.Module):
    """Standard SAE with post-hoc cosine normalization applied at inference.

    Uses the SAME weights as a trained standard SAE, but replaces the encoder
    with cosine similarity + learned global scale:
        pre_acts = scale * cos_sim(x - b_dec, W_enc) + b_enc

    The scale is optimized on eval data to find the best match.
    """

    def __init__(self, standard_sae: BatchTopKSAE, scale: float):
        super().__init__()
        self.d_model = standard_sae.d_model
        self.d_sae = standard_sae.d_sae
        self.k = standard_sae.k

        # Copy weights from standard SAE (frozen, no grad)
        self.W_enc = nn.Parameter(standard_sae.W_enc.data.clone(), requires_grad=False)
        self.b_enc = nn.Parameter(standard_sae.b_enc.data.clone(), requires_grad=False)
        self.W_dec = nn.Parameter(standard_sae.W_dec.data.clone(), requires_grad=False)
        self.b_dec = nn.Parameter(standard_sae.b_dec.data.clone(), requires_grad=False)
        self.register_buffer("threshold", standard_sae.threshold.data.clone())

        # Learnable scale for cosine similarity
        self.scale_b = nn.Parameter(torch.tensor(math.log(scale)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        scale = torch.exp(self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        # Always use threshold (eval mode)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


# =============================================================================
# Scale Optimization
# =============================================================================

def optimize_posthoc_scale(standard_sae, eval_data, n_scales=20):
    """Find the best global scale for post-hoc cosine normalization.

    Sweep over scales and pick the one that maximizes FVE.
    """
    print("    Optimizing post-hoc scale...")
    mean_norm = eval_data.float().norm(dim=-1).mean().item()
    sqrt_d = math.sqrt(D_MODEL)

    # Sweep: log-spaced around mean_norm and sqrt_d
    candidates = np.logspace(
        np.log10(max(sqrt_d * 0.1, 1)),
        np.log10(mean_norm * 5),
        n_scales
    )

    best_fve = -float("inf")
    best_scale = mean_norm

    sample = eval_data[:min(50_000, eval_data.shape[0])]

    for scale in candidates:
        posthoc = PostHocCosineSAE(standard_sae, scale).to(DEVICE)
        posthoc.eval()

        total_var_sum, resid_var_sum = 0.0, 0.0
        for i in range(0, sample.shape[0], BATCH_SIZE):
            batch = sample[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                x_hat, _ = posthoc(batch)
            total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
            resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()

        fve = 1 - resid_var_sum / total_var_sum if total_var_sum > 0 else 0
        if fve > best_fve:
            best_fve = fve
            best_scale = scale

        del posthoc
        torch.cuda.empty_cache()

    print(f"    Best scale: {best_scale:.1f} (FVE={best_fve:.4f})")
    print(f"    For reference: mean_norm={mean_norm:.1f}, sqrt(d)={sqrt_d:.1f}")
    return best_scale, best_fve


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


def collect_activations(model, tokenizer, layer_idx, n_tokens):
    print(f"  Collecting eval activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)

    # Skip 200k docs to match exp17 eval split
    for i, _ in enumerate(text_iter):
        if i >= 200_000:
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
    print(f"  Layer {layer_idx}: {result.shape[0]:,} tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={norms.mean():.1f}, std={norms.std():.1f})")
    return result


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data):
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

    results = {
        "recon_loss_l2": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
        "dead_frac": dead_frac,
    }
    print(f"    [{name:>20s}] FVE={results['fve']:.4f} | dead={dead_frac*100:.1f}% | "
          f"L0={results['l0']:.0f} | cos={results['cos_recon']:.4f}")
    return results


def ablate_feature_kl(model, activation, feature_dir, layer_idx):
    projection = (activation @ feature_dir) * feature_dir
    model_dtype = next(model.parameters()).dtype
    x = activation.unsqueeze(0).unsqueeze(0).to(model_dtype)
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0).to(model_dtype)

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
    print(f"\n    Ablation [{name}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    n_probe = min(100_000, eval_data.shape[0])
    probe = eval_data[:n_probe]
    all_feats = []
    for i in range(0, n_probe, BATCH_SIZE):
        batch = probe[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        with torch.no_grad():
            _, f = sae(batch)
        all_feats.append(f.detach().cpu())
    all_feats = torch.cat(all_feats, dim=0)

    freq = (all_feats > 0).float().mean(dim=0)
    alive_mask = freq > 0
    n_alive = alive_mask.sum().item()
    print(f"    [{name}] {n_alive} alive features (of {D_SAE})")

    n_to_select = min(N_ABLATION_FEATURES, n_alive)
    if n_to_select == 0:
        return {"features": [], "aggregate": {"n_features": 0}}

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

        cos_arr, norm_arr = np.array(cos_v), np.array(norm_v)
        inner_arr, sae_arr = np.array(inner_v), np.array(sae_v)

        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]

        result = {
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_inner_kl": float(corr_inner),
            "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
        }
        feature_results.append(result)

        if rank < 5:
            print(f"      feat {fi:>5d} | cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f}")

    if not feature_results:
        return {"features": [], "aggregate": {"n_features": 0}}

    n = len(feature_results)
    agg = {
        "n_features": n,
        "cos_kl_mean": float(np.mean([r["corr_cos_kl"] for r in feature_results])),
        "inner_kl_mean": float(np.mean([r["corr_inner_kl"] for r in feature_results])),
        "sae_kl_mean": float(np.mean([r["corr_sae_kl"] for r in feature_results])),
        "cos_wins_inner": sum(r["cos_wins_inner"] for r in feature_results),
    }
    print(f"    [{name}] Summary ({n} features): "
          f"cos→KL={agg['cos_kl_mean']:.4f} | inner→KL={agg['inner_kl_mean']:.4f} | "
          f"cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# =============================================================================
# Feature Overlap Analysis
# =============================================================================

@torch.no_grad()
def feature_overlap(name_a, sae_a, name_b, sae_b, eval_data):
    """Measure feature overlap between two SAEs: how many features activate
    on the same tokens?"""
    print(f"\n  Feature overlap: {name_a} vs {name_b}")
    sae_a.eval()
    sae_b.eval()

    n_probe = min(50_000, eval_data.shape[0])
    alive_a = torch.zeros(D_SAE, dtype=torch.bool)
    alive_b = torch.zeros(D_SAE, dtype=torch.bool)

    # Collect which features are alive
    for i in range(0, n_probe, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        _, fa = sae_a(batch)
        _, fb = sae_b(batch)
        alive_a |= (fa > 0).any(dim=0).cpu()
        alive_b |= (fb > 0).any(dim=0).cpu()

    n_alive_a = alive_a.sum().item()
    n_alive_b = alive_b.sum().item()
    both = (alive_a & alive_b).sum().item()

    print(f"    {name_a}: {n_alive_a} alive | {name_b}: {n_alive_b} alive | "
          f"Both: {both} | Jaccard: {both / max(n_alive_a + n_alive_b - both, 1):.3f}")

    # Decoder direction similarity: for each alive feature in A, find most
    # similar feature direction in B
    if n_alive_a > 0 and n_alive_b > 0:
        dirs_a = F.normalize(sae_a.W_dec[alive_a].float(), dim=1)
        dirs_b = F.normalize(sae_b.W_dec[alive_b].float(), dim=1)
        cos_matrix = dirs_a @ dirs_b.T  # [n_alive_a, n_alive_b]
        max_cos, _ = cos_matrix.max(dim=1)
        print(f"    Decoder direction overlap (A→B max cos): "
              f"mean={max_cos.mean():.3f}, median={max_cos.median():.3f}, "
              f">0.9: {(max_cos > 0.9).sum().item()}, >0.95: {(max_cos > 0.95).sum().item()}")

    return {
        "alive_a": n_alive_a, "alive_b": n_alive_b, "both_alive": both,
        "jaccard": both / max(n_alive_a + n_alive_b - both, 1),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 29: Post-Hoc Encoder Normalization")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Question: Does normalizing a trained standard SAE's encoder at")
    print(f"inference recover cosine-<author>el performance?")

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

    all_results = {
        "config": {
            "model_name": MODEL_NAME,
            "layers": LAYERS,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "n_eval_tokens": N_EVAL_TOKENS,
            "checkpoint_source": "exp17 (50M tokens)",
        },
        "layers": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Check checkpoint exists
        std_path = CHECKPOINT_DIR / f"standard_L{layer_idx}_final.pt"
        ada_path = CHECKPOINT_DIR / f"adaptive_l2_L{layer_idx}_final.pt"
        if not std_path.exists():
            print(f"  WARNING: {std_path} not found, skipping layer {layer_idx}")
            continue

        # Collect eval activations
        eval_data = collect_activations(model, tokenizer, layer_idx, N_EVAL_TOKENS)
        mean_norm = eval_data.float().norm(dim=-1).mean().item()

        # Load standard SAE
        print(f"\n  Loading standard SAE from {std_path}")
        std_sae = BatchTopKSAE(D_MODEL, D_SAE, K)
        std_sae.load_state_dict(torch.load(std_path, map_location="cpu", weights_only=True))
        std_sae = std_sae.to(DEVICE).eval()

        # Load adaptive_l2 SAE (reference)
        ada_sae = None
        if ada_path.exists():
            print(f"  Loading adaptive_l2 SAE from {ada_path}")
            ada_sae = AdaptiveCosineBatchTopKSAE(D_MODEL, D_SAE, K)
            ada_sae.load_state_dict(torch.load(ada_path, map_location="cpu", weights_only=True))
            ada_sae = ada_sae.to(DEVICE).eval()

        # Optimize post-hoc scale
        best_scale, _ = optimize_posthoc_scale(std_sae, eval_data)

        # Create post-hoc cosine SAE
        posthoc_sae = PostHocCosineSAE(std_sae, best_scale).to(DEVICE).eval()

        layer_results = {
            "mean_norm": mean_norm,
            "posthoc_scale": best_scale,
        }

        # === Evaluate all variants ===
        variants = [
            ("standard", std_sae),
            ("posthoc_cosine", posthoc_sae),
        ]
        if ada_sae is not None:
            variants.append(("adaptive_l2_ref", ada_sae))

        print(f"\n  --- Reconstruction ---")
        for vname, sae in variants:
            recon = evaluate_reconstruction(vname, sae, eval_data)
            layer_results.setdefault(vname, {})["reconstruction"] = recon

        print(f"\n  --- Ablation ---")
        for vname, sae in variants:
            abl = evaluate_ablation(vname, model, sae, eval_data, layer_idx)
            layer_results[vname]["ablation"] = abl

        # Feature overlap: standard vs posthoc
        overlap_sp = feature_overlap("standard", std_sae, "posthoc_cosine", posthoc_sae, eval_data)
        layer_results["overlap_std_vs_posthoc"] = overlap_sp

        if ada_sae is not None:
            overlap_sa = feature_overlap("standard", std_sae, "adaptive_l2", ada_sae, eval_data)
            layer_results["overlap_std_vs_adaptive"] = overlap_sa

        all_results["layers"][str(layer_idx)] = layer_results

        # Save incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        del std_sae, posthoc_sae, ada_sae, eval_data
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Post-Hoc Normalization")
    print(f"{'='*70}")
    print(f"\n  Question: Does post-hoc cosine normalization recover cosine SAE performance?")

    print(f"\n  {'Layer':>5s}  {'Variant':<22s} {'FVE':>7s} {'Dead%':>7s} {'L0':>6s} "
          f"{'cos>inn':>8s}")
    print(f"  {'-'*5}  {'-'*22} {'-'*7} {'-'*7} {'-'*6} {'-'*8}")

    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        for vname in ["standard", "posthoc_cosine", "adaptive_l2_ref"]:
            r = lr.get(vname, {})
            recon = r.get("reconstruction", {})
            abl_agg = r.get("ablation", {}).get("aggregate", {})
            fve = recon.get("fve", 0)
            dead = recon.get("dead_frac", 1)
            l0 = recon.get("l0", 0)
            cos_wins = abl_agg.get("cos_wins_inner", 0)
            n_feats = abl_agg.get("n_features", 0)
            cos_str = f"{cos_wins}/{n_feats}" if n_feats > 0 else "N/A"
            print(f"  {layer_idx:>5d}  {vname:<22s} {fve:>7.4f} {dead*100:>6.1f}% "
                  f"{l0:>6.0f} {cos_str:>8s}")

    # Verdict
    print(f"\n  VERDICT:")
    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        std_fve = lr.get("standard", {}).get("reconstruction", {}).get("fve", 0)
        ph_fve = lr.get("posthoc_cosine", {}).get("reconstruction", {}).get("fve", 0)
        ada_fve = lr.get("adaptive_l2_ref", {}).get("reconstruction", {}).get("fve", 0)

        std_dead = lr.get("standard", {}).get("reconstruction", {}).get("dead_frac", 1)
        ph_dead = lr.get("posthoc_cosine", {}).get("reconstruction", {}).get("dead_frac", 1)
        ada_dead = lr.get("adaptive_l2_ref", {}).get("reconstruction", {}).get("dead_frac", 1)

        recovery = (ph_fve - std_fve) / max(ada_fve - std_fve, 1e-8) * 100 if ada_fve != std_fve else 0

        print(f"    L{layer_idx}: posthoc recovers {recovery:.0f}% of the cosine FVE gap "
              f"(std={std_fve:.4f} → posthoc={ph_fve:.4f} vs adaptive={ada_fve:.4f})")
        print(f"          dead: std={std_dead*100:.1f}% → posthoc={ph_dead*100:.1f}% "
              f"vs adaptive={ada_dead*100:.1f}%")

    print(f"\nResults: {RESULTS_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
