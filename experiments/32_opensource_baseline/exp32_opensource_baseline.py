"""
Experiment 32: Open-Source Standard SAE Baseline
==================================================

**Why this experiment matters:**

Every comparison in this project uses our own BatchTopKSAE implementation as
the "standard" baseline. If our implementation has a bug, suboptimal
hyperparameters, or training issue, the entire cosine advantage is an artifact
of comparing against a weak baseline.

The adamkarvonen/qwen3-8b-saes on HuggingFace are community-standard
BatchTopK SAEs trained on 500M tokens (10x our 50M) using the well-tested
saprmarks/dictionary_learning codebase. They use the same model, same layers,
same architecture (BatchTopK), same d_sae (16384), same k (80).

**What this gives us:**

1. **Implementation sanity check:** Our 50M-token standard SAE should have
   worse FVE than the 500M-token reference (less data), but should be in
   the right ballpark. If our SAE is dramatically worse, we have a bug.

2. **Gold standard reference point:** We can compare our cosine SAE directly
   against the community reference. If adaptive_l2 beats even the 500M-token
   standard SAE, that's a much stronger claim than beating our own standard.

3. **Dead feature context:** Our standard has 77% dead at L27 (50M tokens).
   The reference (500M tokens) should have far fewer dead features. This
   tells us how much of our dead feature problem is data budget vs architecture.

4. **The killer comparison:** If our adaptive_l2 (50M tokens, cosine) beats
   the reference standard (500M tokens, inner product) on any metric, that's
   a 10x sample efficiency claim for the paper.

**Design:**

- Load adamkarvonen trainer_0 (d_sae=16384, k=80) at L9, L18, L27
- Load our exp17 standard and adaptive_l2 checkpoints (50M tokens)
- Run identical eval on all: FVE, dead%, L0, cos>inner ablation
- Also run feature overlap between reference and our SAEs
- All eval on fresh FineWeb data (same collection as exp17)

Run on <gpu-server> GPU 1.

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/exp32_opensource_baseline.py > experiments/exp32_output.log 2>&1 &
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
from huggingface_hub import hf_hub_download


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
STORAGE_DTYPE = torch.float16

# --- Model ---
MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]
D_MODEL = 4096

# --- SAE architecture ---
D_SAE = 16384  # 4x d_model
K = 80

# --- Data ---
N_EVAL_TOKENS = 500_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0

# --- Ablation ---
N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 50
BATCH_SIZE = 4096

# --- Reference SAE ---
HF_REPO = "adamkarvonen/qwen3-8b-saes"
# trainer_0 = d_sae=16384, k=80 (exact match for our setup)
TRAINER_ID = 0

# --- Our checkpoints ---
EXP17_DIR = Path("checkpoints/exp17")

# --- Output ---
RESULTS_PATH = "experiments/exp32_results.json"

# --- Seed ---
SEED = 42


# =============================================================================
# SAE Architectures (matching exp17 training code exactly)
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

    def encode(self, x):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


class AdaptiveCosineBatchTopKSAE(nn.Module):
    """Cosine encoder with adaptive per-token scale."""

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

    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


# =============================================================================
# Load SAEs
# =============================================================================

def load_adamkarvonen_sae(layer_idx):
    """Load the reference SAE from HuggingFace."""
    filename = (
        f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{layer_idx}"
        f"/trainer_{TRAINER_ID}/ae.pt"
    )
    print(f"    Downloading {filename}...")
    path = hf_hub_download(repo_id=HF_REPO, filename=filename)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Build a lightweight wrapper
    W_enc = ckpt["encoder.weight"]  # (d_sae, d_in)
    b_enc = ckpt["encoder.bias"]    # (d_sae,)
    W_dec_raw = ckpt["decoder.weight"]  # (d_in, d_sae) — transposed!
    b_dec = ckpt["b_dec"]           # (d_in,)
    k = int(ckpt["k"].item())
    threshold = float(ckpt["threshold"].item())

    sae = BatchTopKSAE(D_MODEL, D_SAE, K)
    with torch.no_grad():
        sae.W_enc.copy_(W_enc)
        sae.b_enc.copy_(b_enc)
        sae.W_dec.copy_(W_dec_raw.T)  # Transpose to (d_sae, d_in)
        sae.b_dec.copy_(b_dec)
        sae.threshold.fill_(threshold)

    sae = sae.to(DEVICE).eval()

    print(f"    Loaded: d_sae={W_enc.shape[0]}, k={k}, threshold={threshold:.4f}")
    print(f"    W_dec norms: mean={sae.W_dec.norm(dim=1).mean():.4f}, "
          f"std={sae.W_dec.norm(dim=1).std():.4f}")
    return sae


def load_exp17_sae(variant, layer_idx):
    """Load our trained SAE checkpoint."""
    path = EXP17_DIR / f"{variant}_L{layer_idx}_final.pt"
    if not path.exists():
        print(f"    WARNING: {path} not found, skipping")
        return None

    if variant == "standard":
        sae = BatchTopKSAE(D_MODEL, D_SAE, K)
    elif variant == "adaptive_l2":
        sae = AdaptiveCosineBatchTopKSAE(D_MODEL, D_SAE, K)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    sae.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    sae = sae.to(DEVICE).eval()

    print(f"    Loaded {variant} L{layer_idx}: threshold={sae.threshold.item():.4f}")
    if hasattr(sae, "scale_a"):
        print(f"    scale_a={sae.scale_a.item():.4f}, scale_b_exp={sae.scale_b.exp().item():.1f}")
    return sae


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


def collect_activations(model, tokenizer, layer_idx, n_tokens, skip_docs=200_000):
    """Collect eval activations (skip train docs to avoid overlap)."""
    print(f"  Collecting eval activations for layer {layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)

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
    print(f"  Layer {layer_idx}: {result.shape[0]:,} eval tokens in {time.time()-t0:.1f}s "
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
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0

    results = {
        "recon_loss_l2": float(np.mean(recon_losses)),
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "fve": float(1 - resid_var_sum / total_var_sum) if total_var_sum > 0 else 0,
        "dead_frac": dead_frac,
        "alive_count": alive_count,
    }
    print(f"    [{name:>20s}] FVE={results['fve']:.4f} | "
          f"cos={results['cos_recon']:.4f} | L0={results['l0']:.0f} | "
          f"dead={dead_frac:.3f} ({dead_frac*100:.1f}%) | alive={alive_count}")
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
            "corr_cos_kl": float(corr_cos), "corr_inner_kl": float(corr_inner),
            "corr_sae_kl": float(corr_sae), "corr_norm_kl": float(corr_norm),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
        }
        feature_results.append(result)

        if rank < 5 or rank % 10 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos->KL={corr_cos:.3f} | inner->KL={corr_inner:.3f} | "
                  f"SAE->KL={corr_sae:.3f}")

    if not feature_results:
        return {"features": [], "aggregate": {"n_features": 0}}

    n = len(feature_results)
    agg = {
        "n_features": n,
        "cos_kl_mean": float(np.mean([r["corr_cos_kl"] for r in feature_results])),
        "inner_kl_mean": float(np.mean([r["corr_inner_kl"] for r in feature_results])),
        "sae_kl_mean": float(np.mean([r["corr_sae_kl"] for r in feature_results])),
        "norm_kl_mean": float(np.mean([r["corr_norm_kl"] for r in feature_results])),
        "cos_wins_inner": sum(r["cos_wins_inner"] for r in feature_results),
    }

    print(f"    [{name}] Summary ({n} features): "
          f"cos->KL={agg['cos_kl_mean']:.4f} | inner->KL={agg['inner_kl_mean']:.4f} | "
          f"cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


@torch.no_grad()
def feature_overlap(sae_a, sae_b, eval_data, name_a, name_b):
    """Compare alive feature sets and decoder directions between two SAEs."""
    print(f"\n    Feature overlap: {name_a} vs {name_b}")

    n_probe = min(100_000, eval_data.shape[0])
    probe = eval_data[:n_probe]

    alive_a = torch.zeros(D_SAE, dtype=torch.bool)
    alive_b = torch.zeros(D_SAE, dtype=torch.bool)

    for i in range(0, n_probe, BATCH_SIZE):
        batch = probe[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        _, fa = sae_a(batch)
        _, fb = sae_b(batch)
        alive_a |= (fa > 0).any(dim=0).cpu()
        alive_b |= (fb > 0).any(dim=0).cpu()

    n_a = alive_a.sum().item()
    n_b = alive_b.sum().item()
    both = (alive_a & alive_b).sum().item()
    union = (alive_a | alive_b).sum().item()
    jaccard = both / union if union > 0 else 0

    # Decoder direction similarity: for each alive feature in A,
    # find its max cosine similarity to any alive feature in B
    a_idx = torch.where(alive_a)[0]
    b_idx = torch.where(alive_b)[0]
    if len(a_idx) > 0 and len(b_idx) > 0:
        dec_a = F.normalize(sae_a.W_dec[a_idx].float(), dim=-1)
        dec_b = F.normalize(sae_b.W_dec[b_idx].float(), dim=-1)
        # Compute max cosine similarity for each A feature to any B feature
        # Process in chunks to avoid OOM
        max_cos_a_to_b = []
        chunk_size = 1000
        for i in range(0, len(dec_a), chunk_size):
            chunk = dec_a[i:i+chunk_size]
            sims = chunk @ dec_b.T  # (chunk, n_b)
            max_cos_a_to_b.append(sims.max(dim=1).values)
        max_cos_a_to_b = torch.cat(max_cos_a_to_b)
        dec_cos_mean = max_cos_a_to_b.mean().item()
        dec_cos_median = max_cos_a_to_b.median().item()
        high_match = (max_cos_a_to_b > 0.9).float().mean().item()
    else:
        dec_cos_mean = float("nan")
        dec_cos_median = float("nan")
        high_match = float("nan")

    result = {
        "alive_a": n_a, "alive_b": n_b,
        "both_alive": both, "jaccard": jaccard,
        "dec_max_cos_mean": dec_cos_mean,
        "dec_max_cos_median": dec_cos_median,
        "frac_matched_gt90": high_match,
    }

    print(f"      {name_a}: {n_a} alive | {name_b}: {n_b} alive | "
          f"Jaccard: {jaccard:.3f} | max_cos: mean={dec_cos_mean:.3f}, "
          f"median={dec_cos_median:.3f} | >0.9 match: {high_match:.1%}")
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 32: Open-Source Standard SAE Baseline")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Reference: {HF_REPO} trainer_{TRAINER_ID} (d_sae=16384, k=80, 500M tokens)")
    print(f"Our SAEs: exp17 checkpoints (d_sae=16384, k=80, 50M tokens)")
    print(f"Eval tokens: {N_EVAL_TOKENS:,}")
    print(f"Key question: Is our standard SAE implementation competitive?")
    print(f"Killer question: Does our cosine SAE (50M) beat reference standard (500M)?")

    torch.manual_seed(SEED)

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
            "experiment": "opensource_baseline",
            "layers": LAYERS,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "k": K,
            "n_eval_tokens": N_EVAL_TOKENS,
            "reference_repo": HF_REPO,
            "reference_trainer": TRAINER_ID,
            "reference_tokens": "500M",
            "our_tokens": "50M",
        },
        "layers": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Collect eval data
        eval_data = collect_activations(model, tokenizer, layer_idx, N_EVAL_TOKENS)
        layer_results = {}

        # Load all SAEs for this layer
        print(f"\n  Loading SAEs for L{layer_idx}...")
        saes = {}

        print(f"\n  --- Reference (adamkarvonen, 500M tokens) ---")
        saes["reference_500M"] = load_adamkarvonen_sae(layer_idx)

        print(f"\n  --- Our standard (exp17, 50M tokens) ---")
        sae = load_exp17_sae("standard", layer_idx)
        if sae is not None:
            saes["our_standard_50M"] = sae

        print(f"\n  --- Our adaptive_l2 (exp17, 50M tokens) ---")
        sae = load_exp17_sae("adaptive_l2", layer_idx)
        if sae is not None:
            saes["our_cosine_50M"] = sae

        # Evaluate all
        for name, sae in saes.items():
            print(f"\n  Reconstruction -- {name}")
            recon = evaluate_reconstruction(name, sae, eval_data)

            abl = evaluate_ablation(name, model, sae, eval_data, layer_idx)

            layer_results[name] = {
                "reconstruction": recon,
                "ablation": abl,
            }
            if hasattr(sae, "scale_a"):
                layer_results[name]["scale_a"] = sae.scale_a.item()
                layer_results[name]["scale_b_exp"] = sae.scale_b.exp().item()

        # Feature overlap comparisons
        print(f"\n  --- FEATURE OVERLAP (L{layer_idx}) ---")

        if "reference_500M" in saes and "our_standard_50M" in saes:
            layer_results["overlap_ref_vs_our_std"] = feature_overlap(
                saes["reference_500M"], saes["our_standard_50M"],
                eval_data, "reference_500M", "our_standard_50M"
            )

        if "reference_500M" in saes and "our_cosine_50M" in saes:
            layer_results["overlap_ref_vs_our_cos"] = feature_overlap(
                saes["reference_500M"], saes["our_cosine_50M"],
                eval_data, "reference_500M", "our_cosine_50M"
            )

        if "our_standard_50M" in saes and "our_cosine_50M" in saes:
            layer_results["overlap_our_std_vs_cos"] = feature_overlap(
                saes["our_standard_50M"], saes["our_cosine_50M"],
                eval_data, "our_standard_50M", "our_cosine_50M"
            )

        # Save incrementally
        all_results["layers"][str(layer_idx)] = layer_results
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Clean up
        for sae in saes.values():
            del sae
        del saes, eval_data
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary Table
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Open-Source Baseline Comparison")
    print(f"{'='*70}")

    print(f"\n  {'Layer':>5s}  {'SAE':<22s} {'Tokens':>7s} {'FVE':>7s} {'Dead%':>7s} "
          f"{'Alive':>6s} {'L0':>5s} {'cos->KL':>8s} {'cos>inn':>8s}")
    print(f"  {'-'*5}  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*8} {'-'*8}")

    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        for name, tokens in [
            ("reference_500M", "500M"),
            ("our_standard_50M", "50M"),
            ("our_cosine_50M", "50M"),
        ]:
            r = lr.get(name, {})
            recon = r.get("reconstruction", {})
            abl_agg = r.get("ablation", {}).get("aggregate", {})

            fve = recon.get("fve", 0)
            dead = recon.get("dead_frac", 1)
            alive = recon.get("alive_count", 0)
            l0 = recon.get("l0", 0)
            cos_kl = abl_agg.get("cos_kl_mean", 0)
            cos_wins = abl_agg.get("cos_wins_inner", 0)
            n_feats = abl_agg.get("n_features", 0)
            cos_win_str = f"{cos_wins}/{n_feats}" if n_feats > 0 else "N/A"

            print(f"  {layer_idx:>5d}  {name:<22s} {tokens:>7s} {fve:>7.4f} "
                  f"{dead*100:>6.1f}% {alive:>6d} {l0:>5.0f} "
                  f"{cos_kl:>8.4f} {cos_win_str:>8s}")

    # Key comparisons
    print(f"\n  KEY COMPARISONS:")
    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        ref = lr.get("reference_500M", {}).get("reconstruction", {})
        std = lr.get("our_standard_50M", {}).get("reconstruction", {})
        cos = lr.get("our_cosine_50M", {}).get("reconstruction", {})

        ref_fve = ref.get("fve", 0)
        std_fve = std.get("fve", 0)
        cos_fve = cos.get("fve", 0)
        ref_alive = ref.get("alive_count", 0)
        std_alive = std.get("alive_count", 0)
        cos_alive = cos.get("alive_count", 0)

        print(f"\n    L{layer_idx}:")
        print(f"      Our standard vs reference: FVE {std_fve:.4f} vs {ref_fve:.4f} "
              f"({std_fve-ref_fve:+.4f}), alive {std_alive} vs {ref_alive}")
        print(f"      Our cosine vs reference:   FVE {cos_fve:.4f} vs {ref_fve:.4f} "
              f"({cos_fve-ref_fve:+.4f}), alive {cos_alive} vs {ref_alive}")
        if cos_fve > ref_fve:
            print(f"      *** COSINE (50M) BEATS REFERENCE (500M) ON FVE ***")
        if cos_alive > ref_alive:
            print(f"      *** COSINE (50M) HAS MORE ALIVE FEATURES THAN REFERENCE (500M) ***")

    # Feature overlap summary
    print(f"\n  FEATURE OVERLAP:")
    print(f"  {'Layer':>5s}  {'Comparison':<35s} {'Jaccard':>8s} {'MaxCos':>8s} {'>0.9':>6s}")
    print(f"  {'-'*5}  {'-'*35} {'-'*8} {'-'*8} {'-'*6}")
    for layer_idx in LAYERS:
        lr = all_results["layers"].get(str(layer_idx), {})
        for key, label in [
            ("overlap_ref_vs_our_std", "reference vs our_standard"),
            ("overlap_ref_vs_our_cos", "reference vs our_cosine"),
            ("overlap_our_std_vs_cos", "our_standard vs our_cosine"),
        ]:
            ov = lr.get(key, {})
            if ov:
                print(f"  {layer_idx:>5d}  {label:<35s} {ov.get('jaccard',0):>8.3f} "
                      f"{ov.get('dec_max_cos_mean',0):>8.3f} "
                      f"{ov.get('frac_matched_gt90',0):>5.1%}")

    print(f"\nResults: {RESULTS_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
