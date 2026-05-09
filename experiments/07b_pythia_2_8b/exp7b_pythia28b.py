"""
Experiment 7b: Pythia-2.8B Replication (LayerNorm)
====================================================
Replaces exp7 (Pythia-70M) with a much larger LayerNorm model.
- Model: Pythia-2.8B-deduped (32 layers, d_model=2560, LayerNorm)
- SAE: jacobdunefsky/pythia-2.8B-saes (TopK, k=60, 61k features, resid_pre)

Tests whether the RNH holds on a larger LayerNorm model.
Pythia-70M showed weak results (47% cos>inner) — does this scale?
"""

import json
import time
import gc
import torch
import torch.nn as nn
import numpy as np
from transformer_lens import HookedTransformer
from safetensors import safe_open
from huggingface_hub import snapshot_download
from datasets import load_dataset

DEVICE = "cuda"
DTYPE = torch.float32
MODEL_NAME = "pythia-2.8b-deduped"
SAE_REPO = "jacobdunefsky/pythia-2.8B-saes"

LAYERS = [8, 16, 24]  # early, mid, late out of 32
NUM_FEATURES = 50
NUM_ABLATION_SAMPLES = 100
MAX_SEQ_LEN = 128
NUM_TEXT_SAMPLES = 500


# ── Custom SAE loader ──────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_in, num_features, d_out, top_k):
        super().__init__()
        self.d_in = d_in
        self.d_sae = num_features
        self.d_out = d_out
        self.top_k = top_k
        self.W_enc = nn.Parameter(torch.empty(d_in, num_features))
        self.b_enc = nn.Parameter(torch.empty(num_features))
        self.W_dec = nn.Parameter(torch.empty(num_features, d_out))
        self.b_dec = nn.Parameter(torch.empty(d_out))

    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        top_vals, top_idx = torch.topk(pre_acts, k=self.top_k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, top_idx, top_vals)
        return acts

    @classmethod
    def load_from_hf(cls, repo_id, layer, device="cpu"):
        dir_name = f"pythia-2.8B-dun-resid-sae{layer}"
        local_dir = snapshot_download(repo_id, allow_patterns=[f"{dir_name}/*"])
        sae_dir = f"{local_dir}/{dir_name}"

        with open(f"{sae_dir}/sae.json") as f:
            cfg = json.load(f)

        sae = cls(
            d_in=cfg["d_in"], num_features=cfg["num_features"],
            d_out=cfg["d_out"], top_k=cfg["top_k"],
        )
        with safe_open(f"{sae_dir}/sae.safetensors", framework="pt", device=str(device)) as f:
            sae.W_enc.data = f.get_tensor("W_enc")
            sae.W_dec.data = f.get_tensor("W_dec")
            sae.b_enc.data = f.get_tensor("b_enc")
            sae.b_dec.data = f.get_tensor("b_dec")

        return sae.float().to(device)  # ensure float32 regardless of safetensors dtype


# ── Data loading ───────────────────────────────────────────────────

def load_fineweb_samples(n_samples):
    print(f"  Loading {n_samples} FineWeb samples...")
    t0 = time.time()
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    texts = []
    for row in ds:
        if len(row["text"]) > 100:
            texts.append(row["text"][:512])
        if len(texts) >= n_samples:
            break
    print(f"  Loaded {len(texts)} samples in {time.time()-t0:.1f}s")
    return texts


# ── Activation collection (resid_pre) ─────────────────────────────

def collect_activations(model, texts, layer_idx, batch_size=8):
    """Collect resid_pre activations (what the SAE was trained on)."""
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    all_acts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)[:, :MAX_SEQ_LEN]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        acts = cache[hook_name]
        for b in range(acts.shape[0]):
            all_acts.append(acts[b, 1:].detach())  # skip BOS
        del cache
        torch.cuda.empty_cache()

    return torch.cat(all_acts, dim=0)


def filter_attention_sinks(activations, threshold_multiplier=10.0):
    norms = activations.norm(dim=-1)
    mask = norms < (norms.median() * threshold_multiplier)
    n_filtered = (~mask).sum().item()
    return activations[mask], n_filtered


# ── Ablation ───────────────────────────────────────────────────────

def ablate_feature_kl(model, activation, feature_dir, layer_idx):
    x = activation.clone()
    fd = feature_dir.to(x.dtype)
    projection = (x @ fd) * fd
    x_ablated = x - projection

    bos = model.to_tokens("", prepend_bos=True)[:, :1]
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"

    def orig_hook(act, hook):
        act[:, -1, :] = x
        return act

    with torch.no_grad():
        orig_logits = model.run_with_hooks(
            bos, fwd_hooks=[(hook_name, orig_hook)]
        )[:, -1, :].float()

    def ablated_hook(act, hook):
        act[:, -1, :] = x_ablated
        return act

    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            bos, fwd_hooks=[(hook_name, ablated_hook)]
        )[:, -1, :].float()

    orig_probs = torch.softmax(orig_logits, dim=-1)
    ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)
    kl = torch.where(
        orig_probs > 0,
        orig_probs * (orig_probs.log() - ablated_log_probs),
        torch.zeros_like(orig_probs)
    ).sum().item()

    return {"kl": kl}


# ── Analysis ───────────────────────────────────────────────────────

def analyze_layer(model, sae, activations, layer_idx):
    print(f"\n  Computing feature frequencies...")
    t0 = time.time()
    n_features = sae.d_sae
    feature_freq = torch.zeros(n_features, device=DEVICE)
    n_tokens = len(activations)
    chunk_size = 1024

    for i in range(0, n_tokens, chunk_size):
        batch = activations[i:i+chunk_size].float()  # SAE is float32
        encoded = sae.encode(batch)
        feature_freq += (encoded > 0).float().sum(dim=0)
        del encoded
    feature_freq /= n_tokens
    print(f"  Freq computation: {time.time()-t0:.1f}s")

    top_features = feature_freq.topk(NUM_FEATURES).indices
    print(f"  Top {NUM_FEATURES} features: freq [{feature_freq[top_features[-1]]:.4f}, {feature_freq[top_features[0]]:.4f}]")

    print(f"  Computing top-feature SAE activations...")
    top_feat_set = set(top_features.tolist())
    feat_acts = {f: torch.zeros(n_tokens, device=DEVICE) for f in top_feat_set}
    for i in range(0, n_tokens, chunk_size):
        batch = activations[i:i+chunk_size].float()  # SAE is float32
        encoded = sae.encode(batch)
        for f in top_feat_set:
            feat_acts[f][i:i+len(batch)] = encoded[:, f]
        del encoded

    feature_results = []

    for feat_rank, feat_idx in enumerate(top_features):
        feat_idx = feat_idx.item()
        feature_dir = sae.W_dec[feat_idx]
        feature_dir = feature_dir / feature_dir.norm()
        feature_dir_cast = feature_dir.to(activations.dtype)

        cos_sims = torch.nn.functional.cosine_similarity(
            activations, feature_dir_cast.unsqueeze(0), dim=-1
        )
        norms = activations.norm(dim=-1)
        inner_prods = activations @ feature_dir_cast
        sae_feat_acts = feat_acts[feat_idx]

        active_mask = sae_feat_acts > 0
        n_active = active_mask.sum().item()
        if n_active < 30:
            continue

        active_indices = torch.where(active_mask)[0]
        n_sample = min(NUM_ABLATION_SAMPLES, len(active_indices))
        perm = torch.randperm(len(active_indices))[:n_sample]
        sample_indices = active_indices[perm]

        cos_vals, norm_vals, inner_vals, sae_vals, kl_vals = [], [], [], [], []

        for idx in sample_indices:
            result = ablate_feature_kl(model, activations[idx], feature_dir, layer_idx)
            if result is None or result["kl"] < 0 or np.isnan(result["kl"]):
                continue
            cos_vals.append(cos_sims[idx].item())
            norm_vals.append(norms[idx].item())
            inner_vals.append(inner_prods[idx].item())
            sae_vals.append(sae_feat_acts[idx].item())
            kl_vals.append(result["kl"])

        if len(kl_vals) < 15:
            continue

        cos_arr = np.array(cos_vals)
        norm_arr = np.array(norm_vals)
        inner_arr = np.array(inner_vals)
        sae_arr = np.array(sae_vals)
        kl_arr = np.array(kl_vals)

        if kl_arr.std() < 1e-10:
            continue

        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]

        feat_result = {
            "feature_idx": feat_idx, "n_active": n_active, "n_ablated": len(kl_vals),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "cos_wins_sae": bool(abs(corr_cos) > abs(corr_sae)),
        }
        feature_results.append(feat_result)

        if feat_rank < 5 or feat_rank % 10 == 0:
            print(f"    Feature {feat_idx:>5d} | n={len(kl_vals):>3d} | "
                  f"cos→KL={corr_cos:>6.3f} | norm→KL={corr_norm:>6.3f} | "
                  f"inner→KL={corr_inner:>6.3f} | SAE→KL={corr_sae:>6.3f}")

    if feature_results:
        n = len(feature_results)
        cos_kls = [r["corr_cos_kl"] for r in feature_results]
        norm_kls = [r["corr_norm_kl"] for r in feature_results]
        inner_kls = [r["corr_inner_kl"] for r in feature_results]
        sae_kls = [r["corr_sae_kl"] for r in feature_results]
        cos_wins_inner = sum(1 for r in feature_results if r["cos_wins_inner"])
        cos_wins_sae = sum(1 for r in feature_results if r["cos_wins_sae"])

        print(f"\n  === Layer {layer_idx} Summary ({n} features) ===")
        print(f"  corr(cos, KL):   {np.mean(cos_kls):.4f} ± {np.std(cos_kls):.4f}")
        print(f"  corr(norm, KL):  {np.mean(norm_kls):.4f} ± {np.std(norm_kls):.4f}")
        print(f"  corr(inner, KL): {np.mean(inner_kls):.4f} ± {np.std(inner_kls):.4f}")
        print(f"  corr(SAE, KL):   {np.mean(sae_kls):.4f} ± {np.std(sae_kls):.4f}")
        print(f"  cos > inner: {cos_wins_inner}/{n} | cos > SAE: {cos_wins_sae}/{n}")

        return {
            "layer": layer_idx, "features": feature_results,
            "aggregate": {
                "n": n,
                "cos_mean": float(np.mean(cos_kls)), "norm_mean": float(np.mean(norm_kls)),
                "inner_mean": float(np.mean(inner_kls)), "sae_mean": float(np.mean(sae_kls)),
                "cos_wins_inner": cos_wins_inner, "cos_wins_sae": cos_wins_sae,
            },
        }
    return {"layer": layer_idx, "features": [], "aggregate": {"n": 0}}


def main():
    print("Experiment 7b: Pythia-2.8B Replication (LayerNorm)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME} (32 layers, d_model=2560, LayerNorm)")
    print(f"SAE: {SAE_REPO} (TopK k=60, 61k features, resid_pre)")
    print(f"Layers: {LAYERS}")

    texts = load_fineweb_samples(NUM_TEXT_SAMPLES)

    print("\nLoading model...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

    all_results = {
        "model": MODEL_NAME, "corpus": "FineWeb", "normalization": "LayerNorm",
        "sae": SAE_REPO, "sae_type": "TopK (k=60)", "layers": [],
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        print(f"  Loading SAE for layer {layer_idx}...")
        t0 = time.time()
        sae = TopKSAE.load_from_hf(SAE_REPO, layer_idx, device=DEVICE)
        print(f"  SAE loaded in {time.time()-t0:.1f}s ({sae.d_sae} features)")

        print(f"  Collecting activations (resid_pre)...")
        t0 = time.time()
        activations = collect_activations(model, texts, layer_idx)
        activations, n_filtered = filter_attention_sinks(activations)
        print(f"  {activations.shape[0]} tokens ({n_filtered} filtered) in {time.time()-t0:.1f}s")

        layer_result = analyze_layer(model, sae, activations, layer_idx)
        all_results["layers"].append(layer_result)

        del sae, activations
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY (Pythia-2.8B, LayerNorm)")
    print("=" * 70)

    total_cos_inner, total_cos_sae, total_n = 0, 0, 0
    for lr in all_results["layers"]:
        agg = lr["aggregate"]
        if agg["n"] == 0:
            continue
        total_cos_inner += agg["cos_wins_inner"]
        total_cos_sae += agg["cos_wins_sae"]
        total_n += agg["n"]
        print(f"  Layer {lr['layer']:>2d}: cos→KL={agg['cos_mean']:.3f} | "
              f"norm→KL={agg['norm_mean']:.3f} | inner→KL={agg['inner_mean']:.3f} | "
              f"SAE→KL={agg['sae_mean']:.3f} | "
              f"cos>inner={agg['cos_wins_inner']}/{agg['n']} | "
              f"cos>SAE={agg['cos_wins_sae']}/{agg['n']}")

    if total_n > 0:
        print(f"\n  Total: cos>inner={total_cos_inner}/{total_n} ({100*total_cos_inner/total_n:.0f}%) | "
              f"cos>SAE={total_cos_sae}/{total_n} ({100*total_cos_sae/total_n:.0f}%)")

    with open("experiments/exp7b_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to experiments/exp7b_results.json")


if __name__ == "__main__":
    main()
