"""
Experiment 7: Pythia-70M Replication
=====================================
Replicates the core RNH test on a completely different model and SAE:
- Model: Pythia-70M-deduped (6 layers, d_model=512)
- SAE: SAE-Lens pretrained residual stream SAEs (standard architecture, 32k features)

This addresses: "single model" and "single SAE family" weaknesses.
Different architecture, different scale, different SAE library.

Layers tested: 1, 3, 5 (early, mid, late out of 6 total)
"""

import json
import time
import gc
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

DEVICE = "cuda"
DTYPE = torch.float32  # Pythia-70M is small, no need for bfloat16
MODEL_NAME = "pythia-70m-deduped"
LAYERS = [1, 3, 5]
SAE_RELEASE = "pythia-70m-deduped-res-sm"

NUM_FEATURES = 50
NUM_ABLATION_SAMPLES = 100
MAX_SEQ_LEN = 128
NUM_TEXT_SAMPLES = 500


def load_fineweb_samples(n_samples):
    print(f"  Loading {n_samples} FineWeb samples...")
    t0 = time.time()
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    texts = []
    for row in ds:
        text = row["text"]
        if len(text) > 100:
            texts.append(text[:512])
        if len(texts) >= n_samples:
            break
    print(f"  Loaded {len(texts)} samples in {time.time()-t0:.1f}s")
    return texts


def collect_activations_tl(model, texts, layer_idx, batch_size=16):
    """Collect residual stream activations using TransformerLens."""
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    all_acts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        tokens = tokens[:, :MAX_SEQ_LEN]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        acts = cache[hook_name]  # (batch, seq_len, d_model)
        # Flatten, skip BOS token
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


def ablate_feature_kl_pythia(model, activation, feature_dir, layer_idx):
    """
    Ablate a feature from the residual stream at layer_idx in Pythia.
    Uses TransformerLens hooks.
    """
    x = activation.clone()
    projection = (x @ feature_dir) * feature_dir
    x_ablated = x - projection

    # We need a dummy input to run through the model
    # Use a single BOS token and hook to inject our activation
    bos = model.to_tokens("", prepend_bos=True)[:, :1]  # (1, 1)

    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Original logits
    def orig_hook(act, hook):
        act[:, -1, :] = x
        return act

    with torch.no_grad():
        orig_logits = model.run_with_hooks(
            bos, fwd_hooks=[(hook_name, orig_hook)]
        )[:, -1, :].float()

    # Ablated logits
    def ablated_hook(act, hook):
        act[:, -1, :] = x_ablated
        return act

    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            bos, fwd_hooks=[(hook_name, ablated_hook)]
        )[:, -1, :].float()

    orig_probs = torch.softmax(orig_logits, dim=-1)
    ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)
    # Safe KL: entries where p=0 contribute 0 by convention (avoid 0 * -inf = NaN)
    kl = torch.where(
        orig_probs > 0,
        orig_probs * (orig_probs.log() - ablated_log_probs),
        torch.zeros_like(orig_probs)
    ).sum().item()

    return {"kl": kl}


def analyze_layer(model, sae, activations, layer_idx):
    # Compute feature frequencies in chunks (memory safe)
    print(f"\n  Computing feature frequencies...")
    t0 = time.time()
    n_features = sae.cfg.d_sae
    feature_freq = torch.zeros(n_features, device=DEVICE)
    n_tokens = len(activations)
    chunk_size = 2048

    for i in range(0, n_tokens, chunk_size):
        batch = activations[i:i+chunk_size]
        encoded = sae.encode(batch)
        feature_freq += (encoded > 0).float().sum(dim=0)
        del encoded
    feature_freq /= n_tokens
    print(f"  Freq computation: {time.time()-t0:.1f}s")

    top_features = feature_freq.topk(NUM_FEATURES).indices
    print(f"  Top {NUM_FEATURES} features: freq [{feature_freq[top_features[-1]]:.4f}, {feature_freq[top_features[0]]:.4f}]")

    # Compute per-feature activations for top features
    print(f"  Computing top-feature SAE activations...")
    top_feat_set = set(top_features.tolist())
    feat_acts = {f: torch.zeros(n_tokens, device=DEVICE) for f in top_feat_set}
    for i in range(0, n_tokens, chunk_size):
        batch = activations[i:i+chunk_size]
        encoded = sae.encode(batch)
        for f in top_feat_set:
            feat_acts[f][i:i+len(batch)] = encoded[:, f]
        del encoded

    feature_results = []

    for feat_rank, feat_idx in enumerate(top_features):
        feat_idx = feat_idx.item()
        feature_dir = sae.W_dec[feat_idx]  # unit vector
        feature_dir = feature_dir / feature_dir.norm()  # ensure unit

        cos_sims = torch.nn.functional.cosine_similarity(
            activations, feature_dir.unsqueeze(0), dim=-1
        )
        norms = activations.norm(dim=-1)
        inner_prods = activations @ feature_dir
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
            result = ablate_feature_kl_pythia(model, activations[idx], feature_dir, layer_idx)
            if result is None or result["kl"] < 0 or np.isnan(result["kl"]):
                continue
            cos_vals.append(cos_sims[idx].item())
            norm_vals.append(norms[idx].item())
            inner_vals.append(inner_prods[idx].item())
            sae_vals.append(sae_feat_acts[idx].item())
            kl_vals.append(result["kl"])

        if len(kl_vals) < 15:
            continue

        cos_arr, norm_arr = np.array(cos_vals), np.array(norm_vals)
        inner_arr, sae_arr, kl_arr = np.array(inner_vals), np.array(sae_vals), np.array(kl_vals)

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
    print("Experiment 7: Pythia-70M Replication")
    print("=" * 70)
    print(f"Model: {MODEL_NAME} (6 layers, d_model=512)")
    print(f"SAE: {SAE_RELEASE} (standard SAE, 32k features)")
    print(f"Layers: {LAYERS}")

    texts = load_fineweb_samples(NUM_TEXT_SAMPLES)

    print("\nLoading model...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

    all_results = {"model": MODEL_NAME, "corpus": "FineWeb", "layers": []}

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        sae_id = f"blocks.{layer_idx}.hook_resid_post"
        print(f"  Loading SAE: {sae_id}")
        sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)

        print(f"  Collecting activations...")
        t0 = time.time()
        activations = collect_activations_tl(model, texts, layer_idx)
        activations, n_filtered = filter_attention_sinks(activations)
        print(f"  {activations.shape[0]} tokens ({n_filtered} filtered) in {time.time()-t0:.1f}s")

        layer_result = analyze_layer(model, sae, activations, layer_idx)
        all_results["layers"].append(layer_result)

        del sae, activations
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY (Pythia-70M)")
    print("=" * 70)

    total_cos_inner, total_cos_sae, total_n = 0, 0, 0
    for lr in all_results["layers"]:
        agg = lr["aggregate"]
        if agg["n"] == 0:
            continue
        total_cos_inner += agg["cos_wins_inner"]
        total_cos_sae += agg["cos_wins_sae"]
        total_n += agg["n"]
        print(f"  Layer {lr['layer']}: cos→KL={agg['cos_mean']:.3f} | "
              f"norm→KL={agg['norm_mean']:.3f} | inner→KL={agg['inner_mean']:.3f} | "
              f"SAE→KL={agg['sae_mean']:.3f} | "
              f"cos>inner={agg['cos_wins_inner']}/{agg['n']} | "
              f"cos>SAE={agg['cos_wins_sae']}/{agg['n']}")

    if total_n > 0:
        print(f"\n  Total: cos>inner={total_cos_inner}/{total_n} ({100*total_cos_inner/total_n:.0f}%) | "
              f"cos>SAE={total_cos_sae}/{total_n} ({100*total_cos_sae/total_n:.0f}%)")

    with open("experiments/exp7_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to experiments/exp7_results.json")


if __name__ == "__main__":
    main()
