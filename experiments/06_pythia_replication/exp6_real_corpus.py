"""
Experiment 6: Real Corpus Replication
======================================
Exact same methodology as Experiment 5 (multi-layer, logit-<author>el KL ablation)
but on 10k samples from FineWeb instead of hand-written prompts.

This addresses the strongest methodological objection: "results could be an
artifact of prompt selection."
"""

import json
import time
import gc
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import interp_tools.saes.batch_topk_sae as batch_topk_sae
import interp_tools.model_utils as model_utils

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
SAE_REPO = "adamkarvonen/qwen3-8b-saes"

LAYERS = [9, 18, 27]
SAE_FILENAMES = {
    9:  "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_9/trainer_2/ae.pt",
    18: "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/trainer_2/ae.pt",
    27: "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_27/trainer_2/ae.pt",
}

NUM_FEATURES = 50
NUM_ABLATION_SAMPLES = 100
MAX_SEQ_LEN = 256
NUM_TEXT_SAMPLES = 500  # from FineWeb — each ~256 tokens → ~100k+ tokens total


def load_fineweb_samples(n_samples):
    """Load real text from FineWeb."""
    print(f"  Downloading {n_samples} samples from FineWeb...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    texts = []
    for i, row in enumerate(ds):
        text = row["text"]
        if len(text) > 100:  # skip very short docs
            texts.append(text[:1024])  # cap length
        if len(texts) >= n_samples:
            break
    print(f"  Loaded {len(texts)} samples in {time.time()-t0:.1f}s")
    return texts


def collect_activations(model, tokenizer, texts, layer_idx, batch_size=4):
    all_acts = []
    submodule = model_utils.get_submodule(model, layer_idx)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to(DEVICE)
        acts = model_utils.collect_activations(model, submodule, inputs)
        mask = inputs["attention_mask"]
        for b in range(acts.shape[0]):
            valid = acts[b][mask[b].bool()]
            all_acts.append(valid.detach())
    return torch.cat(all_acts, dim=0)


def filter_attention_sinks(activations, threshold_multiplier=10.0):
    norms = activations.norm(dim=-1)
    mask = norms < (norms.median() * threshold_multiplier)
    n_filtered = (~mask).sum().item()
    return activations[mask], n_filtered


def ablate_feature_kl(model, activation, feature_dir, layer_idx):
    x = activation.unsqueeze(0).unsqueeze(0)
    projection = (activation @ feature_dir) * feature_dir
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0)

    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

    def make_hook(replacement):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                return (replacement,) + outputs[1:]
            return replacement
        return hook

    handle = model.model.layers[layer_idx].register_forward_hook(make_hook(x))
    with torch.no_grad():
        try:
            orig_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            handle.remove()
            return None
    handle.remove()

    handle = model.model.layers[layer_idx].register_forward_hook(make_hook(x_ablated))
    with torch.no_grad():
        try:
            ablated_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            handle.remove()
            return None
    handle.remove()

    orig_probs = torch.softmax(orig_logits, dim=-1)
    ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)
    kl = torch.sum(orig_probs * (orig_probs.log() - ablated_log_probs)).item()

    return {"kl": kl}


def analyze_layer(model, tokenizer, sae, activations, layer_idx):
    # With 100k tokens and 65k features, we can't hold all SAE acts in memory.
    # Compute feature frequency in chunks, then only store per-feature acts for top features.
    print(f"\n  Computing feature frequencies (chunked)...")
    t0 = time.time()
    n_features = sae.W_dec.shape[0]
    feature_freq = torch.zeros(n_features, device=DEVICE)
    n_tokens = len(activations)
    chunk_size = 1024

    for i in range(0, n_tokens, chunk_size):
        batch = activations[i:i+chunk_size]
        encoded = sae.encode(batch)
        feature_freq += (encoded > 0).float().sum(dim=0)
        del encoded
    feature_freq /= n_tokens
    print(f"  Feature freq computation: {time.time()-t0:.1f}s")

    top_features = feature_freq.topk(NUM_FEATURES).indices
    print(f"  Top {NUM_FEATURES} features: freq range "
          f"[{feature_freq[top_features[-1]]:.4f}, {feature_freq[top_features[0]]:.4f}]")

    # Now compute SAE activations only for top features
    print(f"  Computing SAE activations for top features...")
    t0 = time.time()
    top_feat_set = set(top_features.tolist())
    # Store per-feature activations: (n_tokens,) per feature
    feat_acts = {f: torch.zeros(n_tokens, device=DEVICE) for f in top_feat_set}
    for i in range(0, n_tokens, chunk_size):
        batch = activations[i:i+chunk_size]
        encoded = sae.encode(batch)  # (chunk, d_sae)
        for f in top_feat_set:
            feat_acts[f][i:i+len(batch)] = encoded[:, f]
        del encoded
    all_sae_acts = feat_acts  # dict mapping feature_idx -> (n_tokens,) tensor
    print(f"  Top-feature SAE encoding: {time.time()-t0:.1f}s")

    feature_results = []

    for feat_rank, feat_idx in enumerate(top_features):
        feat_idx = feat_idx.item()
        feature_dir = sae.W_dec[feat_idx]

        cos_sims = torch.nn.functional.cosine_similarity(
            activations, feature_dir.unsqueeze(0), dim=-1
        )
        norms = activations.norm(dim=-1)
        inner_prods = (activations @ feature_dir)
        sae_feat_acts = all_sae_acts[feat_idx]

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
            "feature_idx": feat_idx,
            "n_active": n_active,
            "n_ablated": len(kl_vals),
            "corr_cos_kl": float(corr_cos),
            "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner),
            "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "cos_wins_sae": bool(abs(corr_cos) > abs(corr_sae)),
        }
        feature_results.append(feat_result)

        if feat_rank < 5 or feat_rank % 10 == 0:
            print(f"    Feature {feat_idx:>5d} | n={len(kl_vals):>3d} | "
                  f"cos→KL={corr_cos:>6.3f} | norm→KL={corr_norm:>6.3f} | "
                  f"inner→KL={corr_inner:>6.3f} | SAE→KL={corr_sae:>6.3f}")

    if feature_results:
        cos_kls = [r["corr_cos_kl"] for r in feature_results]
        norm_kls = [r["corr_norm_kl"] for r in feature_results]
        inner_kls = [r["corr_inner_kl"] for r in feature_results]
        sae_kls = [r["corr_sae_kl"] for r in feature_results]
        n = len(feature_results)
        cos_wins_inner = sum(1 for r in feature_results if r["cos_wins_inner"])
        cos_wins_sae = sum(1 for r in feature_results if r["cos_wins_sae"])

        print(f"\n  === Layer {layer_idx} Summary ({n} features) ===")
        print(f"  corr(cos, KL):   {np.mean(cos_kls):.4f} ± {np.std(cos_kls):.4f}")
        print(f"  corr(norm, KL):  {np.mean(norm_kls):.4f} ± {np.std(norm_kls):.4f}")
        print(f"  corr(inner, KL): {np.mean(inner_kls):.4f} ± {np.std(inner_kls):.4f}")
        print(f"  corr(SAE, KL):   {np.mean(sae_kls):.4f} ± {np.std(sae_kls):.4f}")
        print(f"  cos > inner: {cos_wins_inner}/{n} | cos > SAE: {cos_wins_sae}/{n}")

        return {
            "layer": layer_idx,
            "features": feature_results,
            "aggregate": {
                "n": n,
                "cos_mean": float(np.mean(cos_kls)), "cos_std": float(np.std(cos_kls)),
                "norm_mean": float(np.mean(norm_kls)), "norm_std": float(np.std(norm_kls)),
                "inner_mean": float(np.mean(inner_kls)), "inner_std": float(np.std(inner_kls)),
                "sae_mean": float(np.mean(sae_kls)), "sae_std": float(np.std(sae_kls)),
                "cos_wins_inner": cos_wins_inner,
                "cos_wins_sae": cos_wins_sae,
            },
        }
    return {"layer": layer_idx, "features": [], "aggregate": {"n": 0}}


def main():
    print("Experiment 6: Real Corpus Replication (FineWeb)")
    print("=" * 70)

    # Load real data
    texts = load_fineweb_samples(NUM_TEXT_SAMPLES)

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    all_results = {"model": MODEL_NAME, "corpus": "FineWeb sample-10BT", "layers": []}

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Load SAE for this layer
        print(f"  Loading SAE for layer {layer_idx}...")
        sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
            repo_id=SAE_REPO, filename=SAE_FILENAMES[layer_idx],
            model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE,
        )

        # Collect activations
        print(f"  Collecting activations...")
        t0 = time.time()
        activations = collect_activations(model, tokenizer, texts, layer_idx)
        activations, n_filtered = filter_attention_sinks(activations)
        print(f"  {activations.shape[0]} tokens ({n_filtered} sinks filtered) in {time.time()-t0:.1f}s")

        # Analyze
        layer_result = analyze_layer(model, tokenizer, sae, activations, layer_idx)
        all_results["layers"].append(layer_result)

        # Free SAE memory
        del sae, activations
        gc.collect()
        torch.cuda.empty_cache()

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY (FineWeb corpus)")
    print("=" * 70)

    total_cos_wins_inner = 0
    total_cos_wins_sae = 0
    total_n = 0

    for lr in all_results["layers"]:
        agg = lr["aggregate"]
        if agg["n"] == 0:
            continue
        total_cos_wins_inner += agg["cos_wins_inner"]
        total_cos_wins_sae += agg["cos_wins_sae"]
        total_n += agg["n"]
        print(f"  Layer {lr['layer']:>2d}: cos→KL={agg['cos_mean']:.3f} | "
              f"norm→KL={agg['norm_mean']:.3f} | inner→KL={agg['inner_mean']:.3f} | "
              f"SAE→KL={agg['sae_mean']:.3f} | "
              f"cos>inner={agg['cos_wins_inner']}/{agg['n']} | "
              f"cos>SAE={agg['cos_wins_sae']}/{agg['n']}")

    if total_n > 0:
        print(f"\n  Total: cos>inner={total_cos_wins_inner}/{total_n} ({100*total_cos_wins_inner/total_n:.0f}%) | "
              f"cos>SAE={total_cos_wins_sae}/{total_n} ({100*total_cos_wins_sae/total_n:.0f}%)")

    with open("experiments/exp6_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to experiments/exp6_results.json")


if __name__ == "__main__":
    main()
