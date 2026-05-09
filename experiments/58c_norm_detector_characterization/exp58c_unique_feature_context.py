"""
Experiment 58c: Unique Feature Norm Context Analysis

Tests WHY cosine discovers ~950 more unique rare features than standard.
Hypothesis: cosine finds features at tokens where standard's dot product
is dominated by input norm rather than direction. If true, cosine-unique
features should activate disproportionately on high-norm tokens.

Method:
1. Recompute feature matching (100K tokens, corr >= 0.7, decoder cos_sim > 0.7)
2. Identify unique features for each SAE (no strong match in the other)
3. For unique features: characterize the tokens they fire on
   - Input norm distribution (Q1-Q4) for tokens where unique features activate
   - Compare to norm distribution for matched-feature activations
   - Compare to overall token norm distribution
4. For the SAME tokens where cosine-unique features fire, check what
   standard fires on — does standard use a different feature, or nothing?

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 experiments/exp58c_unique_feature_context.py \
        2>&1 | tee experiments/exp58c_output.log
"""

import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_cudnn_sdp(False)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL, D_SAE, K = 4096, 65536, 80
LAYER = 18

CKPTS = {
    "standard": "/data/checkpoints/exp40/standard_L18_final.pt",
    "perfeature_l2": "/data/checkpoints/exp40/perfeature_l2_L18_final.pt",
}

N_TOKENS = 100000
CORRELATION_THRESHOLD = 0.7
DECODER_SIM_THRESHOLD = 0.7
RESULTS_PATH = "experiments/exp58c_results.json"
SEED = 42


# ─── SAE Architectures ──────────────────────────────────────────────────────

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
    def encode(self, x):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.threshold < 0: return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)
    def decode(self, f): return f @ self.W_dec + self.b_dec


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
    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        log_norm = torch.log(input_norm)
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.threshold < 0: return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)
    def decode(self, f): return f @ self.W_dec + self.b_dec


SAE_CLASSES = {"standard": BatchTopKSAE, "perfeature_l2": PerFeatureAdaptiveCosineSAE}


def load_sae(name):
    cls = SAE_CLASSES[name]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(CKPTS[name], map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def collect_activations(n_tokens):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()

    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    all_acts = []
    total = 0

    print(f"  Collecting {n_tokens} tokens...")
    for sample in ds:
        tokens = tokenizer(sample["text"], return_tensors="pt", truncation=True,
                          max_length=128).to(DEVICE)
        if tokens["input_ids"].shape[1] < 16:
            continue

        captured = {}
        def hook(module, inp, out):
            captured["act"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        handle = model.model.layers[LAYER].register_forward_hook(hook)
        with torch.no_grad():
            model(**tokens)
        handle.remove()

        acts = captured["act"][0].to(dtype=DTYPE)
        all_acts.append(acts.cpu())
        total += acts.shape[0]
        if total >= n_tokens:
            break

    all_acts = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"  Collected {all_acts.shape[0]} tokens")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return all_acts


def find_matched_and_unique(acts_cpu, sae_std, sae_cos):
    n_tokens = acts_cpu.shape[0]
    batch_size = 512

    print("  Encoding through both SAEs...")
    std_feats_list, cos_feats_list = [], []
    for i in range(0, n_tokens, batch_size):
        batch = acts_cpu[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            sf = sae_std.encode(batch)
            cf = sae_cos.encode(batch)
        std_feats_list.append(sf.cpu())
        cos_feats_list.append(cf.cpu())

    std_feats = torch.cat(std_feats_list, dim=0).float()
    cos_feats = torch.cat(cos_feats_list, dim=0).float()

    std_alive_mask = std_feats.sum(0) > 0
    cos_alive_mask = cos_feats.sum(0) > 0
    std_alive = std_alive_mask.nonzero(as_tuple=True)[0]
    cos_alive = cos_alive_mask.nonzero(as_tuple=True)[0]
    print(f"  Alive: standard={len(std_alive)}, cosine={len(cos_alive)}")

    std_feats_alive = std_feats[:, std_alive]
    cos_feats_alive = cos_feats[:, cos_alive]

    std_normed = std_feats_alive - std_feats_alive.mean(0, keepdim=True)
    cos_normed = cos_feats_alive - cos_feats_alive.mean(0, keepdim=True)
    std_normed = std_normed / (std_normed.norm(dim=0, keepdim=True) + 1e-8)
    cos_normed = cos_normed / (cos_normed.norm(dim=0, keepdim=True) + 1e-8)

    # GPU-accelerated correlations: move normalized matrices to GPU in half precision
    # (100K, 64K) in float16 = ~12GB, fits in 89GB free on GPU 0
    print("  Moving to GPU for correlation computation...")
    std_normed_gpu = std_normed.half().to(DEVICE)
    cos_normed_gpu = cos_normed.half().to(DEVICE)

    n_std = len(std_alive)
    best_corr_s2c = torch.zeros(n_std)
    best_match_s2c = torch.zeros(n_std, dtype=torch.long)

    GPU_CHUNK = 4096
    print(f"  Computing std→cos correlations (GPU, chunks of {GPU_CHUNK})...")
    for i in range(0, n_std, GPU_CHUNK):
        end = min(i + GPU_CHUNK, n_std)
        corr = std_normed_gpu[:, i:end].T @ cos_normed_gpu  # (chunk, n_cos) on GPU
        mc, mi = corr.max(dim=1)
        best_corr_s2c[i:end] = mc.float().cpu()
        best_match_s2c[i:end] = mi.cpu()
        if i % (GPU_CHUNK * 4) == 0:
            print(f"    {i}/{n_std}...")

    n_cos = len(cos_alive)
    best_corr_c2s = torch.zeros(n_cos)
    best_match_c2s = torch.zeros(n_cos, dtype=torch.long)

    print(f"  Computing cos→std correlations (GPU, chunks of {GPU_CHUNK})...")
    for i in range(0, n_cos, GPU_CHUNK):
        end = min(i + GPU_CHUNK, n_cos)
        corr = cos_normed_gpu[:, i:end].T @ std_normed_gpu
        mc, mi = corr.max(dim=1)
        best_corr_c2s[i:end] = mc.float().cpu()
        best_match_c2s[i:end] = mi.cpu()
        if i % (GPU_CHUNK * 4) == 0:
            print(f"    {i}/{n_cos}...")

    del std_normed_gpu, cos_normed_gpu
    torch.cuda.empty_cache()

    # Batched decoder cosine similarity filtering
    W_dec_std = sae_std.W_dec.detach().float()  # keep on GPU
    W_dec_cos = sae_cos.W_dec.detach().float()
    W_dec_std_normed = F.normalize(W_dec_std, dim=-1)
    W_dec_cos_normed = F.normalize(W_dec_cos, dim=-1)

    print("  Filtering by decoder cosine similarity (batched)...")
    # Std→cos: for each std feature with corr >= threshold, check decoder sim
    s2c_candidates = (best_corr_s2c >= CORRELATION_THRESHOLD).nonzero(as_tuple=True)[0]
    matched_std_local = []
    if len(s2c_candidates) > 0:
        si_global = std_alive[s2c_candidates]
        ci_global = cos_alive[best_match_s2c[s2c_candidates]]
        dsims = (W_dec_std_normed[si_global] * W_dec_cos_normed[ci_global]).sum(dim=-1)
        pass_mask = dsims > DECODER_SIM_THRESHOLD
        matched_std_local = s2c_candidates[pass_mask.cpu()].tolist()

    matched_std_global = set(std_alive[matched_std_local].tolist())

    # Cos→std
    c2s_candidates = (best_corr_c2s >= CORRELATION_THRESHOLD).nonzero(as_tuple=True)[0]
    matched_cos_local = []
    if len(c2s_candidates) > 0:
        ci_global = cos_alive[c2s_candidates]
        si_global = std_alive[best_match_c2s[c2s_candidates]]
        dsims = (W_dec_cos_normed[ci_global] * W_dec_std_normed[si_global]).sum(dim=-1)
        pass_mask = dsims > DECODER_SIM_THRESHOLD
        matched_cos_local = c2s_candidates[pass_mask.cpu()].tolist()

    matched_cos_global = set(cos_alive[matched_cos_local].tolist())

    del W_dec_std, W_dec_cos, W_dec_std_normed, W_dec_cos_normed
    torch.cuda.empty_cache()

    unique_std = set(std_alive.tolist()) - matched_std_global
    unique_cos = set(cos_alive.tolist()) - matched_cos_global

    print(f"  Matched: std={len(matched_std_global)}, cos={len(matched_cos_global)}")
    print(f"  Unique:  std={len(unique_std)}, cos={len(unique_cos)}")

    return (std_feats, cos_feats, matched_std_global, matched_cos_global,
            unique_std, unique_cos, std_alive.tolist(), cos_alive.tolist())


def analyze_norm_context(acts_cpu, std_feats, cos_feats, matched_std, matched_cos,
                         unique_std, unique_cos):
    """For unique vs matched features, characterize the input norms of tokens they fire on."""

    norms = acts_cpu.float().norm(dim=-1)  # (n_tokens,)
    q25, q50, q75 = torch.quantile(norms, torch.tensor([0.25, 0.5, 0.75]))
    print(f"  Token norm quartiles: Q1={q25:.1f} Q2={q50:.1f} Q3={q75:.1f}")
    print(f"  Token norm range: min={norms.min():.1f} max={norms.max():.1f} mean={norms.mean():.1f}")

    quartile_bounds = [float(q25), float(q50), float(q75)]

    def get_activation_norm_dist(feats, feature_set, label=""):
        """Vectorized: for a set of features, compute norm stats of tokens they fire on."""
        feature_list = sorted(list(feature_set))
        if not feature_list:
            return {}

        # Vectorized: slice all features at once, compute per-token "any active" mask
        feat_subset = feats[:, feature_list]  # (n_tokens, n_features)
        active_mask = feat_subset > 0  # bool (n_tokens, n_features)

        # Per-feature frequency
        per_feat_count = active_mask.sum(dim=0).float()  # (n_features,)
        n_tokens_total = feats.shape[0]
        feature_freq = per_feat_count / n_tokens_total

        # Weighted norm: for each activation, the token's norm contributes
        # Weight each token by how many features fired on it from this set
        n_active_per_token = active_mask.sum(dim=1)  # (n_tokens,) — how many features fired
        total_activations = int(n_active_per_token.sum().item())

        if total_activations == 0:
            return {}

        # Weighted norm distribution (weight = number of features active on that token)
        weights = n_active_per_token.float()
        weighted_norms = norms * weights
        mean_norm = weighted_norms.sum() / weights.sum()

        # For quartile distribution: count activations in each norm quartile
        q_fracs = []
        bounds = [0.0] + quartile_bounds + [float(norms.max()) + 1]
        for qi in range(4):
            lo, hi = bounds[qi], bounds[qi + 1]
            in_quartile = ((norms >= lo) & (norms < hi))
            q_count = (weights * in_quartile.float()).sum().item()
            q_fracs.append(q_count / total_activations)

        # Median: approximate from tokens with any activation
        any_active = n_active_per_token > 0
        active_token_norms = norms[any_active]

        active_freqs = feature_freq[feature_freq > 0]
        print(f"    [{label}] {len(feature_list)} features, {total_activations} activations, "
              f"mean_norm={mean_norm:.1f}, quartiles={[f'{x:.3f}' for x in q_fracs]}")

        return {
            "n_features": len(feature_list),
            "n_features_active": int((per_feat_count > 0).sum().item()),
            "mean_freq": float(active_freqs.mean()) if len(active_freqs) > 0 else 0.0,
            "median_freq": float(active_freqs.median()) if len(active_freqs) > 0 else 0.0,
            "activation_norm_mean": float(mean_norm),
            "activation_norm_median": float(active_token_norms.median()) if len(active_token_norms) > 0 else 0.0,
            "activation_norm_std": float(active_token_norms.std()) if len(active_token_norms) > 0 else 0.0,
            "quartile_fracs": q_fracs,
            "n_activations": total_activations,
        }

    print("\n  Analyzing cosine-unique features...")
    cos_unique_stats = get_activation_norm_dist(cos_feats, unique_cos, "cos-unique")

    print("  Analyzing standard-unique features...")
    std_unique_stats = get_activation_norm_dist(std_feats, unique_std, "std-unique")

    print("  Analyzing cosine-matched features...")
    cos_matched_stats = get_activation_norm_dist(cos_feats, matched_cos, "cos-matched")

    print("  Analyzing standard-matched features...")
    std_matched_stats = get_activation_norm_dist(std_feats, matched_std, "std-matched")

    # Cross-architecture: on tokens where cosine-unique features fire, what does standard do?
    print("\n  Cross-architecture token analysis...")
    cos_unique_indices = sorted(list(unique_cos))
    feat_subset = cos_feats[:, cos_unique_indices]
    any_cos_unique_active = (feat_subset > 0).any(dim=1)
    cos_unique_token_idx = any_cos_unique_active.nonzero(as_tuple=True)[0]

    if len(cos_unique_token_idx) > 50000:
        cos_unique_token_idx = cos_unique_token_idx[:50000]

    if len(cos_unique_token_idx) > 0:
        std_on_cos_tokens = std_feats[cos_unique_token_idx]
        std_active_per_token = (std_on_cos_tokens > 0).sum(dim=1).float()
        cos_on_cos_tokens = cos_feats[cos_unique_token_idx]
        cos_active_per_token = (cos_on_cos_tokens > 0).sum(dim=1).float()

        cross_stats = {
            "n_tokens_sampled": len(cos_unique_token_idx),
            "std_features_active_mean": float(std_active_per_token.mean()),
            "cos_features_active_mean": float(cos_active_per_token.mean()),
            "std_features_active_median": float(std_active_per_token.median()),
        }
        print(f"    On {len(cos_unique_token_idx)} tokens where cosine-unique features fire:")
        print(f"      Standard fires {std_active_per_token.mean():.1f} features/token (median {std_active_per_token.median():.0f})")
        print(f"      Cosine fires {cos_active_per_token.mean():.1f} features/token (median {cos_active_per_token.median():.0f})")
    else:
        cross_stats = {}

    return {
        "token_norm_quartiles": [float(q25), float(q50), float(q75)],
        "token_norm_mean": float(norms.mean()),
        "cosine_unique": cos_unique_stats,
        "standard_unique": std_unique_stats,
        "cosine_matched": cos_matched_stats,
        "standard_matched": std_matched_stats,
        "cross_architecture": cross_stats,
    }


def main():
    print("=" * 70)
    print("Exp 58c: Unique Feature Norm Context Analysis")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\nStep 1: Load SAEs")
    sae_std = load_sae("standard")
    sae_cos = load_sae("perfeature_l2")

    print("\nStep 2: Collect activations and find matched/unique features")
    acts = collect_activations(N_TOKENS)
    (std_feats, cos_feats, matched_std, matched_cos,
     unique_std, unique_cos, std_alive, cos_alive) = find_matched_and_unique(
        acts, sae_std, sae_cos
    )

    print(f"\nStep 3: Analyze norm context")
    norm_results = analyze_norm_context(
        acts, std_feats, cos_feats,
        matched_std, matched_cos, unique_std, unique_cos
    )

    results = {
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "n_tokens": N_TOKENS,
            "corr_threshold": CORRELATION_THRESHOLD,
            "decoder_sim_threshold": DECODER_SIM_THRESHOLD,
        },
        "feature_counts": {
            "std_alive": len(std_alive),
            "cos_alive": len(cos_alive),
            "std_matched": len(matched_std),
            "cos_matched": len(matched_cos),
            "std_unique": len(unique_std),
            "cos_unique": len(unique_cos),
        },
        "norm_analysis": norm_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Standard: {len(std_alive)} alive, {len(matched_std)} matched, {len(unique_std)} unique")
    print(f"  Cosine:   {len(cos_alive)} alive, {len(matched_cos)} matched, {len(unique_cos)} unique")

    cu = norm_results.get("cosine_unique", {})
    su = norm_results.get("standard_unique", {})
    if cu and su:
        print(f"\n  Cosine-unique activation norm: mean={cu['activation_norm_mean']:.1f}, quartiles={cu['quartile_fracs']}")
        print(f"  Standard-unique activation norm: mean={su['activation_norm_mean']:.1f}, quartiles={su['quartile_fracs']}")
        print(f"  Overall token norm: mean={norm_results['token_norm_mean']:.1f}")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
