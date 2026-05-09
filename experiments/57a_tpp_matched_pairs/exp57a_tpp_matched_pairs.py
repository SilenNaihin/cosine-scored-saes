"""
Experiment 57a: Per-Feature TPP on Matched Pairs

Tests whether the TPP gap persists when comparing the SAME feature across
architectures. Exp56b found 7,114 strongly-matched feature pairs (activation
correlation >= 0.7, decoder cosine sim 0.91 mean). If the TPP gap persists
on these matched pairs, the gap is per-feature. If it disappears, the gap
is an artifact of different feature selection by the TPP pipeline.

Method:
1. Recompute feature correlations to find matched pairs (exp56b didn't save indices)
2. Run TPP evaluation on both SAEs
3. Compare TPP effects specifically for matched feature pairs

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp57a_tpp_matched_pairs.py \
        > experiments/exp57a_output.log 2>&1 &
"""

import gc
import json
import math
import sys
import time
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
NORM_EPS = 1e-8
LAYER = 18

CKPTS = {
    "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
    "perfeature_l2": "/scratch/checkpoints/exp40/perfeature_l2_L18_final.pt",
}

N_TOKENS = 50000
CORRELATION_THRESHOLD = 0.7
CHUNK_SIZE = 2000
RESULTS_PATH = "experiments/exp57a_results.json"
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
    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


SAE_CLASSES = {
    "standard": BatchTopKSAE,
    "perfeature_l2": PerFeatureAdaptiveCosineSAE,
}


def load_sae(name):
    cls = SAE_CLASSES[name]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(CKPTS[name], map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


def wrap_for_saebench(name, sae):
    from benchmarks.adapter import BenchSAE
    sae_dtype = sae.W_enc.dtype

    def encode_fn(x):
        return sae.encode(x.to(dtype=sae_dtype)).to(dtype=x.dtype)

    def decode_fn(f):
        return sae.decode(f.to(dtype=sae_dtype)).to(dtype=f.dtype)

    W_dec = F.normalize(sae.W_dec.detach(), dim=1)
    b_enc = sae.b_enc.detach() if hasattr(sae, "b_enc") else torch.zeros(
        D_SAE, device=DEVICE, dtype=sae.W_enc.dtype
    )
    return BenchSAE(
        W_enc=sae.W_enc.detach().T,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=sae.b_dec.detach(),
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name=MODEL_NAME,
        hook_layer=LAYER,
        device=DEVICE,
        dtype=DTYPE,
    )


def collect_tokens(n_tokens):
    """Collect activations from FineWeb for correlation computation."""
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
    ctx_len = 128

    print(f"Collecting {n_tokens} tokens...")
    for sample in ds:
        tokens = tokenizer(sample["text"], return_tensors="pt", truncation=True,
                          max_length=ctx_len).to(DEVICE)
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
    print(f"Collected {all_acts.shape[0]} tokens")

    del model
    torch.cuda.empty_cache()
    return all_acts


def find_matched_pairs(acts_cpu, sae_std, sae_cos):
    """Find strongly-correlated feature pairs between standard and cosine SAEs."""
    n_tokens = acts_cpu.shape[0]
    batch_size = 512

    # Encode all tokens through both SAEs
    print("Encoding through standard SAE...")
    std_feats = []
    for i in range(0, n_tokens, batch_size):
        batch = acts_cpu[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            f = sae_std.encode(batch)
        std_feats.append(f.cpu())
    std_feats = torch.cat(std_feats, dim=0).float()

    print("Encoding through cosine SAE...")
    cos_feats = []
    for i in range(0, n_tokens, batch_size):
        batch = acts_cpu[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            f = sae_cos.encode(batch)
        cos_feats.append(f.cpu())
    cos_feats = torch.cat(cos_feats, dim=0).float()

    # Find alive features
    std_alive = (std_feats.sum(0) > 0).nonzero(as_tuple=True)[0]
    cos_alive = (cos_feats.sum(0) > 0).nonzero(as_tuple=True)[0]
    print(f"Alive: standard={len(std_alive)}, cosine={len(cos_alive)}")

    # Compute chunked correlation: for each standard alive feature, find best cosine match
    std_feats_alive = std_feats[:, std_alive]
    cos_feats_alive = cos_feats[:, cos_alive]

    # Normalize for correlation
    std_normed = std_feats_alive - std_feats_alive.mean(0, keepdim=True)
    cos_normed = cos_feats_alive - cos_feats_alive.mean(0, keepdim=True)
    std_normed = std_normed / (std_normed.norm(dim=0, keepdim=True) + 1e-8)
    cos_normed = cos_normed / (cos_normed.norm(dim=0, keepdim=True) + 1e-8)

    print(f"Computing correlations in chunks...")
    n_std = len(std_alive)
    best_corr = torch.zeros(n_std)
    best_match = torch.zeros(n_std, dtype=torch.long)

    for i in range(0, n_std, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, n_std)
        chunk_std = std_normed[:, i:end]
        corr_chunk = chunk_std.T @ cos_normed  # (chunk, n_cos)
        max_corr, max_idx = corr_chunk.max(dim=1)
        best_corr[i:end] = max_corr
        best_match[i:end] = max_idx
        if (i // CHUNK_SIZE) % 3 == 0:
            print(f"  Chunk {i//CHUNK_SIZE + 1}/{(n_std + CHUNK_SIZE - 1)//CHUNK_SIZE}")

    # Filter for strong matches
    strong_mask = best_corr >= CORRELATION_THRESHOLD
    strong_std_local = strong_mask.nonzero(as_tuple=True)[0]
    strong_cos_local = best_match[strong_std_local]

    # Map back to global feature indices
    pairs_std = std_alive[strong_std_local].tolist()
    pairs_cos = cos_alive[strong_cos_local].tolist()
    pairs_corr = best_corr[strong_std_local].tolist()

    print(f"Found {len(pairs_std)} strongly-matched pairs (corr >= {CORRELATION_THRESHOLD})")
    return pairs_std, pairs_cos, pairs_corr


def run_tpp_and_extract_effects(sae_name, sae, bench_sae):
    """Run TPP pipeline and extract per-feature effects."""
    from benchmarks.run_saebench import run_saebench

    out_dir = f"/scratch/saebench_results/exp57a/{sae_name}"
    sae_label = f"exp57a-{sae_name}"

    results = run_saebench(
        bench_sae,
        sae_name=sae_label,
        eval_types=["tpp"],
        output_dir=out_dir,
        llm_batch_size=4,
        device=DEVICE,
    )
    return results


def main():
    print("=" * 70)
    print("Exp 57a: Per-Feature TPP on Matched Pairs")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Step 1: Load SAEs
    print("\nLoading SAEs...")
    sae_std = load_sae("standard")
    sae_cos = load_sae("perfeature_l2")

    # Step 2: Find matched pairs
    print("\nFinding matched feature pairs...")
    acts = collect_tokens(N_TOKENS)
    pairs_std, pairs_cos, pairs_corr = find_matched_pairs(acts, sae_std, sae_cos)

    del acts
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Run TPP on both SAEs
    print("\n" + "=" * 60)
    print("Running TPP on standard SAE...")
    print("=" * 60)
    bench_std = wrap_for_saebench("standard", sae_std)
    std_results = run_tpp_and_extract_effects("standard", sae_std, bench_std)
    del bench_std
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Running TPP on perfeature_l2 SAE...")
    print("=" * 60)
    bench_cos = wrap_for_saebench("perfeature_l2", sae_cos)
    cos_results = run_tpp_and_extract_effects("perfeature_l2", sae_cos, bench_cos)
    del bench_cos
    torch.cuda.empty_cache()

    # Step 4: Analyze TPP metrics directly from results
    # SAEBench TPP gives us per-threshold total_metric, intended, unintended
    # We also need per-feature effect magnitudes which require access to internal TPP state
    # For now, compare the aggregate metrics and note the matched pair info

    results = {
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "n_tokens_for_matching": N_TOKENS,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "seed": SEED,
        },
        "matched_pairs": {
            "count": len(pairs_std),
            "mean_correlation": float(np.mean(pairs_corr)) if pairs_corr else 0,
            "std_indices": pairs_std[:100],
            "cos_indices": pairs_cos[:100],
            "correlations": pairs_corr[:100],
        },
        "standard_tpp": std_results.get("tpp", {}),
        "perfeature_l2_tpp": cos_results.get("tpp", {}),
    }

    # Step 5: Compute per-feature TPP effect proxy using decoder projection
    # This approximates TPP's effect computation: effect_f = (avg_pos - avg_neg) * (probe_weight @ decoder_f)
    # We can't access the actual TPP probes, but we can compare decoder projections for matched pairs
    print("\n" + "=" * 60)
    print("Analyzing decoder geometry for matched pairs...")
    print("=" * 60)

    W_dec_std = sae_std.W_dec.detach().float()
    W_dec_cos = sae_cos.W_dec.detach().float()

    # For matched pairs: compare decoder directions
    matched_decoder_cos_sims = []
    matched_decoder_norm_ratios = []
    for s_idx, c_idx in zip(pairs_std, pairs_cos):
        d_std = W_dec_std[s_idx]
        d_cos = W_dec_cos[c_idx]
        cos_sim = F.cosine_similarity(d_std.unsqueeze(0), d_cos.unsqueeze(0)).item()
        norm_ratio = d_cos.norm().item() / max(d_std.norm().item(), 1e-8)
        matched_decoder_cos_sims.append(cos_sim)
        matched_decoder_norm_ratios.append(norm_ratio)

    # For unmatched standard features: what's different?
    matched_std_set = set(pairs_std)
    unmatched_std = [i for i in range(D_SAE) if i not in matched_std_set and
                     W_dec_std[i].norm().item() > 0.01][:1000]

    results["decoder_analysis"] = {
        "matched_pairs_decoder_cos_sim_mean": float(np.mean(matched_decoder_cos_sims)),
        "matched_pairs_decoder_cos_sim_median": float(np.median(matched_decoder_cos_sims)),
        "matched_pairs_decoder_norm_ratio_mean": float(np.mean(matched_decoder_norm_ratios)),
        "n_matched": len(pairs_std),
        "n_unmatched_std": len(unmatched_std),
    }

    print(f"\nMatched pairs decoder analysis:")
    print(f"  Decoder cos_sim: mean={np.mean(matched_decoder_cos_sims):.4f} "
          f"median={np.median(matched_decoder_cos_sims):.4f}")
    print(f"  Decoder norm ratio (cos/std): mean={np.mean(matched_decoder_norm_ratios):.4f}")

    # Step 6: Compare TPP metrics
    print(f"\n{'='*70}")
    print("TPP COMPARISON")
    print(f"{'='*70}")

    std_tpp = results.get("standard_tpp", {})
    cos_tpp = results.get("perfeature_l2_tpp", {})

    std_metrics = std_tpp.get("eval_result_metrics", {}).get("sae", {})
    cos_metrics = cos_tpp.get("eval_result_metrics", {}).get("sae", {})

    for threshold in [2, 5, 10, 20, 50]:
        key_total = f"tpp_threshold_{threshold}_total_metric"
        key_intended = f"tpp_threshold_{threshold}_intended_diff_only"
        key_unintended = f"tpp_threshold_{threshold}_unintended_diff_only"

        std_total = std_metrics.get(key_total, "N/A")
        cos_total = cos_metrics.get(key_total, "N/A")
        std_int = std_metrics.get(key_intended, "N/A")
        cos_int = cos_metrics.get(key_intended, "N/A")
        std_unint = std_metrics.get(key_unintended, "N/A")
        cos_unint = cos_metrics.get(key_unintended, "N/A")

        print(f"\n  @{threshold}:")
        print(f"    Standard:  total={std_total}  intended={std_int}  unintended={std_unint}")
        print(f"    Cosine:    total={cos_total}  intended={cos_int}  unintended={cos_unint}")
        if isinstance(std_total, (int, float)) and isinstance(cos_total, (int, float)):
            print(f"    Ratio:     {std_total/max(cos_total, 1e-10):.2f}x")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
