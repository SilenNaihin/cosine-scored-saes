"""
Experiment 58b: Shared-Feature Sparse Probing

Tests whether the sparse probing advantage comes from feature DISCOVERY
(~950 extra unique features) or feature space SEPARABILITY (the same
features create a more separable space in cosine SAEs).

Method:
1. Recompute feature matching with 100K tokens (reproducing exp56b's 7,114 pairs)
2. Create masked SAEs that only expose the ~7,114 matched features
3. Run SAEBench sparse probing on both masked SAEs
4. If gap persists on shared features → separability (cosine feature space
   is inherently more separable even for the same features)
5. If gap disappears → pure discovery (the extra ~950 features drive it)

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 experiments/exp58b_shared_feature_probing.py \
        2>&1 | tee experiments/exp58b_output.log
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
LAYER = 18

CKPTS = {
    "standard": "/data/checkpoints/exp40/standard_L18_final.pt",
    "perfeature_l2": "/data/checkpoints/exp40/perfeature_l2_L18_final.pt",
}

N_TOKENS = 100000
CORRELATION_THRESHOLD = 0.7
DECODER_SIM_THRESHOLD = 0.7
RESULTS_PATH = "experiments/exp58b_results.json"
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


def find_matched_pairs(acts_cpu, sae_std, sae_cos):
    n_tokens = acts_cpu.shape[0]
    batch_size = 512

    print("  Encoding through standard SAE...")
    std_feats = []
    for i in range(0, n_tokens, batch_size):
        batch = acts_cpu[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            f = sae_std.encode(batch)
        std_feats.append(f.cpu())
    std_feats = torch.cat(std_feats, dim=0).float()

    print("  Encoding through cosine SAE...")
    cos_feats = []
    for i in range(0, n_tokens, batch_size):
        batch = acts_cpu[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            f = sae_cos.encode(batch)
        cos_feats.append(f.cpu())
    cos_feats = torch.cat(cos_feats, dim=0).float()

    std_alive = (std_feats.sum(0) > 0).nonzero(as_tuple=True)[0]
    cos_alive = (cos_feats.sum(0) > 0).nonzero(as_tuple=True)[0]
    print(f"  Alive: standard={len(std_alive)}, cosine={len(cos_alive)}")

    std_feats_alive = std_feats[:, std_alive]
    cos_feats_alive = cos_feats[:, cos_alive]

    std_normed = std_feats_alive - std_feats_alive.mean(0, keepdim=True)
    cos_normed = cos_feats_alive - cos_feats_alive.mean(0, keepdim=True)
    std_normed = std_normed / (std_normed.norm(dim=0, keepdim=True) + 1e-8)
    cos_normed = cos_normed / (cos_normed.norm(dim=0, keepdim=True) + 1e-8)

    # GPU-accelerated correlations
    print("  Moving to GPU for correlation...")
    std_normed_gpu = std_normed.half().to(DEVICE)
    cos_normed_gpu = cos_normed.half().to(DEVICE)

    n_std = len(std_alive)
    best_corr_std = torch.zeros(n_std)
    best_match_std = torch.zeros(n_std, dtype=torch.long)

    GPU_CHUNK = 4096
    print(f"  Computing std→cos correlations (GPU)...")
    for i in range(0, n_std, GPU_CHUNK):
        end = min(i + GPU_CHUNK, n_std)
        corr = std_normed_gpu[:, i:end].T @ cos_normed_gpu
        max_corr, max_idx = corr.max(dim=1)
        best_corr_std[i:end] = max_corr.float().cpu()
        best_match_std[i:end] = max_idx.cpu()

    del std_normed_gpu, cos_normed_gpu
    torch.cuda.empty_cache()

    # Filter by decoder cosine similarity
    strong_mask = best_corr_std >= CORRELATION_THRESHOLD
    strong_std_local = strong_mask.nonzero(as_tuple=True)[0]
    strong_cos_local = best_match_std[strong_std_local]

    si_global = std_alive[strong_std_local]
    ci_global = cos_alive[strong_cos_local]

    W_dec_std_normed = F.normalize(sae_std.W_dec.detach().float(), dim=-1)
    W_dec_cos_normed = F.normalize(sae_cos.W_dec.detach().float(), dim=-1)
    dsims = (W_dec_std_normed[si_global] * W_dec_cos_normed[ci_global]).sum(dim=-1)
    pass_mask = dsims > DECODER_SIM_THRESHOLD

    pairs_std_global = si_global[pass_mask.cpu()].tolist()
    pairs_cos_global = ci_global[pass_mask.cpu()].tolist()
    pairs_corr = best_corr_std[strong_std_local][pass_mask.cpu()].tolist()

    print(f"  Found {len(pairs_std_global)} strongly-matched pairs (corr >= {CORRELATION_THRESHOLD}, dec_sim > {DECODER_SIM_THRESHOLD})")
    return pairs_std_global, pairs_cos_global, pairs_corr, std_alive.tolist(), cos_alive.tolist()


def wrap_masked_sae(name, sae, allowed_indices):
    """Wrap SAE for SAEBench but mask encode to only return allowed features."""
    from benchmarks.adapter import BenchSAE
    sae_dtype = sae.W_enc.dtype
    mask = torch.zeros(D_SAE, device=DEVICE, dtype=torch.bool)
    mask[allowed_indices] = True

    def encode_fn(x):
        f = sae.encode(x.to(dtype=sae_dtype))
        f[..., ~mask] = 0.0
        return f.to(dtype=x.dtype)

    def decode_fn(f):
        return sae.decode(f.to(dtype=sae_dtype)).to(dtype=f.dtype)

    W_dec = F.normalize(sae.W_dec.detach(), dim=1)
    b_enc = sae.b_enc.detach()
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


def run_sparse_probing(bench_sae, label):
    from benchmarks.run_saebench import run_saebench
    out_dir = f"/data/saebench_results/exp58b/{label}"
    return run_saebench(
        bench_sae,
        sae_name=f"exp58b-{label}",
        eval_types=["sparse_probing"],
        output_dir=out_dir,
        llm_batch_size=4,
        device=DEVICE,
    )


def main():
    print("=" * 70)
    print("Exp 58b: Shared-Feature Sparse Probing")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\nStep 1: Load SAEs")
    sae_std = load_sae("standard")
    sae_cos = load_sae("perfeature_l2")

    print("\nStep 2: Find matched pairs (100K tokens)")
    acts = collect_activations(N_TOKENS)
    pairs_std, pairs_cos, pairs_corr, std_alive, cos_alive = find_matched_pairs(
        acts, sae_std, sae_cos
    )
    del acts
    gc.collect()
    torch.cuda.empty_cache()

    # Verify with decoder cos_sim
    W_dec_std = sae_std.W_dec.detach().float()
    W_dec_cos = sae_cos.W_dec.detach().float()
    dec_sims = []
    for si, ci in zip(pairs_std, pairs_cos):
        sim = F.cosine_similarity(W_dec_std[si].unsqueeze(0), W_dec_cos[ci].unsqueeze(0)).item()
        dec_sims.append(sim)
    print(f"  Decoder cos_sim for matched pairs: mean={np.mean(dec_sims):.3f} median={np.median(dec_sims):.3f}")

    # Filter to only pairs with high decoder similarity too
    good_pairs = [(s, c, corr, dsim) for s, c, corr, dsim in zip(pairs_std, pairs_cos, pairs_corr, dec_sims) if dsim > 0.7]
    print(f"  Pairs with decoder cos_sim > 0.7: {len(good_pairs)} / {len(pairs_std)}")

    good_std = [p[0] for p in good_pairs]
    good_cos = [p[1] for p in good_pairs]

    print(f"\nStep 3: Run sparse probing on matched-only SAEs ({len(good_std)} features each)")

    print("\n  Running sparse probing on standard (matched features only)...")
    bench_std_matched = wrap_masked_sae("standard", sae_std, good_std)
    t0 = time.time()
    std_matched_results = run_sparse_probing(bench_std_matched, "standard_matched")
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    del bench_std_matched
    torch.cuda.empty_cache()

    print("\n  Running sparse probing on cosine (matched features only)...")
    bench_cos_matched = wrap_masked_sae("perfeature_l2", sae_cos, good_cos)
    t0 = time.time()
    cos_matched_results = run_sparse_probing(bench_cos_matched, "perfeature_l2_matched")
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    del bench_cos_matched
    torch.cuda.empty_cache()

    print(f"\nStep 4: Run sparse probing on full SAEs (control)")

    print("\n  Running sparse probing on standard (all features)...")
    bench_std_full = wrap_masked_sae("standard", sae_std, std_alive)
    t0 = time.time()
    std_full_results = run_sparse_probing(bench_std_full, "standard_full")
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    del bench_std_full
    torch.cuda.empty_cache()

    print("\n  Running sparse probing on cosine (all features)...")
    bench_cos_full = wrap_masked_sae("perfeature_l2", sae_cos, cos_alive)
    t0 = time.time()
    cos_full_results = run_sparse_probing(bench_cos_full, "perfeature_l2_full")
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

    results = {
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "n_tokens": N_TOKENS,
            "corr_threshold": CORRELATION_THRESHOLD,
            "seed": SEED,
        },
        "matching": {
            "total_matched_pairs": len(pairs_std),
            "good_pairs_decoder_gt_0_7": len(good_pairs),
            "mean_corr": float(np.mean(pairs_corr)),
            "mean_decoder_sim": float(np.mean(dec_sims)),
            "std_alive": len(std_alive),
            "cos_alive": len(cos_alive),
        },
        "sparse_probing": {
            "standard_matched": std_matched_results,
            "perfeature_l2_matched": cos_matched_results,
            "standard_full": std_full_results,
            "perfeature_l2_full": cos_full_results,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Matched pairs: {len(good_pairs)} (corr >= {CORRELATION_THRESHOLD}, decoder cos_sim > 0.7)")
    print(f"  Standard alive: {len(std_alive)}, Cosine alive: {len(cos_alive)}")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
