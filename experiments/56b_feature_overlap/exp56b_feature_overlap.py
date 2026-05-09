"""
Experiment 56b: Feature Overlap Analysis (Quality vs Discovery)

Do cosine and standard SAEs learn the same features differently, or different
features entirely? We compute activation correlations between standard and
perfeature_l2 features on shared data, build a bipartite matching, and
classify features as matched, weakly matched, or unique.

Run on <gpu-server> GPU 1:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python3 experiments/exp56b_feature_overlap.py \
        > experiments/exp56b_output.log 2>&1 &
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL = 4096
D_SAE = 65536
K = 80
NORM_EPS = 1e-8
LAYER = 18
CONTEXT_LENGTH = 256

CKPTS = {
    "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
    "adaptive_l2": "/scratch/checkpoints/exp40/adaptive_l2_L18_final.pt",
    "perfeature_l2": "/scratch/checkpoints/exp40/perfeature_l2_L18_final.pt",
    "no_C": "/scratch/checkpoints/exp42c/no_C_L18_final.pt",
}

# We compare standard vs perfeature_l2 (our best cosine variant on sparse probing)
SAE_A = "standard"
SAE_B = "perfeature_l2"

N_TOKENS = 100_000
COLLECTION_BATCH_SIZE = 16
SKIP_DOCS = 200_000

# Matching thresholds
MATCH_STRONG = 0.7
MATCH_WEAK = 0.3
ALIVE_THRESHOLD = 0.001

# Feature activation correlation is computed on alive features only
# to avoid the trivial correlation of dead features (all zeros)
CORR_CHUNK_SIZE = 2000

RESULTS_PATH = "experiments/exp56b_results.json"
SEED = 42


# ─── SAE Architectures (same as exp56a) ──────────────────────────────────────

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
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def forward(self, x):
        f = self.encode(x)
        return f @ self.W_dec + self.b_dec, f


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

    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def forward(self, x):
        f = self.encode(x)
        return f @ self.W_dec + self.b_dec, f


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
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def forward(self, x):
        f = self.encode(x)
        return f @ self.W_dec + self.b_dec, f


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

    def encode(self, x):
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
        return encoded

    def forward(self, x):
        f = self.encode(x)
        w_u = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w_u
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_norm = getattr(self, "_cached_x_norm", None)
        if x_norm is not None:
            x_raw = x_raw * (x_norm / raw_norm)
        return x_raw + self.b_dec, f


SAE_CLASSES = {
    "standard": BatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
    "perfeature_l2": PerFeatureAdaptiveCosineSAE,
    "no_C": NoCBatchTopKSAE,
}


def load_sae(name):
    cls = SAE_CLASSES[name]
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(CKPTS[name], map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


# ─── Data Collection ──────────────────────────────────────────────────────────

class _EarlyStop(Exception):
    pass


def collect_layer_acts(model, layer_idx, inputs):
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


def collect_activations(model, tokenizer, n_tokens):
    """Collect layer activations from FineWeb."""
    from datasets import load_dataset

    print(f"Streaming FineWeb (skipping first {SKIP_DOCS:,} docs)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= SKIP_DOCS:
            break

    all_acts = []
    tokens_collected = 0
    batch_texts = []

    while tokens_collected < n_tokens:
        try:
            row = next(text_iter)
        except StopIteration:
            break
        text = row["text"]
        if len(text) < 50:
            continue
        batch_texts.append(text[:2048])

        if len(batch_texts) >= COLLECTION_BATCH_SIZE:
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=CONTEXT_LENGTH,
            ).to(DEVICE)

            acts = collect_layer_acts(model, LAYER, inputs)
            mask = inputs["attention_mask"].bool()
            flat = acts[mask]  # [n_valid, d_model]

            # Filter outliers
            norms = flat.float().norm(dim=-1)
            median = norms.median()
            if median > 0:
                keep = norms < median * 10.0
                flat = flat[keep]

            all_acts.append(flat.cpu().float())
            tokens_collected += flat.shape[0]
            batch_texts = []

            if tokens_collected % 10000 < COLLECTION_BATCH_SIZE * CONTEXT_LENGTH:
                print(f"  {tokens_collected:,} / {n_tokens:,} tokens")

    if batch_texts:
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=CONTEXT_LENGTH,
        ).to(DEVICE)
        acts = collect_layer_acts(model, LAYER, inputs)
        mask = inputs["attention_mask"].bool()
        flat = acts[mask]
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            keep = norms < median * 10.0
            flat = flat[keep]
        all_acts.append(flat.cpu().float())

    all_acts = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"Collected {all_acts.shape[0]:,} tokens")
    return all_acts


# ─── Feature Encoding ────────────────────────────────────────────────────────

def encode_all(sae, all_acts, batch_size=4096):
    """Encode all activations through an SAE, return [n_tokens, d_sae] on CPU."""
    chunks = []
    for i in range(0, all_acts.shape[0], batch_size):
        batch = all_acts[i:i+batch_size].to(DEVICE, dtype=DTYPE)
        with torch.no_grad():
            feats = sae.encode(batch)
        chunks.append(feats.cpu().float())
    return torch.cat(chunks, dim=0)


def get_alive_mask(features, threshold=ALIVE_THRESHOLD):
    """Return boolean mask of alive features."""
    fire_rates = (features > 0).float().mean(dim=0)
    return fire_rates > threshold


# ─── Correlation Analysis ─────────────────────────────────────────────────────

def compute_max_correlations_chunked(feats_a, alive_a, feats_b, alive_b, chunk_size=CORR_CHUNK_SIZE):
    """Compute max correlation from each feature in A to features in B.

    Returns:
        max_corr_a_to_b: [n_alive_a] max correlation of each A feature with any B feature
        best_match_a_to_b: [n_alive_a] index into alive_b of the best match
        max_corr_b_to_a: [n_alive_b] max correlation of each B feature with any A feature
        best_match_b_to_a: [n_alive_b] index into alive_a of the best match
    """
    alive_a_idx = alive_a.nonzero(as_tuple=True)[0]
    alive_b_idx = alive_b.nonzero(as_tuple=True)[0]

    n_a = alive_a_idx.shape[0]
    n_b = alive_b_idx.shape[0]
    print(f"  Computing correlations: {n_a} x {n_b} alive features")

    # Extract alive feature columns and standardize
    fa = feats_a[:, alive_a_idx].float()  # [n_tokens, n_a]
    fb = feats_b[:, alive_b_idx].float()  # [n_tokens, n_b]

    # Standardize (zero mean, unit variance)
    fa = fa - fa.mean(dim=0, keepdim=True)
    fb = fb - fb.mean(dim=0, keepdim=True)
    fa_std = fa.std(dim=0, keepdim=True).clamp(min=1e-8)
    fb_std = fb.std(dim=0, keepdim=True).clamp(min=1e-8)
    fa = fa / fa_std
    fb = fb / fb_std

    n_tokens = fa.shape[0]

    # Compute max correlation in chunks to avoid OOM
    max_corr_a = torch.full((n_a,), -1.0)
    best_match_a = torch.zeros(n_a, dtype=torch.long)
    max_corr_b = torch.full((n_b,), -1.0)
    best_match_b = torch.zeros(n_b, dtype=torch.long)

    for i in range(0, n_a, chunk_size):
        i_end = min(i + chunk_size, n_a)
        fa_chunk = fa[:, i:i_end]  # [n_tokens, chunk_a]

        for j in range(0, n_b, chunk_size):
            j_end = min(j + chunk_size, n_b)
            fb_chunk = fb[:, j:j_end]  # [n_tokens, chunk_b]

            # Correlation = (fa_chunk^T @ fb_chunk) / n_tokens
            corr_chunk = (fa_chunk.T @ fb_chunk) / n_tokens  # [chunk_a, chunk_b]

            # Update A->B max
            chunk_max_vals, chunk_max_idx = corr_chunk.max(dim=1)
            improved = chunk_max_vals > max_corr_a[i:i_end]
            max_corr_a[i:i_end] = torch.where(improved, chunk_max_vals, max_corr_a[i:i_end])
            best_match_a[i:i_end] = torch.where(improved, chunk_max_idx + j, best_match_a[i:i_end])

            # Update B->A max
            chunk_max_vals_b, chunk_max_idx_b = corr_chunk.max(dim=0)
            improved_b = chunk_max_vals_b > max_corr_b[j:j_end]
            max_corr_b[j:j_end] = torch.where(improved_b, chunk_max_vals_b, max_corr_b[j:j_end])
            best_match_b[j:j_end] = torch.where(improved_b, chunk_max_idx_b + i, best_match_b[j:j_end])

        if (i // chunk_size) % 5 == 0:
            print(f"    Chunk {i//chunk_size}/{(n_a + chunk_size - 1)//chunk_size}")

    return max_corr_a, best_match_a, max_corr_b, best_match_b, alive_a_idx, alive_b_idx


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print(f"Exp 56b: Feature Overlap Analysis ({SAE_A} vs {SAE_B})")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Collect activations
    print(f"\nCollecting {N_TOKENS:,} tokens of L{LAYER} activations...")
    t0 = time.time()
    all_acts = collect_activations(model, tokenizer, N_TOKENS)
    print(f"Collection done in {time.time()-t0:.1f}s")

    # Free model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Encode through both SAEs
    print(f"\nEncoding through {SAE_A}...")
    t0 = time.time()
    sae_a = load_sae(SAE_A)
    feats_a = encode_all(sae_a, all_acts)
    alive_a = get_alive_mask(feats_a)
    n_alive_a = alive_a.sum().item()
    print(f"  {SAE_A}: {n_alive_a} alive features ({n_alive_a/D_SAE*100:.1f}%) in {time.time()-t0:.1f}s")
    del sae_a
    torch.cuda.empty_cache()

    print(f"\nEncoding through {SAE_B}...")
    t0 = time.time()
    sae_b = load_sae(SAE_B)
    feats_b = encode_all(sae_b, all_acts)
    alive_b = get_alive_mask(feats_b)
    n_alive_b = alive_b.sum().item()
    print(f"  {SAE_B}: {n_alive_b} alive features ({n_alive_b/D_SAE*100:.1f}%) in {time.time()-t0:.1f}s")
    del sae_b
    torch.cuda.empty_cache()

    # Free raw activations
    del all_acts
    gc.collect()

    # Compute correlations
    print(f"\nComputing feature correlations...")
    t0 = time.time()
    (max_corr_a, best_match_a, max_corr_b, best_match_b,
     alive_a_idx, alive_b_idx) = compute_max_correlations_chunked(
        feats_a, alive_a, feats_b, alive_b
    )
    print(f"Correlations computed in {time.time()-t0:.1f}s")

    # Classify matches
    strong_a = (max_corr_a >= MATCH_STRONG).sum().item()
    weak_a = ((max_corr_a >= MATCH_WEAK) & (max_corr_a < MATCH_STRONG)).sum().item()
    unmatched_a = (max_corr_a < MATCH_WEAK).sum().item()

    strong_b = (max_corr_b >= MATCH_STRONG).sum().item()
    weak_b = ((max_corr_b >= MATCH_WEAK) & (max_corr_b < MATCH_STRONG)).sum().item()
    unmatched_b = (max_corr_b < MATCH_WEAK).sum().item()

    print(f"\n{'='*60}")
    print(f"Feature Matching: {SAE_A} -> {SAE_B}")
    print(f"{'='*60}")
    print(f"  Strong matches (corr >= {MATCH_STRONG}): {strong_a} / {n_alive_a} ({strong_a/n_alive_a*100:.1f}%)")
    print(f"  Weak matches   (corr {MATCH_WEAK}-{MATCH_STRONG}): {weak_a} / {n_alive_a} ({weak_a/n_alive_a*100:.1f}%)")
    print(f"  Unmatched      (corr < {MATCH_WEAK}): {unmatched_a} / {n_alive_a} ({unmatched_a/n_alive_a*100:.1f}%)")

    print(f"\nFeature Matching: {SAE_B} -> {SAE_A}")
    print(f"  Strong matches (corr >= {MATCH_STRONG}): {strong_b} / {n_alive_b} ({strong_b/n_alive_b*100:.1f}%)")
    print(f"  Weak matches   (corr {MATCH_WEAK}-{MATCH_STRONG}): {weak_b} / {n_alive_b} ({weak_b/n_alive_b*100:.1f}%)")
    print(f"  Unmatched      (corr < {MATCH_WEAK}): {unmatched_b} / {n_alive_b} ({unmatched_b/n_alive_b*100:.1f}%)")

    # Analyze unmatched features by frequency
    fire_rates_a = (feats_a > 0).float().mean(dim=0)
    fire_rates_b = (feats_b > 0).float().mean(dim=0)

    # Frequency of unmatched A features
    unmatched_a_mask = max_corr_a < MATCH_WEAK
    unmatched_a_feats = alive_a_idx[unmatched_a_mask]
    unmatched_a_freqs = fire_rates_a[unmatched_a_feats].numpy()

    matched_a_mask = max_corr_a >= MATCH_STRONG
    matched_a_feats = alive_a_idx[matched_a_mask]
    matched_a_freqs = fire_rates_a[matched_a_feats].numpy()

    unmatched_b_mask = max_corr_b < MATCH_WEAK
    unmatched_b_feats = alive_b_idx[unmatched_b_mask]
    unmatched_b_freqs = fire_rates_b[unmatched_b_feats].numpy()

    matched_b_mask = max_corr_b >= MATCH_STRONG
    matched_b_feats = alive_b_idx[matched_b_mask]
    matched_b_freqs = fire_rates_b[matched_b_feats].numpy()

    print(f"\nUnmatched {SAE_A} feature frequencies:")
    print(f"  Mean: {np.mean(unmatched_a_freqs):.6f}, Median: {np.median(unmatched_a_freqs):.6f}")
    print(f"Matched {SAE_A} feature frequencies:")
    print(f"  Mean: {np.mean(matched_a_freqs):.6f}, Median: {np.median(matched_a_freqs):.6f}")

    print(f"\nUnmatched {SAE_B} feature frequencies:")
    print(f"  Mean: {np.mean(unmatched_b_freqs):.6f}, Median: {np.median(unmatched_b_freqs):.6f}")
    print(f"Matched {SAE_B} feature frequencies:")
    print(f"  Mean: {np.mean(matched_b_freqs):.6f}, Median: {np.median(matched_b_freqs):.6f}")

    # Correlation distribution statistics
    corr_a_np = max_corr_a.numpy()
    corr_b_np = max_corr_b.numpy()

    # Histogram bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist_a, _ = np.histogram(corr_a_np, bins=bins)
    hist_b, _ = np.histogram(corr_b_np, bins=bins)

    print(f"\nCorrelation distribution ({SAE_A} -> {SAE_B}):")
    for i in range(len(bins)-1):
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist_a[i]:6d} ({hist_a[i]/n_alive_a*100:5.1f}%)")

    print(f"\nCorrelation distribution ({SAE_B} -> {SAE_A}):")
    for i in range(len(bins)-1):
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist_b[i]:6d} ({hist_b[i]/n_alive_b*100:5.1f}%)")

    # Decoder weight similarity for matched pairs
    print(f"\nComputing decoder similarity for matched pairs...")
    sae_a_obj = load_sae(SAE_A)
    sae_b_obj = load_sae(SAE_B)

    W_dec_a = F.normalize(sae_a_obj.W_dec.detach().float(), dim=-1)  # [d_sae, d_model]
    W_dec_b = F.normalize(sae_b_obj.W_dec.detach().float(), dim=-1)

    # For strongly matched pairs, compute decoder cosine similarity
    strong_pairs_a = alive_a_idx[matched_a_mask]
    strong_pairs_b = alive_b_idx[best_match_a[matched_a_mask]]

    if len(strong_pairs_a) > 0:
        dec_a = W_dec_a[strong_pairs_a].cpu()
        dec_b = W_dec_b[strong_pairs_b].cpu()
        dec_cos_sim = (dec_a * dec_b).sum(dim=-1)
        print(f"  Decoder cosine similarity for {len(strong_pairs_a)} strongly-matched pairs:")
        print(f"    Mean: {dec_cos_sim.mean().item():.4f}")
        print(f"    Median: {dec_cos_sim.median().item():.4f}")
        print(f"    Min: {dec_cos_sim.min().item():.4f}")
        print(f"    >0.9: {(dec_cos_sim > 0.9).sum().item()} ({(dec_cos_sim > 0.9).float().mean().item()*100:.1f}%)")
        print(f"    >0.7: {(dec_cos_sim > 0.7).sum().item()} ({(dec_cos_sim > 0.7).float().mean().item()*100:.1f}%)")
        dec_cos_sim_np = dec_cos_sim.numpy()
    else:
        dec_cos_sim_np = np.array([])

    del sae_a_obj, sae_b_obj
    torch.cuda.empty_cache()

    # Jaccard index of alive features (by decoder direction overlap)
    # Two features from different SAEs "overlap" if their decoder directions
    # have cosine similarity > 0.8
    print(f"\nComputing Jaccard-style overlap via decoder directions...")
    # This is expensive for all alive features, so sample
    n_sample = min(5000, n_alive_a, n_alive_b)
    sample_a_idx = alive_a_idx[torch.randperm(n_alive_a)[:n_sample]]
    sample_b_idx = alive_b_idx[torch.randperm(n_alive_b)[:n_sample]]

    sae_a_obj = load_sae(SAE_A)
    sae_b_obj = load_sae(SAE_B)
    dec_a_sample = F.normalize(sae_a_obj.W_dec[sample_a_idx].detach().float(), dim=-1).cpu()
    dec_b_sample = F.normalize(sae_b_obj.W_dec[sample_b_idx].detach().float(), dim=-1).cpu()
    del sae_a_obj, sae_b_obj
    torch.cuda.empty_cache()

    # Compute max decoder cosine sim for each A feature to any B feature
    dec_max_sim_a = torch.zeros(n_sample)
    for i in range(0, n_sample, CORR_CHUNK_SIZE):
        i_end = min(i + CORR_CHUNK_SIZE, n_sample)
        sim_chunk = dec_a_sample[i:i_end] @ dec_b_sample.T  # [chunk, n_sample]
        dec_max_sim_a[i:i_end] = sim_chunk.max(dim=1)[0]

    dec_overlap_80 = (dec_max_sim_a > 0.8).sum().item()
    dec_overlap_90 = (dec_max_sim_a > 0.9).sum().item()
    print(f"  Of {n_sample} sampled {SAE_A} features:")
    print(f"    {dec_overlap_80} ({dec_overlap_80/n_sample*100:.1f}%) have a {SAE_B} feature with decoder cos_sim > 0.8")
    print(f"    {dec_overlap_90} ({dec_overlap_90/n_sample*100:.1f}%) have a {SAE_B} feature with decoder cos_sim > 0.9")

    # ─── Save Results ─────────────────────────────────────────────────────────

    results = {
        "config": {
            "model": MODEL_NAME, "layer": LAYER,
            "sae_a": SAE_A, "sae_b": SAE_B,
            "n_tokens": N_TOKENS, "seed": SEED,
            "match_strong": MATCH_STRONG, "match_weak": MATCH_WEAK,
            "alive_threshold": ALIVE_THRESHOLD,
        },
        "alive_counts": {
            SAE_A: n_alive_a,
            SAE_B: n_alive_b,
        },
        "a_to_b": {
            "strong_matches": strong_a,
            "weak_matches": weak_a,
            "unmatched": unmatched_a,
            "strong_pct": strong_a / n_alive_a,
            "unmatched_pct": unmatched_a / n_alive_a,
            "correlation_histogram": hist_a.tolist(),
            "correlation_mean": float(np.mean(corr_a_np)),
            "correlation_median": float(np.median(corr_a_np)),
            "unmatched_freq_mean": float(np.mean(unmatched_a_freqs)) if len(unmatched_a_freqs) > 0 else 0,
            "unmatched_freq_median": float(np.median(unmatched_a_freqs)) if len(unmatched_a_freqs) > 0 else 0,
            "matched_freq_mean": float(np.mean(matched_a_freqs)) if len(matched_a_freqs) > 0 else 0,
            "matched_freq_median": float(np.median(matched_a_freqs)) if len(matched_a_freqs) > 0 else 0,
        },
        "b_to_a": {
            "strong_matches": strong_b,
            "weak_matches": weak_b,
            "unmatched": unmatched_b,
            "strong_pct": strong_b / n_alive_b,
            "unmatched_pct": unmatched_b / n_alive_b,
            "correlation_histogram": hist_b.tolist(),
            "correlation_mean": float(np.mean(corr_b_np)),
            "correlation_median": float(np.median(corr_b_np)),
            "unmatched_freq_mean": float(np.mean(unmatched_b_freqs)) if len(unmatched_b_freqs) > 0 else 0,
            "unmatched_freq_median": float(np.median(unmatched_b_freqs)) if len(unmatched_b_freqs) > 0 else 0,
            "matched_freq_mean": float(np.mean(matched_b_freqs)) if len(matched_b_freqs) > 0 else 0,
            "matched_freq_median": float(np.median(matched_b_freqs)) if len(matched_b_freqs) > 0 else 0,
        },
        "decoder_similarity": {
            "n_pairs": len(dec_cos_sim_np),
            "mean": float(np.mean(dec_cos_sim_np)) if len(dec_cos_sim_np) > 0 else 0,
            "median": float(np.median(dec_cos_sim_np)) if len(dec_cos_sim_np) > 0 else 0,
            "gt_0_9_pct": float((dec_cos_sim_np > 0.9).mean()) if len(dec_cos_sim_np) > 0 else 0,
            "gt_0_7_pct": float((dec_cos_sim_np > 0.7).mean()) if len(dec_cos_sim_np) > 0 else 0,
        },
        "decoder_direction_overlap": {
            "n_sampled": n_sample,
            "overlap_80_pct": dec_overlap_80 / n_sample,
            "overlap_90_pct": dec_overlap_90 / n_sample,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
