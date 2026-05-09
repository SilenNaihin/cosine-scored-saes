"""
Experiment 60: Decoder Geometry Analysis — Mean-Max-Cosine & Clustering
========================================================================

Measures within-SAE decoder vector disentanglement across architectures.

Key metrics:
  1. Mean-max-cosine (MMC): For each alive feature, compute max cosine similarity
     to any other alive feature. Average over all alive features. Lower = more
     disentangled (features occupy more distinct directions).
  2. Top-k neighbor density: For each feature, count how many neighbors are within
     cos_sim > {0.3, 0.5, 0.7}. Detects local clustering even when mean is low.
  3. Cluster count: Number of feature pairs with cos_sim > 0.5 (tight clusters).
  4. Effective dimensionality: Participation ratio of the singular values of W_dec
     (how many dimensions the decoder actually uses).
  5. Norm-quartile MMC: Split features by their activation frequency / norm
     sensitivity and compute MMC within quartiles to detect if high-norm features
     cluster more.

The hypothesis: cosine SAEs should have LOWER MMC (more disentangled) because
gradient equalization prevents features from converging to a few high-norm attractor
directions. Standard SAEs with 86% Q4-specialized features (exp58c) may cluster
in the high-norm subspace.

Checkpoints analyzed:
  Phase 1 (local, exp47): 5 architectures at L27/50M on Qwen3-8B (d_sae=65536)
  Phase 2 (copied from box 7):
    - exp40: standard/adaptive/perfeature at L18/500M (d_sae=65536, production)
    - exp57: standard/adaptive × {4x, 8x, 16x} expansion on Qwen3-{1.7B, 4B, 8B}
      (tests whether disentanglement scales with dictionary size)

Run on <gpu-server>:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    python3 experiments/exp60_decoder_geometry.py
"""

import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = "/tmp"

import numpy as np
import torch
import torch.nn.functional as F


RESULTS_PATH = Path("experiments/exp60_results.json")
DEVICE = "cuda"
CHUNK_SIZE = 4096


def load_W_dec(path, alive_threshold=None):
    """Load W_dec from checkpoint. Returns (W_dec, num_tokens_since_fired or None)."""
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    W_dec = sd["W_dec"]
    ntsf = ckpt.get("num_tokens_since_fired", sd.get("num_tokens_since_fired", None))
    return W_dec, ntsf


def get_alive_mask(W_dec, ntsf, dead_threshold=10_000_000):
    """Determine alive features. If ntsf available, use it. Otherwise use norm."""
    if ntsf is not None:
        return ntsf < dead_threshold
    norms = W_dec.norm(dim=1)
    return norms > 1e-6


def compute_mmc_chunked(W_dec_normed, chunk_size=CHUNK_SIZE):
    """Compute mean-max-cosine similarity efficiently in chunks.

    For each feature i, finds max(cos_sim(W_dec[i], W_dec[j])) for j != i.
    Returns per-feature max cosine values.
    """
    n = W_dec_normed.shape[0]
    max_cos = torch.zeros(n, device=W_dec_normed.device)

    for i in range(0, n, chunk_size):
        chunk = W_dec_normed[i:i + chunk_size]
        sims = chunk @ W_dec_normed.T
        # Zero out self-similarity
        for k in range(chunk.shape[0]):
            sims[k, i + k] = -1.0
        chunk_max = sims.max(dim=1).values
        max_cos[i:i + chunk_size] = chunk_max

    return max_cos


def compute_neighbor_counts(W_dec_normed, thresholds=(0.3, 0.5, 0.7), chunk_size=CHUNK_SIZE):
    """Count neighbors above each threshold for each feature."""
    n = W_dec_normed.shape[0]
    counts = {t: torch.zeros(n, device=W_dec_normed.device) for t in thresholds}

    for i in range(0, n, chunk_size):
        chunk = W_dec_normed[i:i + chunk_size]
        sims = chunk @ W_dec_normed.T
        # Zero self
        for k in range(chunk.shape[0]):
            sims[k, i + k] = 0.0
        for t in thresholds:
            counts[t][i:i + chunk_size] = (sims > t).sum(dim=1).float()

    return counts


def compute_effective_dim(W_dec):
    """Participation ratio of singular values = (sum(s))^2 / sum(s^2)."""
    _, s, _ = torch.svd_lowrank(W_dec.float(), q=min(W_dec.shape[0], W_dec.shape[1], 512))
    s2 = s ** 2
    return (s2.sum() ** 2 / (s2 ** 2).sum()).item()


def analyze_checkpoint(name, path, device=DEVICE):
    """Full geometry analysis for one checkpoint."""
    print(f"\n  Analyzing: {name}")
    print(f"    Path: {path}")
    t0 = time.time()

    W_dec, ntsf = load_W_dec(path)
    d_sae, d_model = W_dec.shape
    print(f"    Shape: {d_sae} × {d_model}")

    alive_mask = get_alive_mask(W_dec, ntsf)
    n_alive = alive_mask.sum().item()
    n_dead = d_sae - n_alive
    print(f"    Alive: {n_alive:,} ({n_alive/d_sae:.1%}), Dead: {n_dead:,}")

    if n_alive < 100:
        print(f"    SKIP: too few alive features")
        return {
            "name": name, "d_sae": d_sae, "d_model": d_model,
            "n_alive": n_alive, "n_dead": n_dead, "skipped": True,
        }

    W_alive = W_dec[alive_mask].to(device, dtype=torch.float32)
    W_normed = F.normalize(W_alive, dim=1)

    # 1. Mean-max-cosine
    print(f"    Computing MMC ({n_alive:,} features)...", end=" ", flush=True)
    max_cos = compute_mmc_chunked(W_normed)
    mmc = max_cos.mean().item()
    mmc_std = max_cos.std().item()
    mmc_median = max_cos.median().item()
    mmc_p95 = max_cos.quantile(0.95).item()
    mmc_p99 = max_cos.quantile(0.99).item()
    print(f"MMC={mmc:.5f}")

    # 2. Neighbor counts
    print(f"    Computing neighbor counts...", end=" ", flush=True)
    thresholds = [0.3, 0.5, 0.7]
    counts = compute_neighbor_counts(W_normed, thresholds)
    neighbor_stats = {}
    for t in thresholds:
        c = counts[t]
        neighbor_stats[f"mean_neighbors_gt{t}"] = c.mean().item()
        neighbor_stats[f"max_neighbors_gt{t}"] = c.max().item()
        neighbor_stats[f"pct_with_neighbor_gt{t}"] = (c > 0).float().mean().item()
    print("done")

    # 3. Cluster pairs (count pairs > 0.5)
    n_pairs_gt05 = int(counts[0.5].sum().item()) // 2  # each pair counted twice
    n_pairs_gt07 = int(counts[0.7].sum().item()) // 2

    # 4. Effective dimensionality
    print(f"    Computing effective dimensionality...", end=" ", flush=True)
    eff_dim = compute_effective_dim(W_alive)
    eff_dim_ratio = eff_dim / d_model
    print(f"eff_dim={eff_dim:.1f} ({eff_dim_ratio:.2%} of d_model)")

    # 5. Distribution of max-cosine values
    hist_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist = torch.histogram(max_cos.cpu().float(), bins=torch.tensor(hist_edges))
    mmc_histogram = {f"{hist_edges[i]:.1f}-{hist_edges[i+1]:.1f}": int(hist.hist[i].item())
                     for i in range(len(hist_edges) - 1)}

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    result = {
        "name": name,
        "d_sae": d_sae,
        "d_model": d_model,
        "n_alive": n_alive,
        "n_dead": n_dead,
        "dead_frac": n_dead / d_sae,
        "mmc_mean": mmc,
        "mmc_std": mmc_std,
        "mmc_median": mmc_median,
        "mmc_p95": mmc_p95,
        "mmc_p99": mmc_p99,
        "effective_dim": eff_dim,
        "effective_dim_ratio": eff_dim_ratio,
        "n_pairs_gt05": n_pairs_gt05,
        "n_pairs_gt07": n_pairs_gt07,
        "mmc_histogram": mmc_histogram,
        "elapsed_s": elapsed,
    }
    result.update(neighbor_stats)
    return result


def main():
    print("Experiment 60: Decoder Geometry Analysis")
    print("=" * 70)

    results = {"config": {"experiment": "exp60_decoder_geometry"}, "runs": {}}

    # Phase 1: exp47 checkpoints (local on box 3)
    exp47_dir = Path("/mnt/checkpoints/exp47")
    exp47_checkpoints = {
        "exp47_perfeature_original_L27": exp47_dir / "perfeature_original_L27_final.pt",
        "exp47_perfeature_base_delta_L27": exp47_dir / "perfeature_base_delta_L27_final.pt",
        "exp47_perfeature_var_reg_L27": exp47_dir / "perfeature_var_reg_L27_final.pt",
        "exp47_perfeature_gaussian_L27": exp47_dir / "perfeature_gaussian_L27_final.pt",
        "exp47_adaptive_l2_L27": exp47_dir / "adaptive_l2_L27_final.pt",
    }

    # Phase 2: exp40 checkpoints (production 500M, d_sae=65536)
    exp40_dir = Path("/mnt/checkpoints/exp40")
    exp40_checkpoints = {
        "exp40_standard_L18_500M": exp40_dir / "standard_L18_final.pt",
        "exp40_adaptive_l2_L18_500M": exp40_dir / "adaptive_l2_L18_final.pt",
        "exp40_perfeature_l2_L18_500M": exp40_dir / "perfeature_l2_L18_final.pt",
    }

    # Phase 3: exp36 checkpoints (500M, d_sae=16384, multi-layer)
    exp36_dir = Path("/mnt/checkpoints/exp36")
    exp36_checkpoints = {
        "exp36_standard_L9_500M": exp36_dir / "standard_L9_final.pt",
        "exp36_adaptive_l2_L9_500M": exp36_dir / "adaptive_l2_L9_final.pt",
        "exp36_standard_L18_500M": exp36_dir / "standard_L18_final.pt",
        "exp36_adaptive_l2_L18_500M": exp36_dir / "adaptive_l2_L18_final.pt",
        "exp36_standard_L27_500M": exp36_dir / "standard_L27_final.pt",
        "exp36_adaptive_l2_L27_500M": exp36_dir / "adaptive_l2_L27_final.pt",
    }

    # Phase 4: exp57 scaling matrix (standard vs adaptive × expansion ratio)
    exp57_dir = Path("/home/<user>/MechInter--RNH/checkpoints/exp57")
    exp57_checkpoints = {
        "exp57_1.7b_4x_standard": exp57_dir / "1.7b_4x/standard_final.pt",
        "exp57_1.7b_4x_adaptive": exp57_dir / "1.7b_4x/adaptive_l2_final.pt",
        "exp57_1.7b_8x_standard": exp57_dir / "1.7b_8x/standard_final.pt",
        "exp57_1.7b_8x_adaptive": exp57_dir / "1.7b_8x/adaptive_l2_final.pt",
        "exp57_1.7b_16x_standard": exp57_dir / "1.7b_16x/standard_final.pt",
        "exp57_1.7b_16x_adaptive": exp57_dir / "1.7b_16x/adaptive_l2_final.pt",
        "exp57_4b_4x_standard": exp57_dir / "4b_4x/standard_final.pt",
        "exp57_4b_4x_adaptive": exp57_dir / "4b_4x/adaptive_l2_final.pt",
        "exp57_4b_8x_standard": exp57_dir / "4b_8x/standard_final.pt",
        "exp57_4b_8x_adaptive": exp57_dir / "4b_8x/adaptive_l2_final.pt",
        "exp57_4b_16x_standard": exp57_dir / "4b_16x/standard_final.pt",
        "exp57_4b_16x_adaptive": exp57_dir / "4b_16x/adaptive_l2_final.pt",
        "exp57_8b_4x_standard": exp57_dir / "8b_4x/standard_final.pt",
        "exp57_8b_4x_adaptive": exp57_dir / "8b_4x/adaptive_l2_final.pt",
        "exp57_8b_8x_standard": exp57_dir / "8b_8x/standard_final.pt",
        "exp57_8b_8x_adaptive": exp57_dir / "8b_8x/adaptive_l2_final.pt",
    }

    all_checkpoints = {}
    all_checkpoints.update(exp47_checkpoints)
    all_checkpoints.update(exp40_checkpoints)
    all_checkpoints.update(exp36_checkpoints)
    all_checkpoints.update(exp57_checkpoints)

    for name, path in all_checkpoints.items():
        if not path.exists():
            print(f"\n  SKIP (not found): {name} — {path}")
            continue
        try:
            result = analyze_checkpoint(name, path)
            results["runs"][name] = result
        except Exception as e:
            print(f"\n  ERROR on {name}: {e}")
            results["runs"][name] = {"name": name, "error": str(e)}

        # Save incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Summary — Mean-Max-Cosine (lower = more disentangled)")
    print(f"{'=' * 70}")
    print(f"  {'Name':<45s} {'Alive':>7s} {'MMC':>8s} {'p95':>7s} {'EffDim':>7s} {'Pairs>0.5':>10s}")
    print(f"  {'-'*45} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*10}")
    for name, r in sorted(results["runs"].items()):
        if r.get("skipped") or r.get("error"):
            continue
        print(f"  {name:<45s} {r['n_alive']:>7,d} {r['mmc_mean']:>8.5f} "
              f"{r['mmc_p95']:>7.4f} {r['effective_dim']:>7.1f} {r['n_pairs_gt05']:>10,d}")

    print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
