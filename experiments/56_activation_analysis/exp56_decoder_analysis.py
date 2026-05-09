"""
Experiment 56 — Decoder Norm & Orthogonality Analysis

Quick analysis (no GPU needed for inference) of decoder weight properties
across all SAE architectures. Tests whether decoder norms explain the
TPP power gap from exp55a.

Run on dev-box-5 (CPU-only, fast):
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    python3 experiments/exp56_decoder_analysis.py
"""

import json
import torch
import torch.nn.functional as F
import numpy as np

D_MODEL, D_SAE, K = 4096, 65536, 80

CKPTS = {
    "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
    "adaptive_l2": "/scratch/checkpoints/exp40/adaptive_l2_L18_final.pt",
    "perfeature_l2": "/scratch/checkpoints/exp40/perfeature_l2_L18_final.pt",
    "no_C": "/scratch/checkpoints/exp42c/no_C_L18_final.pt",
}

RESULTS_PATH = "experiments/exp56_decoder_analysis.json"


def analyze_sae(name, sd):
    W_dec = sd["W_dec"].float()
    W_enc = sd["W_enc"].float()

    dec_norms = W_dec.norm(dim=-1)
    enc_norms = W_enc.norm(dim=-1)

    dec_unit = F.normalize(W_dec, dim=-1)
    enc_unit = F.normalize(W_enc, dim=-1)

    # Pairwise cosine (sample for speed)
    n_sample = 3000
    idx = torch.randperm(D_SAE)[:n_sample]
    dec_sample = dec_unit[idx]
    cos_matrix = dec_sample @ dec_sample.T
    mask = ~torch.eye(n_sample, dtype=torch.bool)
    pairwise = cos_matrix[mask]

    # Encoder-decoder alignment
    alignment = (enc_unit * dec_unit).sum(dim=-1)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Decoder norms: mean={dec_norms.mean():.4f} std={dec_norms.std():.4f} "
          f"min={dec_norms.min():.4f} max={dec_norms.max():.4f} median={dec_norms.median():.4f}")
    print(f"  Encoder norms: mean={enc_norms.mean():.4f} std={enc_norms.std():.4f} "
          f"min={enc_norms.min():.4f} max={enc_norms.max():.4f} median={enc_norms.median():.4f}")
    print(f"  Pairwise decoder cos_sim: mean={pairwise.mean():.6f} std={pairwise.std():.4f} "
          f"max={pairwise.max():.4f} |>0.5|={((pairwise.abs() > 0.5).float().mean()*100):.2f}%")
    print(f"  Enc-dec alignment: mean={alignment.mean():.4f} std={alignment.std():.4f} "
          f"min={alignment.min():.4f} max={alignment.max():.4f}")

    if "scale_a" in sd:
        sa = sd["scale_a"]
        sb = sd["scale_b"]
        if sa.dim() == 0:
            print(f"  scale_a={sa.item():.4f} scale_b={sb.item():.4f} (global)")
        else:
            print(f"  scale_a: mean={sa.mean():.4f} std={sa.std():.4f} min={sa.min():.4f} max={sa.max():.4f}")
            print(f"  scale_b: mean={sb.mean():.4f} std={sb.std():.4f} min={sb.min():.4f} max={sb.max():.4f}")

    if "threshold" in sd:
        print(f"  Threshold: {sd['threshold'].item():.4f}")

    return {
        "dec_norm_mean": dec_norms.mean().item(),
        "dec_norm_std": dec_norms.std().item(),
        "dec_norm_median": dec_norms.median().item(),
        "dec_norm_min": dec_norms.min().item(),
        "dec_norm_max": dec_norms.max().item(),
        "enc_norm_mean": enc_norms.mean().item(),
        "enc_norm_std": enc_norms.std().item(),
        "pairwise_cos_mean": pairwise.mean().item(),
        "pairwise_cos_std": pairwise.std().item(),
        "pairwise_gt05_pct": (pairwise.abs() > 0.5).float().mean().item(),
        "pairwise_gt03_pct": (pairwise.abs() > 0.3).float().mean().item(),
        "enc_dec_alignment_mean": alignment.mean().item(),
        "enc_dec_alignment_std": alignment.std().item(),
    }


def main():
    results = {}

    for name, path in CKPTS.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        results[name] = analyze_sae(name, sd)

    # adamkarvonen reference
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="adamkarvonen/qwen3-8b-saes",
        filename="saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/trainer_2/ae.pt",
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ref_sd = {
        "W_enc": ckpt["encoder.weight"],
        "W_dec": ckpt["decoder.weight"].T.contiguous(),
        "threshold": ckpt["threshold"],
    }
    results["adamkarvonen_ref"] = analyze_sae("adamkarvonen_ref", ref_sd)
    print(f"  Threshold: {ckpt['threshold'].item():.4f}")
    print(f"  k: {ckpt['k'].item()}")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    header = f"{'SAE':25s} {'dec_norm':>10s} {'enc_norm':>10s} {'pw_cos':>10s} {'|>0.5|%':>8s} {'|>0.3|%':>8s} {'align':>8s}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:25s} {r['dec_norm_mean']:10.4f} {r['enc_norm_mean']:10.4f} "
              f"{r['pairwise_cos_mean']:10.6f} {r['pairwise_gt05_pct']*100:8.2f} "
              f"{r['pairwise_gt03_pct']*100:8.2f} {r['enc_dec_alignment_mean']:8.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
