"""
Experiment 58a: TPP Probe Weight Analysis

Directly inspects the cached TPP probes to test: do probes trained on cosine
SAE features distribute weight more evenly than probes on standard features?

This would explain the TPP gap (standard wins low-N, cosine wins high-N)
WITHOUT any difference in the SAE features themselves.

Method:
1. Load the cached TPP probes (trained on LLM activations, architecture-independent)
2. For each probe, compute probe_weight @ W_dec.T for both SAEs
   -> This gives per-feature "effect direction alignment" for each probe
3. Compare weight concentration: entropy, Gini, top-k fraction
4. If cosine probes distribute weight across more features, ablating one
   feature changes the probe output less (low-N disadvantage) but ablating
   many produces less collateral (high-N advantage)

Note: probes are LINEAR classifiers on LLM activations (d_model=4096),
NOT on SAE features. The per-feature effect is:
    effect_f = (mean_pos_act_f - mean_neg_act_f) * (probe_weight @ decoder_f)
We can compute probe_weight @ decoder_f directly without any data.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 experiments/exp58a_probe_weight_analysis.py \
        2>&1 | tee experiments/exp58a_output.log
"""

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda"
DTYPE = torch.bfloat16
D_MODEL, D_SAE, K = 4096, 65536, 80

CKPTS = {
    "standard": "/data/checkpoints/exp40/standard_L18_final.pt",
    "perfeature_l2": "/data/checkpoints/exp40/perfeature_l2_L18_final.pt",
}

PROBE_DIR = "/data/artifacts/tpp/Qwen/Qwen3-8B/blocks.18.hook_resid_post"
RESULTS_PATH = "experiments/exp58a_results.json"


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
    def encode(self, x):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        return F.relu(pre_acts)
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
    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        log_norm = torch.log(input_norm)
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
        return F.relu(scale * cos_sim + self.b_enc)
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


def load_probes():
    """Load cached TPP probes. These are linear classifiers on d_model activations.
    Structure: {dataset: {'llm_probes': {idx: Probe}, 'llm_test_accuracies': {idx: float}}}
    Each Probe has state_dict with 'net.weight' (n_classes, d_model) and 'net.bias'.
    """
    probes = {}
    for fname in Path(PROBE_DIR).glob("*_probes.pkl"):
        dataset = fname.stem.replace("_probes", "")
        with open(fname, "rb") as f:
            data = pickle.load(f)
        # Extract the actual probe objects from the nested dict
        if isinstance(data, dict) and "llm_probes" in data:
            probes[dataset] = data["llm_probes"]
            print(f"  Loaded {len(probes[dataset])} probes for {dataset}")
        else:
            probes[dataset] = data
            print(f"  Loaded probes for {dataset}: {type(data)}")
    return probes


def analyze_probe_decoder_projections(probes, W_dec_std, W_dec_cos):
    """
    For each probe, compute |probe_weight @ decoder_f| for all features.
    This is the "effect alignment" — how much ablating feature f affects the probe.
    Compare concentration between standard and cosine decoders.
    """
    results = {"per_probe": [], "aggregate": {}}

    all_std_entropies = []
    all_cos_entropies = []
    all_std_ginis = []
    all_cos_ginis = []
    all_std_top10_fracs = []
    all_cos_top10_fracs = []
    all_std_top50_fracs = []
    all_cos_top50_fracs = []
    all_std_eff_dims = []
    all_cos_eff_dims = []

    for dataset_name, probe_data in probes.items():
        print(f"\n  Dataset: {dataset_name}")

        # probe_data structure varies — inspect it
        if isinstance(probe_data, dict):
            probe_items = list(probe_data.items())
        elif isinstance(probe_data, list):
            probe_items = list(enumerate(probe_data))
        else:
            print(f"    Unexpected probe type: {type(probe_data)}")
            continue

        for probe_key, probe_obj in probe_items:
            # Extract the linear weight from the probe
            # SAEBench Probe has state_dict with 'net.weight' and 'net.bias'
            weight = None
            if hasattr(probe_obj, 'net'):
                weight = probe_obj.net.weight.detach().float().to(DEVICE)
            elif hasattr(probe_obj, 'state_dict'):
                sd = probe_obj.state_dict()
                if 'net.weight' in sd:
                    weight = sd['net.weight'].float().to(DEVICE)
                elif 'weight' in sd:
                    weight = sd['weight'].float().to(DEVICE)
            elif hasattr(probe_obj, 'linear'):
                weight = probe_obj.linear.weight.detach().float().to(DEVICE)
            elif isinstance(probe_obj, nn.Module):
                for name, module in probe_obj.named_modules():
                    if isinstance(module, nn.Linear):
                        weight = module.weight.detach().float().to(DEVICE)
                        break

            if weight is None:
                print(f"    Probe {probe_key}: can't extract weight from {type(probe_obj)}")
                continue

            # weight shape: (n_classes, d_model) or (1, d_model)
            # For binary classification, take the decision direction
            if weight.shape[0] == 2:
                probe_dir = (weight[1] - weight[0]).unsqueeze(0)  # (1, d_model)
            elif weight.shape[0] == 1:
                probe_dir = weight
            else:
                # Multi-class: use each row separately
                probe_dir = weight  # (n_classes, d_model)

            for class_idx in range(probe_dir.shape[0]):
                w = probe_dir[class_idx]  # (d_model,)

                # Compute |w @ decoder_f| for all features
                # W_dec shape: (d_sae, d_model)
                std_proj = torch.abs(W_dec_std @ w)  # (d_sae,)
                cos_proj = torch.abs(W_dec_cos @ w)  # (d_sae,)

                # Normalize to get distribution
                std_dist = std_proj / (std_proj.sum() + 1e-10)
                cos_dist = cos_proj / (cos_proj.sum() + 1e-10)

                # Entropy
                std_ent = -(std_dist * torch.log(std_dist + 1e-10)).sum().item()
                cos_ent = -(cos_dist * torch.log(cos_dist + 1e-10)).sum().item()

                # Gini coefficient
                def gini(x):
                    sorted_x = torch.sort(x)[0]
                    n = len(sorted_x)
                    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
                    return (2 * (idx * sorted_x).sum() / (n * sorted_x.sum()) - (n + 1) / n).item()

                std_gini = gini(std_proj)
                cos_gini = gini(cos_proj)

                # Top-k concentration
                std_sorted = torch.sort(std_proj, descending=True)[0]
                cos_sorted = torch.sort(cos_proj, descending=True)[0]
                std_total = std_sorted.sum()
                cos_total = cos_sorted.sum()

                std_top10 = (std_sorted[:10].sum() / std_total).item()
                cos_top10 = (cos_sorted[:10].sum() / cos_total).item()
                std_top50 = (std_sorted[:50].sum() / std_total).item()
                cos_top50 = (cos_sorted[:50].sum() / cos_total).item()

                # Effective dimensionality (exp of entropy)
                std_eff_dim = math.exp(std_ent)
                cos_eff_dim = math.exp(cos_ent)

                all_std_entropies.append(std_ent)
                all_cos_entropies.append(cos_ent)
                all_std_ginis.append(std_gini)
                all_cos_ginis.append(cos_gini)
                all_std_top10_fracs.append(std_top10)
                all_cos_top10_fracs.append(cos_top10)
                all_std_top50_fracs.append(std_top50)
                all_cos_top50_fracs.append(cos_top50)
                all_std_eff_dims.append(std_eff_dim)
                all_cos_eff_dims.append(cos_eff_dim)

                results["per_probe"].append({
                    "dataset": dataset_name,
                    "probe_key": str(probe_key),
                    "class_idx": class_idx,
                    "std_entropy": std_ent,
                    "cos_entropy": cos_ent,
                    "std_gini": std_gini,
                    "cos_gini": cos_gini,
                    "std_top10_frac": std_top10,
                    "cos_top10_frac": cos_top10,
                    "std_top50_frac": std_top50,
                    "cos_top50_frac": cos_top50,
                    "std_eff_dim": std_eff_dim,
                    "cos_eff_dim": cos_eff_dim,
                    "std_total_proj": std_total.item(),
                    "cos_total_proj": cos_total.item(),
                })

    n = len(all_std_entropies)
    if n > 0:
        results["aggregate"] = {
            "n_probes": n,
            "std_entropy_mean": float(np.mean(all_std_entropies)),
            "cos_entropy_mean": float(np.mean(all_cos_entropies)),
            "std_gini_mean": float(np.mean(all_std_ginis)),
            "cos_gini_mean": float(np.mean(all_cos_ginis)),
            "std_top10_frac_mean": float(np.mean(all_std_top10_fracs)),
            "cos_top10_frac_mean": float(np.mean(all_cos_top10_fracs)),
            "std_top50_frac_mean": float(np.mean(all_std_top50_fracs)),
            "cos_top50_frac_mean": float(np.mean(all_cos_top50_fracs)),
            "std_eff_dim_mean": float(np.mean(all_std_eff_dims)),
            "cos_eff_dim_mean": float(np.mean(all_cos_eff_dims)),
            "entropy_ratio_cos_over_std": float(np.mean(all_cos_entropies)) / max(float(np.mean(all_std_entropies)), 1e-10),
            "eff_dim_ratio_cos_over_std": float(np.mean(all_cos_eff_dims)) / max(float(np.mean(all_std_eff_dims)), 1e-10),
        }

        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        print(f"  Probes analyzed: {n}")
        print(f"  Entropy:    std={np.mean(all_std_entropies):.2f}  cos={np.mean(all_cos_entropies):.2f}  ratio={results['aggregate']['entropy_ratio_cos_over_std']:.4f}")
        print(f"  Gini:       std={np.mean(all_std_ginis):.4f}  cos={np.mean(all_cos_ginis):.4f}")
        print(f"  Top-10:     std={np.mean(all_std_top10_fracs):.6f}  cos={np.mean(all_cos_top10_fracs):.6f}")
        print(f"  Top-50:     std={np.mean(all_std_top50_fracs):.6f}  cos={np.mean(all_cos_top50_fracs):.6f}")
        print(f"  Eff dim:    std={np.mean(all_std_eff_dims):.1f}  cos={np.mean(all_cos_eff_dims):.1f}  ratio={results['aggregate']['eff_dim_ratio_cos_over_std']:.4f}")

    return results


def main():
    print("=" * 70)
    print("Exp 58a: TPP Probe Weight Analysis")
    print("=" * 70)

    print("\nLoading SAEs...")
    sae_std = load_sae("standard")
    sae_cos = load_sae("perfeature_l2")

    W_dec_std = sae_std.W_dec.detach().float()
    W_dec_cos = sae_cos.W_dec.detach().float()

    print(f"\nDecoder shapes: std={W_dec_std.shape}, cos={W_dec_cos.shape}")

    print("\nLoading TPP probes...")
    probes = load_probes()

    if not probes:
        print("ERROR: No probes found!")
        return

    print("\nAnalyzing probe-decoder projections...")
    results = analyze_probe_decoder_projections(probes, W_dec_std, W_dec_cos)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
