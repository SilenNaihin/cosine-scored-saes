"""
Experiment 57b: TPP Probe Weight Analysis

Examines WHY TPP gives different results for standard vs cosine SAEs despite
identical per-feature behavioral power (exp55d).

Approach: Run TPP via SAEBench normally for both SAEs, then compute the
per-feature effect vectors analytically:
    effect_f = (avg_pos_acts - avg_neg_acts) * (probe_weight @ decoder_f)

The key insight from the TPP source code: probes are linear classifiers trained
on LLM activations (model space, D_MODEL), NOT on SAE features. The per-feature
effect is the product of:
  1. How much the feature's mean activation differs between positive/negative class
  2. How aligned the feature's decoder direction is with the probe weight direction

We can compute (2) directly from probe_weight and W_dec without any data.

Method:
1. Run TPP on both SAEs to get full metrics
2. Compute probe_weight @ W_dec.T for both SAEs using the SAME probes
   (probes are trained on LLM activations, so they're architecture-independent)
3. Compare effect distributions: concentration, total magnitude, top features

Run on <gpu-server> GPU 1:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python3 experiments/exp57b_tpp_probe_analysis.py \
        > experiments/exp57b_output.log 2>&1 &
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

RESULTS_PATH = "experiments/exp57b_results.json"
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


def run_tpp(sae_name, bench_sae):
    """Run TPP evaluation via SAEBench."""
    from benchmarks.run_saebench import run_saebench

    out_dir = f"/scratch/saebench_results/exp57b/{sae_name}"
    sae_label = f"exp57b-{sae_name}"

    results = run_saebench(
        bench_sae,
        sae_name=sae_label,
        eval_types=["tpp"],
        output_dir=out_dir,
        llm_batch_size=4,
        device=DEVICE,
        force_rerun=True,
    )
    return results


def analyze_decoder_projection(sae, sae_name):
    """Analyze probe_weight @ W_dec.T structure for a given SAE.

    Since TPP probes are trained on LLM activations (model space),
    the same probe is used regardless of SAE architecture. The per-feature
    effect is: effect_f = activation_diff_f * (probe_weight @ decoder_f).

    Here we analyze the second factor: how the decoder directions project
    onto a random direction (proxy for probe weight). This tells us about
    the geometry, not a specific concept.
    """
    W_dec = sae.W_dec.detach().float()  # (D_SAE, D_MODEL)

    # W_dec norms
    dec_norms = W_dec.norm(dim=1)
    alive_mask = dec_norms > 0.01
    n_alive = alive_mask.sum().item()

    # Compute projections of decoder directions onto 100 random probe-like directions
    # This simulates what happens with real probes
    torch.manual_seed(SEED)
    n_probes = 100
    random_probes = torch.randn(n_probes, D_MODEL, device=DEVICE, dtype=torch.float32)
    random_probes = F.normalize(random_probes, dim=1)

    all_proj_stats = []
    for i in range(n_probes):
        probe_dir = random_probes[i]
        projections = (W_dec.to(DEVICE) @ probe_dir).abs()  # (D_SAE,)
        projections_alive = projections[alive_mask.to(DEVICE)]

        sorted_proj, _ = projections_alive.sort(descending=True)
        total = sorted_proj.sum().item()

        top5_frac = sorted_proj[:5].sum().item() / max(total, 1e-8)
        top10_frac = sorted_proj[:10].sum().item() / max(total, 1e-8)
        top20_frac = sorted_proj[:20].sum().item() / max(total, 1e-8)
        top50_frac = sorted_proj[:50].sum().item() / max(total, 1e-8)
        top100_frac = sorted_proj[:100].sum().item() / max(total, 1e-8)

        # Effective dimensionality
        probs = projections_alive / max(projections_alive.sum().item(), 1e-8)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        eff_dim = math.exp(entropy)

        all_proj_stats.append({
            "top5_frac": top5_frac,
            "top10_frac": top10_frac,
            "top20_frac": top20_frac,
            "top50_frac": top50_frac,
            "top100_frac": top100_frac,
            "eff_dim": eff_dim,
            "total_projection": total,
        })

    # Aggregate across random probes
    result = {
        "n_alive": n_alive,
    }
    for key in all_proj_stats[0].keys():
        values = [s[key] for s in all_proj_stats]
        result[f"mean_{key}"] = float(np.mean(values))
        result[f"std_{key}"] = float(np.std(values))

    print(f"\n  {sae_name} decoder projection analysis (100 random probes):")
    print(f"    Alive features: {n_alive}")
    print(f"    Mean total projection: {result['mean_total_projection']:.4f} ± {result['std_total_projection']:.4f}")
    print(f"    Concentration: top5={result['mean_top5_frac']:.4f} top20={result['mean_top20_frac']:.4f} "
          f"top50={result['mean_top50_frac']:.4f} top100={result['mean_top100_frac']:.4f}")
    print(f"    Effective dim: {result['mean_eff_dim']:.1f} ± {result['std_eff_dim']:.1f}")

    return result


def analyze_activation_differences(sae_std, sae_cos):
    """Compare how positive-vs-negative class activation differences look for both SAEs.

    This is the first factor in TPP effects: (avg_pos - avg_neg) per feature.
    If cosine features have smaller class-discriminative activation differences,
    that would directly explain the TPP gap.
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n  Computing activation differences on bias_in_bios...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()

    # Load a concept dataset: male vs female from bias_in_bios
    ds = load_dataset("LabHC/bias_in_bios_class_set1", split="test")
    male_texts = [x["text"] for x in ds if x["class_label"] == 0][:200]
    female_texts = [x["text"] for x in ds if x["class_label"] == 1][:200]

    def get_mean_features(texts, sae):
        all_feats = []
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=128).to(DEVICE)
            captured = {}
            def hook(module, inp, out):
                captured["act"] = out[0].detach() if isinstance(out, tuple) else out.detach()
            handle = model.model.layers[LAYER].register_forward_hook(hook)
            with torch.no_grad():
                model(**tokens)
            handle.remove()

            # Mean-pool over sequence
            act = captured["act"][0].to(dtype=DTYPE)
            with torch.no_grad():
                feats = sae.encode(act)
            mean_feat = feats.mean(dim=0)
            all_feats.append(mean_feat.cpu())

        return torch.stack(all_feats).float()

    print("    Encoding male texts through standard...")
    std_male = get_mean_features(male_texts, sae_std)
    print("    Encoding female texts through standard...")
    std_female = get_mean_features(female_texts, sae_std)
    print("    Encoding male texts through cosine...")
    cos_male = get_mean_features(male_texts, sae_cos)
    print("    Encoding female texts through cosine...")
    cos_female = get_mean_features(female_texts, sae_cos)

    # Activation difference: avg_pos - avg_neg per feature
    std_diff = (std_male.mean(0) - std_female.mean(0)).abs()
    cos_diff = (cos_male.mean(0) - cos_female.mean(0)).abs()

    # Compare differences
    std_nonzero = (std_diff > 1e-6).sum().item()
    cos_nonzero = (cos_diff > 1e-6).sum().item()

    std_total_diff = std_diff.sum().item()
    cos_total_diff = cos_diff.sum().item()

    sorted_std_diff, _ = std_diff.sort(descending=True)
    sorted_cos_diff, _ = cos_diff.sort(descending=True)

    std_top20_diff = sorted_std_diff[:20].sum().item()
    cos_top20_diff = sorted_cos_diff[:20].sum().item()

    std_top20_frac = std_top20_diff / max(std_total_diff, 1e-8)
    cos_top20_frac = cos_top20_diff / max(cos_total_diff, 1e-8)

    result = {
        "std_nonzero_diff_features": std_nonzero,
        "cos_nonzero_diff_features": cos_nonzero,
        "std_total_diff": std_total_diff,
        "cos_total_diff": cos_total_diff,
        "std_top20_diff": std_top20_diff,
        "cos_top20_diff": cos_top20_diff,
        "std_top20_frac": std_top20_frac,
        "cos_top20_frac": cos_top20_frac,
        "ratio_total_diff": std_total_diff / max(cos_total_diff, 1e-8),
        "ratio_top20_diff": std_top20_diff / max(cos_top20_diff, 1e-8),
    }

    print(f"\n    Activation difference (male vs female):")
    print(f"      Standard: {std_nonzero} nonzero, total={std_total_diff:.4f}, "
          f"top20={std_top20_diff:.4f} ({std_top20_frac:.3f})")
    print(f"      Cosine:   {cos_nonzero} nonzero, total={cos_total_diff:.4f}, "
          f"top20={cos_top20_diff:.4f} ({cos_top20_frac:.3f})")
    print(f"      Ratio (std/cos): total={result['ratio_total_diff']:.2f}x, "
          f"top20={result['ratio_top20_diff']:.2f}x")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    print("=" * 70)
    print("Exp 57b: TPP Probe Weight Analysis")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results = {
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "seed": SEED,
        }
    }

    # Load SAEs
    sae_std = load_sae("standard")
    sae_cos = load_sae("perfeature_l2")

    # Part 1: Run TPP on both SAEs
    for sae_name, sae in [("standard", sae_std), ("perfeature_l2", sae_cos)]:
        print(f"\n{'='*60}")
        print(f"Running TPP on {sae_name}...")
        print(f"{'='*60}")
        bench_sae = wrap_for_saebench(sae_name, sae)
        t0 = time.time()
        tpp_results = run_tpp(sae_name, bench_sae)
        results[f"{sae_name}_tpp"] = tpp_results.get("tpp", {})
        print(f"  TPP completed in {(time.time()-t0)/60:.1f} min")
        del bench_sae
        torch.cuda.empty_cache()

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Part 2: Decoder projection analysis
    print(f"\n{'='*60}")
    print("Decoder Projection Analysis")
    print(f"{'='*60}")

    results["decoder_projection"] = {}
    for sae_name, sae in [("standard", sae_std), ("perfeature_l2", sae_cos)]:
        results["decoder_projection"][sae_name] = analyze_decoder_projection(sae, sae_name)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Part 3: Activation difference analysis
    print(f"\n{'='*60}")
    print("Activation Difference Analysis")
    print(f"{'='*60}")

    results["activation_diff"] = analyze_activation_differences(sae_std, sae_cos)

    # Part 4: Summary comparison
    print(f"\n{'='*70}")
    print("TPP METRIC COMPARISON")
    print(f"{'='*70}")

    for threshold in [2, 5, 10, 20, 50]:
        key_total = f"tpp_threshold_{threshold}_total_metric"
        key_intended = f"tpp_threshold_{threshold}_intended_diff_only"
        key_unintended = f"tpp_threshold_{threshold}_unintended_diff_only"

        for label, tpp_key in [("Standard", "standard_tpp"), ("Cosine", "perfeature_l2_tpp")]:
            tpp = results.get(tpp_key, {})
            metrics = tpp.get("eval_result_metrics", {}).get("sae", {})
            total = metrics.get(key_total, "N/A")
            intended = metrics.get(key_intended, "N/A")
            unintended = metrics.get(key_unintended, "N/A")
            if threshold in [5, 20, 50]:
                print(f"  @{threshold} {label}: total={total} intended={intended} unintended={unintended}")

    print(f"\n{'='*70}")
    print("DECODER PROJECTION COMPARISON")
    print(f"{'='*70}")
    for sae_name in ["standard", "perfeature_l2"]:
        dp = results.get("decoder_projection", {}).get(sae_name, {})
        print(f"  {sae_name}:")
        print(f"    Total projection: {dp.get('mean_total_projection', 'N/A')}")
        print(f"    Top-20 concentration: {dp.get('mean_top20_frac', 'N/A')}")
        print(f"    Effective dim: {dp.get('mean_eff_dim', 'N/A')}")

    print(f"\n{'='*70}")
    print("ACTIVATION DIFF COMPARISON")
    print(f"{'='*70}")
    ad = results.get("activation_diff", {})
    print(f"  Total diff ratio (std/cos): {ad.get('ratio_total_diff', 'N/A')}")
    print(f"  Top-20 diff ratio (std/cos): {ad.get('ratio_top20_diff', 'N/A')}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
