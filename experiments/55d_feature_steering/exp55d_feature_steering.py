"""
Experiment 55d: Feature Steering Behavioral Test

Direct test of per-feature intervention power. For well-understood features
(from sparse probing), amplify/ablate individual features and measure
behavioral change (KL divergence, next-token probability shift).

This tests the TPP finding from exp55a: do cosine features really produce
less behavioral change when perturbed? TPP measures probe output change;
this measures actual model output change.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp55d_feature_steering.py \
        > experiments/exp55d_output.log 2>&1 &
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
CONTEXT_LENGTH = 128

CKPTS = {
    "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
    "adaptive_l2": "/scratch/checkpoints/exp40/adaptive_l2_L18_final.pt",
    "perfeature_l2": "/scratch/checkpoints/exp40/perfeature_l2_L18_final.pt",
    "no_C": "/scratch/checkpoints/exp42c/no_C_L18_final.pt",
}

# Steering parameters
AMPLIFY_FACTORS = [0.0, 0.5, 2.0, 5.0]  # 0.0 = ablation, 0.5 = dampen, 2.0/5.0 = amplify
N_PROMPTS = 50
MAX_NEW_TOKENS = 20

RESULTS_PATH = "experiments/exp55d_results.json"
SEED = 42

# Test prompts spanning different domains
TEST_PROMPTS = [
    "The president of France announced that",
    "In a recent scientific study, researchers found that",
    "The stock market today showed signs of",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "El presidente de México declaró que",
    "import numpy as np\nimport pandas as pd\n\ndf = pd.read_csv(",
    "The football match between Manchester United and",
    "In the year 2050, artificial intelligence will",
    "The chemical formula for water is H2O, which means",
    "Les étudiants de l'université de Paris ont",
    "According to the latest climate report,",
    "The Python programming language was created by",
    "In quantum mechanics, the uncertainty principle states",
    "Der deutsche Bundestag hat beschlossen, dass",
    "The recipe calls for two cups of flour,",
    "During the Renaissance period, artists like",
    "The new smartphone features a 6.7-inch display and",
    "In machine learning, gradient descent is used to",
    "The ancient Egyptian pyramids were built as",
    "SELECT * FROM users WHERE age >",
    "The human genome contains approximately",
    "In Tokyo, the cherry blossom season typically begins",
    "The theory of relativity, proposed by Einstein,",
    "npm install --save react-router-dom\n\nimport {",
    "The Amazon rainforest, often called the lungs of",
    "fn main() {\n    let mut v: Vec<i32> = vec![",
    "The orchestra performed Beethoven's Symphony No.",
    "In economics, supply and demand curves intersect at",
    "The COVID-19 pandemic led to significant changes in",
    "#!/bin/bash\nfor file in *.txt; do",
    "According to Buddhist philosophy, the path to",
    "The Mars rover Perseverance discovered evidence of",
    "In neural networks, the backpropagation algorithm",
    "The United Nations General Assembly voted to",
    "class MyComponent extends React.Component {\n    render() {\n        return (",
    "The Hubble Space Telescope captured images of",
    "In traditional Japanese cuisine, sushi is prepared by",
    "The cryptocurrency market experienced a sharp",
    "Según las estadísticas del gobierno,",
    "The mitochondria is often referred to as the",
    "In the game of chess, the opening known as",
    "The Great Wall of China stretches approximately",
    "async function fetchData(url) {\n    const response = await",
    "The Nobel Prize in Physics was awarded to",
    "In machine learning, overfitting occurs when",
    "The Renaissance artist Leonardo da Vinci painted",
    "docker run -d --name myapp -p 8080:80",
    "The speed of light in a vacuum is approximately",
    "In ancient Greek mythology, Zeus was known as",
    "The global population is expected to reach",
]


# ─── SAE Architectures ───────────────────────────────────────────────────────

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
        if self.threshold < 0: encoded = self._batch_topk(post_relu)
        else: encoded = post_relu * (post_relu > self.threshold)
        self._cached_x_norm = x_norm
        return encoded
    def decode(self, f, x_norm=None):
        w_u = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w_u
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        if x_norm is None: x_norm = getattr(self, "_cached_x_norm", None)
        if x_norm is not None: x_raw = x_raw * (x_norm / raw_norm)
        return x_raw + self.b_dec
    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


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


# ─── Steering Logic ──────────────────────────────────────────────────────────

def get_top_features(sae, model, tokenizer, layer, prompts, top_k=20):
    """Find the most consistently active features across prompts."""
    feature_counts = torch.zeros(D_SAE, device="cpu")
    feature_magnitudes = torch.zeros(D_SAE, device="cpu")

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=CONTEXT_LENGTH).to(DEVICE)

        # Get layer activations
        captured = {}
        def hook(module, inp, out):
            captured["act"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        handle = model.model.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        # Last token activation
        act = captured["act"][0, -1:].to(dtype=DTYPE)
        with torch.no_grad():
            feats = sae.encode(act)

        active = (feats[0] > 0)
        feature_counts += active.cpu().float()
        feature_magnitudes += feats[0].cpu().float()

    # Features that fire on >30% of prompts
    rates = feature_counts / len(prompts)
    frequent_mask = rates > 0.3
    frequent_idx = frequent_mask.nonzero(as_tuple=True)[0]

    # Sort by mean activation magnitude
    mean_mag = feature_magnitudes / feature_counts.clamp(min=1)
    sorted_idx = frequent_idx[mean_mag[frequent_idx].argsort(descending=True)]
    return sorted_idx[:top_k].tolist()


def steer_and_measure(model, tokenizer, sae, layer, feature_idx, factor, prompts):
    """Run model with a feature scaled by factor and measure output change.

    Returns per-prompt KL divergence and top-token probability shifts.
    """
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=CONTEXT_LENGTH).to(DEVICE)

        # Baseline: normal forward pass
        with torch.no_grad():
            baseline_out = model(**inputs)
        baseline_logits = baseline_out.logits[0, -1].float()
        baseline_probs = F.softmax(baseline_logits, dim=-1)

        # Steered: hook layer to modify residual stream
        def steering_hook(module, inp, out):
            act = out[0] if isinstance(out, tuple) else out
            # Encode through SAE
            last_act = act[:, -1:].to(dtype=DTYPE)
            with torch.no_grad():
                feats = sae.encode(last_act)

            if feats[0, 0, feature_idx] > 0:
                original_val = feats[0, 0, feature_idx].item()
                # Compute perturbation: (factor - 1) * activation * decoder_direction
                delta = (factor - 1.0) * feats[0, 0, feature_idx] * sae.W_dec[feature_idx].to(act.dtype)
                if isinstance(out, tuple):
                    modified = list(out)
                    modified[0] = act.clone()
                    modified[0][:, -1] = act[:, -1] + delta
                    return tuple(modified)
                else:
                    act_new = act.clone()
                    act_new[:, -1] = act[:, -1] + delta
                    return act_new
            return out

        handle = model.model.layers[layer].register_forward_hook(steering_hook)
        with torch.no_grad():
            steered_out = model(**inputs)
        handle.remove()

        steered_logits = steered_out.logits[0, -1].float()
        steered_probs = F.softmax(steered_logits, dim=-1)

        # KL divergence: KL(steered || baseline)
        kl_div = F.kl_div(
            torch.log(steered_probs + 1e-10),
            baseline_probs,
            reduction="sum"
        ).item()

        # Top-5 token probability shift
        top5_idx = baseline_probs.topk(5).indices
        top5_baseline = baseline_probs[top5_idx].cpu().tolist()
        top5_steered = steered_probs[top5_idx].cpu().tolist()
        prob_shift = sum(abs(s - b) for s, b in zip(top5_steered, top5_baseline))

        results.append({
            "kl_div": kl_div,
            "prob_shift_top5": prob_shift,
        })

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Exp 55d: Feature Steering Behavioral Test")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()

    prompts = TEST_PROMPTS[:N_PROMPTS]

    all_results = {"config": {
        "model": MODEL_NAME, "layer": LAYER,
        "n_prompts": len(prompts), "amplify_factors": AMPLIFY_FACTORS,
        "seed": SEED,
    }}

    for sae_name in ["standard", "perfeature_l2"]:
        print(f"\n{'='*60}")
        print(f"SAE: {sae_name}")
        print(f"{'='*60}")

        sae = load_sae(sae_name)

        # Find top features
        print("  Finding top features...")
        t0 = time.time()
        top_features = get_top_features(sae, model, tokenizer, LAYER, prompts, top_k=20)
        print(f"  Found {len(top_features)} features in {time.time()-t0:.1f}s")
        print(f"  Feature indices: {top_features[:10]}...")

        sae_results = {"top_features": top_features, "per_feature": {}}

        for feat_rank, feat_idx in enumerate(top_features[:10]):
            print(f"\n  Feature {feat_idx} (rank {feat_rank+1}):")

            feat_results = {}
            for factor in AMPLIFY_FACTORS:
                t0 = time.time()
                per_prompt = steer_and_measure(
                    model, tokenizer, sae, LAYER, feat_idx, factor, prompts
                )

                kl_divs = [r["kl_div"] for r in per_prompt]
                prob_shifts = [r["prob_shift_top5"] for r in per_prompt]

                mean_kl = float(np.mean(kl_divs))
                mean_shift = float(np.mean(prob_shifts))
                elapsed = time.time() - t0

                factor_label = f"x{factor}"
                feat_results[factor_label] = {
                    "mean_kl_div": mean_kl,
                    "std_kl_div": float(np.std(kl_divs)),
                    "mean_prob_shift": mean_shift,
                    "std_prob_shift": float(np.std(prob_shifts)),
                    "median_kl_div": float(np.median(kl_divs)),
                }
                print(f"    {factor_label}: KL={mean_kl:.4f} prob_shift={mean_shift:.4f} ({elapsed:.1f}s)")

            sae_results["per_feature"][str(feat_idx)] = feat_results

        # Aggregate across features
        agg = {}
        for factor in AMPLIFY_FACTORS:
            factor_label = f"x{factor}"
            all_kl = []
            all_shift = []
            for feat_data in sae_results["per_feature"].values():
                if factor_label in feat_data:
                    all_kl.append(feat_data[factor_label]["mean_kl_div"])
                    all_shift.append(feat_data[factor_label]["mean_prob_shift"])
            if all_kl:
                agg[factor_label] = {
                    "mean_kl_across_features": float(np.mean(all_kl)),
                    "std_kl_across_features": float(np.std(all_kl)),
                    "mean_shift_across_features": float(np.mean(all_shift)),
                    "std_shift_across_features": float(np.std(all_shift)),
                }
                print(f"\n  Aggregate {factor_label}: "
                      f"KL={agg[factor_label]['mean_kl_across_features']:.4f} ± "
                      f"{agg[factor_label]['std_kl_across_features']:.4f} "
                      f"shift={agg[factor_label]['mean_shift_across_features']:.4f}")

        sae_results["aggregate"] = agg
        all_results[sae_name] = sae_results

        del sae
        torch.cuda.empty_cache()

        # Incremental save
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON: standard vs perfeature_l2")
    print(f"{'='*70}")
    for factor in AMPLIFY_FACTORS:
        fl = f"x{factor}"
        std = all_results.get("standard", {}).get("aggregate", {}).get(fl, {})
        cos = all_results.get("perfeature_l2", {}).get("aggregate", {}).get(fl, {})
        if std and cos:
            kl_ratio = std["mean_kl_across_features"] / max(cos["mean_kl_across_features"], 1e-10)
            print(f"  {fl}: standard KL={std['mean_kl_across_features']:.4f} "
                  f"cosine KL={cos['mean_kl_across_features']:.4f} "
                  f"ratio={kl_ratio:.2f}x")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
