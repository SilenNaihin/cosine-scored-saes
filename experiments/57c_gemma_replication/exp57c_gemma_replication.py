"""
Experiment 57c: Cross-Model Replication on Gemma-2-2B

Replicates the key findings from Qwen3-8B experiments on Gemma-2-2B:
1. Feature overlap analysis (like exp56b): do cosine SAEs discover unique features?
2. Bootstrap sparse probing (like exp55b): is the sparse probing gap robust?
3. Feature steering (like exp55d): are per-feature behavioral effects identical?

Uses exp41d checkpoints: standard and perfeature_l2 at L13 on Gemma-2-2B.
d_model=2304, d_sae=9216, k=80.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp57c_gemma_replication.py \
        > experiments/exp57c_output.log 2>&1 &
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
MODEL_NAME = "google/gemma-2-2b"
D_MODEL, D_SAE, K = 2304, 9216, 80
NORM_EPS = 1e-8
LAYER = 13

CKPTS = {
    "standard": "checkpoints/exp41d/standard_L13_final.pt",
    "perfeature_l2": "checkpoints/exp41d/perfeature_l2_L13_final.pt",
}

N_TOKENS_OVERLAP = 100000
N_PROMPTS_STEERING = 50
CORRELATION_THRESHOLD = 0.7
N_BOOTSTRAP = 5
BOOTSTRAP_SEEDS = [42, 123, 456, 789, 1337]
AMPLIFY_FACTORS = [0.0, 0.5, 2.0, 5.0]
CONTEXT_LENGTH = 128

RESULTS_PATH = "experiments/exp57c_results.json"
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


# ─── Part 1: Feature Overlap ────────────────────────────────────────────────

def feature_overlap_analysis(sae_std, sae_cos):
    """Replicate exp56b feature overlap on Gemma."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("Part 1: Feature Overlap Analysis")
    print("=" * 60)

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

    print(f"  Collecting {N_TOKENS_OVERLAP} tokens...")
    for sample in ds:
        tokens = tokenizer(sample["text"], return_tensors="pt", truncation=True,
                          max_length=CONTEXT_LENGTH).to(DEVICE)
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
        if total >= N_TOKENS_OVERLAP:
            break

    all_acts = torch.cat(all_acts, dim=0)[:N_TOKENS_OVERLAP]
    print(f"  Collected {all_acts.shape[0]} tokens")

    # Encode through both SAEs
    batch_size = 1024
    n = all_acts.shape[0]

    print("  Encoding through standard SAE...")
    std_feats = []
    for i in range(0, n, batch_size):
        batch = all_acts[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            f = sae_std.encode(batch)
        std_feats.append(f.cpu())
    std_feats = torch.cat(std_feats, dim=0).float()

    print("  Encoding through cosine SAE...")
    cos_feats = []
    for i in range(0, n, batch_size):
        batch = all_acts[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            f = sae_cos.encode(batch)
        cos_feats.append(f.cpu())
    cos_feats = torch.cat(cos_feats, dim=0).float()

    # Alive features
    std_alive = (std_feats.sum(0) > 0).nonzero(as_tuple=True)[0]
    cos_alive = (cos_feats.sum(0) > 0).nonzero(as_tuple=True)[0]
    print(f"  Alive: standard={len(std_alive)}, cosine={len(cos_alive)}")

    # Correlation computation
    std_feats_alive = std_feats[:, std_alive]
    cos_feats_alive = cos_feats[:, cos_alive]

    std_normed = std_feats_alive - std_feats_alive.mean(0, keepdim=True)
    cos_normed = cos_feats_alive - cos_feats_alive.mean(0, keepdim=True)
    std_normed = std_normed / (std_normed.norm(dim=0, keepdim=True) + 1e-8)
    cos_normed = cos_normed / (cos_normed.norm(dim=0, keepdim=True) + 1e-8)

    chunk_size = 2000
    n_std = len(std_alive)
    best_corr = torch.zeros(n_std)
    best_match = torch.zeros(n_std, dtype=torch.long)

    print(f"  Computing correlations...")
    for i in range(0, n_std, chunk_size):
        end = min(i + chunk_size, n_std)
        chunk_std = std_normed[:, i:end]
        corr_chunk = chunk_std.T @ cos_normed
        max_corr, max_idx = corr_chunk.max(dim=1)
        best_corr[i:end] = max_corr
        best_match[i:end] = max_idx

    strong = (best_corr >= 0.7).sum().item()
    weak = ((best_corr >= 0.3) & (best_corr < 0.7)).sum().item()
    unmatched = (best_corr < 0.3).sum().item()

    # Also do cosine -> standard
    n_cos = len(cos_alive)
    best_corr_rev = torch.zeros(n_cos)
    for i in range(0, n_cos, chunk_size):
        end = min(i + chunk_size, n_cos)
        chunk_cos = cos_normed[:, i:end]
        corr_chunk = chunk_cos.T @ std_normed
        max_corr_rev, _ = corr_chunk.max(dim=1)
        best_corr_rev[i:end] = max_corr_rev

    cos_strong = (best_corr_rev >= 0.7).sum().item()
    cos_unmatched = (best_corr_rev < 0.3).sum().item()

    # Decoder cosine similarity for matched pairs
    strong_mask = best_corr >= 0.7
    strong_std_local = strong_mask.nonzero(as_tuple=True)[0]
    strong_cos_local = best_match[strong_std_local]
    dec_sims = []
    W_dec_std = sae_std.W_dec.detach().float().cpu()
    W_dec_cos = sae_cos.W_dec.detach().float().cpu()
    for s_local, c_local in zip(strong_std_local[:500].tolist(), strong_cos_local[:500].tolist()):
        s_idx = std_alive[s_local].item()
        c_idx = cos_alive[c_local].item()
        sim = F.cosine_similarity(W_dec_std[s_idx].unsqueeze(0), W_dec_cos[c_idx].unsqueeze(0)).item()
        dec_sims.append(sim)

    result = {
        "std_alive": len(std_alive),
        "cos_alive": len(cos_alive),
        "std_to_cos": {
            "strong": strong,
            "weak": weak,
            "unmatched": unmatched,
            "strong_pct": strong / max(len(std_alive), 1) * 100,
            "unmatched_pct": unmatched / max(len(std_alive), 1) * 100,
        },
        "cos_to_std": {
            "strong": cos_strong,
            "unmatched": cos_unmatched,
            "strong_pct": cos_strong / max(len(cos_alive), 1) * 100,
            "unmatched_pct": cos_unmatched / max(len(cos_alive), 1) * 100,
        },
        "decoder_sim_matched": {
            "mean": float(np.mean(dec_sims)) if dec_sims else 0,
            "median": float(np.median(dec_sims)) if dec_sims else 0,
        },
    }

    print(f"\n  Results:")
    print(f"    std->cos: {strong} strong ({result['std_to_cos']['strong_pct']:.1f}%), "
          f"{unmatched} unmatched ({result['std_to_cos']['unmatched_pct']:.1f}%)")
    print(f"    cos->std: {cos_strong} strong ({result['cos_to_std']['strong_pct']:.1f}%), "
          f"{cos_unmatched} unmatched ({result['cos_to_std']['unmatched_pct']:.1f}%)")
    if dec_sims:
        print(f"    Matched decoder cos_sim: mean={np.mean(dec_sims):.3f} median={np.median(dec_sims):.3f}")

    del model, all_acts, std_feats, cos_feats
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ─── Part 2: Bootstrap Sparse Probing ───────────────────────────────────────

def bootstrap_sparse_probing(sae_std, sae_cos):
    """Run sparse probing with multiple seeds on Gemma."""
    import sae_bench.evals.sparse_probing.main as sp

    print("\n" + "=" * 60)
    print("Part 2: Bootstrap Sparse Probing")
    print("=" * 60)

    all_results = {}

    for sae_name, sae in [("standard", sae_std), ("perfeature_l2", sae_cos)]:
        bench_sae = wrap_for_saebench(sae_name, sae)
        seed_results = []

        for seed_idx, seed in enumerate(BOOTSTRAP_SEEDS):
            print(f"\n  {sae_name} seed {seed_idx+1}/{N_BOOTSTRAP}: {seed}")
            t0 = time.time()

            out_dir = f"/scratch/saebench_results/exp57c/{sae_name}_seed{seed}"
            sae_label = f"exp57c-{sae_name}-seed{seed}"

            config = sp.SparseProbingEvalConfig(
                model_name=MODEL_NAME,
                llm_batch_size=8,
                llm_dtype="bfloat16",
                random_seed=seed,
            )

            sp.run_eval(
                config,
                [(sae_label, bench_sae)],
                DEVICE,
                out_dir,
                force_rerun=True,
            )

            result_file = None
            for p in Path(out_dir).glob("*.json"):
                if sae_label in p.stem:
                    result_file = p
                    break
            if result_file is None:
                for p in Path(out_dir).glob("*.json"):
                    result_file = p
                    break

            if result_file:
                with open(result_file) as f:
                    result_data = json.load(f)

                metrics = result_data.get("eval_result_metrics", {})
                sae_metrics = metrics.get("sae", {})
                top1 = sae_metrics.get("sae_top_1_test_accuracy", None)

                elapsed = time.time() - t0
                print(f"    Top-1: {top1:.4f} ({elapsed:.0f}s)")
                seed_results.append({"seed": seed, "top_1": top1, "elapsed_s": elapsed})
            else:
                print(f"    No result file found")
                seed_results.append({"seed": seed, "error": "no result file"})

        top1s = [r["top_1"] for r in seed_results if r.get("top_1") is not None]
        agg = {
            "mean_top1": float(np.mean(top1s)),
            "std_top1": float(np.std(top1s)),
            "all_top1": top1s,
        } if top1s else {"error": "no valid results"}

        print(f"\n  {sae_name} aggregate: {agg.get('mean_top1', 'N/A'):.4f} ± {agg.get('std_top1', 'N/A'):.4f}")
        all_results[sae_name] = {"seeds": seed_results, "aggregate": agg}

        del bench_sae
        torch.cuda.empty_cache()

    # Gap
    std_top1s = [r["top_1"] for r in all_results["standard"]["seeds"] if r.get("top_1")]
    cos_top1s = [r["top_1"] for r in all_results["perfeature_l2"]["seeds"] if r.get("top_1")]
    if std_top1s and cos_top1s:
        gaps = [c - s for c, s in zip(cos_top1s, std_top1s)]
        all_results["gap"] = {
            "mean_pp": float(np.mean(gaps) * 100),
            "std_pp": float(np.std(gaps) * 100),
            "all_pp": [float(g * 100) for g in gaps],
        }
        print(f"\n  Gap: {np.mean(gaps)*100:.2f}pp ± {np.std(gaps)*100:.2f}pp")

    return all_results


# ─── Part 3: Feature Steering ───────────────────────────────────────────────

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


def feature_steering_analysis(sae_std, sae_cos):
    """Replicate exp55d feature steering on Gemma."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("Part 3: Feature Steering")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()

    prompts = TEST_PROMPTS[:N_PROMPTS_STEERING]
    all_results = {}

    for sae_name, sae in [("standard", sae_std), ("perfeature_l2", sae_cos)]:
        print(f"\n  SAE: {sae_name}")

        # Find top features
        feature_counts = torch.zeros(D_SAE, device="cpu")
        feature_magnitudes = torch.zeros(D_SAE, device="cpu")

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=CONTEXT_LENGTH).to(DEVICE)
            captured = {}
            def hook(module, inp, out):
                captured["act"] = out[0].detach() if isinstance(out, tuple) else out.detach()
            handle = model.model.layers[LAYER].register_forward_hook(hook)
            with torch.no_grad():
                model(**inputs)
            handle.remove()

            act = captured["act"][0, -1:].to(dtype=DTYPE)
            with torch.no_grad():
                feats = sae.encode(act)
            active = (feats[0] > 0)
            feature_counts += active.cpu().float()
            feature_magnitudes += feats[0].cpu().float()

        rates = feature_counts / len(prompts)
        frequent_mask = rates > 0.3
        frequent_idx = frequent_mask.nonzero(as_tuple=True)[0]
        mean_mag = feature_magnitudes / feature_counts.clamp(min=1)
        sorted_idx = frequent_idx[mean_mag[frequent_idx].argsort(descending=True)]
        top_features = sorted_idx[:10].tolist()
        print(f"    Top features: {top_features}")

        # Steer and measure
        sae_agg = {}
        for factor in AMPLIFY_FACTORS:
            all_kl = []
            all_shift = []
            for feat_idx in top_features:
                for prompt in prompts:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                      max_length=CONTEXT_LENGTH).to(DEVICE)

                    with torch.no_grad():
                        baseline_out = model(**inputs)
                    baseline_probs = F.softmax(baseline_out.logits[0, -1].float(), dim=-1)

                    def steering_hook(module, inp, out):
                        act = out[0] if isinstance(out, tuple) else out
                        last_act = act[:, -1:].to(dtype=DTYPE)
                        with torch.no_grad():
                            feats = sae.encode(last_act)
                        if feats[0, 0, feat_idx] > 0:
                            delta = (factor - 1.0) * feats[0, 0, feat_idx] * sae.W_dec[feat_idx].to(act.dtype)
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

                    handle = model.model.layers[LAYER].register_forward_hook(steering_hook)
                    with torch.no_grad():
                        steered_out = model(**inputs)
                    handle.remove()

                    steered_probs = F.softmax(steered_out.logits[0, -1].float(), dim=-1)
                    kl = F.kl_div(torch.log(steered_probs + 1e-10), baseline_probs, reduction="sum").item()
                    top5_idx = baseline_probs.topk(5).indices
                    shift = sum(abs(steered_probs[j].item() - baseline_probs[j].item()) for j in top5_idx)
                    all_kl.append(kl)
                    all_shift.append(shift)

            sae_agg[f"x{factor}"] = {
                "mean_kl": float(np.mean(all_kl)),
                "std_kl": float(np.std(all_kl)),
                "mean_shift": float(np.mean(all_shift)),
            }
            print(f"    x{factor}: KL={np.mean(all_kl):.4f} ± {np.std(all_kl):.4f}")

        all_results[sae_name] = {"top_features": top_features, "aggregate": sae_agg}

    # Comparison
    print(f"\n  Comparison:")
    for factor in AMPLIFY_FACTORS:
        fl = f"x{factor}"
        std_kl = all_results["standard"]["aggregate"][fl]["mean_kl"]
        cos_kl = all_results["perfeature_l2"]["aggregate"][fl]["mean_kl"]
        ratio = std_kl / max(cos_kl, 1e-10)
        print(f"    {fl}: standard={std_kl:.4f} cosine={cos_kl:.4f} ratio={ratio:.2f}x")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Exp 57c: Cross-Model Replication on Gemma-2-2B")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results = {
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "k": K,
            "seed": SEED,
        }
    }

    # Load SAEs
    print("\nLoading SAEs...")
    sae_std = load_sae("standard")
    sae_cos = load_sae("perfeature_l2")

    # Part 1: Feature overlap
    t0 = time.time()
    results["feature_overlap"] = feature_overlap_analysis(sae_std, sae_cos)
    print(f"\nFeature overlap completed in {(time.time()-t0)/60:.1f} min")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Part 2: Bootstrap sparse probing
    t0 = time.time()
    results["sparse_probing"] = bootstrap_sparse_probing(sae_std, sae_cos)
    print(f"\nSparse probing completed in {(time.time()-t0)/60:.1f} min")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Part 3: Feature steering
    t0 = time.time()
    results["feature_steering"] = feature_steering_analysis(sae_std, sae_cos)
    print(f"\nFeature steering completed in {(time.time()-t0)/60:.1f} min")

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    fo = results.get("feature_overlap", {})
    print(f"\nFeature overlap:")
    print(f"  Alive: std={fo.get('std_alive', 'N/A')}, cos={fo.get('cos_alive', 'N/A')}")
    s2c = fo.get("std_to_cos", {})
    print(f"  std->cos: {s2c.get('strong', 'N/A')} strong ({s2c.get('strong_pct', 'N/A'):.1f}%), "
          f"{s2c.get('unmatched', 'N/A')} unmatched ({s2c.get('unmatched_pct', 'N/A'):.1f}%)")

    sp = results.get("sparse_probing", {})
    gap = sp.get("gap", {})
    print(f"\nSparse probing gap: {gap.get('mean_pp', 'N/A'):.2f}pp ± {gap.get('std_pp', 'N/A'):.2f}pp")

    fs = results.get("feature_steering", {})
    for factor in AMPLIFY_FACTORS:
        fl = f"x{factor}"
        std_kl = fs.get("standard", {}).get("aggregate", {}).get(fl, {}).get("mean_kl", "N/A")
        cos_kl = fs.get("perfeature_l2", {}).get("aggregate", {}).get(fl, {}).get("mean_kl", "N/A")
        if isinstance(std_kl, (int, float)) and isinstance(cos_kl, (int, float)):
            print(f"\nSteering {fl}: std={std_kl:.4f} cos={cos_kl:.4f} ratio={std_kl/max(cos_kl, 1e-10):.2f}x")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
