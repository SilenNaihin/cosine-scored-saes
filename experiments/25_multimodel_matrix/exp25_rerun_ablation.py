#!/usr/bin/env python3
"""
Re-run ablation for exp25a/exp25c with dtype fix.
Loads saved SAE checkpoints and existing results, replaces ablation data.

Usage:
  # For exp25a (Pythia-6.9B on H100 GPU 1):
  CUDA_VISIBLE_DEVICES=1 python experiments/exp25_rerun_ablation.py --exp 25a

  # For exp25c (Mistral-7B on A100):
  python experiments/exp25_rerun_ablation.py --exp 25c
"""

import argparse
import gc
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Shared config ──────────────────────────────────────────────────────
BATCH_SIZE = 4096
N_EVAL_TOKENS = 500_000
N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 50
SEED = 42
DEVICE = "cuda"

CONFIGS = {
    "25a": {
        "model_name": "EleutherAI/pythia-6.9b-deduped",
        "d_model": 4096,
        "d_sae": 16384,
        "k": 80,
        "model_dtype": torch.float16,
        "layers": [8, 16, 24],
        "hook_template": "model.gpt_neox.layers[{layer}]",
        "checkpoint_dir": "checkpoints/exp25a",
        "results_path": "experiments/exp25a_results.json",
    },
    "25c": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "d_model": 4096,
        "d_sae": 16384,
        "k": 80,
        "model_dtype": torch.bfloat16,
        "layers": [8, 16, 24],
        "hook_template": "model.model.layers[{layer}]",
        "checkpoint_dir": "checkpoints/exp25c",
        "results_path": "experiments/exp25c_results.json",
    },
}


# ── SAE classes (must match exp25a/exp25c training code exactly) ───────
import math

class BatchTopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int = 50):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.zeros(()))

    def encode(self, x):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = F.relu(pre_acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


class CosineBatchTopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int = 50):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.zeros(()))

    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        scale = torch.exp(self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


class AdaptiveCosineBatchTopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int = 50):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.zeros(()))

    def encode(self, x):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        acts = F.relu(pre_acts)
        return torch.where(acts >= self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


VARIANT_CLASSES = {
    "standard": BatchTopKSAE,
    "cosine": CosineBatchTopKSAE,
    "adaptive_l2": AdaptiveCosineBatchTopKSAE,
}


# ── Data collection ───────────────────────────────────────────────────
def _resolve_layer(model, hook_template, layer_idx):
    """Resolve hook_template like 'model.gpt_neox.layers[{layer}]' to a module."""
    parts = hook_template.replace("[{layer}]", "").split(".")
    mod = model  # skip 'model' prefix — we already have the model object
    for p in parts[1:]:
        mod = getattr(mod, p)
    return mod[layer_idx]


def collect_activations(model, tokenizer, layer_idx, n_tokens, hook_template, skip_docs=0):
    """Collect residual stream activations from the specified layer."""
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    it = iter(ds)
    for _ in range(skip_docs):
        next(it, None)

    activations = []
    total = 0

    layer_mod = _resolve_layer(model, hook_template, layer_idx)

    def hook_fn(module, inputs, outputs):
        out = outputs[0] if isinstance(outputs, tuple) else outputs
        activations.append(out.detach().cpu().to(torch.float16))

    h = layer_mod.register_forward_hook(hook_fn)

    t0 = time.time()
    with torch.no_grad():
        while total < n_tokens:
            batch_texts = []
            for _ in range(8):
                doc = next(it, None)
                if doc is None:
                    break
                batch_texts.append(doc["text"][:2048])
            if not batch_texts:
                break
            enc = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(DEVICE)
            model(**enc)
            total += enc["input_ids"].numel()

    h.remove()
    all_acts = torch.cat(activations, dim=1 if activations[0].dim() == 3 else 0)
    if all_acts.dim() == 3:
        all_acts = all_acts.reshape(-1, all_acts.shape[-1])
    all_acts = all_acts[:n_tokens]
    norms = all_acts.float().norm(dim=-1)
    elapsed = time.time() - t0
    print(f"  Layer {layer_idx}: {all_acts.shape[0]:,} tokens in {elapsed:.1f}s "
          f"(norm: mean={norms.mean():.1f}, std={norms.std():.1f})")
    return all_acts


# ── Ablation ──────────────────────────────────────────────────────────
def ablate_feature_kl(model, activation, feature_dir, layer_idx, hook_template):
    projection = (activation @ feature_dir) * feature_dir
    model_dtype = next(model.parameters()).dtype
    x = activation.unsqueeze(0).unsqueeze(0).to(model_dtype)
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0).to(model_dtype)

    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
    layer_mod = _resolve_layer(model, hook_template, layer_idx)

    def make_hook(replacement):
        def hook(module, inputs, outputs):
            return (replacement,) + outputs[1:] if isinstance(outputs, tuple) else replacement
        return hook

    h = layer_mod.register_forward_hook(make_hook(x))
    with torch.no_grad():
        try:
            orig_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception as e:
            h.remove()
            return None
    h.remove()

    h = layer_mod.register_forward_hook(make_hook(x_ablated))
    with torch.no_grad():
        try:
            abl_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception as e:
            h.remove()
            return None
    h.remove()

    orig_probs = torch.softmax(orig_logits, dim=-1).clamp(min=1e-10)
    abl_log_probs = torch.log_softmax(abl_logits, dim=-1)
    kl = torch.sum(orig_probs * (orig_probs.log() - abl_log_probs)).item()
    if np.isnan(kl) or kl < 0:
        return None
    return kl


def evaluate_ablation(name, model, sae, eval_data, layer_idx, hook_template):
    print(f"\n    Ablation [{name}] ({N_ABLATION_FEATURES} feats, "
          f"{N_ABLATION_SAMPLES} samples)...")
    sae.eval()

    n_probe = min(100_000, eval_data.shape[0])
    probe = eval_data[:n_probe]
    all_feats = []
    for i in range(0, n_probe, BATCH_SIZE):
        batch = probe[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        _, f = sae(batch)
        all_feats.append(f.detach().cpu())
    all_feats = torch.cat(all_feats, dim=0)

    freq = (all_feats > 0).float().mean(dim=0)
    alive_mask = freq > 0
    n_alive = alive_mask.sum().item()
    print(f"    [{name}] {n_alive} alive features (of {sae.d_sae})")

    n_to_select = min(N_ABLATION_FEATURES, n_alive)
    if n_to_select == 0:
        return {"features": [], "aggregate": {"n_features": 0}}

    top_idx = freq.topk(n_to_select).indices
    feature_results = []

    for rank, fi in enumerate(top_idx):
        fi = fi.item()
        feat_dir = sae.W_dec[fi].float()
        feat_dir = feat_dir / feat_dir.norm()

        feat_acts = all_feats[:, fi]
        active = torch.where(feat_acts > 0)[0]
        if len(active) < 20:
            continue

        n_sample = min(N_ABLATION_SAMPLES, len(active))
        chosen = active[torch.randperm(len(active))[:n_sample]]

        cos_v, norm_v, inner_v, sae_v, kl_v = [], [], [], [], []
        for idx in chosen:
            x = probe[idx].to(DEVICE, dtype=torch.float32)
            kl = ablate_feature_kl(model, x, feat_dir, layer_idx, hook_template)
            if kl is None:
                continue
            cos_v.append(F.cosine_similarity(x.unsqueeze(0), feat_dir.unsqueeze(0)).item())
            norm_v.append(x.norm().item())
            inner_v.append((x @ feat_dir).item())
            sae_v.append(feat_acts[idx].item())
            kl_v.append(kl)

        if len(kl_v) < 10:
            continue
        kl_arr = np.array(kl_v)
        if kl_arr.std() < 1e-10:
            continue

        cos_arr = np.array(cos_v)
        inner_arr = np.array(inner_v)
        norm_arr = np.array(norm_v)
        sae_arr = np.array(sae_v)

        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]

        result = {
            "feature_idx": fi, "n_ablated": len(kl_v),
            "corr_cos_kl": float(corr_cos), "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner), "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "cos_wins_sae": bool(abs(corr_cos) > abs(corr_sae)),
            "sae_wins_inner": bool(abs(corr_sae) > abs(corr_inner)),
        }
        feature_results.append(result)

        if rank < 5 or rank % 10 == 0:
            print(f"      feat {fi:>5d} | n={len(kl_v)} | "
                  f"cos→KL={corr_cos:.3f} | inner→KL={corr_inner:.3f} | "
                  f"SAE→KL={corr_sae:.3f} | norm→KL={corr_norm:.3f}")

    if not feature_results:
        print(f"    [{name}] No features with enough data")
        return {"features": [], "aggregate": {"n_features": 0}}

    n = len(feature_results)
    agg = {
        "n_features": n,
        "cos_kl_mean": float(np.mean([r["corr_cos_kl"] for r in feature_results])),
        "inner_kl_mean": float(np.mean([r["corr_inner_kl"] for r in feature_results])),
        "sae_kl_mean": float(np.mean([r["corr_sae_kl"] for r in feature_results])),
        "norm_kl_mean": float(np.mean([r["corr_norm_kl"] for r in feature_results])),
        "cos_wins_inner": sum(r["cos_wins_inner"] for r in feature_results),
        "cos_wins_sae": sum(r["cos_wins_sae"] for r in feature_results),
        "sae_wins_inner": sum(r["sae_wins_inner"] for r in feature_results),
    }

    print(f"    [{name}] Summary ({n} features): "
          f"cos→KL={agg['cos_kl_mean']:.4f} | inner→KL={agg['inner_kl_mean']:.4f} | "
          f"SAE→KL={agg['sae_kl_mean']:.4f} | cos>inner: {agg['cos_wins_inner']}/{n}")
    return {"features": feature_results, "aggregate": agg}


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=["25a", "25c"])
    args = parser.parse_args()

    cfg = CONFIGS[args.exp]
    print(f"Re-running ablation for exp{args.exp}")
    print(f"Model: {cfg['model_name']}")
    print(f"Layers: {cfg['layers']}")
    print(f"Checkpoints: {cfg['checkpoint_dir']}")
    print(f"Results: {cfg['results_path']}")

    # Load existing results
    with open(cfg["results_path"]) as f:
        results = json.load(f)

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=cfg["model_dtype"], device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    for layer_idx in cfg["layers"]:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Collect eval activations
        print(f"  Collecting eval activations for layer {layer_idx}...")
        eval_data = collect_activations(
            model, tokenizer, layer_idx, N_EVAL_TOKENS,
            cfg["hook_template"], skip_docs=200_000,
        )

        for vname in ["standard", "cosine", "adaptive_l2"]:
            print(f"\n  --- VARIANT: {vname} (L{layer_idx}) ---")

            # Load checkpoint
            ckpt_path = Path(cfg["checkpoint_dir"]) / f"{vname}_L{layer_idx}_final.pt"
            if not ckpt_path.exists():
                print(f"    Checkpoint not found: {ckpt_path} — skipping")
                continue

            cls = VARIANT_CLASSES[vname]
            sae = cls(cfg["d_model"], cfg["d_sae"], cfg["k"]).to(DEVICE)
            sae.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            sae.eval()

            # Run ablation
            abl = evaluate_ablation(
                vname, model, sae, eval_data, layer_idx, cfg["hook_template"]
            )

            # Update results
            results["layers"][str(layer_idx)][vname]["ablation"] = abl

            del sae
            gc.collect()
            torch.cuda.empty_cache()

        del eval_data
        gc.collect()
        torch.cuda.empty_cache()

        # Save incrementally
        with open(cfg["results_path"], "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY")
    print(f"{'='*70}")
    for layer_idx in cfg["layers"]:
        for vname in ["standard", "cosine", "adaptive_l2"]:
            abl = results["layers"][str(layer_idx)][vname].get("ablation", {})
            agg = abl.get("aggregate", {})
            n = agg.get("n_features", 0)
            if n > 0:
                print(f"  L{layer_idx} {vname:>14s}: cos→KL={agg['cos_kl_mean']:.4f} | "
                      f"inner→KL={agg['inner_kl_mean']:.4f} | "
                      f"cos>inner: {agg['cos_wins_inner']}/{n}")
            else:
                print(f"  L{layer_idx} {vname:>14s}: no ablation data")

    print(f"\nUpdated: {cfg['results_path']}")
    print("Done!")


if __name__ == "__main__":
    main()
