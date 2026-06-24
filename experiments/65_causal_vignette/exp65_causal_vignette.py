#!/usr/bin/env python3
"""Experiment 65: Causal feature-ablation vignette (A4 tier 2).

Reviewer ask (762k, NTVF, EZEE): go beyond aggregate sparse probing and show a
LEGIBLE causal example -- pick a few discovered content features, intervene, and
demonstrate that cosine features give cleaner causal effects than standard.

This is the worked-example complement to exp57b (aggregate TPP precision). For a
small set of content CONCEPTS (a programming language, a natural language), we:
  1. Rank each dictionary's features by concept selectivity (mean activation on
     concept-positive minus concept-negative prompts).
  2. Ablate the top-K concept features as a SET, sweeping K, and measure the
     effect on the next-token distribution, split into:
       - ON-concept prompts  (intended)   -> want a LARGE effect
       - OFF-concept prompts (collateral)  -> want a SMALL effect
  3. Report precision = on-effect / off-effect per arm, as a function of K.

v1 post-mortem (do NOT regress): single-best-feature was too easy (both arms had
a perfectly selective concept feature, off-effect ~0 -> precision=inf for both,
uninformative) AND raw cross-arm KL is confounded by activation magnitude (the
norm effect the paper is about). v2 fixes both: (a) top-K as a SET so off-effect
is nonzero and the deeper supply of genuine concept features is tested -- the
discovery advantage is aggregate; (b) precision is a WITHIN-arm ratio, so per-arm
magnitude cancels.

Hypothesis: cosine sustains high precision as K grows (its top-K are all genuine
concept content); standard's precision degrades with K because its later
"concept" features are norm-detectors that fire indiscriminately, adding
off-concept collateral. This is exp57b's shape, now at the concept level.

Inference only on existing checkpoints; no training. SAE classes + steering hook
are reused verbatim from exp55d_feature_steering.py.

Env (one GPU):
    CUDA_VISIBLE_DEVICES=0 EXP65_CKPT_DIR=checkpoints/exp61_seed123 \
        PYTHONUNBUFFERED=1 python3 experiments/exp65_causal_vignette.py
"""
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 18
D_MODEL = 4096
D_SAE = 65536
K = 80
DEVICE = "cuda"
DTYPE = torch.float32
CONTEXT_LENGTH = 64
SEED = 42
NORM_EPS = 1e-8

CKPT_DIR = os.environ.get("EXP65_CKPT_DIR", "checkpoints/exp61_seed123")
CKPTS = {
    "standard":      f"{CKPT_DIR}/standard_L18_final.pt",
    "perfeature_l2": f"{CKPT_DIR}/perfeature_l2_L18_final.pt",
}
RESULTS_PATH = os.environ.get("EXP65_RESULTS", "experiments/exp65_results.json")

# Concept probe sets: ON-concept (where the feature should fire / matter) vs
# OFF-concept (collateral controls, matched register/length). Small, legible.
CONCEPTS = {
    "python_code": {
        "on": [
            "import numpy as np\narr = np.",
            "def quicksort(lst):\n    if len(lst) <= 1:\n        return",
            "for i in range(10):\n    print(",
            "df = pd.DataFrame(data)\ndf.",
            "class Model(nn.Module):\n    def __init__(self):\n        super().",
        ],
        "off": [
            "The president of France announced that",
            "In a recent scientific study, researchers found that",
            "The recipe calls for two cups of flour and",
            "During the Renaissance period, artists like",
            "The football match ended with a score of",
        ],
    },
    "french_language": {
        "on": [
            "Le président de la République a déclaré que",
            "Les étudiants de l'université de Paris ont",
            "La cuisine française est réputée pour ses",
            "Selon les statistiques du gouvernement français,",
            "Bonjour, je voudrais réserver une table pour",
        ],
        "off": [
            "The stock market today showed signs of",
            "import numpy as np\narr = np.",
            "The chemical formula for water is",
            "In machine learning, gradient descent is used to",
            "The orchestra performed a symphony by",
        ],
    },
}

# ── SAE classes (verbatim from exp55d) ───────────────────────────────────────
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
        bs = max(acts.shape[0], 1)
        total_k = min(self.k * bs, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)
    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        post = F.relu(pre)
        return self._batch_topk(post) if self.threshold < 0 else post * (post > self.threshold)
    def decode(self, f):
        return f @ self.W_dec + self.b_dec


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
        bs = max(acts.shape[0], 1)
        total_k = min(self.k * bs, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)
    def encode(self, x):
        xc = x - self.b_dec
        xu = F.normalize(xc, dim=-1)
        wu = F.normalize(self.W_enc, dim=-1)
        cos = xu @ wu.T
        norm = xc.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        scale = torch.exp(self.scale_a * torch.log(norm) + self.scale_b)
        post = F.relu(scale * cos + self.b_enc)
        return self._batch_topk(post) if self.threshold < 0 else post * (post > self.threshold)
    def decode(self, f):
        return f @ self.W_dec + self.b_dec


SAE_CLASSES = {"standard": BatchTopKSAE, "perfeature_l2": PerFeatureAdaptiveCosineSAE}


def load_sae(name):
    sae = SAE_CLASSES[name](D_MODEL, D_SAE, K).to(DEVICE)
    ckpt = torch.load(CKPTS[name], map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sae.load_state_dict(sd)
    sae.eval()
    return sae


# ── Activation capture + ablation hook (pattern from exp55d) ──────────────────
def last_token_act(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=CONTEXT_LENGTH).to(DEVICE)
    captured = {}
    def hook(mod, inp, out):
        captured["act"] = (out[0] if isinstance(out, tuple) else out).detach()
    h = model.model.layers[LAYER].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    h.remove()
    return captured["act"][0, -1:].to(dtype=DTYPE), inputs


def rank_concept_features(sae, model, tokenizer, on_prompts, off_prompts):
    """Rank ALL features by concept selectivity (mean on-act minus mean off-act).

    Returns the descending-sorted feature indices and their (on, off, gap) stats.
    v2 uses top-K (not just argmax): the discovery advantage is aggregate, so a
    single easy concept feature exists in both dictionaries -- the question is how
    DEEP the supply of genuine concept features runs before a dictionary starts
    spending ablation budget on norm-detectors that fire indiscriminately.
    """
    on_acts = torch.zeros(D_SAE)
    off_acts = torch.zeros(D_SAE)
    for p in on_prompts:
        act, _ = last_token_act(model, tokenizer, p)
        with torch.no_grad():
            on_acts += sae.encode(act)[0].cpu().float()
    for p in off_prompts:
        act, _ = last_token_act(model, tokenizer, p)
        with torch.no_grad():
            off_acts += sae.encode(act)[0].cpu().float()
    on_acts /= len(on_prompts)
    off_acts /= len(off_prompts)
    gap = on_acts - off_acts
    order = gap.argsort(descending=True)
    return order, on_acts, off_acts, gap


def ablate_set_effect(model, tokenizer, sae, feature_ids, prompts):
    """Mean KL(ablated || baseline) over prompts when a SET of features is zeroed.

    Ablating a SET (not one feature) guarantees a nonzero off-concept effect once
    K is large enough, so the precision ratio below is finite and well-defined.
    """
    feat_ids = torch.tensor(feature_ids, device=DEVICE, dtype=torch.long)
    kls = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=CONTEXT_LENGTH).to(DEVICE)
        with torch.no_grad():
            base = model(**inputs).logits[0, -1].float()
        base_p = F.softmax(base, dim=-1)

        def abl_hook(mod, out_in, out):
            act = out[0] if isinstance(out, tuple) else out
            last = act[:, -1:].to(dtype=DTYPE)
            with torch.no_grad():
                feats = sae.encode(last)
            vals = feats[0, 0, feat_ids]              # [K]
            active = vals > 0
            if active.any():
                # subtract each active feature's decoder contribution
                delta = -(vals[active].unsqueeze(-1)
                          * sae.W_dec[feat_ids[active]].to(DTYPE)).sum(0)
                delta = delta.to(act.dtype)
                if isinstance(out, tuple):
                    mod_out = list(out); a = act.clone(); a[:, -1] = act[:, -1] + delta
                    mod_out[0] = a; return tuple(mod_out)
                a = act.clone(); a[:, -1] = act[:, -1] + delta; return a
            return out

        h = model.model.layers[LAYER].register_forward_hook(abl_hook)
        with torch.no_grad():
            abl = model(**inputs).logits[0, -1].float()
        h.remove()
        abl_p = F.softmax(abl, dim=-1)
        kl = F.kl_div(torch.log(abl_p + 1e-10), base_p, reduction="sum").item()
        kls.append(kl)
    return sum(kls) / len(kls)


def main():
    torch.manual_seed(SEED)
    print(f"[exp65] loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    model.eval()

    saes = {name: load_sae(name) for name in CKPTS}
    print(f"[exp65] SAEs loaded: {list(saes)}  (ckpt dir {CKPT_DIR})")

    results = {"config": {"model": MODEL_NAME, "layer": LAYER, "ckpt_dir": CKPT_DIR},
               "concepts": {}}

    # Ablate the top-K concept features as a SET, sweeping K. Precision =
    # on-concept effect / off-concept effect is a WITHIN-arm ratio, so cross-arm
    # activation-magnitude differences (the norm confound that broke v1) cancel.
    K_SWEEP = [1, 3, 5, 10, 20, 50]
    for concept, sets in CONCEPTS.items():
        print(f"\n[exp65] === concept: {concept} ===")
        results["concepts"][concept] = {}
        for name, sae in saes.items():
            order, on_acts, off_acts, gap = rank_concept_features(
                sae, model, tokenizer, sets["on"], sets["off"])
            per_k = []
            print(f"  -- {name} --")
            for K in K_SWEEP:
                feat_ids = order[:K].tolist()
                on_kl = ablate_set_effect(model, tokenizer, sae, feat_ids, sets["on"])
                off_kl = ablate_set_effect(model, tokenizer, sae, feat_ids, sets["off"])
                precision = on_kl / off_kl if off_kl > 1e-9 else float("inf")
                # mean selectivity gap of the K features chosen (how "real" they are)
                mean_gap = float(gap[order[:K]].mean())
                per_k.append({"K": K, "on_kl": on_kl, "off_kl": off_kl,
                              "precision": precision, "mean_selectivity_gap": mean_gap,
                              "feature_ids": feat_ids if K <= 10 else None})
                pstr = f"{precision:.1f}x" if precision != float("inf") else "inf"
                print(f"    K={K:3d}  on_KL={on_kl:.4f} off_KL={off_kl:.4f}  "
                      f"precision={pstr:>7s}  mean_gap={mean_gap:.3f}")
            results["concepts"][concept][name] = {"k_sweep": per_k}

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[exp65] done -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
