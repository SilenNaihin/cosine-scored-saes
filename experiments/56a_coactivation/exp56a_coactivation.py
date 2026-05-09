"""
Experiment 56a: Feature Co-activation Analysis (Concept Support Size)

Tests the "distributed encoding" hypothesis: do cosine SAEs activate more
features per concept than standard SAEs?

For each SAEBench sparse probing concept, we measure how many SAE features
reliably fire on positive examples (concept support size). If cosine SAEs
have larger support, it explains the precision-vs-power tradeoff from exp55a.

Run on <gpu-server> GPU 0:
    ssh <server>     cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python3 experiments/exp56a_coactivation.py \
        > experiments/exp56a_output.log 2>&1 &
"""

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
CONTEXT_LENGTH = 128

CKPTS = {
    "standard": "/scratch/checkpoints/exp40/standard_L18_final.pt",
    "adaptive_l2": "/scratch/checkpoints/exp40/adaptive_l2_L18_final.pt",
    "perfeature_l2": "/scratch/checkpoints/exp40/perfeature_l2_L18_final.pt",
    "no_C": "/scratch/checkpoints/exp42c/no_C_L18_final.pt",
}

DATASETS = [
    "LabHC/bias_in_bios_class_set1",
    "LabHC/bias_in_bios_class_set2",
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "canrager/amazon_reviews_mcauley_1and5_sentiment",
    "codeparrot/github-code",
    "fancyzhx/ag_news",
    "Helsinki-NLP/europarl",
]

N_EXAMPLES = 1000
LLM_BATCH_SIZE = 8
SUPPORT_POS_THRESHOLD = 0.3
SUPPORT_NEG_THRESHOLD = 0.1

RESULTS_PATH = "experiments/exp56a_results.json"

SEED = 42


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
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

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
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

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
        if self.threshold < 0:
            return self._batch_topk(post_relu)
        return post_relu * (post_relu > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

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
        if self.threshold < 0:
            encoded = self._batch_topk(post_relu)
        else:
            encoded = post_relu * (post_relu > self.threshold)
        self._cached_x_norm = x_norm
        return encoded

    def decode(self, f, x_norm=None):
        w_u = F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        x_raw = f @ w_u
        raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        if x_norm is None:
            x_norm = getattr(self, "_cached_x_norm", None)
        if x_norm is not None:
            x_raw = x_raw * (x_norm / raw_norm)
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


# ─── Data Loading ─────────────────────────────────────────────────────────────

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


def load_dataset_with_labels(dataset_name, tokenizer, n_examples):
    """Load a SAEBench sparse probing dataset and return labeled examples.

    Returns dict of {class_label: list of token_id tensors}.
    """
    from datasets import load_dataset

    if "bias_in_bios" in dataset_name:
        set_num = dataset_name.split("set")[-1]
        ds = load_dataset("LabHC/bias_in_bios", split="test", streaming=True)
        # bias_in_bios class sets map to different profession groupings
        # SAEBench uses the 'profession' field
        examples_by_class = {}
        count = 0
        for row in ds:
            if count >= n_examples:
                break
            label = row.get("profession", row.get("label", "unknown"))
            text = row.get("hard_text", row.get("text", ""))
            if not text:
                continue
            toks = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=CONTEXT_LENGTH, padding=False)
            if label not in examples_by_class:
                examples_by_class[label] = []
            examples_by_class[label].append(toks["input_ids"][0])
            count += 1
        return examples_by_class

    elif "amazon_reviews" in dataset_name:
        if "sentiment" in dataset_name:
            ds = load_dataset("canrager/amazon_reviews_mcauley_1and5", split="test",
                            streaming=True)
            examples_by_class = {}
            count = 0
            for row in ds:
                if count >= n_examples:
                    break
                label = "positive" if row.get("label", 0) == 1 else "negative"
                text = row.get("text", "")
                if not text:
                    continue
                toks = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=CONTEXT_LENGTH, padding=False)
                examples_by_class.setdefault(label, []).append(toks["input_ids"][0])
                count += 1
            return examples_by_class
        else:
            ds = load_dataset("canrager/amazon_reviews_mcauley_1and5", split="test",
                            streaming=True)
            examples_by_class = {}
            count = 0
            for row in ds:
                if count >= n_examples:
                    break
                label = row.get("category", row.get("label", "unknown"))
                text = row.get("text", "")
                if not text:
                    continue
                toks = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=CONTEXT_LENGTH, padding=False)
                examples_by_class.setdefault(str(label), []).append(toks["input_ids"][0])
                count += 1
            return examples_by_class

    elif "github-code" in dataset_name:
        ds = load_dataset("codeparrot/github-code", split="train", streaming=True,
                         languages=["Python", "JavaScript", "Java", "Go", "Ruby", "PHP"])
        examples_by_class = {}
        count = 0
        for row in ds:
            if count >= n_examples:
                break
            label = row.get("language", "unknown")
            text = row.get("code", "")
            if not text or len(text) < 20:
                continue
            toks = tokenizer(text[:1024], return_tensors="pt", truncation=True,
                           max_length=CONTEXT_LENGTH, padding=False)
            examples_by_class.setdefault(label, []).append(toks["input_ids"][0])
            count += 1
        return examples_by_class

    elif "ag_news" in dataset_name:
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        ds = load_dataset("fancyzhx/ag_news", split="test", streaming=True)
        examples_by_class = {}
        count = 0
        for row in ds:
            if count >= n_examples:
                break
            label = label_map.get(row.get("label", -1), "unknown")
            text = row.get("text", "")
            if not text:
                continue
            toks = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=CONTEXT_LENGTH, padding=False)
            examples_by_class.setdefault(label, []).append(toks["input_ids"][0])
            count += 1
        return examples_by_class

    elif "europarl" in dataset_name:
        ds = load_dataset("Helsinki-NLP/europarl", split="train", streaming=True,
                         trust_remote_code=True)
        examples_by_class = {}
        count = 0
        for row in ds:
            if count >= n_examples:
                break
            label = row.get("language", row.get("lang", "unknown"))
            text = row.get("text", "")
            if not text:
                continue
            toks = tokenizer(text[:1024], return_tensors="pt", truncation=True,
                           max_length=CONTEXT_LENGTH, padding=False)
            examples_by_class.setdefault(label, []).append(toks["input_ids"][0])
            count += 1
        return examples_by_class

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ─── Core Analysis ────────────────────────────────────────────────────────────

def compute_concept_support(sae, acts_by_class, sae_name):
    """Compute concept support size for each class.

    For each concept class:
    - Encode positive examples through SAE
    - Compute per-feature activation rate on positives vs negatives
    - Count features that are discriminative (high on positive, low on negative)

    Returns per-concept metrics dict.
    """
    all_classes = list(acts_by_class.keys())
    results = []

    for target_class in all_classes:
        pos_acts = acts_by_class[target_class]
        if len(pos_acts) < 10:
            continue

        neg_acts = []
        for cls, acts in acts_by_class.items():
            if cls != target_class:
                neg_acts.extend(acts)
        if len(neg_acts) < 10:
            continue

        # Encode positives through SAE in batches
        pos_fire_counts = torch.zeros(D_SAE, device="cpu")
        n_pos = 0
        batch_size = 256
        for i in range(0, len(pos_acts), batch_size):
            batch = torch.stack(pos_acts[i:i+batch_size]).to(DEVICE, dtype=DTYPE)
            with torch.no_grad():
                feats = sae.encode(batch)
            fires = (feats > 0).float().sum(dim=0).cpu()
            pos_fire_counts += fires
            n_pos += batch.shape[0]
        pos_rates = pos_fire_counts / max(n_pos, 1)

        # Encode negatives (subsample if too many)
        neg_sample = neg_acts[:min(len(neg_acts), n_pos * 3)]
        neg_fire_counts = torch.zeros(D_SAE, device="cpu")
        n_neg = 0
        for i in range(0, len(neg_sample), batch_size):
            batch = torch.stack(neg_sample[i:i+batch_size]).to(DEVICE, dtype=DTYPE)
            with torch.no_grad():
                feats = sae.encode(batch)
            fires = (feats > 0).float().sum(dim=0).cpu()
            neg_fire_counts += fires
            n_neg += batch.shape[0]
        neg_rates = neg_fire_counts / max(n_neg, 1)

        # Discriminative features: high on positive, low on negative
        discriminative = (pos_rates > SUPPORT_POS_THRESHOLD) & (neg_rates < SUPPORT_NEG_THRESHOLD)
        support_size = discriminative.sum().item()

        # Broader support: features that fire on >20% of positives
        broad_support = (pos_rates > 0.2).sum().item()

        # Activation entropy on positives (how distributed is the encoding?)
        pos_mean_acts = torch.zeros(D_SAE, device="cpu")
        for i in range(0, len(pos_acts), batch_size):
            batch = torch.stack(pos_acts[i:i+batch_size]).to(DEVICE, dtype=DTYPE)
            with torch.no_grad():
                feats = sae.encode(batch)
            pos_mean_acts += feats.float().sum(dim=0).cpu()
        pos_mean_acts /= max(n_pos, 1)

        # Normalize to get a distribution over features
        active_mask = pos_mean_acts > 0
        if active_mask.sum() > 1:
            active_vals = pos_mean_acts[active_mask]
            probs = active_vals / active_vals.sum()
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            max_entropy = math.log(active_mask.sum().item())
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            entropy = 0.0
            normalized_entropy = 0.0

        # Gini coefficient of positive activation magnitudes
        if active_mask.sum() > 1:
            sorted_vals = torch.sort(pos_mean_acts[active_mask])[0]
            n = sorted_vals.shape[0]
            cumsum = torch.cumsum(sorted_vals, dim=0)
            gini = (2.0 * torch.sum((torch.arange(1, n+1).float() * sorted_vals)) / (n * sorted_vals.sum()) - (n + 1) / n).item()
        else:
            gini = 0.0

        results.append({
            "class": str(target_class),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "support_size": support_size,
            "broad_support_20pct": broad_support,
            "n_active_on_positives": int(active_mask.sum().item()),
            "activation_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "gini_coefficient": gini,
        })

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Exp 56a: Feature Co-activation Analysis")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model and tokenizer
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

    all_results = {"config": {
        "model": MODEL_NAME, "layer": LAYER, "n_examples": N_EXAMPLES,
        "support_pos_threshold": SUPPORT_POS_THRESHOLD,
        "support_neg_threshold": SUPPORT_NEG_THRESHOLD,
        "datasets": DATASETS, "seed": SEED,
    }}

    # Process each dataset
    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load and tokenize dataset
        t0 = time.time()
        try:
            examples_by_class = load_dataset_with_labels(ds_name, tokenizer, N_EXAMPLES)
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            import traceback; traceback.print_exc()
            all_results[ds_name] = {"error": str(e)}
            continue

        n_classes = len(examples_by_class)
        n_total = sum(len(v) for v in examples_by_class.values())
        print(f"  Loaded {n_total} examples across {n_classes} classes in {time.time()-t0:.1f}s")
        for cls, exs in sorted(examples_by_class.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"    {cls}: {len(exs)} examples")

        # Collect layer activations for all examples
        print(f"  Collecting L{LAYER} activations...")
        t0 = time.time()
        acts_by_class = {}
        for cls, token_ids_list in examples_by_class.items():
            cls_acts = []
            for i in range(0, len(token_ids_list), LLM_BATCH_SIZE):
                batch_ids = token_ids_list[i:i+LLM_BATCH_SIZE]
                # Pad to same length
                max_len = max(ids.shape[0] for ids in batch_ids)
                padded = torch.full((len(batch_ids), max_len), tokenizer.pad_token_id)
                mask = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
                for j, ids in enumerate(batch_ids):
                    padded[j, :ids.shape[0]] = ids
                    mask[j, :ids.shape[0]] = 1

                inputs = {"input_ids": padded.to(DEVICE), "attention_mask": mask.to(DEVICE)}
                layer_acts = collect_layer_acts(model, LAYER, inputs)

                # Take the last non-padding token's activation for each example
                for j in range(len(batch_ids)):
                    seq_len = mask[j].sum().item()
                    last_act = layer_acts[j, seq_len - 1]  # [d_model]
                    cls_acts.append(last_act.cpu())

            acts_by_class[cls] = cls_acts

        print(f"  Activations collected in {time.time()-t0:.1f}s")

        # Free some memory
        del examples_by_class
        torch.cuda.empty_cache()

        # Analyze each SAE
        ds_results = {"n_classes": n_classes, "n_total": n_total, "sae_results": {}}

        for sae_name in ["standard", "adaptive_l2", "perfeature_l2", "no_C"]:
            print(f"\n  SAE: {sae_name}")
            t0 = time.time()

            sae = load_sae(sae_name)
            concept_results = compute_concept_support(sae, acts_by_class, sae_name)

            # Aggregate stats
            support_sizes = [r["support_size"] for r in concept_results]
            broad_supports = [r["broad_support_20pct"] for r in concept_results]
            entropies = [r["normalized_entropy"] for r in concept_results]
            ginis = [r["gini_coefficient"] for r in concept_results]

            agg = {
                "mean_support_size": float(np.mean(support_sizes)) if support_sizes else 0,
                "median_support_size": float(np.median(support_sizes)) if support_sizes else 0,
                "std_support_size": float(np.std(support_sizes)) if support_sizes else 0,
                "mean_broad_support": float(np.mean(broad_supports)) if broad_supports else 0,
                "mean_normalized_entropy": float(np.mean(entropies)) if entropies else 0,
                "mean_gini": float(np.mean(ginis)) if ginis else 0,
                "n_concepts": len(concept_results),
                "per_concept": concept_results,
            }

            ds_results["sae_results"][sae_name] = agg

            elapsed = time.time() - t0
            print(f"    Support: mean={agg['mean_support_size']:.1f} "
                  f"median={agg['median_support_size']:.1f} "
                  f"broad={agg['mean_broad_support']:.1f} "
                  f"entropy={agg['mean_normalized_entropy']:.3f} "
                  f"gini={agg['mean_gini']:.3f} ({elapsed:.1f}s)")

            del sae
            torch.cuda.empty_cache()

        all_results[ds_name] = ds_results

        # Incremental save
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        del acts_by_class
        torch.cuda.empty_cache()

    # Free model
    del model
    torch.cuda.empty_cache()

    # ─── Summary ──────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("SUMMARY: Mean Concept Support Size Across All Datasets")
    print(f"{'='*70}")

    for sae_name in ["standard", "adaptive_l2", "perfeature_l2", "no_C"]:
        all_supports = []
        all_broad = []
        all_entropy = []
        all_gini = []
        for ds_name in DATASETS:
            ds_data = all_results.get(ds_name, {})
            if "error" in ds_data:
                continue
            sae_data = ds_data.get("sae_results", {}).get(sae_name, {})
            for concept in sae_data.get("per_concept", []):
                all_supports.append(concept["support_size"])
                all_broad.append(concept["broad_support_20pct"])
                all_entropy.append(concept["normalized_entropy"])
                all_gini.append(concept["gini_coefficient"])

        if all_supports:
            print(f"\n  {sae_name}:")
            print(f"    Discriminative support: {np.mean(all_supports):.1f} ± {np.std(all_supports):.1f}")
            print(f"    Broad support (>20%):   {np.mean(all_broad):.1f} ± {np.std(all_broad):.1f}")
            print(f"    Normalized entropy:     {np.mean(all_entropy):.4f} ± {np.std(all_entropy):.4f}")
            print(f"    Gini coefficient:       {np.mean(all_gini):.4f} ± {np.std(all_gini):.4f}")

    # Save final
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
