"""
Experiment 33: Feature Interpretability at 50M Scale
====================================================

LLM-judged describe-then-predict evaluation of standard vs cosine SAE features.
Bridges proxy metrics (FVE, alive count) to ground-truth interpretability.

Two phases:
  --collect  (run on A100): Load model + SAEs, collect 200K tokens, extract
             top-20 activating contexts per feature. Saves exp33_contexts.json.
  --score    (run locally): LLM judge via Bedrock Sonnet 4.6 scores each feature.
             Saves exp33_results.json incrementally.

Usage:
    # Phase 1: on A100
    ssh <server>     cd ~/MechInter--RNH
    python experiments/exp33_feature_interpretability.py --collect

    # Phase 2: locally (after scp-ing exp33_contexts.json)
    python experiments/exp33_feature_interpretability.py --score
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "Qwen/Qwen3-8B"
LAYER_IDX = 27
D_MODEL = 4096
D_SAE = 16384
K = 80

# Collection
N_COLLECT_TOKENS = 200_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0
SKIP_DOCS = 200_000          # Skip training data overlap
ALIVE_THRESHOLD = 0.001      # Feature fires on >0.1% of tokens

# Feature sampling
N_FEATURES_PER_SAE = 200
N_HIGH_FREQ = 50
N_MED_FREQ = 100
N_LOW_FREQ = 50
TOP_K_CONTEXTS = 20          # Top activating contexts per feature
CONTEXT_WINDOW = 50          # Tokens of surrounding context

# LLM scoring
LLM_MODEL = "bedrock/us.anthropic.claude-sonnet-4-6"
N_DESCRIBE = 10              # Contexts shown to generate description
N_PREDICT = 10               # Held-out contexts for prediction
MAX_RETRIES = 3
RETRY_DELAY = 5

# Paths
CHECKPOINT_DIR = "checkpoints/exp17"
CONTEXTS_PATH = "experiments/exp33_contexts.json"
RESULTS_PATH = "experiments/exp33_results.json"
ANALYSIS_PATH = "experiments/exp33_analysis.md"

VARIANTS = {
    "standard": {
        "checkpoint": f"{CHECKPOINT_DIR}/standard_L27_final.pt",
        "class": "BatchTopKSAE",
    },
    "adaptive_l2": {
        "checkpoint": f"{CHECKPOINT_DIR}/adaptive_l2_L27_final.pt",
        "class": "AdaptiveCosineBatchTopKSAE",
    },
}

SEED = 42


# =============================================================================
# SAE Architectures (from exp17, self-contained)
# =============================================================================

class BatchTopKSAE(nn.Module):
    def __init__(self, d_model=D_MODEL, d_sae=D_SAE, k=K):
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
        return self.decode(f), f


class AdaptiveCosineBatchTopKSAE(nn.Module):
    def __init__(self, d_model=D_MODEL, d_sae=D_SAE, k=K):
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
        return self.decode(f), f


SAE_CLASSES = {
    "BatchTopKSAE": BatchTopKSAE,
    "AdaptiveCosineBatchTopKSAE": AdaptiveCosineBatchTopKSAE,
}


# =============================================================================
# Collection phase (GPU)
# =============================================================================

class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
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


def collect_contexts():
    """Phase 1: collect activations and extract top-K contexts per feature."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("Exp 33: Feature Interpretability — Collection Phase")
    print("=" * 70)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load model ----
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

    # ---- Collect texts and tokenize ----
    print(f"\nStreaming FineWeb (skipping first {SKIP_DOCS:,} docs)...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    # Skip to avoid train/eval overlap
    for i, _ in enumerate(text_iter):
        if i >= SKIP_DOCS:
            break

    # Collect raw texts and their token IDs together
    all_texts = []       # raw text strings
    all_token_ids = []   # list of 1D tensors (token IDs per doc)
    tokens_collected = 0

    while tokens_collected < N_COLLECT_TOKENS:
        try:
            row = next(text_iter)
        except StopIteration:
            break
        text = row["text"]
        if len(text) < 50:
            continue
        text = text[:2048]

        toks = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=CTX_LEN, add_special_tokens=True)
        ids = toks["input_ids"][0]   # [seq_len]
        all_texts.append(text)
        all_token_ids.append(ids)
        tokens_collected += ids.shape[0]

    print(f"Collected {len(all_texts):,} docs, ~{tokens_collected:,} tokens "
          f"in {time.time()-t0:.1f}s")

    # ---- Collect activations + map tokens to (doc_idx, pos) ----
    print(f"\nCollecting layer {LAYER_IDX} activations...")
    t0 = time.time()

    # Process in batches and store per-token: activation, doc_idx, position
    all_acts = []          # [N, d_model]
    all_positions = []     # list of (doc_idx, token_pos) tuples
    global_idx = 0

    for batch_start in range(0, len(all_texts), COLLECTION_BATCH_SIZE):
        batch_end = min(batch_start + COLLECTION_BATCH_SIZE, len(all_texts))
        batch_texts = all_texts[batch_start:batch_end]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=CTX_LEN,
        ).to(DEVICE)

        acts = _collect_layer_acts(model, LAYER_IDX, inputs)  # [B, seq, d_model]
        mask = inputs["attention_mask"].bool()                 # [B, seq]

        for b in range(acts.shape[0]):
            doc_idx = batch_start + b
            for pos in range(acts.shape[1]):
                if mask[b, pos]:
                    all_positions.append((doc_idx, pos))

        flat = acts[mask]  # [n_valid, d_model]

        # Filter attention sinks
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            keep = norms < median * OUTLIER_MULTIPLIER
            flat = flat[keep]
            # Filter positions too
            positions_batch = [all_positions[global_idx + i]
                              for i in range(len(keep)) if keep[i]]
            # Replace tail of all_positions
            all_positions = all_positions[:global_idx] + positions_batch
        else:
            positions_batch_len = flat.shape[0]

        all_acts.append(flat.to("cpu", dtype=torch.float32))
        global_idx = len(all_positions)

    all_acts = torch.cat(all_acts, dim=0)[:N_COLLECT_TOKENS]
    all_positions = all_positions[:N_COLLECT_TOKENS]
    n_tokens = all_acts.shape[0]
    print(f"Collected {n_tokens:,} tokens in {time.time()-t0:.1f}s")

    # Free model memory
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ---- Decode token strings for context windows ----
    # Pre-decode all token IDs for fast context window extraction
    all_decoded_tokens = []  # list of lists of strings, per doc
    for ids in all_token_ids:
        decoded = [tokenizer.decode([tid], skip_special_tokens=False) for tid in ids]
        all_decoded_tokens.append(decoded)

    # ---- Process each SAE ----
    output = {"config": {
        "model": MODEL_NAME, "layer": LAYER_IDX,
        "n_tokens": n_tokens, "n_features_per_sae": N_FEATURES_PER_SAE,
        "top_k_contexts": TOP_K_CONTEXTS, "context_window": CONTEXT_WINDOW,
        "alive_threshold": ALIVE_THRESHOLD, "seed": SEED,
    }}

    for variant_name, variant_info in VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"Processing {variant_name}")
        print(f"{'='*50}")

        # Load SAE
        cls = SAE_CLASSES[variant_info["class"]]
        sae = cls(d_model=D_MODEL, d_sae=D_SAE, k=K)
        state = torch.load(variant_info["checkpoint"], map_location="cpu",
                          weights_only=True)
        sae.load_state_dict(state)
        sae.eval()
        print(f"Loaded {variant_info['checkpoint']}")

        # Encode all tokens in batches
        print("Encoding all tokens...")
        t0 = time.time()
        encode_batch = 4096
        all_features = []
        for i in range(0, n_tokens, encode_batch):
            batch = all_acts[i:i+encode_batch]
            with torch.no_grad():
                feats = sae.encode(batch)
            all_features.append(feats)
        all_features = torch.cat(all_features, dim=0)  # [n_tokens, d_sae]
        print(f"Encoded in {time.time()-t0:.1f}s")

        # Compute per-feature activation frequency
        fire_counts = (all_features > 0).float().sum(dim=0)  # [d_sae]
        fire_rates = fire_counts / n_tokens
        alive_mask = fire_rates > ALIVE_THRESHOLD
        alive_indices = alive_mask.nonzero(as_tuple=True)[0].tolist()
        alive_count = len(alive_indices)
        total_dead = D_SAE - alive_count
        print(f"Alive features: {alive_count} / {D_SAE} "
              f"({alive_count/D_SAE*100:.1f}%), dead: {total_dead}")

        # Sort alive features by frequency
        alive_freqs = [(idx, fire_rates[idx].item()) for idx in alive_indices]
        alive_freqs.sort(key=lambda x: x[1])

        # Stratified sampling
        n_alive = len(alive_freqs)
        q25 = n_alive // 4
        q75 = 3 * n_alive // 4

        low_pool = alive_freqs[:q25]
        med_pool = alive_freqs[q25:q75]
        high_pool = alive_freqs[q75:]

        random.seed(SEED)
        sampled_low = random.sample(low_pool, min(N_LOW_FREQ, len(low_pool)))
        sampled_med = random.sample(med_pool, min(N_MED_FREQ, len(med_pool)))
        sampled_high = random.sample(high_pool, min(N_HIGH_FREQ, len(high_pool)))
        sampled = sampled_low + sampled_med + sampled_high
        print(f"Sampled {len(sampled)} features "
              f"(low={len(sampled_low)}, med={len(sampled_med)}, "
              f"high={len(sampled_high)})")

        # Extract top-K contexts per feature
        print("Extracting top activating contexts...")
        t0 = time.time()
        feature_data = []

        for feat_idx, feat_freq in sampled:
            # Get top-K activations for this feature
            feat_acts = all_features[:, feat_idx]  # [n_tokens]
            topk_vals, topk_indices = torch.topk(feat_acts, TOP_K_CONTEXTS)

            contexts = []
            for rank, (val, tok_idx) in enumerate(
                    zip(topk_vals.tolist(), topk_indices.tolist())):
                doc_idx, token_pos = all_positions[tok_idx]
                doc_tokens = all_decoded_tokens[doc_idx]
                doc_len = len(doc_tokens)

                # Context window
                start = max(0, token_pos - CONTEXT_WINDOW // 2)
                end = min(doc_len, token_pos + CONTEXT_WINDOW // 2 + 1)

                # Build context with marked activating token
                prefix = "".join(doc_tokens[start:token_pos])
                target = doc_tokens[token_pos]
                suffix = "".join(doc_tokens[token_pos+1:end])

                contexts.append({
                    "rank": rank,
                    "activation": val,
                    "prefix": prefix,
                    "target_token": target,
                    "suffix": suffix,
                    "doc_idx": doc_idx,
                    "token_pos": token_pos,
                })

            freq_band = ("low" if feat_freq <= alive_freqs[q25-1][1]
                        else "high" if feat_freq >= alive_freqs[q75][1]
                        else "medium")

            feature_data.append({
                "feature_idx": feat_idx,
                "frequency": feat_freq,
                "frequency_band": freq_band,
                "contexts": contexts,
            })

        print(f"Extracted contexts in {time.time()-t0:.1f}s")

        output[variant_name] = {
            "alive_count": alive_count,
            "dead_count": total_dead,
            "sampled_count": len(sampled),
            "features": feature_data,
        }

        del all_features, sae
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # Save
    with open(CONTEXTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nContexts saved to {CONTEXTS_PATH}")
    print(f"Total features: {sum(len(output[v]['features']) for v in VARIANTS)}")


# =============================================================================
# Scoring phase (local, LLM judge)
# =============================================================================

def call_llm(messages, max_tokens=512):
    """Call Bedrock Sonnet 4.6 via litellm with retries."""
    import litellm

    os.environ.setdefault("AWS_REGION_NAME", "us-west-1")

    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  LLM call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def format_context(ctx, mark_target=True):
    """Format a context entry as readable text with optional target marking."""
    if mark_target:
        return f"{ctx['prefix']}**{ctx['target_token']}**{ctx['suffix']}"
    else:
        return f"{ctx['prefix']}{ctx['target_token']}{ctx['suffix']}"


def describe_feature(contexts):
    """Ask LLM to describe what pattern a feature detects."""
    examples = "\n\n---\n\n".join(
        f"Excerpt {i+1}:\n{format_context(ctx, mark_target=True)}"
        for i, ctx in enumerate(contexts[:N_DESCRIBE])
    )

    prompt = f"""These text excerpts all strongly activate a particular neuron in a language model. The activating token in each excerpt is marked with **asterisks**.

{examples}

What pattern or concept does this neuron detect? Give a concise description (1-2 sentences). Focus on what the marked tokens have in common — it could be semantic (a topic/concept), syntactic (a grammatical role), lexical (specific words/subwords), or positional."""

    response = call_llm([{"role": "user", "content": prompt}], max_tokens=200)
    return response.strip()


def predict_tokens(description, contexts):
    """Ask LLM to predict activating tokens given the description."""
    excerpts = []
    for i, ctx in enumerate(contexts[:N_PREDICT]):
        # Show context WITHOUT marking the target
        text = format_context(ctx, mark_target=False)
        excerpts.append(f"Excerpt {i+1}:\n{text}")

    examples_text = "\n\n---\n\n".join(excerpts)

    prompt = f"""A neuron in a language model detects the following pattern:

"{description}"

For each excerpt below, identify which single token is most likely the one that activates this neuron. Respond with ONLY a JSON list of {N_PREDICT} strings, one per excerpt — each string is your predicted activating token (the exact token text, not a description). If uncertain, give your best guess.

{examples_text}

Respond with a JSON list of exactly {N_PREDICT} strings. Example format: ["token1", "token2", ...]"""

    response = call_llm([{"role": "user", "content": prompt}], max_tokens=300)

    # Parse JSON list from response
    try:
        # Find the JSON array in the response
        text = response.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            predictions = json.loads(text[start:end])
            if isinstance(predictions, list):
                return predictions[:N_PREDICT]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try line-by-line parsing
    lines = [l.strip().strip('"').strip("'").strip(",") for l in response.strip().split("\n") if l.strip()]
    return lines[:N_PREDICT]


def score_predictions(predictions, contexts):
    """Score predictions against actual target tokens."""
    correct = 0
    details = []
    for i, (pred, ctx) in enumerate(zip(predictions, contexts[:N_PREDICT])):
        actual = ctx["target_token"].strip()
        pred_clean = pred.strip()
        # Flexible matching: exact, stripped, or substring
        match = (pred_clean == actual or
                 pred_clean.strip() == actual.strip() or
                 pred_clean.lower() == actual.lower() or
                 actual.strip() in pred_clean or
                 pred_clean in actual.strip())
        if match:
            correct += 1
        details.append({
            "predicted": pred_clean,
            "actual": actual,
            "correct": match,
        })
    return correct / max(len(predictions), 1), details


def score_features():
    """Phase 2: LLM-judge scoring of collected feature contexts."""
    print("=" * 70)
    print("Exp 33: Feature Interpretability — Scoring Phase")
    print(f"LLM: {LLM_MODEL}")
    print("=" * 70)

    # Load contexts
    with open(CONTEXTS_PATH) as f:
        data = json.load(f)

    config = data["config"]
    print(f"Loaded contexts: {config['n_tokens']:,} tokens, "
          f"layer {config['layer']}")

    # Load existing results for incremental saves
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Resuming from {RESULTS_PATH}")

    results["config"] = {
        **config,
        "llm_model": LLM_MODEL,
        "n_describe": N_DESCRIBE,
        "n_predict": N_PREDICT,
        "interpretability_threshold": 0.5,
    }

    for variant_name in VARIANTS:
        print(f"\n{'='*50}")
        print(f"Scoring {variant_name}")
        print(f"{'='*50}")

        variant_data = data[variant_name]
        features = variant_data["features"]

        # Check what's already scored
        if variant_name in results and "features" in results[variant_name]:
            scored_indices = {f["feature_idx"]
                            for f in results[variant_name]["features"]}
        else:
            scored_indices = set()
            results[variant_name] = {
                "alive_count": variant_data["alive_count"],
                "dead_count": variant_data["dead_count"],
                "sampled_count": variant_data["sampled_count"],
                "features": [],
            }

        n_total = len(features)
        n_done = len(scored_indices)
        print(f"Features: {n_total} total, {n_done} already scored, "
              f"{n_total - n_done} remaining")

        for i, feat in enumerate(features):
            if feat["feature_idx"] in scored_indices:
                continue

            t0 = time.time()
            feat_idx = feat["feature_idx"]
            contexts = feat["contexts"]

            if len(contexts) < N_DESCRIBE + N_PREDICT:
                print(f"  [{i+1}/{n_total}] Feature {feat_idx}: "
                      f"only {len(contexts)} contexts, skipping")
                continue

            # Split into describe and predict sets
            describe_contexts = contexts[:N_DESCRIBE]
            predict_contexts = contexts[N_DESCRIBE:N_DESCRIBE + N_PREDICT]

            # Step 1: Describe
            try:
                description = describe_feature(describe_contexts)
            except Exception as e:
                print(f"  [{i+1}/{n_total}] Feature {feat_idx}: "
                      f"describe failed ({e}), skipping")
                continue

            # Step 2: Predict
            try:
                predictions = predict_tokens(description, predict_contexts)
            except Exception as e:
                print(f"  [{i+1}/{n_total}] Feature {feat_idx}: "
                      f"predict failed ({e}), skipping")
                continue

            # Step 3: Score
            accuracy, details = score_predictions(predictions, predict_contexts)
            is_interpretable = accuracy >= 0.5

            result_entry = {
                "feature_idx": feat_idx,
                "frequency": feat["frequency"],
                "frequency_band": feat["frequency_band"],
                "description": description,
                "prediction_accuracy": accuracy,
                "interpretable": is_interpretable,
                "prediction_details": details,
            }
            results[variant_name]["features"].append(result_entry)

            status = "YES" if is_interpretable else "no"
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_total}] Feature {feat_idx} "
                  f"({feat['frequency_band']}): {accuracy:.0%} "
                  f"[{status}] ({elapsed:.1f}s) — {description[:80]}")

            # Incremental save every 10 features
            if (len(results[variant_name]["features"]) % 10 == 0):
                _save_results(results)

        # Final save for this variant
        _compute_variant_stats(results, variant_name)
        _save_results(results)

    # Compute comparison
    _compute_comparison(results)
    _save_results(results)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Print summary
    _print_summary(results)


def _compute_variant_stats(results, variant_name):
    """Compute aggregate stats for a variant."""
    features = results[variant_name]["features"]
    if not features:
        return

    accuracies = [f["prediction_accuracy"] for f in features]
    interpretable = [f for f in features if f["interpretable"]]

    results[variant_name]["interpretability_rate"] = len(interpretable) / len(features)
    results[variant_name]["mean_prediction_accuracy"] = np.mean(accuracies)
    results[variant_name]["median_prediction_accuracy"] = float(np.median(accuracies))

    # Per frequency band
    for band in ["low", "medium", "high"]:
        band_feats = [f for f in features if f["frequency_band"] == band]
        if band_feats:
            band_accs = [f["prediction_accuracy"] for f in band_feats]
            band_interp = [f for f in band_feats if f["interpretable"]]
            results[variant_name][f"{band}_freq_rate"] = len(band_interp) / len(band_feats)
            results[variant_name][f"{band}_freq_mean_acc"] = float(np.mean(band_accs))
            results[variant_name][f"{band}_freq_count"] = len(band_feats)

    # Extrapolate to all alive features
    alive = results[variant_name]["alive_count"]
    rate = results[variant_name]["interpretability_rate"]
    results[variant_name]["estimated_total_interpretable"] = int(alive * rate)


def _compute_comparison(results):
    """Compute head-to-head comparison."""
    comparison = {}
    if "standard" in results and "adaptive_l2" in results:
        std = results["standard"]
        cos = results["adaptive_l2"]
        if "interpretability_rate" in std and "interpretability_rate" in cos:
            comparison["std_interp_rate"] = std["interpretability_rate"]
            comparison["cos_interp_rate"] = cos["interpretability_rate"]
            comparison["rate_ratio"] = (cos["interpretability_rate"] /
                                       max(std["interpretability_rate"], 1e-6))
            comparison["std_mean_accuracy"] = std["mean_prediction_accuracy"]
            comparison["cos_mean_accuracy"] = cos["mean_prediction_accuracy"]
            comparison["accuracy_delta"] = (cos["mean_prediction_accuracy"] -
                                           std["mean_prediction_accuracy"])
            if "estimated_total_interpretable" in std and "estimated_total_interpretable" in cos:
                comparison["std_total_interpretable"] = std["estimated_total_interpretable"]
                comparison["cos_total_interpretable"] = cos["estimated_total_interpretable"]
                comparison["total_ratio"] = (cos["estimated_total_interpretable"] /
                                            max(std["estimated_total_interpretable"], 1))
    results["comparison"] = comparison


def _save_results(results):
    """Save results to JSON."""
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=convert)


def _print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for variant_name in VARIANTS:
        if variant_name not in results:
            continue
        v = results[variant_name]
        print(f"\n{variant_name}:")
        print(f"  Alive features: {v.get('alive_count', '?')}")
        print(f"  Sampled: {v.get('sampled_count', '?')}")
        if "interpretability_rate" in v:
            print(f"  Interpretability rate: {v['interpretability_rate']:.1%}")
            print(f"  Mean prediction accuracy: {v['mean_prediction_accuracy']:.1%}")
            for band in ["low", "medium", "high"]:
                if f"{band}_freq_rate" in v:
                    print(f"  {band.capitalize()} freq rate: "
                          f"{v[f'{band}_freq_rate']:.1%} "
                          f"(n={v[f'{band}_freq_count']})")
            if "estimated_total_interpretable" in v:
                print(f"  Est. total interpretable: "
                      f"{v['estimated_total_interpretable']:,}")

    if "comparison" in results and results["comparison"]:
        c = results["comparison"]
        print(f"\nComparison:")
        print(f"  Rate ratio (cosine/standard): {c.get('rate_ratio', '?'):.2f}x")
        print(f"  Accuracy delta: {c.get('accuracy_delta', '?'):+.1%}")
        if "total_ratio" in c:
            print(f"  Total interpretable ratio: {c['total_ratio']:.2f}x")
            print(f"    Standard: {c['std_total_interpretable']:,}")
            print(f"    Cosine:   {c['cos_total_interpretable']:,}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 33: Feature Interpretability")
    parser.add_argument("--collect", action="store_true",
                       help="Run collection phase (GPU required)")
    parser.add_argument("--score", action="store_true",
                       help="Run scoring phase (LLM API, no GPU)")
    args = parser.parse_args()

    if not args.collect and not args.score:
        parser.error("Specify --collect or --score")

    if args.collect:
        collect_contexts()
    if args.score:
        score_features()
