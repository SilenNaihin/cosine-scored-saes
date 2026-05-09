"""
Experiment 2b: Outlier Analysis — Causal Failure Cases of SAE vs Cosine
========================================================================
Goal: Find specific token activations where the SAE (inner product) and cosine
similarity *disagree* about feature importance, then run a causal ablation at
the logit <author>el to show which one is right.

Outlier types:
  A) "Matched SAE, different cosine" pairs — two tokens with ~equal SAE activation
     but different cosine similarity. If ablation effect tracks cosine (not SAE),
     the SAE is measuring noise.
  B) High-norm false positives — SAE says strong, cosine says weak, ablation confirms
     the feature doesn't actually matter.
  C) Low-norm misses — cosine says strong, SAE says weak, ablation confirms the
     feature actually matters a lot.

Causal test: full forward pass with vs without the feature, measure KL divergence
at the logit <author>el (not just next-layer cosine).

Model: Qwen3-8B
SAE: adamkarvonen/qwen3-8b-saes (layer 18, 65k, BatchTopK k=80)
"""

import json
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import interp_tools.saes.batch_topk_sae as batch_topk_sae
import interp_tools.model_utils as model_utils

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
SAE_REPO = "adamkarvonen/qwen3-8b-saes"
SAE_FILENAME = "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/trainer_2/ae.pt"
LAYER = 18
MAX_SEQ_LEN = 256
NUM_FEATURES = 10  # features to analyze deeply
NUM_OUTLIER_PAIRS = 20  # pairs per feature


PROMPTS = [
    "The theory of general relativity describes how gravity works as a curvature of spacetime caused by mass and energy. Einstein published this theory in 1915, fundamentally changing our understanding of the universe.",
    "In Python, you can use list comprehensions to create new lists based on existing ones. For example, squares = [x**2 for x in range(10)] creates a list of squares. This is more concise than a traditional for loop.",
    "The French Revolution began in 1789 when the people of France rose up against the monarchy. The storming of the Bastille on July 14th became a symbol of the revolution and is still celebrated as a national holiday.",
    "Machine learning models are trained by optimizing a loss function over a dataset. The model learns to map inputs to outputs by adjusting its parameters through gradient descent. Regularization techniques help prevent overfitting.",
    "Once upon a time, there was a small village nestled in a valley between two great mountains. The villagers lived simple lives, farming the rich soil and trading goods at the weekly market in the town square.",
    "The mitochondria is the powerhouse of the cell, producing adenosine triphosphate (ATP) through oxidative phosphorylation. This process involves the electron transport chain in the inner mitochondrial membrane.",
    "According to quantum mechanics, particles can exist in superposition states until measured. The famous Schrodinger's cat thought experiment illustrates this principle, where a cat in a box is both alive and dead simultaneously.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "The stock market crashed in 2008 because of a housing bubble fueled by subprime mortgages and complex financial derivatives. Banks had taken on excessive risk, and when housing prices fell, the entire financial system nearly collapsed.",
    "In a clinical trial, researchers found that the new drug reduced symptoms by 40% compared to placebo. The study was double-blind and randomized, with 500 participants across 12 medical centers.",
    "Shakespeare wrote Romeo and Juliet in the late 16th century. The play tells the tragic story of two young lovers from feuding families in Verona, Italy. It remains one of the most performed plays in the world.",
    "The neural network architecture consists of multiple layers of interconnected nodes. Each connection has a weight that is learned during training. Activation functions introduce nonlinearity, enabling the network to learn complex patterns.",
    "Climate change is primarily caused by the burning of fossil fuels, which releases carbon dioxide and other greenhouse gases into the atmosphere. These gases trap heat, leading to a gradual increase in global temperatures.",
    "In mathematics, a prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The fundamental theorem of arithmetic states that every integer can be uniquely factored into primes.",
    "The human genome contains approximately 3 billion base pairs of DNA organized into 23 pairs of chromosomes. Only about 1.5% of the genome codes for proteins, while the rest includes regulatory sequences and other elements.",
    "Photosynthesis converts sunlight into chemical energy in plants. The process occurs in chloroplasts, where light-dependent reactions in the thylakoid membranes produce ATP and NADPH, which drive the Calvin cycle.",
    "The United States Constitution was ratified in 1788 after extensive debate between Federalists and Anti-Federalists. The Bill of Rights, comprising the first ten amendments, was added in 1791 to protect individual liberties.",
    "import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):\n    def __init__(self, d_model, nhead, num_layers):\n        super().__init__()\n        self.encoder = nn.TransformerEncoder(\n            nn.TransformerEncoderLayer(d_model, nhead), num_layers)\n",
    "Water exists in three states: solid (ice), liquid (water), and gas (steam). The transitions between these states occur at specific temperatures and pressures, governed by the phase diagram of water.",
    "The Renaissance was a period of cultural rebirth in Europe, spanning roughly from the 14th to the 17th century. It saw tremendous advances in art, science, literature, and philosophy, with figures like Leonardo da Vinci and Galileo.",
    "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller subproblems. Each recursive call should move toward a base case that stops the recursion.",
    "The Amazon rainforest covers approximately 5.5 million square kilometers and produces about 20% of the world's oxygen. It is home to an estimated 10% of all known species on Earth.",
    "Quantum computing uses qubits instead of classical bits. Unlike classical bits which are either 0 or 1, qubits can exist in superposition, allowing quantum computers to explore many solutions simultaneously.",
    "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing, transportation, and society. Steam power, mechanized textile production, and iron smelting drove unprecedented economic growth.",
]


def collect_activations_with_context(model, tokenizer, prompts, layer_idx, batch_size=4):
    """Collect activations with token identity preserved."""
    all_data = []
    submodule = model_utils.get_submodule(model, layer_idx)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(DEVICE)

        acts = model_utils.collect_activations(model, submodule, inputs)
        attention_mask = inputs["attention_mask"]

        for b in range(acts.shape[0]):
            mask = attention_mask[b].bool()
            tokens = inputs["input_ids"][b][mask]
            token_acts = acts[b][mask]
            token_strs = [tokenizer.decode([t]) for t in tokens]
            for pos, (tok_str, act) in enumerate(zip(token_strs, token_acts)):
                all_data.append({
                    "prompt_idx": i + b,
                    "position": pos,
                    "token": tok_str,
                    "activation": act.detach(),
                })

    return all_data


def filter_sinks(data, threshold_multiplier=10.0):
    """Filter attention sink tokens."""
    norms = torch.stack([d["activation"] for d in data]).norm(dim=-1)
    median_norm = norms.median()
    mask = norms < (median_norm * threshold_multiplier)
    return [d for d, m in zip(data, mask) if m]


def ablate_and_measure_kl(model, activation, feature_dir, layer_idx):
    """
    Given a single activation at layer_idx, run the rest of the model with
    and without the feature, return KL divergence at the logit <author>el.

    We hook layer_idx to replace its output, then run the remaining layers.
    """
    # Original: full forward from this activation
    # Ablated: remove feature projection from activation, then full forward

    x = activation.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    projection = (activation @ feature_dir) * feature_dir
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # Run through remaining layers manually
        hidden_orig = x
        hidden_ablated = x_ablated

        for l in range(layer_idx + 1, len(model.model.layers)):
            layer_module = model.model.layers[l]
            # Apply input layernorm
            normed_orig = layer_module.input_layernorm(hidden_orig)
            normed_ablated = layer_module.input_layernorm(hidden_ablated)

            # Self-attention (no causal mask needed for single token)
            # We need to handle this carefully — use the layer directly
            # but we can't easily run a single-token through a decoder layer
            # without position embeddings. Use hook approach instead.
            pass

        # Simpler: use hook-based approach. Replace layer output and run full model.
        # We'll use a dummy input and hook to inject our activation.

    # Hook approach: run model on a dummy input, intercept at layer_idx
    # to inject our activation, then collect logits
    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

    def make_replace_hook(replacement):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                return (replacement,) + outputs[1:]
            return replacement
        return hook

    # Original logits
    handle = model.model.layers[layer_idx].register_forward_hook(
        make_replace_hook(x)
    )
    with torch.no_grad():
        try:
            orig_logits = model(dummy_input).logits[0, -1, :]
        except Exception:
            handle.remove()
            return None
    handle.remove()

    # Ablated logits
    handle = model.model.layers[layer_idx].register_forward_hook(
        make_replace_hook(x_ablated)
    )
    with torch.no_grad():
        try:
            ablated_logits = model(dummy_input).logits[0, -1, :]
        except Exception:
            handle.remove()
            return None
    handle.remove()

    # KL divergence
    orig_probs = torch.softmax(orig_logits.float(), dim=-1)
    ablated_probs = torch.softmax(ablated_logits.float(), dim=-1)

    # KL(orig || ablated) — how much does ablation change the distribution?
    kl = torch.nn.functional.kl_div(
        ablated_probs.log(), orig_probs, reduction="sum"
    ).item()

    # Also get top-token info
    orig_top = orig_logits.argmax().item()
    ablated_top = ablated_logits.argmax().item()

    # Cosine similarity of logit vectors
    logit_cos = torch.nn.functional.cosine_similarity(
        orig_logits.unsqueeze(0).float(), ablated_logits.unsqueeze(0).float()
    ).item()

    return {
        "kl_divergence": kl,
        "logit_cosine": logit_cos,
        "top_token_match": orig_top == ablated_top,
    }


def find_outlier_pairs(cos_sims, sae_acts, norms, n_pairs=20):
    """
    Find pairs of tokens where SAE activation is ~equal but cosine differs.
    These are the cases where norm is compensating for direction.
    """
    pairs = []

    # Sort by SAE activation
    order = np.argsort(sae_acts)

    # Slide a window along SAE-activation-sorted tokens, find pairs with max cosine diff
    window = max(len(sae_acts) // 20, 50)

    for i in range(0, len(order) - window, window // 2):
        chunk_indices = order[i : i + window]
        chunk_sae = sae_acts[chunk_indices]
        chunk_cos = cos_sims[chunk_indices]
        chunk_norms = norms[chunk_indices]

        # Within this chunk, SAE activation is similar. Find max cosine difference.
        if chunk_cos.max() - chunk_cos.min() < 0.05:
            continue

        high_cos_idx = chunk_indices[np.argmax(chunk_cos)]
        low_cos_idx = chunk_indices[np.argmin(chunk_cos)]

        # Check that SAE activations are actually close
        sae_diff = abs(sae_acts[high_cos_idx] - sae_acts[low_cos_idx])
        sae_mean = (sae_acts[high_cos_idx] + sae_acts[low_cos_idx]) / 2
        if sae_mean > 0 and sae_diff / sae_mean < 0.5:  # within 50% of each other
            pairs.append({
                "high_cos_idx": int(high_cos_idx),
                "low_cos_idx": int(low_cos_idx),
                "high_cos": float(cos_sims[high_cos_idx]),
                "low_cos": float(cos_sims[low_cos_idx]),
                "high_sae": float(sae_acts[high_cos_idx]),
                "low_sae": float(sae_acts[low_cos_idx]),
                "high_norm": float(norms[high_cos_idx]),
                "low_norm": float(norms[low_cos_idx]),
                "cos_gap": float(cos_sims[high_cos_idx] - cos_sims[low_cos_idx]),
            })

    # Sort by cosine gap (most disagreement first)
    pairs.sort(key=lambda p: -p["cos_gap"])
    return pairs[:n_pairs]


def find_high_norm_false_positives(cos_sims, sae_acts, norms, top_k=20):
    """
    Tokens where SAE activation is high but cosine is low (norm inflated it).
    """
    # High SAE, low cosine
    sae_rank = np.argsort(-sae_acts)  # descending
    cos_rank = np.argsort(cos_sims)  # ascending

    # Score: high SAE rank + low cosine rank = big disagreement
    n = len(sae_acts)
    sae_ranks = np.zeros(n)
    cos_ranks = np.zeros(n)
    sae_ranks[sae_rank] = np.arange(n)
    cos_ranks[cos_rank] = np.arange(n)

    # Want low sae_rank (high activation) and low cos_rank (low cosine)
    disagreement = sae_ranks + cos_ranks  # lower = more disagreement
    candidates = np.argsort(disagreement)

    results = []
    for idx in candidates[:top_k]:
        idx = int(idx)
        if sae_acts[idx] > 0:
            results.append({
                "idx": idx,
                "cos": float(cos_sims[idx]),
                "sae_act": float(sae_acts[idx]),
                "norm": float(norms[idx]),
                "inner_prod": float(cos_sims[idx] * norms[idx]),
            })
    return results


def find_low_norm_misses(cos_sims, sae_acts, norms, top_k=20):
    """
    Tokens where cosine is high but SAE activation is low (norm deflated it).
    """
    n = len(sae_acts)
    sae_rank = np.argsort(sae_acts)  # ascending (low activation)
    cos_rank = np.argsort(-cos_sims)  # descending (high cosine)

    sae_ranks = np.zeros(n)
    cos_ranks = np.zeros(n)
    sae_ranks[sae_rank] = np.arange(n)
    cos_ranks[cos_rank] = np.arange(n)

    disagreement = sae_ranks + cos_ranks
    candidates = np.argsort(disagreement)

    results = []
    for idx in candidates[:top_k]:
        idx = int(idx)
        results.append({
            "idx": idx,
            "cos": float(cos_sims[idx]),
            "sae_act": float(sae_acts[idx]),
            "norm": float(norms[idx]),
            "inner_prod": float(cos_sims[idx] * norms[idx]),
        })
    return results


def main():
    print("Experiment 2b: Outlier Analysis")
    print("=" * 70)

    # Load model + SAE
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print("Loading SAE...")
    sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
        repo_id=SAE_REPO, filename=SAE_FILENAME,
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE,
    )

    # Collect activations with token context
    print(f"Collecting activations from {len(PROMPTS)} prompts...")
    data = collect_activations_with_context(model, tokenizer, PROMPTS, LAYER)
    data = filter_sinks(data)
    print(f"{len(data)} tokens after filtering sinks")

    activations = torch.stack([d["activation"] for d in data])  # (N, d_model)

    # Encode all
    print("Computing SAE activations...")
    with torch.no_grad():
        all_sae_acts = sae.encode(activations)  # (N, d_sae)

    # Find most active features
    feature_freq = (all_sae_acts > 0).float().mean(dim=0)
    top_features = feature_freq.topk(NUM_FEATURES).indices.tolist()

    all_results = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "n_tokens": len(data),
        "features": [],
    }

    for feat_idx in top_features:
        print(f"\n{'='*70}")
        print(f"  Feature {feat_idx}")
        print(f"{'='*70}")

        feature_dir = sae.W_dec[feat_idx]  # unit vector

        # Compute metrics for all tokens
        cos_sims = torch.nn.functional.cosine_similarity(
            activations, feature_dir.unsqueeze(0), dim=-1
        ).detach().float().cpu().numpy()

        norms = activations.norm(dim=-1).detach().float().cpu().numpy()
        sae_feat_acts = all_sae_acts[:, feat_idx].detach().float().cpu().numpy()

        active_mask = sae_feat_acts > 0
        n_active = active_mask.sum()
        print(f"  Active on {n_active}/{len(data)} tokens")

        if n_active < 40:
            print("  Skipping: too few active tokens")
            continue

        feature_result = {
            "feature_idx": feat_idx,
            "n_active": int(n_active),
            "outlier_pairs": [],
            "false_positives": [],
            "low_norm_misses": [],
        }

        # === TYPE A: Matched SAE, different cosine ===
        print("\n  --- Type A: Matched SAE activation, different cosine ---")
        # Only consider active tokens
        active_indices = np.where(active_mask)[0]
        active_cos = cos_sims[active_indices]
        active_sae = sae_feat_acts[active_indices]
        active_norms = norms[active_indices]

        pairs = find_outlier_pairs(active_cos, active_sae, active_norms, NUM_OUTLIER_PAIRS)

        for p_idx, pair in enumerate(pairs[:5]):
            hi_idx = active_indices[pair["high_cos_idx"]] if pair["high_cos_idx"] < len(active_indices) else pair["high_cos_idx"]
            lo_idx = active_indices[pair["low_cos_idx"]] if pair["low_cos_idx"] < len(active_indices) else pair["low_cos_idx"]

            # CAUSAL TEST: ablate and measure KL divergence
            hi_result = ablate_and_measure_kl(
                model, data[hi_idx]["activation"], feature_dir, LAYER
            )
            lo_result = ablate_and_measure_kl(
                model, data[lo_idx]["activation"], feature_dir, LAYER
            )

            if hi_result is None or lo_result is None:
                continue

            # RNH predicts: high-cosine token should have LARGER ablation effect
            rnh_correct = hi_result["kl_divergence"] > lo_result["kl_divergence"]

            pair_result = {
                "high_cos_token": data[hi_idx]["token"],
                "low_cos_token": data[lo_idx]["token"],
                "high_cos": pair["high_cos"],
                "low_cos": pair["low_cos"],
                "high_sae_act": pair["high_sae"],
                "low_sae_act": pair["low_sae"],
                "high_norm": pair["high_norm"],
                "low_norm": pair["low_norm"],
                "high_kl": hi_result["kl_divergence"],
                "low_kl": lo_result["kl_divergence"],
                "high_logit_cos": hi_result["logit_cosine"],
                "low_logit_cos": lo_result["logit_cosine"],
                "rnh_correct": rnh_correct,
            }
            feature_result["outlier_pairs"].append(pair_result)

            marker = "RNH CORRECT" if rnh_correct else "RNH WRONG"
            print(
                f"  Pair {p_idx}: "
                f"'{data[hi_idx]['token']}' (cos={pair['high_cos']:.3f}, SAE={pair['high_sae']:.2f}, KL={hi_result['kl_divergence']:.4f}) vs "
                f"'{data[lo_idx]['token']}' (cos={pair['low_cos']:.3f}, SAE={pair['low_sae']:.2f}, KL={lo_result['kl_divergence']:.4f}) "
                f"→ {marker}"
            )

        # === TYPE B: High-norm false positives ===
        print("\n  --- Type B: High-norm false positives (SAE high, cosine low) ---")
        fps = find_high_norm_false_positives(cos_sims, sae_feat_acts, norms, 10)

        for fp in fps[:5]:
            idx = fp["idx"]
            result = ablate_and_measure_kl(
                model, data[idx]["activation"], feature_dir, LAYER
            )
            if result is None:
                continue

            fp["token"] = data[idx]["token"]
            fp["kl_divergence"] = result["kl_divergence"]
            fp["logit_cosine"] = result["logit_cosine"]
            feature_result["false_positives"].append(fp)

            print(
                f"    '{data[idx]['token']}' | cos={fp['cos']:.3f} | "
                f"SAE={fp['sae_act']:.2f} | norm={fp['norm']:.1f} | "
                f"KL={result['kl_divergence']:.4f}"
            )

        # === TYPE C: Low-norm misses (cosine high, SAE low) ===
        print("\n  --- Type C: Low-norm misses (cosine high, SAE low) ---")
        misses = find_low_norm_misses(cos_sims, sae_feat_acts, norms, 10)

        for miss in misses[:5]:
            idx = miss["idx"]
            result = ablate_and_measure_kl(
                model, data[idx]["activation"], feature_dir, LAYER
            )
            if result is None:
                continue

            miss["token"] = data[idx]["token"]
            miss["kl_divergence"] = result["kl_divergence"]
            miss["logit_cosine"] = result["logit_cosine"]
            feature_result["low_norm_misses"].append(miss)

            print(
                f"    '{data[idx]['token']}' | cos={miss['cos']:.3f} | "
                f"SAE={miss['sae_act']:.2f} | norm={miss['norm']:.1f} | "
                f"KL={result['kl_divergence']:.4f}"
            )

        all_results["features"].append(feature_result)

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total_pairs = 0
    rnh_correct = 0
    for feat in all_results["features"]:
        for pair in feat["outlier_pairs"]:
            total_pairs += 1
            if pair["rnh_correct"]:
                rnh_correct += 1

    if total_pairs > 0:
        print(f"\n  Type A (matched SAE, different cosine) pairs:")
        print(f"    RNH correct: {rnh_correct}/{total_pairs} ({100*rnh_correct/total_pairs:.1f}%)")
        print(f"    (Higher cosine → larger ablation effect, as RNH predicts)")

    # Type B: false positives should have LOW KL (small effect despite high SAE act)
    all_fp_kls = []
    all_fp_sae = []
    all_miss_kls = []
    all_miss_cos = []

    for feat in all_results["features"]:
        for fp in feat["false_positives"]:
            if "kl_divergence" in fp:
                all_fp_kls.append(fp["kl_divergence"])
                all_fp_sae.append(fp["sae_act"])
        for miss in feat["low_norm_misses"]:
            if "kl_divergence" in miss:
                all_miss_kls.append(miss["kl_divergence"])
                all_miss_cos.append(miss["cos"])

    if all_fp_kls:
        print(f"\n  Type B (high-norm false positives): {len(all_fp_kls)} tokens")
        print(f"    Mean SAE activation: {np.mean(all_fp_sae):.2f}")
        print(f"    Mean KL divergence: {np.mean(all_fp_kls):.4f}")
        print(f"    (High SAE act but low actual impact → SAE fooled by magnitude)")

    if all_miss_kls:
        print(f"\n  Type C (low-norm misses): {len(all_miss_kls)} tokens")
        print(f"    Mean cosine similarity: {np.mean(all_miss_cos):.3f}")
        print(f"    Mean KL divergence: {np.mean(all_miss_kls):.4f}")
        print(f"    (High cosine but low SAE act → SAE missed due to low magnitude)")

    if all_fp_kls and all_miss_kls:
        print(f"\n  If RNH is right: Type C (cosine-detected) should have HIGHER KL than Type B (SAE-detected)")
        print(f"    Type B mean KL: {np.mean(all_fp_kls):.4f}")
        print(f"    Type C mean KL: {np.mean(all_miss_kls):.4f}")
        if np.mean(all_miss_kls) > np.mean(all_fp_kls):
            print(f"    → CONFIRMED: cosine-detected features have more causal impact")
        else:
            print(f"    → NOT confirmed: SAE-detected features have more causal impact")

    # Save
    output_path = "experiments/exp2b_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
