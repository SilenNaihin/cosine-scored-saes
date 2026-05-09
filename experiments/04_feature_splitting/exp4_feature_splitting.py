"""
Experiment 4: SAE Feature Splitting by Magnitude
=================================================
Does the SAE waste capacity by learning near-duplicate feature directions?

If the RNH is correct, magnitude is noise — but the SAE (using inner product)
might learn multiple features for the same *direction* at different magnitude
thresholds. This would be direct evidence of wasted capacity due to the
linear representation hypothesis.

Part 1 — Geometry (weights only, fast):
  1. Compute pairwise cosine similarity of all decoder directions
  2. Find near-duplicate pairs (cos > 0.95)
  3. For these pairs, compare encoder weights and biases
     - Similar encoder direction but different bias → magnitude threshold bins
     - This is the SAE's way of implementing "same concept, different norm range"

Part 2 — Activation patterns (needs model):
  For near-duplicate pairs:
  1. Collect activations, run SAE encoding
  2. Check co-occurrence: do both fire on the same token?
  3. When only one fires, is it the input norm that discriminates?
  4. When both fire, do they activate on different norm ranges?

Model: Qwen3-8B
SAE: adamkarvonen/qwen3-8b-saes (layer 18, 65k features, BatchTopK k=80)
"""

import json
import time
import torch
import numpy as np
from collections import defaultdict
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

# Thresholds for "near-duplicate"
COS_THRESHOLD_HIGH = 0.95  # very similar direction
COS_THRESHOLD_MED = 0.90   # moderately similar

# Prompts for activation collection
PROMPTS = [
    "The theory of general relativity describes how gravity works as a curvature of spacetime caused by mass and energy. Einstein published this theory in 1915, fundamentally changing our understanding of the universe and leading to predictions about black holes and gravitational waves.",
    "In Python, you can use list comprehensions to create new lists based on existing ones. For example, squares = [x**2 for x in range(10)] creates a list of squares. This is more concise than a traditional for loop and is considered more Pythonic.",
    "The French Revolution began in 1789 when the people of France rose up against the monarchy. The storming of the Bastille on July 14th became a symbol of the revolution and is still celebrated as a national holiday in France today.",
    "Machine learning models are trained by optimizing a loss function over a dataset. The model learns to map inputs to outputs by adjusting its parameters through gradient descent. Regularization techniques like dropout and weight decay help prevent overfitting to the training data.",
    "Once upon a time, there was a small village nestled in a valley between two great mountains. The villagers lived simple lives, farming the rich soil and trading goods at the weekly market in the town square. One day a stranger arrived carrying a mysterious wooden box.",
    "The mitochondria is the powerhouse of the cell, producing adenosine triphosphate (ATP) through oxidative phosphorylation. This process involves the electron transport chain in the inner mitochondrial membrane and is essential for aerobic life.",
    "According to quantum mechanics, particles can exist in superposition states until measured. The famous Schrodinger's cat thought experiment illustrates this principle, where a cat in a box is both alive and dead simultaneously until the box is opened.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "The stock market crashed in 2008 because of a housing bubble fueled by subprime mortgages and complex financial derivatives. Banks had taken on excessive risk, and when housing prices fell, the entire financial system nearly collapsed, leading to the Great Recession.",
    "In a clinical trial, researchers found that the new drug reduced symptoms by 40% compared to placebo. The study was double-blind and randomized, with 500 participants across 12 medical centers over a period of two years.",
    "Shakespeare wrote Romeo and Juliet in the late 16th century. The play tells the tragic story of two young lovers from feuding families in Verona, Italy. It remains one of the most performed plays in the world and has been adapted into countless films and operas.",
    "The neural network architecture consists of multiple layers of interconnected nodes. Each connection has a weight that is learned during training. Activation functions like ReLU introduce nonlinearity, enabling the network to learn complex patterns in the data.",
    "Climate change is primarily caused by the burning of fossil fuels, which releases carbon dioxide and other greenhouse gases into the atmosphere. These gases trap heat, leading to a gradual increase in global temperatures and more extreme weather events worldwide.",
    "In mathematics, a prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The fundamental theorem of arithmetic states that every integer greater than 1 can be uniquely factored into primes.",
    "The human genome contains approximately 3 billion base pairs of DNA organized into 23 pairs of chromosomes. Only about 1.5% of the genome codes for proteins, while the rest includes regulatory sequences, introns, and other non-coding elements.",
    "Photosynthesis converts sunlight into chemical energy in plants. The process occurs in chloroplasts, where light-dependent reactions in the thylakoid membranes produce ATP and NADPH, which then drive the Calvin cycle to fix carbon dioxide into glucose.",
    "The United States Constitution was ratified in 1788 after extensive debate between Federalists and Anti-Federalists. The Bill of Rights, comprising the first ten amendments, was added in 1791 to protect individual liberties from government overreach.",
    "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, nhead):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, nhead)\n        self.norm1 = nn.LayerNorm(d_model)\n        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))",
    "Water exists in three states: solid (ice), liquid (water), and gas (steam). The transitions between these states occur at specific temperatures and pressures, governed by the phase diagram of water. At standard pressure, water freezes at 0°C and boils at 100°C.",
    "The Renaissance was a period of cultural rebirth in Europe, spanning roughly from the 14th to the 17th century. It saw tremendous advances in art, science, literature, and philosophy, with figures like Leonardo da Vinci, Michelangelo, and Galileo leading the way.",
    "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller subproblems. Each recursive call should move toward a base case that stops the recursion, preventing infinite loops and stack overflow errors.",
    "The Amazon rainforest covers approximately 5.5 million square kilometers and produces about 20% of the world's oxygen. It is home to an estimated 10% of all known species on Earth, including jaguars, river dolphins, and thousands of bird species.",
    "Quantum computing uses qubits instead of classical bits. Unlike classical bits which are either 0 or 1, qubits can exist in superposition, allowing quantum computers to explore many solutions simultaneously through quantum parallelism and entanglement.",
    "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing, transportation, and society. Steam power, mechanized textile production, and iron smelting drove unprecedented economic growth and urbanization across Europe.",
]


# ============================================================
#  PART 1: GEOMETRIC ANALYSIS (weights only)
# ============================================================

def find_near_duplicates(W_dec, threshold=COS_THRESHOLD_HIGH, batch_size=4096):
    """
    Find pairs of decoder features with cosine similarity above threshold.
    W_dec is (n_features, d_model), rows are unit-normalized.
    Since rows are unit-norm, cosine similarity = dot product.
    """
    n_features = W_dec.shape[0]
    pairs = []

    print(f"  Computing pairwise cosine similarity for {n_features} features...")
    print(f"  (batched, {batch_size} rows at a time)")

    for i in range(0, n_features, batch_size):
        end_i = min(i + batch_size, n_features)
        chunk = W_dec[i:end_i]  # (chunk_size, d_model)

        # Only compare with features that come after (avoid duplicates)
        for j in range(i, n_features, batch_size):
            end_j = min(j + batch_size, n_features)
            other = W_dec[j:end_j]  # (other_size, d_model)

            # Cosine similarity = dot product (since unit-normed)
            sims = chunk @ other.T  # (chunk_size, other_size)

            # Find pairs above threshold
            if i == j:
                # Same chunk: only upper triangle (no self-pairs)
                mask = torch.triu(sims > threshold, diagonal=1)
            else:
                mask = sims > threshold

            indices = torch.nonzero(mask)
            for idx in indices:
                fi = i + idx[0].item()
                fj = j + idx[1].item()
                if fi != fj:
                    sim_val = sims[idx[0], idx[1]].item()
                    pairs.append((fi, fj, sim_val))

        if (i // batch_size) % 4 == 0:
            print(f"    processed {end_i}/{n_features} rows, {len(pairs)} pairs found so far")

    return pairs


def analyze_encoder_for_pairs(sae, pairs):
    """
    For near-duplicate decoder pairs, compare their encoder weights and biases.
    If encoder directions are similar but biases differ, the SAE is implementing
    magnitude threshold bins for the same concept.
    """
    W_enc = sae.W_enc  # (n_features, d_model)
    b_enc = sae.b_enc  # (n_features,)

    results = []
    for fi, fj, dec_cos in pairs:
        enc_i = W_enc[fi]
        enc_j = W_enc[fj]

        # Encoder direction similarity
        enc_cos = torch.nn.functional.cosine_similarity(
            enc_i.unsqueeze(0), enc_j.unsqueeze(0)
        ).item()

        # Encoder magnitude ratio
        enc_norm_i = enc_i.norm().item()
        enc_norm_j = enc_j.norm().item()
        enc_norm_ratio = max(enc_norm_i, enc_norm_j) / (min(enc_norm_i, enc_norm_j) + 1e-8)

        # Bias difference
        bias_i = b_enc[fi].item()
        bias_j = b_enc[fj].item()
        bias_diff = abs(bias_i - bias_j)
        bias_mean = (abs(bias_i) + abs(bias_j)) / 2

        results.append({
            "feature_i": fi,
            "feature_j": fj,
            "dec_cosine": dec_cos,
            "enc_cosine": enc_cos,
            "enc_norm_i": enc_norm_i,
            "enc_norm_j": enc_norm_j,
            "enc_norm_ratio": enc_norm_ratio,
            "bias_i": bias_i,
            "bias_j": bias_j,
            "bias_diff": bias_diff,
            "bias_relative_diff": bias_diff / (bias_mean + 1e-8),
        })

    return results


def part1_geometry(sae):
    """Run the full geometric analysis on SAE weights."""
    print("\n" + "=" * 70)
    print("  PART 1: GEOMETRIC ANALYSIS (decoder weight similarity)")
    print("=" * 70)

    W_dec = sae.W_dec  # (n_features, d_model)
    n_features, d_model = W_dec.shape
    print(f"  W_dec: {n_features} features x {d_model} dims")

    # Verify unit normalization
    norms = W_dec.norm(dim=-1)
    print(f"  Decoder norm range: [{norms.min():.6f}, {norms.max():.6f}] (should be ~1.0)")

    # Find near-duplicates at multiple thresholds
    pair_counts = {}
    pairs_high = []
    pairs_med = []

    for threshold in [0.95, 0.90, 0.80, 0.70]:
        print(f"\n  --- Finding pairs with cos > {threshold} ---")
        pairs = find_near_duplicates(W_dec, threshold=threshold)
        pair_counts[threshold] = len(pairs)
        print(f"  Found {len(pairs)} pairs with cos > {threshold}")
        if threshold == COS_THRESHOLD_HIGH:
            pairs_high = pairs
        if threshold == COS_THRESHOLD_MED:
            pairs_med = pairs
        # If we found a lot at this threshold, skip lower ones
        if len(pairs) > 10000:
            print(f"  (skipping lower thresholds — enough data)")
            break

    # Distribution of cosine similarities (sample-based)
    print("\n  --- Cosine similarity distribution (sampled) ---")
    n_sample = min(5000, n_features)
    sample_idx = torch.randperm(n_features)[:n_sample]
    sample = W_dec[sample_idx]
    sample_sims = (sample @ sample.T).detach().cpu().float().numpy()
    # Upper triangle only (exclude diagonal)
    upper = sample_sims[np.triu_indices(n_sample, k=1)]
    percentiles = np.percentile(upper, [50, 90, 95, 99, 99.5, 99.9])
    print(f"  Sampled {n_sample} features, {len(upper)} pairs")
    print(f"  Percentiles: 50th={percentiles[0]:.4f}, 90th={percentiles[1]:.4f}, "
          f"95th={percentiles[2]:.4f}, 99th={percentiles[3]:.4f}, "
          f"99.5th={percentiles[4]:.4f}, 99.9th={percentiles[5]:.4f}")
    print(f"  Mean={upper.mean():.4f}, Std={upper.std():.4f}, Max={upper.max():.4f}")

    # Analyze encoder weights for high-similarity pairs
    if pairs_high:
        print(f"\n  --- Encoder analysis for {len(pairs_high)} high-similarity pairs ---")
        enc_analysis = analyze_encoder_for_pairs(sae, pairs_high)

        enc_cosines = [r["enc_cosine"] for r in enc_analysis]
        bias_diffs = [r["bias_diff"] for r in enc_analysis]
        bias_rel = [r["bias_relative_diff"] for r in enc_analysis]
        enc_ratios = [r["enc_norm_ratio"] for r in enc_analysis]

        print(f"  Encoder direction cosine: mean={np.mean(enc_cosines):.4f}, "
              f"std={np.std(enc_cosines):.4f}")
        print(f"  Encoder norm ratio: mean={np.mean(enc_ratios):.4f}, "
              f"std={np.std(enc_ratios):.4f}")
        print(f"  Bias absolute diff: mean={np.mean(bias_diffs):.4f}, "
              f"std={np.std(bias_diffs):.4f}")
        print(f"  Bias relative diff: mean={np.mean(bias_rel):.4f}, "
              f"std={np.std(bias_rel):.4f}")

        # Categorize: are these "same encoder, different bias" or "different encoder"?
        same_enc = sum(1 for c in enc_cosines if c > 0.9)
        diff_enc = sum(1 for c in enc_cosines if c < 0.5)
        print(f"\n  Classification of {len(pairs_high)} high-cosine decoder pairs:")
        print(f"    Encoder also similar (cos > 0.9): {same_enc} "
              f"({100*same_enc/len(pairs_high):.1f}%)")
        print(f"    Encoder quite different (cos < 0.5): {diff_enc} "
              f"({100*diff_enc/len(pairs_high):.1f}%)")
        print(f"    In between: {len(pairs_high) - same_enc - diff_enc}")

        # For same-encoder pairs: are biases different? (magnitude binning)
        same_enc_pairs = [r for r, c in zip(enc_analysis, enc_cosines) if c > 0.9]
        if same_enc_pairs:
            se_bias_diffs = [r["bias_diff"] for r in same_enc_pairs]
            se_bias_rel = [r["bias_relative_diff"] for r in same_enc_pairs]
            print(f"\n  Among {len(same_enc_pairs)} same-encoder pairs:")
            print(f"    Bias diff: mean={np.mean(se_bias_diffs):.4f}, "
                  f"std={np.std(se_bias_diffs):.4f}")
            print(f"    Bias relative diff: mean={np.mean(se_bias_rel):.4f}")
            if np.mean(se_bias_rel) > 0.1:
                print(f"    → EVIDENCE of magnitude binning: same direction, different thresholds")
            else:
                print(f"    → Biases are similar too — true duplicates, not magnitude bins")

        # Print top 10 most similar pairs
        pairs_sorted = sorted(enc_analysis, key=lambda r: -r["dec_cosine"])
        print(f"\n  Top 10 most similar decoder pairs:")
        for r in pairs_sorted[:10]:
            print(f"    f{r['feature_i']:>5d} & f{r['feature_j']:>5d}: "
                  f"dec_cos={r['dec_cosine']:.4f}, enc_cos={r['enc_cosine']:.4f}, "
                  f"bias_i={r['bias_i']:.3f}, bias_j={r['bias_j']:.3f}, "
                  f"enc_norms={r['enc_norm_i']:.3f}/{r['enc_norm_j']:.3f}")
    else:
        enc_analysis = []

    return {
        "n_features": n_features,
        "d_model": d_model,
        "n_pairs_high": len(pairs_high),
        "n_pairs_med": len(pairs_med),
        "pair_counts_by_threshold": {str(k): v for k, v in pair_counts.items()},
        "cos_distribution": {
            "n_sampled": n_sample,
            "mean": float(upper.mean()),
            "std": float(upper.std()),
            "max": float(upper.max()),
            "percentiles": {
                "p50": float(percentiles[0]),
                "p90": float(percentiles[1]),
                "p95": float(percentiles[2]),
                "p99": float(percentiles[3]),
                "p99.5": float(percentiles[4]),
                "p99.9": float(percentiles[5]),
            },
        },
        "encoder_analysis": enc_analysis[:100],  # save top 100
        "pairs_high": [(fi, fj, float(s)) for fi, fj, s in pairs_high],
    }


# ============================================================
#  PART 2: ACTIVATION ANALYSIS (needs model)
# ============================================================

def collect_activations(model, tokenizer, prompts, layer_idx, batch_size=4):
    """Collect residual stream activations."""
    all_acts = []
    submodule = model_utils.get_submodule(model, layer_idx)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to(DEVICE)
        acts = model_utils.collect_activations(model, submodule, inputs)
        mask = inputs["attention_mask"]
        for b in range(acts.shape[0]):
            valid = acts[b][mask[b].bool()]
            all_acts.append(valid)

    return torch.cat(all_acts, dim=0)


def filter_sinks(activations, threshold_multiplier=10.0):
    """Remove attention sink tokens."""
    norms = activations.norm(dim=-1)
    mask = norms < (norms.median() * threshold_multiplier)
    return activations[mask], (~mask).sum().item()


def analyze_pair_activations(sae, activations, fi, fj):
    """
    For a near-duplicate pair (fi, fj), analyze their activation patterns.
    Are they magnitude bins of the same direction?
    """
    norms = activations.norm(dim=-1).cpu().numpy()

    # SAE activations for both features
    with torch.no_grad():
        sae_acts = sae.encode(activations)
    acts_i = sae_acts[:, fi].cpu().numpy()
    acts_j = sae_acts[:, fj].cpu().numpy()

    # Firing patterns
    fire_i = acts_i > 0
    fire_j = acts_j > 0
    fire_both = fire_i & fire_j
    fire_only_i = fire_i & ~fire_j
    fire_only_j = ~fire_i & fire_j
    fire_neither = ~fire_i & ~fire_j

    n = len(activations)
    n_both = fire_both.sum()
    n_only_i = fire_only_i.sum()
    n_only_j = fire_only_j.sum()

    result = {
        "feature_i": fi,
        "feature_j": fj,
        "n_tokens": n,
        "n_fire_i": int(fire_i.sum()),
        "n_fire_j": int(fire_j.sum()),
        "n_fire_both": int(n_both),
        "n_fire_only_i": int(n_only_i),
        "n_fire_only_j": int(n_only_j),
        "n_fire_neither": int(fire_neither.sum()),
    }

    # Jaccard similarity of firing patterns
    union = (fire_i | fire_j).sum()
    result["jaccard"] = float(n_both / union) if union > 0 else 0.0

    # Norm distributions for each firing pattern
    if n_both > 10:
        result["norm_both_mean"] = float(norms[fire_both].mean())
        result["norm_both_std"] = float(norms[fire_both].std())
    if n_only_i > 10:
        result["norm_only_i_mean"] = float(norms[fire_only_i].mean())
        result["norm_only_i_std"] = float(norms[fire_only_i].std())
    if n_only_j > 10:
        result["norm_only_j_mean"] = float(norms[fire_only_j].mean())
        result["norm_only_j_std"] = float(norms[fire_only_j].std())

    # KEY TEST: when only one fires, is the discriminator the input norm?
    if n_only_i > 5 and n_only_j > 5:
        norms_only_i = norms[fire_only_i]
        norms_only_j = norms[fire_only_j]
        result["norm_only_i_median"] = float(np.median(norms_only_i))
        result["norm_only_j_median"] = float(np.median(norms_only_j))
        result["norm_separation"] = float(
            abs(np.median(norms_only_i) - np.median(norms_only_j))
            / (np.std(np.concatenate([norms_only_i, norms_only_j])) + 1e-8)
        )
        # High norm_separation → the pair is discriminating by input norm → magnitude bins
    else:
        result["norm_separation"] = None

    # Activation magnitude correlation when both fire
    if n_both > 10:
        corr = np.corrcoef(acts_i[fire_both], acts_j[fire_both])[0, 1]
        result["act_correlation_when_both"] = float(corr)
    else:
        result["act_correlation_when_both"] = None

    return result


def part2_activations(sae, model, tokenizer, pairs_to_test):
    """Run activation analysis on near-duplicate pairs."""
    print("\n" + "=" * 70)
    print("  PART 2: ACTIVATION PATTERN ANALYSIS")
    print("=" * 70)

    if not pairs_to_test:
        print("  No near-duplicate pairs to test. Skipping Part 2.")
        return []

    # Collect activations
    print(f"  Collecting activations from {len(PROMPTS)} prompts...")
    t0 = time.time()
    activations = collect_activations(model, tokenizer, PROMPTS, LAYER)
    activations, n_filtered = filter_sinks(activations)
    print(f"  {activations.shape[0]} tokens after filtering ({n_filtered} sinks removed)")
    print(f"  Collection took {time.time() - t0:.1f}s")

    # Analyze each pair
    print(f"\n  Analyzing {len(pairs_to_test)} near-duplicate pairs...")
    results = []

    for idx, (fi, fj, dec_cos) in enumerate(pairs_to_test):
        result = analyze_pair_activations(sae, activations, fi, fj)
        result["dec_cosine"] = dec_cos
        results.append(result)

        if idx < 20 or idx % 50 == 0:
            sep = result.get("norm_separation")
            sep_str = f"{sep:.2f}" if sep is not None else "N/A"
            jac = result["jaccard"]
            print(
                f"    Pair f{fi:>5d} & f{fj:>5d}: "
                f"dec_cos={dec_cos:.3f}, "
                f"jaccard={jac:.3f}, "
                f"fire_both={result['n_fire_both']}, "
                f"only_i={result['n_fire_only_i']}, "
                f"only_j={result['n_fire_only_j']}, "
                f"norm_sep={sep_str}"
            )

    # Summary statistics
    if results:
        jaccards = [r["jaccard"] for r in results]
        norm_seps = [r["norm_separation"] for r in results if r["norm_separation"] is not None]
        act_corrs = [r["act_correlation_when_both"] for r in results
                     if r["act_correlation_when_both"] is not None]

        print(f"\n  --- Summary over {len(results)} pairs ---")
        print(f"  Jaccard overlap: mean={np.mean(jaccards):.4f}, "
              f"std={np.std(jaccards):.4f}")
        if norm_seps:
            print(f"  Norm separation (exclusive firing): mean={np.mean(norm_seps):.4f}, "
                  f"std={np.std(norm_seps):.4f}")
            high_sep = sum(1 for s in norm_seps if s > 1.0)
            print(f"  Pairs with norm_separation > 1.0 (clear magnitude binning): "
                  f"{high_sep}/{len(norm_seps)} ({100*high_sep/len(norm_seps):.1f}%)")
        if act_corrs:
            print(f"  Activation correlation (when both fire): mean={np.mean(act_corrs):.4f}, "
                  f"std={np.std(act_corrs):.4f}")

    return results


def main():
    print("Experiment 4: SAE Feature Splitting by Magnitude")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"Device: {DEVICE}")
    print(f"Thresholds: high={COS_THRESHOLD_HIGH}, med={COS_THRESHOLD_MED}")

    # Load SAE
    print("\nLoading SAE...")
    t0 = time.time()
    sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
        repo_id=SAE_REPO, filename=SAE_FILENAME,
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE,
    )
    print(f"SAE loaded in {time.time() - t0:.1f}s")
    print(f"  {sae.W_dec.shape[0]} features, d_model={sae.W_dec.shape[1]}")

    # Part 1: Geometric analysis (weights only)
    geom_results = part1_geometry(sae)

    # Part 2: Activation analysis (needs model)
    # Use whatever pairs we found, even at lower thresholds
    pairs_to_test = geom_results["pairs_high"]
    if not pairs_to_test:
        # Fall back to any pairs found at any threshold
        all_pairs = []
        for threshold_pairs in [geom_results.get("pairs_all", [])]:
            all_pairs.extend(threshold_pairs)
        if not all_pairs:
            # Use pairs_med or reconstruct from pair_counts
            pairs_to_test = geom_results.get("pairs_med", [])
        else:
            pairs_to_test = all_pairs

    if pairs_to_test:
        # Limit to reasonable number for activation testing
        pairs_to_test = pairs_to_test[:200]

        print(f"\nLoading model for Part 2 ({len(pairs_to_test)} pairs to test)...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=DTYPE, device_map=DEVICE
        )
        model.eval()
        print(f"Model loaded in {time.time() - t0:.1f}s")

        act_results = part2_activations(sae, model, tokenizer, pairs_to_test)
    else:
        print("\nNo near-duplicate decoder pairs found. Skipping Part 2.")
        act_results = []

    # Save results
    all_results = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "cos_threshold_high": COS_THRESHOLD_HIGH,
        "cos_threshold_med": COS_THRESHOLD_MED,
        "geometry": {
            "n_features": geom_results["n_features"],
            "d_model": geom_results["d_model"],
            "n_pairs_high": geom_results["n_pairs_high"],
            "n_pairs_med": geom_results["n_pairs_med"],
            "cos_distribution": geom_results["cos_distribution"],
            "encoder_analysis": geom_results["encoder_analysis"],
        },
        "activation_analysis": act_results[:100],  # cap to keep file reasonable
    }

    output_path = "experiments/exp4_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Near-duplicate decoder pairs (cos > {COS_THRESHOLD_HIGH}): {geom_results['n_pairs_high']}")
    print(f"  Near-duplicate decoder pairs (cos > {COS_THRESHOLD_MED}): {geom_results['n_pairs_med']}")

    if geom_results['n_pairs_high'] > 0:
        enc_analysis = geom_results["encoder_analysis"]
        if enc_analysis:
            same_enc = sum(1 for r in enc_analysis if r["enc_cosine"] > 0.9)
            print(f"  Of high-cos pairs: {same_enc}/{len(enc_analysis)} also have similar encoders")
            print(f"  → These are potential magnitude-binning duplicates")
    else:
        print(f"  No near-duplicate pairs found — SAE features are well-separated")
        print(f"  This would mean the SAE is NOT wasting capacity on magnitude")

    if act_results:
        norm_seps = [r["norm_separation"] for r in act_results if r["norm_separation"] is not None]
        if norm_seps:
            high_sep = sum(1 for s in norm_seps if s > 1.0)
            print(f"  Pairs with clear magnitude binning (norm_sep > 1.0): "
                  f"{high_sep}/{len(norm_seps)}")


if __name__ == "__main__":
    main()
