"""
Experiment 2: Magnitude-Confounded Feature Pairs
=================================================
Core test of the Relative Norm Hypothesis.

The question: When two activations have the same cosine similarity to an SAE
feature direction (same "presence" under RNH) but different norms, does the
model's downstream behavior actually differ?

Method:
  1. Collect residual stream activations at layer 18 across many tokens
  2. For each SAE feature that fires, compute:
     - cos(x, f_dec): cosine similarity to the feature direction (RNH measure)
     - SAE_activation: the SAE's feature activation (linear rep hypothesis measure)
     - ||x||: activation norm
  3. Within cosine-similarity bins, check whether activation norm predicts:
     (a) SAE feature activation (yes, by construction)
     (b) Downstream model behavior (logit change when feature is ablated)
  4. If norm does NOT predict downstream behavior within a cosine bin,
     then magnitude is noise and RNH is supported.

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
NUM_FEATURES_TO_ANALYZE = 50  # top features by activation frequency
NUM_PROMPTS = 500
MAX_SEQ_LEN = 256

# Prompts: diverse text for collecting activations
SEED_PROMPTS = [
    "The theory of general relativity describes",
    "In Python, you can use list comprehensions to",
    "The French Revolution began in 1789 when",
    "Machine learning models are trained by",
    "Once upon a time, there was a small village",
    "The mitochondria is the powerhouse of",
    "According to quantum mechanics, particles can",
    "def quicksort(arr):\n    if len(arr) <= 1:",
    "The stock market crashed in 2008 because",
    "In a clinical trial, researchers found that",
    "The quick brown fox jumps over the lazy",
    "import numpy as np\nimport torch\n",
    "The United Nations was established in",
    "Water boils at 100 degrees Celsius at",
    "Shakespeare wrote Romeo and Juliet in",
    "The neural network architecture consists of",
    "Climate change is primarily caused by",
    "In mathematics, a prime number is",
    "The human genome contains approximately",
    "Photosynthesis converts sunlight into",
]


def collect_activations_batch(model, tokenizer, prompts, layer_idx, batch_size=8):
    """Collect residual stream activations at layer_idx for all prompts."""
    all_activations = []
    all_input_ids = []
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
        # acts shape: (batch, seq_len, d_model)

        attention_mask = inputs["attention_mask"]
        for b in range(acts.shape[0]):
            mask = attention_mask[b].bool()
            valid_acts = acts[b][mask]  # (valid_seq_len, d_model)
            all_activations.append(valid_acts)
            all_input_ids.append(inputs["input_ids"][b][mask])

    return all_activations, all_input_ids


def filter_attention_sinks(activations, threshold_multiplier=10.0):
    """Filter out attention sink tokens (outlier norms) per adamkarvonen's guidance."""
    norms = activations.norm(dim=-1)
    median_norm = norms.median()
    mask = norms < (median_norm * threshold_multiplier)
    filtered = activations[mask]
    n_filtered = (~mask).sum().item()
    return filtered, n_filtered


def analyze_feature_magnitude_confound(
    sae, activations, feature_idx, feature_dir
):
    """
    For a single feature, analyze whether activation norm confounds
    the SAE's feature activation beyond what cosine similarity captures.

    Returns dict with correlation/regression statistics.
    """
    # Compute cosine similarity to feature direction for each activation
    # feature_dir: (d_model,), activations: (N, d_model)
    cos_sims = torch.nn.functional.cosine_similarity(
        activations, feature_dir.unsqueeze(0), dim=-1
    )  # (N,)

    # Compute activation norms
    norms = activations.norm(dim=-1)  # (N,)

    # Compute SAE feature activations
    sae_acts = sae.encode(activations)[:, feature_idx]  # (N,)

    # Inner product with feature direction (what the linear rep hypothesis uses)
    inner_prods = (activations @ feature_dir).squeeze()  # (N,)

    # Only analyze tokens where the feature is active (SAE activation > 0)
    active_mask = sae_acts > 0
    n_active = active_mask.sum().item()

    if n_active < 20:
        return None  # Not enough data

    cos_active = cos_sims[active_mask].float().detach().cpu().numpy()
    norms_active = norms[active_mask].float().detach().cpu().numpy()
    sae_active = sae_acts[active_mask].float().detach().cpu().numpy()
    inner_active = inner_prods[active_mask].float().detach().cpu().numpy()

    # Correlation: cosine vs SAE activation
    corr_cos_sae = np.corrcoef(cos_active, sae_active)[0, 1]

    # Correlation: norm vs SAE activation (among active tokens)
    corr_norm_sae = np.corrcoef(norms_active, sae_active)[0, 1]

    # Correlation: inner product vs SAE activation
    corr_inner_sae = np.corrcoef(inner_active, sae_active)[0, 1]

    # Correlation: cosine vs norm (are they independent?)
    corr_cos_norm = np.corrcoef(cos_active, norms_active)[0, 1]

    # Key statistic: within cosine-similarity bins, how much does norm vary?
    # And does that norm variation correlate with SAE activation?
    cos_bins = np.linspace(cos_active.min(), cos_active.max(), 6)
    within_bin_norm_sae_corrs = []

    for j in range(len(cos_bins) - 1):
        bin_mask = (cos_active >= cos_bins[j]) & (cos_active < cos_bins[j + 1])
        if bin_mask.sum() < 10:
            continue
        bin_norms = norms_active[bin_mask]
        bin_sae = sae_active[bin_mask]
        if bin_norms.std() > 0 and bin_sae.std() > 0:
            c = np.corrcoef(bin_norms, bin_sae)[0, 1]
            within_bin_norm_sae_corrs.append(c)

    # Norm variation statistics
    norm_cv = norms_active.std() / norms_active.mean()  # coefficient of variation

    return {
        "feature_idx": int(feature_idx),
        "n_active": n_active,
        "n_total": len(cos_sims),
        "corr_cos_sae": float(corr_cos_sae),
        "corr_norm_sae": float(corr_norm_sae),
        "corr_inner_sae": float(corr_inner_sae),
        "corr_cos_norm": float(corr_cos_norm),
        "within_bin_norm_sae_corrs": [float(c) for c in within_bin_norm_sae_corrs],
        "within_bin_norm_sae_corr_mean": (
            float(np.mean(within_bin_norm_sae_corrs))
            if within_bin_norm_sae_corrs
            else None
        ),
        "norm_cv": float(norm_cv),
        "cos_active_mean": float(cos_active.mean()),
        "cos_active_std": float(cos_active.std()),
        "norm_active_mean": float(norms_active.mean()),
        "norm_active_std": float(norms_active.std()),
        "sae_active_mean": float(sae_active.mean()),
        "sae_active_std": float(sae_active.std()),
    }


def ablation_test(
    model, tokenizer, sae, activations, input_ids_list, feature_idx, feature_dir,
    layer_idx, n_samples=50
):
    """
    For tokens where this feature is active, ablate the feature from the residual
    stream and measure the downstream effect. Then check whether cosine similarity
    or SAE activation (which includes norm) better predicts the effect size.

    We ablate by removing the feature's projection: x' = x - (x @ f) * f
    """
    # Flatten all activations and find active tokens
    all_acts = activations  # (N, d_model)
    sae_acts = sae.encode(all_acts)[:, feature_idx]
    active_mask = sae_acts > 0
    active_indices = torch.where(active_mask)[0]

    if len(active_indices) < 20:
        return None

    # Sample indices to test
    sample_size = min(n_samples, len(active_indices))
    perm = torch.randperm(len(active_indices))[:sample_size]
    sample_indices = active_indices[perm]

    cos_sims = []
    norms = []
    sae_activations = []
    logit_diffs = []  # KL divergence between original and ablated

    submodule = model_utils.get_submodule(model, layer_idx)

    for idx in sample_indices:
        x = all_acts[idx]  # (d_model,)
        cos_sim = torch.nn.functional.cosine_similarity(
            x.unsqueeze(0), feature_dir.unsqueeze(0)
        ).item()
        norm = x.norm().item()
        sae_act = sae_acts[idx].item()

        # Compute original logits for this activation
        # We'll use the model's layers after layer_idx
        # Feed x through remaining layers + final norm + lm_head
        x_input = x.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
        x_ablated = x - (x @ feature_dir) * feature_dir
        x_ablated = x_ablated.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)

        # Forward through remaining layers
        orig_hidden = x_input
        ablated_hidden = x_ablated

        with torch.no_grad():
            for l in range(layer_idx + 1, len(model.model.layers)):
                layer_module = model.model.layers[l]
                # Need to create a fake position_ids and pass through
                # This is tricky with transformer layers, let's use the hook approach instead
                pass

        # Simpler approach: measure the cosine distance between original and ablated
        # activations after RMSNorm (what the next layer actually sees)
        rmsnorm = model.model.layers[layer_idx + 1].input_layernorm
        with torch.no_grad():
            orig_normed = rmsnorm(x_input)
            ablated_normed = rmsnorm(x_ablated)

        post_norm_cos = torch.nn.functional.cosine_similarity(
            orig_normed.flatten().unsqueeze(0),
            ablated_normed.flatten().unsqueeze(0),
        ).item()

        # The "effect" is 1 - cosine_similarity (how much the ablation changed things)
        effect = 1.0 - post_norm_cos

        cos_sims.append(cos_sim)
        norms.append(norm)
        sae_activations.append(sae_act)
        logit_diffs.append(effect)

    cos_sims = np.array(cos_sims)
    norms = np.array(norms)
    sae_activations = np.array(sae_activations)
    effects = np.array(logit_diffs)

    if effects.std() == 0:
        return None

    # Key question: what predicts the ablation effect?
    corr_cos_effect = np.corrcoef(cos_sims, effects)[0, 1]
    corr_norm_effect = np.corrcoef(norms, effects)[0, 1]
    corr_sae_effect = np.corrcoef(sae_activations, effects)[0, 1]
    corr_inner_effect = np.corrcoef(cos_sims * norms, effects)[0, 1]

    return {
        "feature_idx": int(feature_idx),
        "n_samples": sample_size,
        "corr_cos_effect": float(corr_cos_effect),
        "corr_norm_effect": float(corr_norm_effect),
        "corr_sae_effect": float(corr_sae_effect),
        "corr_inner_effect": float(corr_inner_effect),
        "effect_mean": float(effects.mean()),
        "effect_std": float(effects.std()),
        "cos_range": [float(cos_sims.min()), float(cos_sims.max())],
        "norm_range": [float(norms.min()), float(norms.max())],
    }


def main():
    print("Experiment 2: Magnitude-Confounded Feature Pairs")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"Device: {DEVICE}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load SAE
    print("\nLoading SAE...")
    sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
        repo_id=SAE_REPO,
        filename=SAE_FILENAME,
        model_name=MODEL_NAME,
        device=DEVICE,
        dtype=DTYPE,
    )
    print(f"SAE loaded: {sae.W_dec.shape[0]} features, d_model={sae.W_dec.shape[1]}")

    # Generate prompts: use longer, more diverse text
    # Extend seed prompts into longer paragraphs for more tokens per prompt
    extended_prompts = []
    extensions = [
        " and this has significant implications for our understanding of the world. "
        "Researchers have long debated the fundamental nature of these phenomena, "
        "and recent advances in technology have enabled new approaches to studying them. "
        "The results suggest that previous assumptions may need to be revised.",

        " which is a common pattern in many programming languages and frameworks. "
        "The implementation details can vary significantly depending on the specific "
        "requirements and constraints of the project. Consider the following example "
        "that demonstrates the key concepts involved in this approach.",

        ". This historical event had far-reaching consequences that shaped the modern "
        "world in ways that are still being understood today. Scholars continue to "
        "analyze the complex interplay of economic, social, and political factors "
        "that contributed to these developments across multiple continents.",

        " through a process that involves multiple stages of computation and "
        "optimization. The mathematical foundations of these methods draw from "
        "linear algebra, probability theory, and calculus. Understanding these "
        "foundations is essential for developing new algorithms and techniques.",
    ]
    for seed in SEED_PROMPTS:
        for ext in extensions:
            extended_prompts.append(seed + ext)
    prompts = (extended_prompts * (NUM_PROMPTS // len(extended_prompts) + 1))[:NUM_PROMPTS]

    # Collect activations
    print(f"\nCollecting activations from {len(prompts)} prompts...")
    t0 = time.time()
    act_list, id_list = collect_activations_batch(model, tokenizer, prompts, LAYER)
    all_activations = torch.cat(act_list, dim=0)  # (total_tokens, d_model)
    print(f"Collected {all_activations.shape[0]} token activations in {time.time() - t0:.1f}s")

    # Filter attention sinks
    all_activations, n_filtered = filter_attention_sinks(all_activations)
    print(f"Filtered {n_filtered} attention sink tokens, {all_activations.shape[0]} remaining")

    # Compute SAE activations for all tokens to find most-active features
    print("\nComputing SAE activations...")
    t0 = time.time()
    batch_size = 512
    all_sae_acts = []
    for i in range(0, len(all_activations), batch_size):
        batch = all_activations[i : i + batch_size]
        encoded = sae.encode(batch)
        all_sae_acts.append(encoded)
    all_sae_acts = torch.cat(all_sae_acts, dim=0)  # (N, d_sae)
    print(f"SAE encoding done in {time.time() - t0:.1f}s")

    # Find most frequently active features
    feature_freq = (all_sae_acts > 0).float().mean(dim=0)  # (d_sae,)
    top_features = feature_freq.topk(NUM_FEATURES_TO_ANALYZE).indices
    print(f"\nTop {NUM_FEATURES_TO_ANALYZE} features by activation frequency:")
    print(f"  Frequency range: {feature_freq[top_features[-1]]:.4f} to {feature_freq[top_features[0]]:.4f}")

    # Part A: Analyze magnitude confound statistics
    print("\n" + "=" * 60)
    print("  PART A: Magnitude Confound Analysis")
    print("=" * 60)

    confound_results = []
    for i, feat_idx in enumerate(top_features):
        feat_idx = feat_idx.item()
        feature_dir = sae.W_dec[feat_idx]  # (d_model,) unit vector

        result = analyze_feature_magnitude_confound(
            sae, all_activations, feat_idx, feature_dir
        )
        if result is not None:
            confound_results.append(result)

            if i < 10:  # Print first 10
                print(
                    f"  Feature {feat_idx:>5d} | "
                    f"n_active={result['n_active']:>5d} | "
                    f"corr(cos,SAE)={result['corr_cos_sae']:>6.3f} | "
                    f"corr(norm,SAE)={result['corr_norm_sae']:>6.3f} | "
                    f"corr(inner,SAE)={result['corr_inner_sae']:>6.3f} | "
                    f"within-bin corr(norm,SAE)={result['within_bin_norm_sae_corr_mean'] or 'N/A':>6}"
                )

    # Part A summary
    if confound_results:
        cos_sae_corrs = [r["corr_cos_sae"] for r in confound_results]
        norm_sae_corrs = [r["corr_norm_sae"] for r in confound_results]
        inner_sae_corrs = [r["corr_inner_sae"] for r in confound_results]
        within_corrs = [
            r["within_bin_norm_sae_corr_mean"]
            for r in confound_results
            if r["within_bin_norm_sae_corr_mean"] is not None
        ]

        print(f"\n  Summary over {len(confound_results)} features:")
        print(f"    corr(cos, SAE_act):   mean={np.mean(cos_sae_corrs):.4f}, std={np.std(cos_sae_corrs):.4f}")
        print(f"    corr(norm, SAE_act):  mean={np.mean(norm_sae_corrs):.4f}, std={np.std(norm_sae_corrs):.4f}")
        print(f"    corr(inner, SAE_act): mean={np.mean(inner_sae_corrs):.4f}, std={np.std(inner_sae_corrs):.4f}")
        if within_corrs:
            print(f"    within-bin corr(norm, SAE_act): mean={np.mean(within_corrs):.4f}, std={np.std(within_corrs):.4f}")
        print()
        print("    If within-bin corr(norm, SAE) is high, the SAE is encoding magnitude")
        print("    as additional signal beyond cosine similarity (supporting RNH claim).")

    # Part B: Ablation test — does norm predict downstream effect?
    print("\n" + "=" * 60)
    print("  PART B: Ablation Test — Does Norm Predict Downstream Effect?")
    print("=" * 60)

    ablation_results = []
    for i, feat_idx in enumerate(top_features[:20]):  # Test top 20 features
        feat_idx = feat_idx.item()
        feature_dir = sae.W_dec[feat_idx]

        result = ablation_test(
            model, tokenizer, sae, all_activations, id_list,
            feat_idx, feature_dir, LAYER, n_samples=100
        )
        if result is not None:
            ablation_results.append(result)
            print(
                f"  Feature {feat_idx:>5d} | "
                f"corr(cos,effect)={result['corr_cos_effect']:>6.3f} | "
                f"corr(norm,effect)={result['corr_norm_effect']:>6.3f} | "
                f"corr(SAE,effect)={result['corr_sae_effect']:>6.3f} | "
                f"corr(inner,effect)={result['corr_inner_effect']:>6.3f} | "
                f"effect_mean={result['effect_mean']:.4e}"
            )

    # Part B summary
    if ablation_results:
        cos_eff = [r["corr_cos_effect"] for r in ablation_results]
        norm_eff = [r["corr_norm_effect"] for r in ablation_results]
        sae_eff = [r["corr_sae_effect"] for r in ablation_results]
        inner_eff = [r["corr_inner_effect"] for r in ablation_results]

        print(f"\n  Summary over {len(ablation_results)} features:")
        print(f"    corr(cos, effect):   mean={np.mean(cos_eff):.4f}, std={np.std(cos_eff):.4f}")
        print(f"    corr(norm, effect):  mean={np.mean(norm_eff):.4f}, std={np.std(norm_eff):.4f}")
        print(f"    corr(SAE, effect):   mean={np.mean(sae_eff):.4f}, std={np.std(sae_eff):.4f}")
        print(f"    corr(inner, effect): mean={np.mean(inner_eff):.4f}, std={np.std(inner_eff):.4f}")
        print()
        print("    RNH prediction: corr(cos, effect) > corr(norm, effect)")
        print("    i.e. cosine similarity better predicts ablation impact than norm does.")
        print()
        cos_wins = sum(
            1 for c, n in zip(cos_eff, norm_eff) if abs(c) > abs(n)
        )
        print(f"    |corr(cos,effect)| > |corr(norm,effect)| for {cos_wins}/{len(ablation_results)} features")

        inner_vs_cos = sum(
            1 for c, ip in zip(cos_eff, inner_eff) if abs(c) > abs(ip)
        )
        print(f"    |corr(cos,effect)| > |corr(inner,effect)| for {inner_vs_cos}/{len(ablation_results)} features")

    # Save all results
    all_results = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "sae_repo": SAE_REPO,
        "n_tokens": int(all_activations.shape[0]),
        "n_features_analyzed": len(confound_results),
        "confound_analysis": confound_results,
        "ablation_tests": ablation_results,
    }

    output_path = "experiments/exp2_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
