"""
Experiment 3: LLM Interpretation of RNH-Critical Features
==========================================================
For features where cosine and SAE disagree, use Claude (Bedrock) to:
  1. Interpret the feature from top-activating examples (by cosine similarity)
  2. Test whether cosine-detected tokens (SAE misses) match the interpretation
  3. Test whether SAE false positives (high SAE, low cosine) match the interpretation

If cosine-detected tokens are more interpretable (match the explanation better),
this provides further evidence for the RNH via a qualitative/LLM-as-judge test.

Model: Qwen3-8B with SAE at layer 18
Judge: Claude via AWS Bedrock
"""

import json
import time
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import interp_tools.saes.batch_topk_sae as batch_topk_sae
import interp_tools.model_utils as model_utils

# AWS Bedrock config — credentials from ~/.aws/credentials default profile
AWS_REGION = "us-west-2"
CLAUDE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
SAE_REPO = "adamkarvonen/qwen3-8b-saes"
SAE_FILENAME = "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/trainer_2/ae.pt"
LAYER = 18
MAX_SEQ_LEN = 256
TOP_K_EXAMPLES = 15  # top activating examples for interpretation
NUM_FEATURES = 10

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


def get_bedrock_client():
    """Create Bedrock client with credentials."""
    import boto3
    from botocore.config import Config
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        config=Config(read_timeout=120, connect_timeout=10, retries={"max_attempts": 3}),
    )


def call_claude(client, system_prompt, user_prompt, max_tokens=1024):
    """Call Claude via Bedrock."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.3,
    })

    response = client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def collect_activations_with_context(model, tokenizer, prompts, layer_idx, batch_size=4):
    """Collect activations with full token and surrounding context."""
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
            prompt_text = prompts[i + b]
            token_strs = [tokenizer.decode([t]) for t in tokens]
            for pos, (tok_str, act) in enumerate(zip(token_strs, token_acts)):
                # Get surrounding context (5 tokens before and after)
                context_start = max(0, pos - 5)
                context_end = min(len(token_strs), pos + 6)
                context_tokens = token_strs[context_start:context_end]
                highlighted_pos = pos - context_start

                # Build context string with ** around target token
                context_parts = []
                for ci, ct in enumerate(context_tokens):
                    if ci == highlighted_pos:
                        context_parts.append(f"**{ct}**")
                    else:
                        context_parts.append(ct)
                context_str = "".join(context_parts)

                all_data.append({
                    "prompt_idx": i + b,
                    "position": pos,
                    "token": tok_str,
                    "context": context_str,
                    "activation": act.detach(),
                    "prompt_text": prompt_text[:100],
                })

    return all_data


def filter_sinks(data, threshold_multiplier=10.0):
    norms = torch.stack([d["activation"] for d in data]).norm(dim=-1)
    median_norm = norms.median()
    mask = norms < (median_norm * threshold_multiplier)
    return [d for d, m in zip(data, mask) if m]


def get_top_activating_examples(data, feature_dir, metric="cosine", top_k=15):
    """Get top activating examples by cosine similarity or SAE activation."""
    activations = torch.stack([d["activation"] for d in data])

    if metric == "cosine":
        scores = torch.nn.functional.cosine_similarity(
            activations, feature_dir.unsqueeze(0), dim=-1
        ).detach().float().cpu().numpy()
    elif metric == "inner_product":
        scores = (activations @ feature_dir).detach().float().cpu().numpy()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    top_indices = np.argsort(-scores)[:top_k]
    examples = []
    for idx in top_indices:
        examples.append({
            "token": data[idx]["token"],
            "context": data[idx]["context"],
            "score": float(scores[idx]),
            "prompt_snippet": data[idx]["prompt_text"],
        })
    return examples


def format_examples_for_llm(examples, metric_name="activation"):
    """Format top-activating examples for the LLM prompt."""
    lines = []
    for i, ex in enumerate(examples):
        lines.append(
            f"{i+1}. [{metric_name}={ex['score']:.4f}] "
            f"Context: ...{ex['context']}..."
        )
    return "\n".join(lines)


def main():
    print("Experiment 3: LLM Interpretation of RNH-Critical Features")
    print("=" * 70)

    # Test Bedrock connection first
    print("Testing Bedrock connection...")
    bedrock = get_bedrock_client()
    test_response = call_claude(bedrock, "You are a helpful assistant.", "Say 'hello' in one word.", max_tokens=10)
    print(f"Bedrock OK: {test_response.strip()}")

    # Load model + SAE
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

    print("Loading SAE...")
    sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
        repo_id=SAE_REPO, filename=SAE_FILENAME,
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE,
    )

    # Collect activations
    print(f"Collecting activations from {len(PROMPTS)} prompts...")
    data = collect_activations_with_context(model, tokenizer, PROMPTS, LAYER)
    data = filter_sinks(data)
    print(f"{len(data)} tokens after filtering")

    activations = torch.stack([d["activation"] for d in data])
    with torch.no_grad():
        all_sae_acts = sae.encode(activations)

    # Get top features
    feature_freq = (all_sae_acts > 0).float().mean(dim=0)
    top_features = feature_freq.topk(NUM_FEATURES).indices.tolist()

    all_results = {"features": []}

    INTERPRET_SYSTEM = (
        "You are an expert in mechanistic interpretability. "
        "I will show you examples where a specific feature in a language model "
        "has high activation. The target token is highlighted with **double stars**. "
        "Provide a single concise explanation (1-2 sentences) of what concept or "
        "pattern this feature detects. Be specific and concrete."
    )

    JUDGE_SYSTEM = (
        "You are evaluating whether tokens match a feature description. "
        "I'll give you a feature explanation and a list of tokens with their context. "
        "For each token, respond with YES if the token clearly matches the explanation, "
        "PARTIAL if it somewhat matches, or NO if it doesn't match. "
        "Return a JSON object with keys 'judgments' (list of YES/PARTIAL/NO) and "
        "'match_rate' (fraction of YES+PARTIAL out of total)."
    )

    for feat_idx in top_features:
        print(f"\n{'='*70}")
        print(f"  Feature {feat_idx}")
        print(f"{'='*70}")

        feature_dir = sae.W_dec[feat_idx]

        # Compute scores
        cos_sims = torch.nn.functional.cosine_similarity(
            activations, feature_dir.unsqueeze(0), dim=-1
        ).detach().float().cpu().numpy()

        norms = activations.norm(dim=-1).detach().float().cpu().numpy()
        sae_feat_acts = all_sae_acts[:, feat_idx].detach().float().cpu().numpy()

        # Step 1: Get top examples by COSINE SIMILARITY and interpret
        top_cos_examples = get_top_activating_examples(
            data, feature_dir, metric="cosine", top_k=TOP_K_EXAMPLES
        )

        print("\n  Top tokens by cosine similarity:")
        for ex in top_cos_examples[:5]:
            print(f"    cos={ex['score']:.4f} | {ex['context']}")

        cos_examples_text = format_examples_for_llm(top_cos_examples, "cosine_sim")

        # Generate interpretation from cosine-top examples
        interpret_prompt = (
            f"Here are the top activating examples for this feature "
            f"(ranked by cosine similarity to the feature direction):\n\n"
            f"{cos_examples_text}\n\n"
            f"What concept or pattern does this feature detect?"
        )

        print("\n  Generating interpretation from cosine-top examples...")
        cos_interpretation = call_claude(bedrock, INTERPRET_SYSTEM, interpret_prompt)
        print(f"  Interpretation (cosine): {cos_interpretation.strip()}")

        # Also generate interpretation from SAE-top examples for comparison
        top_sae_examples = []
        sae_order = np.argsort(-sae_feat_acts)[:TOP_K_EXAMPLES]
        for idx in sae_order:
            top_sae_examples.append({
                "token": data[idx]["token"],
                "context": data[idx]["context"],
                "score": float(sae_feat_acts[idx]),
                "prompt_snippet": data[idx]["prompt_text"],
            })

        sae_examples_text = format_examples_for_llm(top_sae_examples, "sae_activation")

        interpret_prompt_sae = (
            f"Here are the top activating examples for this feature "
            f"(ranked by SAE feature activation):\n\n"
            f"{sae_examples_text}\n\n"
            f"What concept or pattern does this feature detect?"
        )

        print("  Generating interpretation from SAE-top examples...")
        sae_interpretation = call_claude(bedrock, INTERPRET_SYSTEM, interpret_prompt_sae)
        print(f"  Interpretation (SAE): {sae_interpretation.strip()}")

        # Step 2: Test the cosine interpretation against different token groups

        # Group A: HIGH cosine, LOW SAE (tokens cosine detects but SAE misses)
        cosine_detected = []
        for idx in np.argsort(-cos_sims):
            if sae_feat_acts[idx] == 0 and cos_sims[idx] > np.median(cos_sims):
                cosine_detected.append({
                    "token": data[idx]["token"],
                    "context": data[idx]["context"],
                    "cos": float(cos_sims[idx]),
                    "sae_act": float(sae_feat_acts[idx]),
                })
                if len(cosine_detected) >= 15:
                    break

        # Group B: HIGH SAE, LOW cosine (tokens SAE detects but cosine says are weak)
        sae_detected = []
        active_mask = sae_feat_acts > 0
        active_indices = np.where(active_mask)[0]
        if len(active_indices) > 0:
            # Among active tokens, find those with lowest cosine
            active_cos = cos_sims[active_indices]
            low_cos_order = np.argsort(active_cos)
            for oi in low_cos_order:
                idx = active_indices[oi]
                sae_detected.append({
                    "token": data[idx]["token"],
                    "context": data[idx]["context"],
                    "cos": float(cos_sims[idx]),
                    "sae_act": float(sae_feat_acts[idx]),
                })
                if len(sae_detected) >= 15:
                    break

        feature_result = {
            "feature_idx": feat_idx,
            "cos_interpretation": cos_interpretation.strip(),
            "sae_interpretation": sae_interpretation.strip(),
        }

        # Judge: do cosine-detected tokens match the cosine interpretation?
        if cosine_detected:
            cos_detected_text = "\n".join(
                f"{i+1}. Token: '{t['token']}' | Context: ...{t['context']}... | cos={t['cos']:.4f}, SAE={t['sae_act']:.1f}"
                for i, t in enumerate(cosine_detected)
            )

            judge_prompt = (
                f"Feature explanation: {cos_interpretation.strip()}\n\n"
                f"Tokens to evaluate (these were detected by cosine similarity "
                f"but MISSED by the SAE — SAE activation is 0):\n\n"
                f"{cos_detected_text}\n\n"
                f"For each token, does it match the feature explanation? "
                f"Return JSON with 'judgments' list and 'match_rate'."
            )

            print("\n  Judging cosine-detected tokens (SAE misses)...")
            cos_judgment = call_claude(bedrock, JUDGE_SYSTEM, judge_prompt, max_tokens=2048)
            print(f"  Judgment: {cos_judgment[:200]}...")
            feature_result["cosine_detected_judgment"] = cos_judgment.strip()
            feature_result["cosine_detected_tokens"] = cosine_detected

        # Judge: do SAE false-positive tokens match the cosine interpretation?
        if sae_detected:
            sae_detected_text = "\n".join(
                f"{i+1}. Token: '{t['token']}' | Context: ...{t['context']}... | cos={t['cos']:.4f}, SAE={t['sae_act']:.1f}"
                for i, t in enumerate(sae_detected)
            )

            judge_prompt = (
                f"Feature explanation: {cos_interpretation.strip()}\n\n"
                f"Tokens to evaluate (these were detected by SAE activation "
                f"but have LOW cosine similarity to the feature direction):\n\n"
                f"{sae_detected_text}\n\n"
                f"For each token, does it match the feature explanation? "
                f"Return JSON with 'judgments' list and 'match_rate'."
            )

            print("  Judging SAE false-positive tokens (low cosine)...")
            sae_judgment = call_claude(bedrock, JUDGE_SYSTEM, judge_prompt, max_tokens=2048)
            print(f"  Judgment: {sae_judgment[:200]}...")
            feature_result["sae_detected_judgment"] = sae_judgment.strip()
            feature_result["sae_detected_tokens"] = sae_detected

        all_results["features"].append(feature_result)

    # Summary: parse match rates
    print("\n" + "=" * 70)
    print("  SUMMARY: Interpretability by Detection Method")
    print("=" * 70)

    cos_match_rates = []
    sae_match_rates = []

    for feat in all_results["features"]:
        # Try to extract match_rate from JSON in judgment
        for key, rates_list in [
            ("cosine_detected_judgment", cos_match_rates),
            ("sae_detected_judgment", sae_match_rates),
        ]:
            if key in feat:
                text = feat[key]
                try:
                    # Find JSON in response
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(text[start:end])
                        if "match_rate" in parsed:
                            rates_list.append(float(parsed["match_rate"]))
                        elif "judgments" in parsed:
                            judgments = parsed["judgments"]
                            matches = sum(1 for j in judgments if j in ("YES", "PARTIAL"))
                            rates_list.append(matches / len(judgments) if judgments else 0)
                except (json.JSONDecodeError, ValueError):
                    pass

    print(f"\n  Cosine-detected tokens (SAE misses) match rate:")
    if cos_match_rates:
        print(f"    Mean: {np.mean(cos_match_rates):.2%} (n={len(cos_match_rates)} features)")
        print(f"    Per-feature: {[f'{r:.2%}' for r in cos_match_rates]}")
    else:
        print("    Could not parse match rates")

    print(f"\n  SAE false-positive tokens (low cosine) match rate:")
    if sae_match_rates:
        print(f"    Mean: {np.mean(sae_match_rates):.2%} (n={len(sae_match_rates)} features)")
        print(f"    Per-feature: {[f'{r:.2%}' for r in sae_match_rates]}")
    else:
        print("    Could not parse match rates")

    if cos_match_rates and sae_match_rates:
        print(f"\n  Delta: cosine-detected = {np.mean(cos_match_rates):.2%}, "
              f"SAE false-pos = {np.mean(sae_match_rates):.2%}")
        if np.mean(cos_match_rates) > np.mean(sae_match_rates):
            print("  → Cosine-detected tokens are MORE interpretable (supports RNH)")
        else:
            print("  → SAE-detected tokens are more interpretable")

    # Save
    output_path = "experiments/exp3_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
