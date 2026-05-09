"""
Experiment 1: LayerNorm Erasure Sanity Check
=============================================
Goal: Demonstrate that LayerNorm/RMSNorm erases magnitude from the residual stream,
validating the core mechanism behind the Relative Norm Hypothesis.

Method:
  1. Collect residual stream activations at layer l
  2. Scale them by different factors (0.5x, 2x, 5x, 10x)
  3. Pass through the actual RMSNorm of layer l+1
  4. Measure cosine similarity between post-norm representations
  5. Also measure how model output (logits) change with scaled vs original activations

Expected: Post-norm representations should be identical regardless of input magnitude
(cosine sim = 1.0), confirming that the model literally cannot see magnitude differences.

Model: Qwen3-8B
"""

import json
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS_TO_TEST = [9, 18, 27]  # 25%, 50%, 75% depth (36 layers total)
SCALE_FACTORS = [0.1, 0.5, 2.0, 5.0, 10.0, 100.0]

TEST_PROMPTS = [
    "The capital of France is",
    "In a surprising turn of events, the",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The theory of relativity states that",
    "Once upon a time in a land far away,",
    "import torch\nmodel = AutoModelForCausalLM.from_pretrained(",
    "The quick brown fox jumps over the lazy",
    "According to recent studies, climate change",
]


def get_rmsnorm_at_layer(model, layer_idx):
    """Get the RMSNorm (input_layernorm) at the start of a given layer."""
    return model.model.layers[layer_idx].input_layernorm


def collect_residual_stream(model, tokenizer, prompts, layer_idx):
    """Collect residual stream activations at the output of a given layer."""
    activations = []

    def hook_fn(module, inputs, outputs):
        if isinstance(outputs, tuple):
            activations.append(outputs[0].detach())
        else:
            activations.append(outputs.detach())

    submodule = model.model.layers[layer_idx]
    handle = submodule.register_forward_hook(hook_fn)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)

    handle.remove()
    return activations


def test_rmsnorm_magnitude_invariance(model, tokenizer, prompts, layer_idx):
    """Core test: scale activations, pass through RMSNorm, check if output is identical."""
    print(f"\n{'='*60}")
    print(f"  Layer {layer_idx} -> RMSNorm of Layer {layer_idx + 1}")
    print(f"{'='*60}")

    # Collect activations at layer l
    acts = collect_residual_stream(model, tokenizer, prompts, layer_idx)
    rmsnorm = get_rmsnorm_at_layer(model, layer_idx + 1)

    results = []

    for prompt_idx, (prompt, act) in enumerate(zip(prompts, acts)):
        # act shape: (1, seq_len, d_model)
        original_normed = rmsnorm(act)

        for scale in SCALE_FACTORS:
            scaled_act = act * scale
            scaled_normed = rmsnorm(scaled_act)

            # Cosine similarity between post-norm representations (per token)
            cos_sim = torch.nn.functional.cosine_similarity(
                original_normed.reshape(-1, original_normed.shape[-1]),
                scaled_normed.reshape(-1, scaled_normed.shape[-1]),
                dim=-1,
            )

            # L2 distance between post-norm representations
            l2_dist = torch.norm(original_normed - scaled_normed, dim=-1).mean().item()

            # Max absolute difference
            max_diff = (original_normed - scaled_normed).abs().max().item()

            result = {
                "prompt_idx": prompt_idx,
                "layer": layer_idx,
                "scale_factor": scale,
                "cosine_sim_mean": cos_sim.mean().item(),
                "cosine_sim_min": cos_sim.min().item(),
                "cosine_sim_std": cos_sim.std().item(),
                "l2_distance": l2_dist,
                "max_abs_diff": max_diff,
                "original_norm": act.norm(dim=-1).mean().item(),
                "scaled_norm": scaled_act.norm(dim=-1).mean().item(),
            }
            results.append(result)

            if prompt_idx == 0:  # Print details for first prompt only
                print(
                    f"  scale={scale:>6.1f}x | "
                    f"cos_sim={cos_sim.mean().item():.10f} | "
                    f"l2_dist={l2_dist:.2e} | "
                    f"max_diff={max_diff:.2e}"
                )

    return results


def test_logit_invariance(model, tokenizer, prompts, layer_idx):
    """
    Stronger test: intervene on the residual stream by scaling it at layer l,
    then check if the model's final output logits change.
    """
    print(f"\n{'='*60}")
    print(f"  Logit Invariance: Scale at Layer {layer_idx}")
    print(f"{'='*60}")

    results = []

    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Get original logits
        with torch.no_grad():
            original_logits = model(**inputs).logits  # (1, seq_len, vocab_size)

        for scale in SCALE_FACTORS:
            # Hook to scale residual stream at layer l
            def scale_hook(module, inputs, outputs, s=scale):
                if isinstance(outputs, tuple):
                    return (outputs[0] * s,) + outputs[1:]
                return outputs * s

            handle = model.model.layers[layer_idx].register_forward_hook(scale_hook)

            with torch.no_grad():
                scaled_logits = model(**inputs).logits

            handle.remove()

            # Compare logits
            # Use last token logits for next-token prediction comparison
            orig_last = original_logits[0, -1, :]
            scaled_last = scaled_logits[0, -1, :]

            cos_sim = torch.nn.functional.cosine_similarity(
                orig_last.unsqueeze(0), scaled_last.unsqueeze(0)
            ).item()

            # KL divergence between output distributions
            orig_probs = torch.softmax(orig_last, dim=-1)
            scaled_probs = torch.softmax(scaled_last, dim=-1)
            kl_div = (
                torch.nn.functional.kl_div(
                    scaled_probs.log(), orig_probs, reduction="sum"
                )
                .item()
            )

            # Top-1 token agreement
            orig_top = orig_last.argmax().item()
            scaled_top = scaled_last.argmax().item()
            top1_match = orig_top == scaled_top

            result = {
                "prompt_idx": prompt_idx,
                "layer": layer_idx,
                "scale_factor": scale,
                "logit_cosine_sim": cos_sim,
                "kl_divergence": kl_div,
                "top1_match": top1_match,
                "orig_top_token": tokenizer.decode([orig_top]),
                "scaled_top_token": tokenizer.decode([scaled_top]),
            }
            results.append(result)

            if prompt_idx == 0:
                match_str = "MATCH" if top1_match else "DIFF"
                print(
                    f"  scale={scale:>6.1f}x | "
                    f"logit_cos={cos_sim:.6f} | "
                    f"KL={kl_div:.4e} | "
                    f"top1={match_str} "
                    f"('{tokenizer.decode([orig_top])}' vs '{tokenizer.decode([scaled_top])}')"
                )

    return results


def main():
    print("Experiment 1: LayerNorm Erasure Sanity Check")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS_TO_TEST}")
    print(f"Scale factors: {SCALE_FACTORS}")
    print(f"Device: {DEVICE}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Check: what kind of normalization does Qwen3 use?
    norm_layer = model.model.layers[0].input_layernorm
    print(f"Normalization type: {type(norm_layer).__name__}")
    print(f"Model layers: {len(model.model.layers)}")

    all_results = {
        "model": MODEL_NAME,
        "norm_type": type(norm_layer).__name__,
        "layers_tested": LAYERS_TO_TEST,
        "scale_factors": SCALE_FACTORS,
        "num_prompts": len(TEST_PROMPTS),
        "rmsnorm_tests": [],
        "logit_tests": [],
    }

    # Part A: RMSNorm magnitude invariance
    print("\n" + "=" * 60)
    print("  PART A: RMSNorm Magnitude Invariance")
    print("=" * 60)
    for layer_idx in LAYERS_TO_TEST:
        results = test_rmsnorm_magnitude_invariance(
            model, tokenizer, TEST_PROMPTS, layer_idx
        )
        all_results["rmsnorm_tests"].extend(results)

    # Part B: Full model logit invariance under scaling
    print("\n" + "=" * 60)
    print("  PART B: Logit Invariance Under Residual Stream Scaling")
    print("=" * 60)
    for layer_idx in LAYERS_TO_TEST:
        results = test_logit_invariance(
            model, tokenizer, TEST_PROMPTS, layer_idx
        )
        all_results["logit_tests"].extend(results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print("\nRMSNorm invariance (cosine sim between original and scaled post-norm):")
    for layer_idx in LAYERS_TO_TEST:
        layer_results = [
            r for r in all_results["rmsnorm_tests"] if r["layer"] == layer_idx
        ]
        cos_sims = [r["cosine_sim_mean"] for r in layer_results]
        print(
            f"  Layer {layer_idx}: "
            f"mean={np.mean(cos_sims):.10f}, "
            f"min={np.min(cos_sims):.10f}, "
            f"std={np.std(cos_sims):.2e}"
        )

    print("\nLogit invariance under scaling:")
    for layer_idx in LAYERS_TO_TEST:
        layer_results = [
            r for r in all_results["logit_tests"] if r["layer"] == layer_idx
        ]
        cos_sims = [r["logit_cosine_sim"] for r in layer_results]
        top1_matches = [r["top1_match"] for r in layer_results]
        kl_divs = [r["kl_divergence"] for r in layer_results]
        print(
            f"  Layer {layer_idx}: "
            f"logit_cos_mean={np.mean(cos_sims):.6f}, "
            f"top1_match_rate={np.mean(top1_matches):.2%}, "
            f"kl_div_mean={np.mean(kl_divs):.4e}"
        )

    # Save results
    output_path = "experiments/exp1_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
