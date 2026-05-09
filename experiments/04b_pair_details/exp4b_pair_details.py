"""
Experiment 4b: Detailed analysis of the 18 pairs found at cos > 0.70
Quick follow-up to get encoder weights and activation patterns for
the small number of near-similar pairs.
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

PROMPTS = [
    "The theory of general relativity describes how gravity works as a curvature of spacetime caused by mass and energy. Einstein published this theory in 1915.",
    "In Python, you can use list comprehensions to create new lists. For example, squares = [x**2 for x in range(10)] is more concise than a for loop.",
    "The French Revolution began in 1789. The storming of the Bastille on July 14th became a symbol and is still celebrated as a national holiday.",
    "Machine learning models are trained by optimizing a loss function. The model adjusts parameters through gradient descent. Regularization prevents overfitting.",
    "Once upon a time, there was a small village nestled between two mountains. The villagers lived simple lives, farming the rich soil.",
    "The mitochondria produces ATP through oxidative phosphorylation in the inner mitochondrial membrane. This is essential for aerobic life.",
    "Quantum mechanics says particles exist in superposition until measured. Schrodinger's cat is both alive and dead simultaneously.",
    "The stock market crashed in 2008 due to subprime mortgages and complex derivatives. The entire financial system nearly collapsed.",
    "Shakespeare wrote Romeo and Juliet in the late 16th century. It tells the story of two lovers from feuding families in Verona.",
    "Climate change is caused by burning fossil fuels, releasing greenhouse gases that trap heat and increase global temperatures.",
    "The human genome has 3 billion base pairs in 23 chromosome pairs. Only 1.5% codes for proteins.",
    "The Renaissance spanned the 14th to 17th century with advances in art, science, and philosophy led by da Vinci and Galileo.",
]


def main():
    print("Experiment 4b: Pair Detail Analysis")
    print("=" * 70)

    # Load SAE
    sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
        repo_id=SAE_REPO, filename=SAE_FILENAME,
        model_name=MODEL_NAME, device=DEVICE, dtype=DTYPE,
    )
    W_dec = sae.W_dec  # (65536, 4096)
    W_enc = sae.W_enc  # (65536, 4096)
    b_enc = sae.b_enc  # (65536,)

    # Re-find pairs at cos > 0.70
    print("\nFinding all pairs with cos > 0.70...")
    pairs = []
    n = W_dec.shape[0]
    batch = 4096
    for i in range(0, n, batch):
        chunk = W_dec[i:min(i+batch, n)]
        for j in range(i, n, batch):
            other = W_dec[j:min(j+batch, n)]
            sims = chunk @ other.T
            if i == j:
                mask = torch.triu(sims > 0.70, diagonal=1)
            else:
                mask = sims > 0.70
            for idx in torch.nonzero(mask):
                fi = i + idx[0].item()
                fj = j + idx[1].item()
                if fi != fj:
                    pairs.append((fi, fj, sims[idx[0], idx[1]].item()))

    print(f"Found {len(pairs)} pairs")
    pairs.sort(key=lambda p: -p[2])

    # Encoder analysis for each pair
    print("\n--- Encoder Analysis ---")
    pair_details = []
    for fi, fj, dec_cos in pairs:
        # W_enc shape: (d_model, n_features) — index by column
        enc_i = W_enc[:, fi] if W_enc.shape[0] < W_enc.shape[1] else W_enc[fi]
        enc_j = W_enc[:, fj] if W_enc.shape[0] < W_enc.shape[1] else W_enc[fj]
        enc_cos = torch.nn.functional.cosine_similarity(
            enc_i.unsqueeze(0), enc_j.unsqueeze(0)
        ).item()
        enc_norm_i = enc_i.norm().item()
        enc_norm_j = enc_j.norm().item()
        bi = b_enc[fi].item()
        bj = b_enc[fj].item()

        detail = {
            "fi": fi, "fj": fj,
            "dec_cos": dec_cos, "enc_cos": enc_cos,
            "enc_norm_i": enc_norm_i, "enc_norm_j": enc_norm_j,
            "bias_i": bi, "bias_j": bj,
            "bias_diff": abs(bi - bj),
        }
        pair_details.append(detail)

        print(f"  f{fi:>5d} & f{fj:>5d}: dec_cos={dec_cos:.4f}, enc_cos={enc_cos:.4f}, "
              f"enc_norms={enc_norm_i:.3f}/{enc_norm_j:.3f}, "
              f"biases={bi:.3f}/{bj:.3f}")

    # Summary
    if pair_details:
        enc_cosines = [p["enc_cos"] for p in pair_details]
        bias_diffs = [p["bias_diff"] for p in pair_details]
        print(f"\n  Summary ({len(pair_details)} pairs):")
        print(f"  Encoder cosine: mean={np.mean(enc_cosines):.4f}, range=[{min(enc_cosines):.4f}, {max(enc_cosines):.4f}]")
        print(f"  Bias difference: mean={np.mean(bias_diffs):.4f}, range=[{min(bias_diffs):.4f}, {max(bias_diffs):.4f}]")
        same_enc = sum(1 for c in enc_cosines if c > 0.7)
        print(f"  Pairs where encoder is also similar (cos > 0.7): {same_enc}/{len(pair_details)}")

    # Part 2: Quick activation check for these pairs
    print("\n--- Activation Analysis ---")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device_map=DEVICE)
    model.eval()

    # Collect activations
    submodule = model_utils.get_submodule(model, LAYER)
    all_acts = []
    for i in range(0, len(PROMPTS), 4):
        batch_prompts = PROMPTS[i:i+4]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        acts = model_utils.collect_activations(model, submodule, inputs)
        mask = inputs["attention_mask"]
        for b in range(acts.shape[0]):
            all_acts.append(acts[b][mask[b].bool()])
    activations = torch.cat(all_acts, dim=0)

    # Filter sinks
    norms = activations.norm(dim=-1)
    keep = norms < (norms.median() * 10)
    activations = activations[keep]
    print(f"  {activations.shape[0]} tokens")

    # Encode
    with torch.no_grad():
        sae_acts = sae.encode(activations)

    act_norms = activations.norm(dim=-1).detach().float().cpu().numpy()

    for detail in pair_details:
        fi, fj = detail["fi"], detail["fj"]
        ai = sae_acts[:, fi].detach().float().cpu().numpy()
        aj = sae_acts[:, fj].detach().float().cpu().numpy()

        fire_i = ai > 0
        fire_j = aj > 0
        both = fire_i & fire_j
        only_i = fire_i & ~fire_j
        only_j = ~fire_i & fire_j

        detail["n_fire_i"] = int(fire_i.sum())
        detail["n_fire_j"] = int(fire_j.sum())
        detail["n_both"] = int(both.sum())
        detail["n_only_i"] = int(only_i.sum())
        detail["n_only_j"] = int(only_j.sum())

        union = (fire_i | fire_j).sum()
        detail["jaccard"] = float(both.sum() / union) if union > 0 else 0

        # Norm separation when only one fires
        if only_i.sum() > 3 and only_j.sum() > 3:
            ni = act_norms[only_i]
            nj = act_norms[only_j]
            detail["norm_only_i_median"] = float(np.median(ni))
            detail["norm_only_j_median"] = float(np.median(nj))
            combined_std = np.std(np.concatenate([ni, nj]))
            detail["norm_separation"] = float(abs(np.median(ni) - np.median(nj)) / (combined_std + 1e-8))
        else:
            detail["norm_separation"] = None

        print(f"  f{fi:>5d} & f{fj:>5d}: "
              f"jaccard={detail['jaccard']:.3f}, "
              f"both={detail['n_both']}, only_i={detail['n_only_i']}, only_j={detail['n_only_j']}, "
              f"norm_sep={detail.get('norm_separation', 'N/A')}")

    # Save
    with open("experiments/exp4b_results.json", "w") as f:
        json.dump(pair_details, f, indent=2, default=str)
    print(f"\nSaved to experiments/exp4b_results.json")


if __name__ == "__main__":
    main()
