"""
Experiment 6: Cross-Model Replication on Pythia-70M
====================================================
Replicate the RNH finding (cos > inner > SAE > norm for predicting causal
feature importance) on a completely different model and SAE.

This addresses the "single model" caveat from Experiment 5.

Model: Pythia-70M (6 layers, d_model=512, ~70M params)
SAE: EleutherAI/sae-pythia-70m-32k (TopK, k=16, 32768 features per layer)
     - Different architecture than Qwen3-8B
     - Different scale (70M vs 8B)
     - Different SAE family (EleutherAI TopK vs adamkarvonen BatchTopK)
     - Different normalization (Pythia uses LayerNorm, not RMSNorm)

Layers tested: 1, 3, 5 (early, mid, late out of 6)
Ablation measure: logit-<author>el KL divergence
"""

import json
import time
import gc
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformer_lens import HookedTransformer

DEVICE = "cuda"
MODEL_NAME = "pythia-70m"
SAE_REPO = "EleutherAI/sae-pythia-70m-32k"
LAYERS = [1, 3, 5]  # early, mid, late (out of 0-5)
K = 16  # TopK parameter
D_SAE = 32768
D_MODEL = 512

NUM_FEATURES = 50
NUM_ABLATION_SAMPLES = 100
MAX_SEQ_LEN = 256

# Diverse real text corpus (same as exp5 for consistency)
TEXTS = [
    "The development of quantum field theory in the mid-20th century revolutionized our understanding of fundamental particles. Building on the earlier work of Dirac and others, physicists developed a framework that unified quantum mechanics with special relativity, leading to remarkably precise predictions about the behavior of subatomic particles.",
    "In 1969, the first message was sent over ARPANET, the precursor to the modern internet. The message was supposed to be 'LOGIN', but the system crashed after just two letters, so the first message ever sent over what would become the internet was simply 'LO'. This humble beginning would eventually transform human communication.",
    "The enzyme lactase is responsible for breaking down lactose, the primary sugar found in milk. In most mammals, lactase production decreases after weaning, but a genetic mutation that arose approximately 7,500 years ago in European populations allowed some humans to continue producing lactase into adulthood.",
    "import torch\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader\n\ndef train_epoch(model, loader, optimizer):\n    model.train()\n    total_loss = 0\n    for batch in loader:\n        optimizer.zero_grad()\n        output = model(batch['input_ids'])\n        loss = F.cross_entropy(output.logits, batch['labels'])\n        loss.backward()\n        optimizer.step()\n        total_loss += loss.item()\n    return total_loss / len(loader)",
    "The Great Barrier Reef stretches over 2,300 kilometers along the northeast coast of Australia. It is the world's largest coral reef system, composed of over 2,900 individual reef systems and 900 islands. The reef supports an extraordinary diversity of life, including 1,500 species of fish, 400 types of coral, and numerous species of birds and marine mammals.",
    "Consider the following theorem: For any continuous function f on a closed interval [a,b], there exists at least one point c in (a,b) such that f'(c) = (f(b) - f(a))/(b - a). This is the Mean Value Theorem, a fundamental result in calculus that connects the average rate of change of a function to its instantaneous rate of change.",
    "The manufacturing process for semiconductor chips involves hundreds of steps and takes place in cleanrooms where the air contains fewer than 100 particles per cubic foot. Photolithography, the key patterning step, uses ultraviolet light to transfer circuit designs onto silicon wafers coated with photoresist.",
    "A recent meta-analysis of 47 randomized controlled trials found that cognitive behavioral therapy (CBT) was significantly more effective than waitlist controls for treating generalized anxiety disorder, with a pooled effect size of d = 0.82. However, the analysis also revealed substantial heterogeneity across studies.",
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and established the principle of state sovereignty that forms the foundation of modern international relations. Before this treaty, European politics was dominated by the overlapping authorities of the Holy Roman Empire, the Catholic Church, and various feudal obligations.",
    "SELECT u.username, COUNT(o.order_id) as total_orders, SUM(o.amount) as total_spent\nFROM users u\nLEFT JOIN orders o ON u.user_id = o.user_id\nWHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)\nGROUP BY u.user_id\nHAVING total_orders > 5\nORDER BY total_spent DESC\nLIMIT 100;",
    "Fermentation is one of humanity's oldest biotechnologies. The earliest evidence of intentional fermentation dates back approximately 13,000 years to a site in Israel, where residues of a wheat-and-barley-based beer were found in stone mortars. Today, industrial fermentation produces everything from antibiotics to biofuels.",
    "The philosophical concept of qualia refers to the subjective, conscious experiences that accompany perception. When you see the color red, there is something it is like to have that experience — a redness to it that cannot be fully captured by describing the wavelength of light or the neural activity in your visual cortex.",
    "During photosynthesis, plants use the energy from sunlight to convert carbon dioxide and water into glucose and oxygen. The light-dependent reactions occur in the thylakoid membranes, where photosystems I and II work together to generate ATP and NADPH through a process called the Z-scheme.",
    "The city of Venice was built on 118 small islands connected by over 400 bridges. Founded in the 5th century by Romans fleeing barbarian invasions, it grew into one of the most powerful maritime republics in history. Today, the city faces existential challenges from rising sea <author>els and mass tourism.",
    "In distributed systems, the CAP theorem states that it is impossible for a distributed data store to simultaneously provide more than two of the following three guarantees: Consistency, Availability, and Partition tolerance. This fundamental result shapes the design of every modern database and distributed application.",
    "The aurora borealis, or northern lights, occur when charged particles from the sun interact with gases in Earth's atmosphere. These solar particles are guided by Earth's magnetic field toward the poles, where they collide with oxygen and nitrogen molecules, causing them to emit light in spectacular displays.",
    "Bach's Well-Tempered Clavier, composed in two books (1722 and 1742), contains 48 preludes and fugues in all 24 major and minor keys. It demonstrated that keyboard instruments tuned in a temperament close to equal temperament could play in any key.",
    "The human immune system employs two main strategies: innate immunity and adaptive immunity. Innate immunity provides immediate but non-specific defense through physical barriers, phagocytes, and inflammatory responses. Adaptive immunity develops specific responses through B cells and T cells.",
    "Reinforcement learning differs from supervised learning in that the agent must discover which actions yield the most reward through trial and error. The exploration-exploitation dilemma is central: the agent must balance trying new actions with choosing actions known to give good rewards.",
    "The Silk Road was not a single road but a network of trade routes connecting China to the Mediterranean, spanning over 6,000 kilometers. Beyond silk, merchants traded spices, precious metals, gemstones, and glassware, facilitating the exchange of ideas, religions, technologies, and diseases.",
    "The concept of natural selection describes how organisms with traits better suited to their environment tend to survive and reproduce more successfully. Over many generations, this process leads to the adaptation of species. Modern evolutionary biology includes genetic drift and gene flow.",
    "In functional programming, functions are treated as first-class citizens that can be passed as arguments, returned from other functions, and assigned to variables. Languages like Haskell enforce pure functions with no side effects, making programs easier to reason about.",
    "The Mongol Empire, established by Genghis Khan in 1206, became the largest contiguous land empire in history, stretching from Eastern Europe to the Sea of Japan. The Pax Mongolica facilitated unprecedented trade and cultural exchange along the Silk Road.",
    "Superconductivity is a phenomenon where certain materials exhibit zero electrical resistance when cooled below a critical temperature. Discovered in 1911, conventional superconductors are explained by BCS theory, which describes how electrons form Cooper pairs.",
    "CRISPR-Cas9 is a gene-editing technology derived from a bacterial immune system. The Cas9 protein, guided by synthetic RNA, makes precise cuts in DNA at targeted locations, allowing researchers to modify genes with unprecedented accuracy.",
    "The Fibonacci sequence appears throughout nature: the spiral of sunflower seeds, the branching of trees, the shell of a nautilus. Each number is the sum of the two preceding ones, and the ratio of consecutive terms converges to the golden ratio.",
    "Tectonic plates are massive segments of Earth's lithosphere that move, float, and fracture. Their interaction causes earthquakes, volcanic activity, and mountain building. The theory of plate tectonics, developed in the 1960s, unified observations about continental drift and seafloor spreading.",
    "The JavaScript event loop handles asynchronous operations through a message queue and call stack. When the stack is empty, the loop takes the first message from the queue. Promises and async/await provide more elegant asynchronous code than callbacks.",
    "Black holes are regions where gravity is so strong that nothing can escape past the event horizon. Stellar black holes form from collapsing massive stars, while supermassive black holes reside at galaxy centers.",
    "Bayesian inference updates beliefs using new evidence. Starting with a prior distribution, Bayes' theorem combines this with observed data likelihood to produce a posterior distribution. This approach is fundamental to machine learning and scientific reasoning.",
    "The Antarctic ice sheet contains about 26.5 million cubic kilometers of ice, representing 70% of Earth's fresh water. Ice cores from the sheet provide climate data spanning hundreds of thousands of years.",
    "Blockchain creates a distributed, immutable ledger verified by network consensus. Each block contains a cryptographic hash of the previous block. Beyond cryptocurrency, applications include supply chain tracking and smart contracts.",
    "The human brain contains approximately 86 billion neurons connected through synapses. Neural signals travel at up to 120 meters per second. The prefrontal cortex, responsible for executive functions, doesn't fully mature until the mid-twenties.",
    "Gabriel Garcia Marquez's One Hundred Years of Solitude follows seven generations of the Buendia family in fictional Macondo. The novel blends fantastical elements with Colombian history and has sold over 50 million copies.",
    "The Haber-Bosch process synthesizes ammonia from atmospheric nitrogen under high pressure and temperature. This enabled mass fertilizer production, and it is estimated that nearly half of the nitrogen in human bodies comes from this process.",
    "Docker containers provide lightweight environments for running applications. Unlike virtual machines, containers share the host OS kernel. Kubernetes manages deployment, scaling, and networking of containerized applications across clusters.",
    "General relativity predicts gravitational lensing: the bending of light around massive objects. This effect is confirmed through galaxy cluster observations and is used to study dark matter distribution in the universe.",
    "The Renaissance spanned the 14th to 17th centuries with advances in art, science, and philosophy. Leonardo da Vinci, Michelangelo, and Galileo were among its leading figures. The printing press accelerated the spread of ideas across Europe.",
    "Traditional Chinese medicine encompasses herbal remedies, acupuncture, and dietary therapy based on concepts of qi and yin-yang balance. While some treatments show clinical efficacy, others remain controversial in evidence-based medicine.",
    "The Bauhaus school, founded in 1919, unified art, craft, and technology. Its principles of functional design influenced architecture and industrial design throughout the 20th century. Members included Kandinsky and Mies van der Rohe.",
]


class TopKSAE:
    """Minimal TopK SAE wrapper for EleutherAI weights."""

    def __init__(self, W_dec, W_enc, b_enc, b_dec, k=16):
        self.W_dec = W_dec  # (d_sae, d_model)
        self.W_enc = W_enc  # (d_sae, d_model)
        self.b_enc = b_enc  # (d_sae,)
        self.b_dec = b_dec  # (d_model,)
        self.k = k

    def encode(self, x):
        """TopK encoding: pre_acts = (x - b_dec) @ W_enc.T + b_enc, then keep top-k."""
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc  # (batch, d_sae)
        # TopK: zero out everything except top-k values
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(dim=-1, index=topk_idx, src=torch.relu(topk_vals))
        return acts

    @staticmethod
    def load(repo_id, layer_name, device="cuda"):
        cfg_path = hf_hub_download(repo_id, f"{layer_name}/cfg.json")
        sae_path = hf_hub_download(repo_id, f"{layer_name}/sae.safetensors")
        cfg = json.load(open(cfg_path))
        state = load_file(sae_path, device=device)

        # Check decoder norm
        dec_norms = state["W_dec"].norm(dim=-1)
        if cfg.get("normalize_decoder", True):
            state["W_dec"] = state["W_dec"] / dec_norms.unsqueeze(-1)

        return TopKSAE(
            W_dec=state["W_dec"],
            W_enc=state["encoder.weight"],
            b_enc=state["encoder.bias"],
            b_dec=state["b_dec"],
            k=cfg.get("k", 16),
        )


def collect_activations(model, texts, layer_idx, batch_size=16):
    """Collect residual stream activations using TransformerLens."""
    all_acts = []
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        if tokens.shape[1] > MAX_SEQ_LEN:
            tokens = tokens[:, :MAX_SEQ_LEN]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
            )
        acts = cache[hook_name]  # (batch, seq, d_model)
        # Flatten, skip padding (Pythia doesn't pad, but be safe)
        for b in range(acts.shape[0]):
            all_acts.append(acts[b].detach())

    return torch.cat(all_acts, dim=0)  # (total_tokens, d_model)


def ablate_feature_kl(model, all_acts_at_layer, token_idx, feature_dir, layer_idx):
    """
    Ablate a feature from one token's activation, measure logit-<author>el KL.
    Uses TransformerLens hooks.
    """
    x = all_acts_at_layer[token_idx]  # (d_model,)
    projection = (x @ feature_dir) * feature_dir
    x_ablated = x - projection

    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # We need a dummy forward pass. We'll hook into the model to
    # replace the activation at position 0 of a 1-token input.
    dummy_tokens = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

    def make_hook(replacement):
        def hook_fn(value, hook):
            value[0, 0, :] = replacement
            return value
        return hook_fn

    # Original logits
    with torch.no_grad():
        orig_logits = model.run_with_hooks(
            dummy_tokens,
            fwd_hooks=[(hook_name, make_hook(x))],
        )[0, -1, :].float()

    # Ablated logits
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            dummy_tokens,
            fwd_hooks=[(hook_name, make_hook(x_ablated))],
        )[0, -1, :].float()

    # KL divergence — use numerically stable computation
    # Clamp logits to prevent extreme values causing NaN
    orig_logits = orig_logits.clamp(-100, 100)
    ablated_logits = ablated_logits.clamp(-100, 100)

    orig_log_probs = torch.log_softmax(orig_logits, dim=-1)
    ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)
    kl = torch.sum(
        torch.exp(orig_log_probs) * (orig_log_probs - ablated_log_probs)
    ).item()

    if np.isnan(kl) or np.isinf(kl) or kl < 0:
        return None

    return {"kl": kl}


def analyze_layer(model, sae, activations, layer_idx):
    """Analyze one layer: find top features, run ablation, compute correlations."""
    print(f"\n  Computing SAE activations...")
    t0 = time.time()
    batch_size = 1024
    all_sae_acts = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size]
        encoded = sae.encode(batch)
        all_sae_acts.append(encoded.detach())
    all_sae_acts = torch.cat(all_sae_acts, dim=0)
    print(f"  SAE encoding: {time.time()-t0:.1f}s")

    # Top features by frequency
    feature_freq = (all_sae_acts > 0).float().mean(dim=0)
    top_features = feature_freq.topk(NUM_FEATURES).indices
    print(f"  Top {NUM_FEATURES} features: freq [{feature_freq[top_features[-1]]:.4f}, {feature_freq[top_features[0]]:.4f}]")

    feature_results = []

    for feat_rank, feat_idx in enumerate(top_features):
        feat_idx = feat_idx.item()
        feature_dir = sae.W_dec[feat_idx]  # decoder direction
        fd_norm = feature_dir.norm()
        if fd_norm > 0:
            feature_dir = feature_dir / fd_norm  # ensure unit norm

        # Metrics for all tokens
        cos_sims = torch.nn.functional.cosine_similarity(
            activations, feature_dir.unsqueeze(0), dim=-1
        )
        norms = activations.norm(dim=-1)
        inner_prods = activations @ feature_dir
        sae_feat_acts = all_sae_acts[:, feat_idx]

        active_mask = sae_feat_acts > 0
        n_active = active_mask.sum().item()

        if n_active < 30:
            continue

        active_indices = torch.where(active_mask)[0]
        n_sample = min(NUM_ABLATION_SAMPLES, len(active_indices))
        perm = torch.randperm(len(active_indices))[:n_sample]
        sample_indices = active_indices[perm]

        cos_vals, norm_vals, inner_vals, sae_vals, kl_vals = [], [], [], [], []

        for idx in sample_indices:
            result = ablate_feature_kl(model, activations, idx.item(), feature_dir, layer_idx)
            if result is None:
                continue
            cos_vals.append(cos_sims[idx].item())
            norm_vals.append(norms[idx].item())
            inner_vals.append(inner_prods[idx].item())
            sae_vals.append(sae_feat_acts[idx].item())
            kl_vals.append(result["kl"])

        if len(kl_vals) < 15:
            continue

        cos_arr = np.array(cos_vals)
        norm_arr = np.array(norm_vals)
        inner_arr = np.array(inner_vals)
        sae_arr = np.array(sae_vals)
        kl_arr = np.array(kl_vals)

        if kl_arr.std() < 1e-10:
            continue

        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]

        feat_result = {
            "feature_idx": feat_idx,
            "n_active": n_active,
            "n_ablated": len(kl_vals),
            "kl_mean": float(kl_arr.mean()),
            "corr_cos_kl": float(corr_cos),
            "corr_norm_kl": float(corr_norm),
            "corr_inner_kl": float(corr_inner),
            "corr_sae_kl": float(corr_sae),
            "cos_wins_inner": bool(abs(corr_cos) > abs(corr_inner)),
            "cos_wins_sae": bool(abs(corr_cos) > abs(corr_sae)),
        }
        feature_results.append(feat_result)

        if feat_rank < 5 or feat_rank % 10 == 0:
            print(f"    Feature {feat_idx:>5d} (rank {feat_rank:>2d}) | "
                  f"n={len(kl_vals):>3d} | "
                  f"corr(cos,KL)={corr_cos:>6.3f} | "
                  f"corr(norm,KL)={corr_norm:>6.3f} | "
                  f"corr(inner,KL)={corr_inner:>6.3f} | "
                  f"corr(SAE,KL)={corr_sae:>6.3f}")

    # Aggregate
    if feature_results:
        cos_kls = [r["corr_cos_kl"] for r in feature_results]
        norm_kls = [r["corr_norm_kl"] for r in feature_results]
        inner_kls = [r["corr_inner_kl"] for r in feature_results]
        sae_kls = [r["corr_sae_kl"] for r in feature_results]
        n = len(feature_results)

        cos_wins_inner = sum(1 for r in feature_results if r["cos_wins_inner"])
        cos_wins_sae = sum(1 for r in feature_results if r["cos_wins_sae"])

        aggregate = {
            "n_features_tested": n,
            "corr_cos_kl_mean": float(np.mean(cos_kls)),
            "corr_cos_kl_std": float(np.std(cos_kls)),
            "corr_norm_kl_mean": float(np.mean(norm_kls)),
            "corr_norm_kl_std": float(np.std(norm_kls)),
            "corr_inner_kl_mean": float(np.mean(inner_kls)),
            "corr_inner_kl_std": float(np.std(inner_kls)),
            "corr_sae_kl_mean": float(np.mean(sae_kls)),
            "corr_sae_kl_std": float(np.std(sae_kls)),
            "cos_wins_inner_count": cos_wins_inner,
            "cos_wins_inner_pct": float(100 * cos_wins_inner / n),
            "cos_wins_sae_count": cos_wins_sae,
            "cos_wins_sae_pct": float(100 * cos_wins_sae / n),
        }

        print(f"\n  === Layer {layer_idx} Summary ({n} features) ===")
        print(f"  corr(cos, KL):   mean={np.mean(cos_kls):.4f} ± {np.std(cos_kls):.4f}")
        print(f"  corr(norm, KL):  mean={np.mean(norm_kls):.4f} ± {np.std(norm_kls):.4f}")
        print(f"  corr(inner, KL): mean={np.mean(inner_kls):.4f} ± {np.std(inner_kls):.4f}")
        print(f"  corr(SAE, KL):   mean={np.mean(sae_kls):.4f} ± {np.std(sae_kls):.4f}")
        print(f"  |corr(cos)| > |corr(inner)|: {cos_wins_inner}/{n}")
        print(f"  |corr(cos)| > |corr(SAE)|:   {cos_wins_sae}/{n}")
    else:
        aggregate = {"n_features_tested": 0}

    return {"features": feature_results, "aggregate": aggregate}


def main():
    print("Experiment 6: Pythia-70M Cross-Model Replication")
    print("=" * 70)
    print(f"Model: {MODEL_NAME} (6 layers, d_model=512)")
    print(f"SAE: {SAE_REPO} (TopK, k={K}, 32768 features)")
    print(f"Layers: {LAYERS}")
    print(f"Features per layer: {NUM_FEATURES}")
    print(f"Ablation samples per feature: {NUM_ABLATION_SAMPLES}")
    print(f"Ablation measure: logit-<author>el KL divergence")
    print(f"Note: Pythia uses LayerNorm (not RMSNorm like Qwen3)")

    # Load model
    print("\nLoading Pythia-70M...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Repeat texts to get more tokens (Pythia tokenizer produces fewer tokens)
    texts = TEXTS * 5  # 200 passages → ~17k tokens
    print(f"Using {len(texts)} text passages")

    all_results = {
        "model": MODEL_NAME,
        "sae_repo": SAE_REPO,
        "sae_type": "TopK",
        "sae_k": K,
        "sae_d": D_SAE,
        "layers": LAYERS,
        "n_features_per_layer": NUM_FEATURES,
        "n_ablation_samples": NUM_ABLATION_SAMPLES,
        "n_texts": len(texts),
        "normalization": "LayerNorm",
        "layer_results": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Load SAE
        print(f"  Loading SAE for layer {layer_idx}...")
        sae = TopKSAE.load(SAE_REPO, f"layers.{layer_idx}", device=DEVICE)
        print(f"  SAE loaded: {sae.W_dec.shape[0]} features")

        # Collect activations
        print(f"  Collecting activations...")
        t0 = time.time()
        activations = collect_activations(model, texts, layer_idx)
        print(f"  Collected {activations.shape[0]} tokens in {time.time()-t0:.1f}s")

        # Run analysis
        layer_result = analyze_layer(model, sae, activations, layer_idx)
        all_results["layer_results"][str(layer_idx)] = layer_result

        # Save intermediate
        with open("experiments/exp6_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Intermediate results saved.")

        del sae, activations
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-layer summary
    print(f"\n{'='*70}")
    print(f"  CROSS-LAYER SUMMARY (Pythia-70M)")
    print(f"{'='*70}")

    print(f"\n  {'Layer':>6s} | {'n_feat':>6s} | {'cos→KL':>10s} | {'norm→KL':>10s} | "
          f"{'inner→KL':>10s} | {'SAE→KL':>10s} | {'cos>inner':>10s} | {'cos>SAE':>10s}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    for layer_idx in LAYERS:
        agg = all_results["layer_results"][str(layer_idx)]["aggregate"]
        if agg["n_features_tested"] == 0:
            continue
        print(f"  {layer_idx:>6d} | {agg['n_features_tested']:>6d} | "
              f"{agg['corr_cos_kl_mean']:>10.4f} | {agg['corr_norm_kl_mean']:>10.4f} | "
              f"{agg['corr_inner_kl_mean']:>10.4f} | {agg['corr_sae_kl_mean']:>10.4f} | "
              f"{agg['cos_wins_inner_count']}/{agg['n_features_tested']:>3d}"
              f"{'':>4s} | "
              f"{agg['cos_wins_sae_count']}/{agg['n_features_tested']:>3d}")

    # Cross-model comparison header
    print(f"\n  For comparison with Qwen3-8B (Exp 5):")
    print(f"  Qwen3-8B: cos→KL=0.269, norm→KL=-0.101, inner→KL=0.238, SAE→KL=0.205")

    all_cos = []
    all_inner = []
    for layer_idx in LAYERS:
        agg = all_results["layer_results"][str(layer_idx)]["aggregate"]
        if agg["n_features_tested"] > 0:
            all_cos.append(agg["corr_cos_kl_mean"])
            all_inner.append(agg["corr_inner_kl_mean"])

    if all_cos:
        cos_wins_all = all(c > i for c, i in zip(all_cos, all_inner))
        print(f"\n  Pythia-70M avg: cos→KL={np.mean(all_cos):.4f}, inner→KL={np.mean(all_inner):.4f}")
        print(f"  Cosine beats inner at ALL Pythia layers: {cos_wins_all}")

    print(f"\nFinal results saved to experiments/exp6_results.json")


if __name__ == "__main__":
    main()
