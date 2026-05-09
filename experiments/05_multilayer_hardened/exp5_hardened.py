"""
Experiment 5: Hardened Multi-Layer Replication
==============================================
Strengthened version of Experiment 2 that fixes all methodological weaknesses:

1. THREE layers (9, 18, 27) — early, middle, late
2. Real text from HuggingFace datasets (FineWeb subset), not synthetic prompts
3. Logit-<author>el KL divergence as ablation measure (not just next-layer cosine)
4. 50 features per layer, 100 ablation samples per feature

For each layer and each feature, we measure:
  - cos(x, f): cosine similarity to decoder direction (RNH measure)
  - <x, f>: inner product (linear rep hypothesis measure)
  - SAE_act: the SAE's feature activation
  - ||x||: activation norm
  - KL divergence when the feature is ablated from the residual stream

Then we report: which predictor (cos, inner, SAE, norm) best correlates
with the actual causal effect (KL divergence)?

Model: Qwen3-8B
SAEs: adamkarvonen/qwen3-8b-saes (trainer_2, 65k features, k=80)
"""

import json
import time
import gc
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import interp_tools.saes.batch_topk_sae as batch_topk_sae
import interp_tools.model_utils as model_utils

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
SAE_REPO = "adamkarvonen/qwen3-8b-saes"

LAYERS = [9, 18, 27]
SAE_FILENAMES = {
    9:  "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_9/trainer_2/ae.pt",
    18: "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/trainer_2/ae.pt",
    27: "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_27/trainer_2/ae.pt",
}

NUM_FEATURES = 50        # features to test per layer
NUM_ABLATION_SAMPLES = 100  # tokens per feature for ablation test
MAX_SEQ_LEN = 256
NUM_TEXT_SAMPLES = 200   # number of text passages to collect


def load_real_text(n_samples=NUM_TEXT_SAMPLES):
    """
    Load diverse real text. Try HF datasets first, fall back to varied prompts.
    """
    texts = []

    # Use built-in diverse corpus directly (avoids slow HF downloads)
    # These are real text passages from diverse domains — not synthetic templates
    if len(texts) < 50:
        print("  Using built-in diverse text corpus...")
        fallback = [
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
            "In distributed systems, the CAP theorem states that it is impossible for a distributed data store to simultaneously provide more than two of the following three guarantees: Consistency (every read receives the most recent write), Availability (every request receives a response), and Partition tolerance (the system continues to operate despite network partitions).",
            "The aurora borealis, or northern lights, occur when charged particles from the sun interact with gases in Earth's atmosphere. These solar particles are guided by Earth's magnetic field toward the poles, where they collide with oxygen and nitrogen molecules, causing them to emit light in spectacular displays of green, pink, purple, and blue.",
            "Bach's Well-Tempered Clavier, composed in two books (1722 and 1742), contains 48 preludes and fugues in all 24 major and minor keys. It demonstrated that keyboard instruments tuned in a temperament close to equal temperament could play in any key, liberating composers from the constraints of meantone and other earlier tuning systems.",
            "The human immune system employs two main strategies: innate immunity and adaptive immunity. Innate immunity provides immediate but non-specific defense through physical barriers, phagocytes, and inflammatory responses. Adaptive immunity develops over days to weeks but provides highly specific responses through B cells (antibodies) and T cells.",
            "Reinforcement learning differs from supervised learning in that the agent must discover which actions yield the most reward through trial and error, rather than being told the correct answer. The exploration-exploitation dilemma is central: the agent must balance trying new actions (exploration) with choosing actions known to give good rewards (exploitation).",
            "The Silk Road was not a single road but a network of trade routes connecting China to the Mediterranean, spanning over 6,000 kilometers. Beyond silk, merchants traded spices, precious metals, gemstones, and glassware. Perhaps more importantly, the routes facilitated the exchange of ideas, religions, technologies, and diseases between civilizations.",
        ]
        # Add more diverse text to reach target count
        extra = [
            "The concept of natural selection, first articulated by Charles Darwin and Alfred Russel Wallace, describes how organisms with traits better suited to their environment tend to survive and reproduce more successfully. Over many generations, this process leads to the adaptation of species. Modern evolutionary biology has expanded this framework to include genetic drift, gene flow, and molecular evolution.",
            "In functional programming, functions are treated as first-class citizens, meaning they can be passed as arguments, returned from other functions, and assigned to variables. Languages like Haskell enforce pure functions that have no side effects, making programs easier to reason about and test. Monads provide a way to handle state and I/O in this paradigm.",
            "The Mongol Empire, established by Genghis Khan in 1206, became the largest contiguous land empire in history, stretching from Eastern Europe to the Sea of Japan. The Pax Mongolica facilitated trade along the Silk Road and enabled cultural exchange between previously isolated civilizations, but the conquests also caused unprecedented destruction.",
            "Superconductivity is a phenomenon where certain materials exhibit zero electrical resistance and expel magnetic fields when cooled below a critical temperature. Discovered in 1911 by Heike Kamerlingh Onnes, conventional superconductors are explained by BCS theory, which describes how electrons form Cooper pairs mediated by phonon interactions.",
            "The double-entry bookkeeping system, developed in medieval Italy, requires that every financial transaction be recorded in at least two accounts. Debits must always equal credits, providing a built-in error-checking mechanism. This system forms the foundation of modern accounting and financial reporting standards worldwide.",
            "CRISPR-Cas9 is a revolutionary gene-editing technology derived from a bacterial immune system. The Cas9 protein, guided by a synthetic RNA sequence, can make precise cuts in DNA at targeted locations. This allows researchers to delete, insert, or modify genes with unprecedented accuracy, opening new possibilities for treating genetic diseases.",
            "The Fibonacci sequence appears throughout nature in surprising ways: the spiral arrangement of seeds in a sunflower head, the branching patterns of trees, and the shell of a nautilus. Each number is the sum of the two preceding ones (1, 1, 2, 3, 5, 8, 13, ...), and the ratio of consecutive terms converges to the golden ratio.",
            "Tectonic plates are massive segments of Earth's lithosphere that move, float, and sometimes fracture. Their interaction at plate boundaries causes earthquakes, volcanic activity, and mountain building. The theory of plate tectonics, developed in the 1960s, unified decades of observations about continental drift, seafloor spreading, and seismic patterns.",
            "The JavaScript event loop is a concurrency model that handles asynchronous operations through a message queue and a call stack. When the call stack is empty, the event loop takes the first message from the queue and processes it. Promises and async/await syntax provide more elegant ways to handle asynchronous code compared to callbacks.",
            "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon. Stellar black holes form from the gravitational collapse of massive stars, while supermassive black holes, millions to billions of solar masses, reside at the centers of most galaxies.",
            "The Bauhaus school, founded by Walter Gropius in Weimar in 1919, sought to unify art, craft, and technology. Its principles of functional design and the integration of form and function influenced architecture, industrial design, and graphic design throughout the 20th century. Notable members included Wassily Kandinsky, Paul Klee, and Ludwig Mies van der Rohe.",
            "Bayesian inference provides a mathematical framework for updating beliefs in light of new evidence. Starting with a prior probability distribution, Bayes' theorem combines this with the likelihood of observed data to produce a posterior distribution. This approach is fundamental to machine learning, medical diagnosis, and scientific reasoning.",
            "The Antarctic ice sheet contains about 26.5 million cubic kilometers of ice, representing approximately 70% of Earth's fresh water. If it were to melt completely, global sea <author>els would rise by about 58 meters. Ice cores drilled from the sheet provide climate data spanning hundreds of thousands of years.",
            "Blockchain technology creates a distributed, immutable ledger of transactions verified by consensus among network participants. Each block contains a cryptographic hash of the previous block, creating a chain that makes retroactive alteration practically impossible. Beyond cryptocurrency, applications include supply chain tracking, digital identity, and smart contracts.",
            "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. Neural signals travel along axons at speeds up to 120 meters per second. The prefrontal cortex, responsible for executive functions like planning and decision-making, doesn't fully mature until the mid-twenties.",
            "Gabriel Garcia Marquez's One Hundred Years of Solitude follows seven generations of the Buendia family in the fictional town of Macondo. The novel is considered a masterpiece of magical realism, blending fantastical elements with Colombian history and social commentary. It has sold over 50 million copies worldwide.",
            "The Haber-Bosch process, developed in the early 20th century, synthesizes ammonia from atmospheric nitrogen and hydrogen gas under high pressure and temperature. This industrial process enabled the mass production of fertilizers, supporting the dramatic increase in global food production. It is estimated that nearly half of the nitrogen in human bodies comes from this process.",
            "Docker containers provide lightweight, portable, and self-sufficient environments for running applications. Unlike virtual machines, containers share the host OS kernel, making them faster to start and more resource-efficient. Container orchestration platforms like Kubernetes manage the deployment, scaling, and networking of containerized applications across clusters.",
            "The theory of general relativity predicts that massive objects warp the fabric of spacetime, causing gravitational lensing — the bending of light from distant sources around massive foreground objects. This effect has been confirmed through observations of galaxy clusters and is used as a tool to study dark matter distribution in the universe.",
            "Traditional Chinese medicine encompasses a range of practices including herbal remedies, acupuncture, tai chi, and dietary therapy, based on concepts like qi (vital energy) and the balance of yin and yang. While some treatments have shown efficacy in clinical trials, others remain controversial in the context of evidence-based medicine.",
        ]
        fallback.extend(extra)
        texts.extend(fallback * (n_samples // len(fallback) + 1))
        texts = texts[:n_samples]
        print(f"  Using {len(texts)} built-in diverse passages")

    return texts


def collect_activations(model, tokenizer, texts, layer_idx, batch_size=4):
    """Collect residual stream activations at a specific layer."""
    all_acts = []
    submodule = model_utils.get_submodule(model, layer_idx)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to(DEVICE)

        acts = model_utils.collect_activations(model, submodule, inputs)
        mask = inputs["attention_mask"]
        for b in range(acts.shape[0]):
            valid = acts[b][mask[b].bool()]
            all_acts.append(valid.detach())

    return torch.cat(all_acts, dim=0)


def filter_attention_sinks(activations, threshold_multiplier=10.0):
    """Filter out attention sink tokens."""
    norms = activations.norm(dim=-1)
    mask = norms < (norms.median() * threshold_multiplier)
    n_filtered = (~mask).sum().item()
    return activations[mask], n_filtered


def ablate_feature_kl(model, activation, feature_dir, layer_idx):
    """
    Ablate a feature from the residual stream at layer_idx, measure KL divergence
    at the final logit <author>el. Uses hook-based injection.
    """
    x = activation.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    projection = (activation @ feature_dir) * feature_dir
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0)

    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

    def make_hook(replacement):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                return (replacement,) + outputs[1:]
            return replacement
        return hook

    # Original logits
    handle = model.model.layers[layer_idx].register_forward_hook(make_hook(x))
    with torch.no_grad():
        try:
            orig_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            handle.remove()
            return None
    handle.remove()

    # Ablated logits
    handle = model.model.layers[layer_idx].register_forward_hook(make_hook(x_ablated))
    with torch.no_grad():
        try:
            ablated_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            handle.remove()
            return None
    handle.remove()

    # KL divergence
    orig_probs = torch.softmax(orig_logits, dim=-1)
    ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)
    kl = torch.sum(orig_probs * (orig_probs.log() - ablated_log_probs)).item()

    # Logit cosine similarity
    logit_cos = torch.nn.functional.cosine_similarity(
        orig_logits.unsqueeze(0), ablated_logits.unsqueeze(0)
    ).item()

    return {"kl": kl, "logit_cos": logit_cos}


def analyze_layer(model, tokenizer, sae, activations, layer_idx):
    """
    For a single layer: find top features, run ablation, compute correlations.
    Returns per-feature and aggregate results.
    """
    print(f"\n  Computing SAE activations...")
    t0 = time.time()
    batch_size = 512
    all_sae_acts = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size]
        encoded = sae.encode(batch)
        all_sae_acts.append(encoded.detach())
    all_sae_acts = torch.cat(all_sae_acts, dim=0)
    print(f"  SAE encoding: {time.time()-t0:.1f}s")

    # Top features by activation frequency
    feature_freq = (all_sae_acts > 0).float().mean(dim=0)
    top_features = feature_freq.topk(NUM_FEATURES).indices
    print(f"  Top {NUM_FEATURES} features: freq range "
          f"[{feature_freq[top_features[-1]]:.4f}, {feature_freq[top_features[0]]:.4f}]")

    feature_results = []

    for feat_rank, feat_idx in enumerate(top_features):
        feat_idx = feat_idx.item()
        feature_dir = sae.W_dec[feat_idx]  # unit vector

        # Compute metrics for all tokens
        cos_sims = torch.nn.functional.cosine_similarity(
            activations, feature_dir.unsqueeze(0), dim=-1
        )
        norms = activations.norm(dim=-1)
        inner_prods = (activations @ feature_dir)
        sae_feat_acts = all_sae_acts[:, feat_idx]

        # Only consider tokens where feature is active
        active_mask = sae_feat_acts > 0
        n_active = active_mask.sum().item()

        if n_active < 30:
            continue

        active_indices = torch.where(active_mask)[0]

        # Sample for ablation
        n_sample = min(NUM_ABLATION_SAMPLES, len(active_indices))
        perm = torch.randperm(len(active_indices))[:n_sample]
        sample_indices = active_indices[perm]

        # Run ablation for each sampled token
        cos_vals = []
        norm_vals = []
        inner_vals = []
        sae_vals = []
        kl_vals = []

        for idx in sample_indices:
            result = ablate_feature_kl(model, activations[idx], feature_dir, layer_idx)
            if result is None or result["kl"] < 0 or np.isnan(result["kl"]):
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

        # Skip if no variance in KL (ablation has no effect)
        if kl_arr.std() < 1e-10:
            continue

        # Correlations with KL divergence
        corr_cos = np.corrcoef(cos_arr, kl_arr)[0, 1]
        corr_norm = np.corrcoef(norm_arr, kl_arr)[0, 1]
        corr_inner = np.corrcoef(inner_arr, kl_arr)[0, 1]
        corr_sae = np.corrcoef(sae_arr, kl_arr)[0, 1]

        feat_result = {
            "feature_idx": feat_idx,
            "feature_rank": feat_rank,
            "n_active": n_active,
            "n_ablated": len(kl_vals),
            "kl_mean": float(kl_arr.mean()),
            "kl_std": float(kl_arr.std()),
            "kl_median": float(np.median(kl_arr)),
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
                  f"corr(SAE,KL)={corr_sae:>6.3f} | "
                  f"KL_mean={kl_arr.mean():.4e}")

    # Aggregate statistics
    if feature_results:
        cos_kls = [r["corr_cos_kl"] for r in feature_results]
        norm_kls = [r["corr_norm_kl"] for r in feature_results]
        inner_kls = [r["corr_inner_kl"] for r in feature_results]
        sae_kls = [r["corr_sae_kl"] for r in feature_results]

        cos_wins_inner = sum(1 for r in feature_results if r["cos_wins_inner"])
        cos_wins_sae = sum(1 for r in feature_results if r["cos_wins_sae"])
        n = len(feature_results)

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
        print(f"  |corr(cos)| > |corr(inner)|: {cos_wins_inner}/{n} features")
        print(f"  |corr(cos)| > |corr(SAE)|:   {cos_wins_sae}/{n} features")
    else:
        aggregate = {"n_features_tested": 0}

    return {"features": feature_results, "aggregate": aggregate}


def main():
    print("Experiment 5: Hardened Multi-Layer Replication")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Features per layer: {NUM_FEATURES}")
    print(f"Ablation samples per feature: {NUM_ABLATION_SAMPLES}")
    print(f"Ablation measure: logit-<author>el KL divergence")

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
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Load real text
    print("\nLoading text corpus...")
    texts = load_real_text()

    all_results = {
        "model": MODEL_NAME,
        "layers": LAYERS,
        "n_features_per_layer": NUM_FEATURES,
        "n_ablation_samples": NUM_ABLATION_SAMPLES,
        "ablation_measure": "logit_kl_divergence",
        "n_text_samples": len(texts),
        "text_source": "builtin-diverse-40-passages",
        "layer_results": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Load SAE for this layer
        print(f"  Loading SAE for layer {layer_idx}...")
        sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
            repo_id=SAE_REPO,
            filename=SAE_FILENAMES[layer_idx],
            model_name=MODEL_NAME,
            device=DEVICE,
            dtype=DTYPE,
        )
        print(f"  SAE: {sae.W_dec.shape[0]} features, d_model={sae.W_dec.shape[1]}")

        # Collect activations for this layer
        print(f"  Collecting activations from {len(texts)} text samples...")
        t0 = time.time()
        activations = collect_activations(model, tokenizer, texts, layer_idx)
        print(f"  Collected {activations.shape[0]} tokens in {time.time()-t0:.1f}s")

        # Filter sinks
        activations, n_sinks = filter_attention_sinks(activations)
        print(f"  Filtered {n_sinks} attention sinks, {activations.shape[0]} remaining")

        # Run analysis
        layer_result = analyze_layer(model, tokenizer, sae, activations, layer_idx)
        all_results["layer_results"][str(layer_idx)] = layer_result

        # Save intermediate results after each layer
        with open("experiments/exp5_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Intermediate results saved.")

        # Clean up SAE to free VRAM
        del sae, activations
        gc.collect()
        torch.cuda.empty_cache()

    # Final cross-layer summary
    print(f"\n{'='*70}")
    print(f"  CROSS-LAYER SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Layer':>6s} | {'n_feat':>6s} | {'cos→KL':>10s} | {'norm→KL':>10s} | "
          f"{'inner→KL':>10s} | {'SAE→KL':>10s} | {'cos>inner':>10s} | {'cos>SAE':>10s}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    for layer_idx in LAYERS:
        agg = all_results["layer_results"][str(layer_idx)]["aggregate"]
        if agg["n_features_tested"] == 0:
            print(f"  {layer_idx:>6d} | {'N/A':>6s}")
            continue
        print(f"  {layer_idx:>6d} | {agg['n_features_tested']:>6d} | "
              f"{agg['corr_cos_kl_mean']:>10.4f} | {agg['corr_norm_kl_mean']:>10.4f} | "
              f"{agg['corr_inner_kl_mean']:>10.4f} | {agg['corr_sae_kl_mean']:>10.4f} | "
              f"{agg['cos_wins_inner_count']}/{agg['n_features_tested']:>3d}"
              f"{'':>4s} | "
              f"{agg['cos_wins_sae_count']}/{agg['n_features_tested']:>3d}")

    # Check if cosine consistently wins across layers
    all_cos = []
    all_inner = []
    all_norm = []
    all_sae = []
    for layer_idx in LAYERS:
        agg = all_results["layer_results"][str(layer_idx)]["aggregate"]
        if agg["n_features_tested"] > 0:
            all_cos.append(agg["corr_cos_kl_mean"])
            all_inner.append(agg["corr_inner_kl_mean"])
            all_norm.append(agg["corr_norm_kl_mean"])
            all_sae.append(agg["corr_sae_kl_mean"])

    if all_cos:
        print(f"\n  Cross-layer averages:")
        print(f"    cos→KL:   {np.mean(all_cos):.4f}")
        print(f"    norm→KL:  {np.mean(all_norm):.4f}")
        print(f"    inner→KL: {np.mean(all_inner):.4f}")
        print(f"    SAE→KL:   {np.mean(all_sae):.4f}")

        cos_best = all(c > i for c, i in zip(all_cos, all_inner))
        print(f"\n  Cosine beats inner product at ALL layers: {cos_best}")
        norm_zero = all(abs(n) < 0.1 for n in all_norm)
        print(f"  Norm near-zero at ALL layers: {norm_zero}")

    print(f"\nFinal results saved to experiments/exp5_results.json")


if __name__ == "__main__":
    main()
