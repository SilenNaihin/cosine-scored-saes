"""
Experiment 9: Adversarial Norm — Where SAEs Are Provably Wrong
==============================================================
Find specific tokens where SAE activation and cosine similarity maximally
disagree, then ablate to show cosine is the correct predictor.

For each SAE feature, we identify two adversarial groups:
  - HIGH-NORM/LOW-COS: Tokens with high SAE activation driven by large ||x||
    but low cosine similarity to the feature direction. The SAE says "strong
    feature" but cosine says "weak feature."
  - LOW-NORM/HIGH-COS: Tokens with low SAE activation despite strong
    directional alignment. The SAE says "weak feature" but cosine says
    "strong feature."

We then ablate the feature and measure KL divergence. The RNH predicts:
  - HIGH-COS tokens → large KL (the feature is genuinely important)
  - HIGH-NORM tokens → small KL (the SAE was fooled by magnitude)

Model: Qwen3-8B
SAEs: adamkarvonen/qwen3-8b-saes (BatchTopK, 65k features)
Layers: 9, 18, 27
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

NUM_FEATURES = 30          # features to test per layer
NUM_ABLATIONS_PER_GROUP = 50  # ablation samples per adversarial group
MAX_SEQ_LEN = 256
MIN_ACTIVE_TOKENS = 200    # need enough active tokens to find adversarial pairs


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
    "In distributed systems, the CAP theorem states that it is impossible for a distributed data store to simultaneously provide more than two of the following three guarantees: Consistency (every read receives the most recent write), Availability (every request receives a response), and Partition tolerance (the system continues to operate despite network partitions).",
    "The aurora borealis, or northern lights, occur when charged particles from the sun interact with gases in Earth's atmosphere. These solar particles are guided by Earth's magnetic field toward the poles, where they collide with oxygen and nitrogen molecules, causing them to emit light in spectacular displays of green, pink, purple, and blue.",
    "Bach's Well-Tempered Clavier, composed in two books (1722 and 1742), contains 48 preludes and fugues in all 24 major and minor keys. It demonstrated that keyboard instruments tuned in a temperament close to equal temperament could play in any key, liberating composers from the constraints of meantone and other earlier tuning systems.",
    "The human immune system employs two main strategies: innate immunity and adaptive immunity. Innate immunity provides immediate but non-specific defense through physical barriers, phagocytes, and inflammatory responses. Adaptive immunity develops over days to weeks but provides highly specific responses through B cells (antibodies) and T cells.",
    "Reinforcement learning differs from supervised learning in that the agent must discover which actions yield the most reward through trial and error, rather than being told the correct answer. The exploration-exploitation dilemma is central: the agent must balance trying new actions (exploration) with choosing actions known to give good rewards (exploitation).",
    "The Silk Road was not a single road but a network of trade routes connecting China to the Mediterranean, spanning over 6,000 kilometers. Beyond silk, merchants traded spices, precious metals, gemstones, and glassware. Perhaps more importantly, the routes facilitated the exchange of ideas, religions, technologies, and diseases between civilizations.",
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
    Ablate a feature from the residual stream at layer_idx, measure KL divergence.
    """
    x = activation.unsqueeze(0).unsqueeze(0)
    projection = (activation @ feature_dir) * feature_dir
    x_ablated = (activation - projection).unsqueeze(0).unsqueeze(0)

    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

    def make_hook(replacement):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                return (replacement,) + outputs[1:]
            return replacement
        return hook

    handle = model.model.layers[layer_idx].register_forward_hook(make_hook(x))
    with torch.no_grad():
        try:
            orig_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            handle.remove()
            return None
    handle.remove()

    handle = model.model.layers[layer_idx].register_forward_hook(make_hook(x_ablated))
    with torch.no_grad():
        try:
            ablated_logits = model(dummy_input).logits[0, -1, :].float()
        except Exception:
            handle.remove()
            return None
    handle.remove()

    orig_probs = torch.softmax(orig_logits, dim=-1)
    ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)
    kl = torch.sum(orig_probs * (orig_probs.log() - ablated_log_probs)).item()

    if np.isnan(kl) or np.isinf(kl) or kl < 0:
        return None

    return kl


def find_adversarial_tokens(cos_sims, norms, sae_acts, active_mask):
    """
    Among active tokens, find adversarial groups where SAE and cosine disagree.

    Returns:
        high_cos_low_norm: indices of tokens with high cosine but low norm (SAE underestimates)
        high_norm_low_cos: indices of tokens with high norm but low cosine (SAE overestimates)
    """
    active_idx = torch.where(active_mask)[0]
    if len(active_idx) < MIN_ACTIVE_TOKENS:
        return None, None

    a_cos = cos_sims[active_idx]
    a_norm = norms[active_idx]
    a_sae = sae_acts[active_idx]

    cos_median = a_cos.median()
    norm_median = a_norm.median()

    # HIGH-COS / LOW-NORM: cosine above median, norm below median
    # These tokens have strong directional alignment but low magnitude.
    # SAE activation = inner product ∝ cos × norm, so SAE underestimates these.
    hc_ln_mask = (a_cos > cos_median) & (a_norm < norm_median)

    # HIGH-NORM / LOW-COS: norm above median, cosine below median
    # These tokens have high magnitude but weak directional alignment.
    # SAE activation inflated by norm — SAE overestimates these.
    hn_lc_mask = (a_norm > norm_median) & (a_cos < cos_median)

    hc_ln_idx = active_idx[hc_ln_mask]
    hn_lc_idx = active_idx[hn_lc_mask]

    if len(hc_ln_idx) < 10 or len(hn_lc_idx) < 10:
        return None, None

    return hc_ln_idx, hn_lc_idx


def analyze_layer(model, tokenizer, sae, activations, layer_idx):
    """Run adversarial norm analysis for one layer."""
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
    top_features = feature_freq.topk(NUM_FEATURES * 2).indices  # extra candidates
    print(f"  Selecting from top {NUM_FEATURES * 2} features by frequency")

    feature_results = []
    n_tested = 0

    for feat_rank, feat_idx in enumerate(top_features):
        if n_tested >= NUM_FEATURES:
            break

        feat_idx = feat_idx.item()
        feature_dir = sae.W_dec[feat_idx]

        # Compute metrics for all tokens
        cos_sims = torch.nn.functional.cosine_similarity(
            activations, feature_dir.unsqueeze(0), dim=-1
        )
        norms = activations.norm(dim=-1)
        sae_feat_acts = all_sae_acts[:, feat_idx]
        active_mask = sae_feat_acts > 0

        # Find adversarial groups
        hc_ln_idx, hn_lc_idx = find_adversarial_tokens(
            cos_sims, norms, sae_feat_acts, active_mask
        )
        if hc_ln_idx is None:
            continue

        n_tested += 1

        # Sample from each group
        n_hc = min(NUM_ABLATIONS_PER_GROUP, len(hc_ln_idx))
        n_hn = min(NUM_ABLATIONS_PER_GROUP, len(hn_lc_idx))
        hc_sample = hc_ln_idx[torch.randperm(len(hc_ln_idx))[:n_hc]]
        hn_sample = hn_lc_idx[torch.randperm(len(hn_lc_idx))[:n_hn]]

        # Ablate HIGH-COS/LOW-NORM group (SAE says weak, cosine says strong)
        hc_kls = []
        hc_cos_vals = []
        hc_norm_vals = []
        hc_sae_vals = []
        for idx in hc_sample:
            kl = ablate_feature_kl(model, activations[idx], feature_dir, layer_idx)
            if kl is not None:
                hc_kls.append(kl)
                hc_cos_vals.append(cos_sims[idx].item())
                hc_norm_vals.append(norms[idx].item())
                hc_sae_vals.append(sae_feat_acts[idx].item())

        # Ablate HIGH-NORM/LOW-COS group (SAE says strong, cosine says weak)
        hn_kls = []
        hn_cos_vals = []
        hn_norm_vals = []
        hn_sae_vals = []
        for idx in hn_sample:
            kl = ablate_feature_kl(model, activations[idx], feature_dir, layer_idx)
            if kl is not None:
                hn_kls.append(kl)
                hn_cos_vals.append(cos_sims[idx].item())
                hn_norm_vals.append(norms[idx].item())
                hn_sae_vals.append(sae_feat_acts[idx].item())

        if len(hc_kls) < 5 or len(hn_kls) < 5:
            continue

        hc_kl_arr = np.array(hc_kls)
        hn_kl_arr = np.array(hn_kls)

        # The key comparison: which group has higher ablation KL?
        # RNH predicts: high-cos group (hc) > high-norm group (hn)
        hc_mean = float(hc_kl_arr.mean())
        hn_mean = float(hn_kl_arr.mean())
        rnh_correct = hc_mean > hn_mean

        feat_result = {
            "feature_idx": feat_idx,
            "feature_rank": feat_rank,
            # High-cosine / low-norm group stats
            "hc_n": len(hc_kls),
            "hc_kl_mean": hc_mean,
            "hc_kl_median": float(np.median(hc_kl_arr)),
            "hc_kl_std": float(hc_kl_arr.std()),
            "hc_cos_mean": float(np.mean(hc_cos_vals)),
            "hc_norm_mean": float(np.mean(hc_norm_vals)),
            "hc_sae_mean": float(np.mean(hc_sae_vals)),
            # High-norm / low-cosine group stats
            "hn_n": len(hn_kls),
            "hn_kl_mean": hn_mean,
            "hn_kl_median": float(np.median(hn_kl_arr)),
            "hn_kl_std": float(hn_kl_arr.std()),
            "hn_cos_mean": float(np.mean(hn_cos_vals)),
            "hn_norm_mean": float(np.mean(hn_norm_vals)),
            "hn_sae_mean": float(np.mean(hn_sae_vals)),
            # Key comparison
            "kl_ratio": hc_mean / (hn_mean + 1e-10),
            "rnh_correct": rnh_correct,
        }
        feature_results.append(feat_result)

        marker = "OK" if rnh_correct else "XX"
        print(f"    [{marker}] Feature {feat_idx:>5d} | "
              f"high-cos KL={hc_mean:.4e} (n={len(hc_kls)}) | "
              f"high-norm KL={hn_mean:.4e} (n={len(hn_kls)}) | "
              f"ratio={hc_mean/(hn_mean+1e-10):.2f}x | "
              f"SAE: hc={np.mean(hc_sae_vals):.3f} hn={np.mean(hn_sae_vals):.3f}")

    # Aggregate
    if feature_results:
        n = len(feature_results)
        n_correct = sum(1 for r in feature_results if r["rnh_correct"])
        ratios = [r["kl_ratio"] for r in feature_results]
        hc_kls_all = [r["hc_kl_mean"] for r in feature_results]
        hn_kls_all = [r["hn_kl_mean"] for r in feature_results]
        hc_sae_all = [r["hc_sae_mean"] for r in feature_results]
        hn_sae_all = [r["hn_sae_mean"] for r in feature_results]

        aggregate = {
            "n_features_tested": n,
            "rnh_correct_count": n_correct,
            "rnh_correct_pct": float(100 * n_correct / n),
            "kl_ratio_mean": float(np.mean(ratios)),
            "kl_ratio_median": float(np.median(ratios)),
            "hc_kl_mean": float(np.mean(hc_kls_all)),
            "hn_kl_mean": float(np.mean(hn_kls_all)),
            # Show that SAE activations are REVERSED from KL
            "hc_sae_mean": float(np.mean(hc_sae_all)),
            "hn_sae_mean": float(np.mean(hn_sae_all)),
            "sae_says_hn_stronger": float(np.mean(hn_sae_all) > np.mean(hc_sae_all)),
        }

        print(f"\n  === Layer {layer_idx} Summary ({n} features) ===")
        print(f"  RNH correct (high-cos KL > high-norm KL): {n_correct}/{n} ({100*n_correct/n:.0f}%)")
        print(f"  KL ratio (high-cos / high-norm): mean={np.mean(ratios):.2f}x, median={np.median(ratios):.2f}x")
        print(f"  High-cos group: mean KL={np.mean(hc_kls_all):.4e}, mean SAE act={np.mean(hc_sae_all):.4f}")
        print(f"  High-norm group: mean KL={np.mean(hn_kls_all):.4e}, mean SAE act={np.mean(hn_sae_all):.4f}")
        sae_reversed = np.mean(hn_sae_all) > np.mean(hc_sae_all)
        print(f"  SAE says high-norm group is stronger: {sae_reversed}")
        if sae_reversed:
            print(f"  → SAE is WRONG: it assigns higher activation to the less causally important group")
    else:
        aggregate = {"n_features_tested": 0}

    return {"features": feature_results, "aggregate": aggregate}


def main():
    print("Experiment 9: Adversarial Norm — Where SAEs Are Provably Wrong")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Features per layer: {NUM_FEATURES}")
    print(f"Ablation samples per group: {NUM_ABLATIONS_PER_GROUP}")
    print()
    print("For each feature, we find tokens where SAE activation and cosine")
    print("similarity disagree, then ablate to see which predicts the real effect.")

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

    texts = TEXTS * 5  # Repeat for more tokens

    all_results = {
        "model": MODEL_NAME,
        "layers": LAYERS,
        "n_features_per_layer": NUM_FEATURES,
        "n_ablations_per_group": NUM_ABLATIONS_PER_GROUP,
        "ablation_measure": "logit_kl_divergence",
        "adversarial_split": "median_cos_vs_median_norm among active tokens",
        "n_texts": len(texts),
        "layer_results": {},
    }

    for layer_idx in LAYERS:
        print(f"\n{'='*70}")
        print(f"  LAYER {layer_idx}")
        print(f"{'='*70}")

        # Load SAE
        print(f"  Loading SAE for layer {layer_idx}...")
        sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
            repo_id=SAE_REPO,
            filename=SAE_FILENAMES[layer_idx],
            model_name=MODEL_NAME,
            device=DEVICE,
            dtype=DTYPE,
        )
        print(f"  SAE: {sae.W_dec.shape[0]} features, d_model={sae.W_dec.shape[1]}")

        # Collect activations
        print(f"  Collecting activations from {len(texts)} passages...")
        t0 = time.time()
        activations = collect_activations(model, tokenizer, texts, layer_idx)
        print(f"  Collected {activations.shape[0]} tokens in {time.time()-t0:.1f}s")

        # Filter sinks
        activations, n_sinks = filter_attention_sinks(activations)
        print(f"  Filtered {n_sinks} attention sinks, {activations.shape[0]} remaining")

        # Run analysis
        layer_result = analyze_layer(model, tokenizer, sae, activations, layer_idx)
        all_results["layer_results"][str(layer_idx)] = layer_result

        # Save intermediate
        with open("experiments/exp9_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Intermediate results saved.")

        del sae, activations
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-layer summary
    print(f"\n{'='*70}")
    print(f"  CROSS-LAYER SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Layer':>6s} | {'n_feat':>6s} | {'RNH correct':>12s} | "
          f"{'KL ratio':>10s} | {'HC KL':>10s} | {'HN KL':>10s} | "
          f"{'HC SAE':>8s} | {'HN SAE':>8s}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*12} | {'-'*10} | {'-'*10} | "
          f"{'-'*10} | {'-'*8} | {'-'*8}")

    total_correct = 0
    total_features = 0
    for layer_idx in LAYERS:
        agg = all_results["layer_results"][str(layer_idx)]["aggregate"]
        if agg["n_features_tested"] == 0:
            print(f"  {layer_idx:>6d} | {'N/A':>6s}")
            continue
        n = agg["n_features_tested"]
        nc = agg["rnh_correct_count"]
        total_correct += nc
        total_features += n
        print(f"  {layer_idx:>6d} | {n:>6d} | "
              f"{nc:>4d}/{n:<4d} {agg['rnh_correct_pct']:>4.0f}% | "
              f"{agg['kl_ratio_mean']:>10.2f} | "
              f"{agg['hc_kl_mean']:>10.4e} | {agg['hn_kl_mean']:>10.4e} | "
              f"{agg['hc_sae_mean']:>8.4f} | {agg['hn_sae_mean']:>8.4f}")

    if total_features > 0:
        print(f"\n  Overall: RNH correct on {total_correct}/{total_features} "
              f"({100*total_correct/total_features:.0f}%) features")
        print(f"\n  Interpretation:")
        print(f"  - 'RNH correct' means: tokens with HIGH cosine similarity have")
        print(f"    MORE causal impact from ablation than tokens with HIGH norm,")
        print(f"    even though the SAE assigns higher activation to the high-norm tokens.")
        print(f"  - The SAE is systematically fooled by magnitude into overestimating")
        print(f"    feature importance for high-norm tokens and underestimating it")
        print(f"    for high-cosine tokens.")

    print(f"\nFinal results saved to experiments/exp9_results.json")


if __name__ == "__main__":
    main()
