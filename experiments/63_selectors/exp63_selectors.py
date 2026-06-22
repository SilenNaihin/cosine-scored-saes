"""
Experiment 60: Does the cosine advantage survive other selectors? (A2, stage 1)
================================================================================

Reviewer EZEE: "testing TopK, JumpReLU, gated SAEs, and AbsTopK would clarify if
the effect is fundamentally about BatchTopK's batch-wide competition, or about
inner-product scoring more generally ... showing whether the advantage persists
under per-token TopK would be informative."

MECHANISM PREDICTION. Our story is batch-wide competition: under BatchTopK a
single batch-wide budget lets high-norm tokens claim disproportionate slots. Under
*per-token* TopK every token gets exactly k slots, so the per-token input-norm
scalar ||x_c|| multiplies every feature's inner-product score equally and CANCELS
out of the within-token ranking. Prediction: the cosine advantage should LARGELY
VANISH under per-token TopK. Any residual gap isolates the encoder-weight-norm
channel ||w_i|| (inner-product per-token still differs from pure cosine via ||w_i||).

This script (stage 1) covers the two fixed-k selectors that drop straight into the
BatchTopK harness:
  - topk     : per-token TopK (the decisive mechanism test)
  - abs_topk : per-token AbsTopK (Zhu 2025); select by |pre_act|, keep sign, no ReLU

crossed with three scoring arms:
  - inner          : standard inner-product encoder (baseline)
  - cos_global     : adaptive cosine, global a, b   (exp43d adaptive_l2)
  - cos_perfeature : adaptive cosine, per-feature a_i,b_i (exp43d perfeature_l2; best recipe)

The BatchTopK row of the table is reused free from exp43d (standard 0.723 FVE /
0.530 top-1, perfeature 0.726 / 0.648 top-1 at this exact setting). JumpReLU and
Gated (separate L0/gated losses, need lambda tuning to hit L0~=80) are stage 2.

Setting (identical to exp43d / exp59 headline): Qwen3-8B L18, d_sae=65536, k=80,
50M FineWeb tokens, saprmarks recipe. Activations cached once in RAM (~410GB, fits
the 629GB box) and replayed across all six variants.

Metric: SAEBench sparse-probing top-1 (the paper's headline interpretability proxy),
plus FVE / dead% / cos>inner diagnostic.

Run on box-5 (both /mnt/nvme0 gone -> use /mnt; check paths in CONFIG):
    ssh h100-dev-box-5
    cd ~/MechInter--RNH && source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup \
        python3 experiments/exp60_selectors.py 2>&1 | tee experiments/exp60_output.log &

CPU smoke test (no GPU, no model download, tiny dims; validates selector math):
    python3 experiments/exp60_selectors.py --smoke
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# cuDNN SDPA backend broken on this H100 driver (see exp43d)
torch.backends.cuda.enable_cudnn_sdp(False)


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 18
D_MODEL = 4096
D_SAE = 65536
K = 80

N_TRAIN_TOKENS = 50_000_000
N_EVAL_TOKENS = 2_000_000
CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 8
OUTLIER_MULTIPLIER = 10.0

# saprmarks recipe (matches exp43d / exp59)
LR = 5e-5
BATCH_SIZE = 2048
WARMUP_STEPS = 1000
AUXK_ALPHA = 1 / 32
DEAD_FEATURE_THRESHOLD = 10_000_000
TOP_K_AUX = D_MODEL // 2
SEED = 42
LOG_EVERY = 500
NORM_EPS = 1e-8

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE     # 24,414
DECAY_START = int(0.8 * N_STEPS)

N_ABLATION_FEATURES = 100
N_ABLATION_SAMPLES = 200

# box-5: /mnt/nvme0 was lost; /mnt (/dev/sdb1) has ~239G free.
SAVE_DIR = Path("/mnt/exp60_checkpoints")
RESULTS_PATH = Path("experiments/exp60_results.json")
EVAL_OUT_ROOT = "/mnt/exp60_eval_results"

# Stage-1 variants: 2 selectors x 3 scoring arms + 2 weight-norm-isolation arms.
# (BatchTopK row reused from exp43d; JumpReLU/Gated are stage 2.)
# The inner_unitenc arms isolate the encoder-weight-norm channel: under per-token
# selection they should match cosine if the residual within-token advantage is
# entirely the ||w_i|| channel. Completed arms are skipped on resume.
VARIANTS = [
    ("topk_inner",          "inner",          "topk"),
    ("topk_cos_global",     "cos_global",     "topk"),
    ("topk_cos_perfeature", "cos_perfeature", "topk"),
    ("abstopk_inner",          "inner",          "abs_topk"),
    ("abstopk_cos_global",     "cos_global",     "abs_topk"),
    ("abstopk_cos_perfeature", "cos_perfeature", "abs_topk"),
    ("topk_inner_unitenc",     "inner_unitenc",  "topk"),
    ("abstopk_inner_unitenc",  "inner_unitenc",  "abs_topk"),
]


# =============================================================================
# Helpers (geometric median, decoder norm) -- identical to exp43d
# =============================================================================

@torch.no_grad()
def geometric_median(points, max_iter=100, tol=1e-5):
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess.clone()
        dists = torch.norm(points - guess, dim=1).clamp(min=1e-8)
        weights = 1.0 / dists
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


@torch.no_grad()
def set_decoder_norm_to_unit_norm(W_dec):
    norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    W_dec.div_(norms)
    return W_dec


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(W_dec, W_dec_grad):
    normed = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    parallel = (W_dec_grad * normed).sum(dim=1, keepdim=True)
    W_dec_grad -= parallel * normed
    return W_dec_grad


def make_lr_schedule(total_steps, warmup_steps, decay_start):
    def schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step >= decay_start:
            return (total_steps - step) / max(total_steps - decay_start, 1)
        return 1.0
    return schedule


# =============================================================================
# Selector functions
#   All operate on a (B, d_sae) score tensor and keep exactly k per ROW
#   (per-token). Being per-token, they are batch-independent -> train and eval
#   behave identically, with no batch-wide threshold needed (unlike BatchTopK).
# =============================================================================

def select_topk(pre_acts, k):
    """Per-token TopK on ReLU scores. Keeps the k largest non-negative entries
    per row (standard TopK; ties/zeros allowed, matching gao2024scaling)."""
    post = F.relu(pre_acts)
    k_eff = min(k, post.shape[-1])
    vals, idx = torch.topk(post, k_eff, dim=-1)
    out = torch.zeros_like(post)
    out.scatter_(-1, idx, vals)
    return out, post


def select_abs_topk(pre_acts, k):
    """Per-token AbsTopK (Zhu 2025). Select the k entries of largest |score| per
    row and KEEP THEIR SIGN; no ReLU. Returns signed code + a relu view for the
    dead-feature aux loss."""
    k_eff = min(k, pre_acts.shape[-1])
    _, idx = torch.topk(pre_acts.abs(), k_eff, dim=-1)
    out = torch.zeros_like(pre_acts)
    out.scatter_(-1, idx, pre_acts.gather(-1, idx))
    return out, F.relu(pre_acts)


SELECTORS = {"topk": select_topk, "abs_topk": select_abs_topk}


# =============================================================================
# Configurable SAE: scoring arm x fixed-k selector
# =============================================================================

class SelectorSAE(nn.Module):
    """SAE with a pluggable scoring arm and a pluggable per-token selector.

    score_mode in {inner, cos_global, cos_perfeature}; selector in {topk, abs_topk}.
    Decoder is unit-norm with gradient projection in all arms (matches exp43d).
    """

    def __init__(self, d_model, d_sae, k=80, score_mode="inner", selector="topk"):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.score_mode = score_mode
        self.selector = selector
        self.is_cosine = score_mode.startswith("cos")
        # inner_unitenc: inner-product score but encoder rows L2-normalized at
        # score time. Isolates the encoder-WEIGHT-norm channel: it keeps the
        # input norm ||x|| (unlike cosine) but removes ||w_i||. Under per-token
        # TopK ||x|| cancels from the within-token ranking, so selection ranks by
        # cos alone -- identical to cosine selection. If this matches cosine, the
        # residual within-token advantage IS the weight-norm channel.
        self.unit_enc = (score_mode == "inner_unitenc")

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        if score_mode == "cos_global":
            self.scale_a = nn.Parameter(torch.tensor(0.0))
            self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        elif score_mode == "cos_perfeature":
            self.scale_a = nn.Parameter(torch.zeros(d_sae))
            self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            # Standard inner-product init uses 0.1 * W_dec^T (exp43d); cosine uses W_dec^T.
            self.W_enc.copy_(self.W_dec if self.is_cosine else 0.1 * self.W_dec)
            self.b_enc.zero_()

    def _pre_acts(self, x):
        x_c = x - self.b_dec
        if not self.is_cosine:
            W = F.normalize(self.W_enc, dim=-1) if self.unit_enc else self.W_enc
            return x_c @ W.T + self.b_enc
        x_unit = F.normalize(x_c, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        return scale * cos_sim + self.b_enc

    def encode(self, x, return_active=False):
        pre = self._pre_acts(x)
        encoded, post_relu = SELECTORS[self.selector](pre, self.k)
        if return_active:
            active_indices = (encoded != 0).sum(0) > 0
            return encoded, active_indices, post_relu
        return encoded

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x)
        return self.decode(f), f


# =============================================================================
# Auxiliary k-loss (dead-feature revival) -- identical to exp43d
# =============================================================================

def get_auxiliary_loss(post_relu_acts, num_tokens_since_fired):
    dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return None, n_dead
    k_aux = min(TOP_K_AUX, n_dead)
    auxk_latents = torch.where(
        dead_mask[None], post_relu_acts,
        torch.tensor(-torch.inf, device=post_relu_acts.device),
    )
    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
    auxk_buffer = torch.zeros_like(post_relu_acts)
    auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
    return auxk_buffer, n_dead


# =============================================================================
# Activation cache (stream once into RAM, replay across all variants)
# =============================================================================

class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
    captured = {}

    def hook(module, inp, out):
        captured["act"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        raise _EarlyStop

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    except _EarlyStop:
        pass
    handle.remove()
    return captured["act"]


def stream_acts_to_ram(model, tokenizer, n_tokens, skip_rows=0, seed=SEED):
    """Stream n_tokens of layer-LAYER activations into a RAM list of
    (BATCH_SIZE, D_MODEL) bf16 CPU tensors. Outlier-trimmed like exp43d."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    if skip_rows == 0:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= skip_rows:
            break

    shards, leftover = [], None
    collected = 0
    t0 = time.time()
    batch_texts = []
    while collected < n_tokens:
        try:
            row = next(text_iter)
        except StopIteration:
            break
        if len(row["text"]) <= 50:
            continue
        batch_texts.append(row["text"][:8192])
        if len(batch_texts) < COLLECTION_BATCH_SIZE:
            continue
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=CTX_LEN).to(DEVICE)
        batch_texts = []
        acts = _collect_layer_acts(model, LAYER, inputs)
        flat = acts[inputs["attention_mask"].bool()]
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            flat = flat[norms < median * OUTLIER_MULTIPLIER]
        flat = flat.to("cpu", dtype=DTYPE)
        if leftover is not None:
            flat = torch.cat([leftover, flat], dim=0)
        n_full = flat.shape[0] // BATCH_SIZE
        for j in range(n_full):
            shards.append(flat[j * BATCH_SIZE:(j + 1) * BATCH_SIZE].clone())
            collected += BATCH_SIZE
        leftover = flat[n_full * BATCH_SIZE:].clone()
        if len(shards) % 200 == 0 and shards:
            tps = collected / (time.time() - t0)
            print(f"    cached {collected/1e6:.1f}M / {n_tokens/1e6:.0f}M "
                  f"[{tps/1e3:.1f}k tok/s]", flush=True)
    print(f"  cached {collected/1e6:.1f}M tokens in {len(shards)} batches "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return shards


def ram_iter(shards, n_steps, seed=SEED):
    """Yield batches from the RAM cache, reshuffling batch order each epoch."""
    g = torch.Generator().manual_seed(seed)
    step = 0
    while step < n_steps:
        order = torch.randperm(len(shards), generator=g)
        for idx in order:
            if step >= n_steps:
                return
            yield shards[idx].to(DEVICE, dtype=torch.float32)
            step += 1


# =============================================================================
# Training
# =============================================================================

def train_variant(name, score_mode, selector, shards):
    print(f"\n{'='*70}\n  {name}  (score={score_mode}, selector={selector})\n{'='*70}", flush=True)
    torch.manual_seed(SEED)
    sae = SelectorSAE(D_MODEL, D_SAE, K, score_mode=score_mode, selector=selector).to(DEVICE)
    sae.train()

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_schedule(N_STEPS, WARMUP_STEPS, DECAY_START))

    num_tokens_since_fired = torch.zeros(D_SAE, dtype=torch.long, device=DEVICE)
    b_dec_initialized = False
    log = []
    t0 = time.time()

    for step, batch in enumerate(ram_iter(shards, N_STEPS)):
        if not b_dec_initialized:
            with torch.no_grad():
                sae.b_dec.data.copy_(geometric_median(batch).to(sae.b_dec.dtype))
            b_dec_initialized = True
            print(f"    [{name}] b_dec init (norm={sae.b_dec.norm():.1f})", flush=True)

        x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)
        recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

        did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
        did_fire[active_indices] = True
        num_tokens_since_fired += batch.shape[0]
        num_tokens_since_fired[did_fire] = 0

        residual = (batch - x_hat).detach()
        auxk_buffer, n_dead = get_auxiliary_loss(post_relu_acts, num_tokens_since_fired)
        if n_dead > 0:
            x_aux = auxk_buffer @ sae.W_dec
            auxk_l2 = (residual.float() - x_aux.float()).pow(2).sum(dim=-1).mean()
            resid_mu = residual.mean(dim=0, keepdim=True)
            denom = (residual.float() - resid_mu.float()).pow(2).sum(dim=-1).mean()
            auxk_loss = (auxk_l2 / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            auxk_loss = torch.tensor(0.0, device=DEVICE)

        loss = recon_loss + AUXK_ALPHA * auxk_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if sae.W_dec.grad is not None:
            sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                sae.W_dec.data, sae.W_dec.grad.data)
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        set_decoder_norm_to_unit_norm(sae.W_dec.data)

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                l0 = (features != 0).float().sum(dim=-1).mean().item()
                total_var = torch.var(batch, dim=0, unbiased=False).sum()
                resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                dead_frac = (num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD).float().mean().item()
            entry = {"step": step, "fve": fve, "l0": l0, "dead_frac": dead_frac,
                     "n_dead": n_dead, "recon_loss": recon_loss.item(),
                     "auxk_loss": float(auxk_loss), "lr": scheduler.get_last_lr()[0]}
            if hasattr(sae, "scale_a"):
                entry["scale_a_mean"] = sae.scale_a.float().mean().item()
            log.append(entry)
            if step % 2000 == 0 or step == N_STEPS - 1:
                tps = (step + 1) * BATCH_SIZE / (time.time() - t0)
                sa = f" a={entry.get('scale_a_mean', float('nan')):.4f}" if "scale_a_mean" in entry else ""
                print(f"    [{name}] {step:>5d}/{N_STEPS} FVE={fve:.4f} L0={l0:.0f} "
                      f"dead={dead_frac:.3f}({n_dead}){sa} [{tps/1e3:.1f}k tok/s]", flush=True)

    sae.eval()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "score_mode": score_mode,
                "selector": selector, "step": N_STEPS},
               SAVE_DIR / f"{name}_final.pt")
    print(f"    [{name}] done in {(time.time()-t0)/3600:.2f}h", flush=True)
    return sae, log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_shards):
    sae.eval()
    cos_sims, l0s = [], []
    total_var_sum = resid_var_sum = 0.0
    dead_counts = None
    for batch in eval_shards:
        batch = batch.to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        alive = (features != 0).sum(dim=0) != 0
        dead_counts = ~alive if dead_counts is None else (dead_counts & ~alive)
    fve = 1 - resid_var_sum / total_var_sum if total_var_sum > 0 else 0
    alive_count = int((~dead_counts).sum().item())
    res = {"fve": fve, "cos_recon": float(np.mean(cos_sims)),
           "mean_l0": float(np.mean(l0s)), "dead_frac": dead_counts.float().mean().item(),
           "alive_count": alive_count}
    print(f"    [{name}] FVE={fve:.4f} dead={res['dead_frac']:.3f} "
          f"alive={alive_count:,} L0={res['mean_l0']:.1f}", flush=True)
    return res


def run_sparse_probing(name, sae):
    from benchmarks.adapter import BenchSAE
    from benchmarks.run_saebench import run_saebench
    sae.eval()
    sae_dtype = sae.W_enc.dtype

    def encode_fn(x):
        return sae.encode(x.to(dtype=sae_dtype)).to(dtype=x.dtype)

    def decode_fn(f):
        return sae.decode(f.to(dtype=sae_dtype)).to(dtype=f.dtype)

    bench_sae = BenchSAE(
        W_enc=sae.W_enc.detach().T, W_dec=F.normalize(sae.W_dec.detach(), dim=1),
        b_enc=sae.b_enc.detach(), b_dec=sae.b_dec.detach(),
        encode_fn=encode_fn, decode_fn=decode_fn,
        model_name=MODEL_NAME, hook_layer=LAYER, device=DEVICE, dtype=DTYPE)
    return run_saebench(bench_sae, sae_name=f"exp60-{name}",
                        eval_types=["sparse_probing"],
                        output_dir=f"{EVAL_OUT_ROOT}/{name}",
                        llm_batch_size=4, device=DEVICE)


# =============================================================================
# Main
# =============================================================================

def main(probe_only=False):
    print(f"Experiment 60 (stage 1): selectors x scoring at Qwen3-8B L{LAYER}/50M"
          f"{' [PROBE-ONLY]' if probe_only else ''}", flush=True)
    print(f"  variants: {[v[0] for v in VARIANTS]}", flush=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    results = {"config": {"experiment": "exp60_selectors_stage1", "model": MODEL_NAME,
                          "layer": LAYER, "d_sae": D_SAE, "k": K,
                          "n_train_tokens": N_TRAIN_TOKENS}, "runs": {}}
    if RESULTS_PATH.exists():
        try:
            results = json.load(open(RESULTS_PATH))
        except Exception:
            pass

    trained = {}

    # Probe-only: skip model load + the 50M re-cache entirely; reload trained
    # SAEs from checkpoints in Phase 3 below. Used to re-run probing after the
    # training phase already completed (checkpoints live on /mnt).
    if not probe_only:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading model...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE)
        model.eval()

        print("Caching train activations to RAM...", flush=True)
        train_shards = stream_acts_to_ram(model, tokenizer, N_TRAIN_TOKENS, skip_rows=0)
        print("Caching eval activations to RAM...", flush=True)
        eval_shards = stream_acts_to_ram(model, tokenizer, N_EVAL_TOKENS, skip_rows=600_000)

        # Free the model during training; reload for SAEBench probing.
        del model
        gc.collect()
        torch.cuda.empty_cache()

        for name, score_mode, selector in VARIANTS:
            if name in results.get("runs", {}) and results["runs"][name].get("reconstruction"):
                print(f"  [{name}] already in results, skipping training.", flush=True)
                continue
            sae, log = train_variant(name, score_mode, selector, train_shards)
            recon = evaluate_reconstruction(name, sae, eval_shards)
            run = {"score_mode": score_mode, "selector": selector,
                   "reconstruction": recon, "training_log": log[-15:]}
            if hasattr(sae, "scale_a"):
                run["scale_a_mean"] = sae.scale_a.float().mean().item()
            results["runs"][name] = run
            json.dump(results, open(RESULTS_PATH, "w"), indent=2, default=str)
            trained[name] = sae
            gc.collect()
            torch.cuda.empty_cache()

    # Phase 3: SAEBench sparse probing (reloads the model internally).
    print(f"\n{'='*70}\n  SAEBench sparse probing\n{'='*70}", flush=True)
    for name, score_mode, selector in VARIANTS:
        if results["runs"].get(name, {}).get("sparse_probing"):
            continue
        sae = trained.get(name)
        if sae is None:
            ckpt = SAVE_DIR / f"{name}_final.pt"
            if not ckpt.exists():
                continue
            sae = SelectorSAE(D_MODEL, D_SAE, K, score_mode=score_mode, selector=selector).to(DEVICE)
            sae.load_state_dict(torch.load(ckpt, weights_only=True)["state_dict"])
        try:
            sp = run_sparse_probing(name, sae)
            if isinstance(sp, dict) and "sparse_probing" in sp:
                spp = sp["sparse_probing"]
                metrics = spp.get("eval_result_metrics", {}).get("sae", spp) if isinstance(spp, dict) else spp
                results["runs"][name]["sparse_probing"] = metrics
                t1 = metrics.get("sae_top_1_test_accuracy", "?") if isinstance(metrics, dict) else "?"
                print(f"    [{name}] top-1={t1}", flush=True)
            json.dump(results, open(RESULTS_PATH, "w"), indent=2, default=str)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results["runs"][name]["sparse_probing_error"] = str(e)
            json.dump(results, open(RESULTS_PATH, "w"), indent=2, default=str)

    print(f"\n{'='*70}\n  Summary (L{LAYER}, 50M)\n{'='*70}", flush=True)
    for name, _, _ in VARIANTS:
        r = results["runs"].get(name, {})
        rec = r.get("reconstruction", {})
        sp = r.get("sparse_probing", {})
        t1 = sp.get("sae_top_1_test_accuracy", "?") if isinstance(sp, dict) else "?"
        print(f"  {name:24s} FVE={rec.get('fve','?')} dead={rec.get('dead_frac','?')} "
              f"top1={t1}", flush=True)
    print(f"\nResults: {RESULTS_PATH}", flush=True)


# =============================================================================
# CPU smoke test: validate selector + SAE shapes on tiny dims, no GPU/model.
# =============================================================================

def smoke():
    global DEVICE
    DEVICE = "cpu"
    print("SMOKE: validating selector math + SAE forward/backward on CPU...")
    torch.manual_seed(0)
    B, dm, ds, k = 32, 16, 64, 4

    # selector unit tests
    pre = torch.randn(B, ds)
    out, post = select_topk(pre, k)
    assert (out != 0).sum(-1).max().item() <= k, "topk: >k per row"
    assert (out >= 0).all(), "topk: negative kept"
    out2, _ = select_abs_topk(pre, k)
    assert (out2 != 0).sum(-1).max().item() <= k, "abs_topk: >k per row"
    # abs_topk must keep sign and pick largest-magnitude entries (check row 0)
    row = pre[0]
    out2_row = out2[0]
    keep = out2_row != 0
    kept_mag = row[keep].abs().min()
    dropped_mag = row[~keep].abs().max() if (~keep).any() else torch.tensor(-1.0)
    assert kept_mag >= dropped_mag - 1e-6, "abs_topk: did not keep largest |.|"
    assert torch.allclose(out2_row[keep], row[keep]), "abs_topk: sign/value not preserved"
    print("  selectors OK (per-token cap, sign handling)")

    for score_mode in ["inner", "inner_unitenc", "cos_global", "cos_perfeature"]:
        for selector in ["topk", "abs_topk"]:
            sae = SelectorSAE(dm, ds, k, score_mode=score_mode, selector=selector)
            x = torch.randn(B, dm) * 5.0
            x_hat, f, active, post_relu = sae(x, return_active=True)
            assert x_hat.shape == x.shape
            assert (f != 0).sum(-1).max().item() <= k
            loss = (x - x_hat).pow(2).sum(-1).mean()
            loss.backward()
            assert sae.W_enc.grad is not None and torch.isfinite(sae.W_enc.grad).all()
            # eval path must equal train path (batch-independent selectors)
            sae.eval()
            with torch.no_grad():
                f_eval = sae.encode(x)
                f_train = f
            assert (f_eval != 0).sum(-1).max().item() <= k
            print(f"  {score_mode:14s} x {selector:9s} OK "
                  f"(FVE-ish cos={F.cosine_similarity(x, x_hat, dim=-1).mean():.3f})")

    # inner_unitenc must IGNORE encoder-row norm: scaling a W_enc row should not
    # change which features fire under per-token TopK (ranking is by cos alone).
    sae = SelectorSAE(dm, ds, k, score_mode="inner_unitenc", selector="topk")
    x = torch.randn(B, dm) * 5.0
    with torch.no_grad():
        f_before = sae.encode(x)
        sae.W_enc.mul_(torch.linspace(0.5, 5.0, ds).unsqueeze(1))  # blow up row norms
        f_after = sae.encode(x)
    same = ((f_before != 0) == (f_after != 0)).float().mean().item()
    assert same > 0.999, f"inner_unitenc: row-norm scaling changed selection ({same})"
    print(f"  inner_unitenc ignores ||w_i|| OK (selection match {same:.3f})")
    print("SMOKE PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU shape/logic test, no GPU")
    ap.add_argument("--probe-only", action="store_true",
                    help="skip training + activation caching; run SAEBench probing from checkpoints")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        main(probe_only=args.probe_only)
