"""
Experiment 57: Dictionary Size x Model Size Scaling Matrix
==========================================================
Trains standard + adaptive_l2 SAEs across model sizes (1.7B/4B/8B) and expansion
ratios (4x/8x/16x). Saprmarks recipe, 50M tokens, K=80. SAEBench sparse probing.
Results accumulate into experiments/exp57_results.json.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp57_scaling_matrix.py --model 1.7b --all
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp57_scaling_matrix.py --model 4b --expansion 8
    CUDA_VISIBLE_DEVICES=0 python3 experiments/exp57_scaling_matrix.py --model 1.7b --all --eval-only
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

# Disable cuDNN SDPA backend — broken on H100 with driver 595.58 / cuDNN 9.1
torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# === Model Configurations ===

MODEL_CONFIGS = {
    "1.7b": {"model_name": "Qwen/Qwen3-1.7B", "d_model": 2048, "num_layers": 28, "hook_layer": 14},
    "4b": {"model_name": "Qwen/Qwen3-4B", "d_model": 2560, "num_layers": 36, "hook_layer": 18},
    "8b": {"model_name": "Qwen/Qwen3-8B", "d_model": 4096, "num_layers": 64, "hook_layer": 18},
}

EXPANSION_RATIOS = [4, 8, 16]

# === Training Constants (saprmarks recipe) ===

DEVICE = "cuda"
DTYPE = torch.bfloat16

K = 80
LR = 5e-5
BATCH_SIZE = 2048
WARMUP_STEPS = 1000
AUXK_ALPHA = 1 / 32
DEAD_FEATURE_THRESHOLD = 10_000_000
THRESHOLD_BETA = 0.999
THRESHOLD_START_STEP = 1000
SEED = 42
LOG_EVERY = 500
NORM_EPS = 1e-8

N_TRAIN_TOKENS = 50_000_000
N_EVAL_TOKENS = 2_000_000
CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 4
OUTLIER_MULTIPLIER = 10.0
BUFFER_TOKENS = 500_000

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
DECAY_START = int(0.8 * N_STEPS)
BUFFER_BATCHES = BUFFER_TOKENS // BATCH_SIZE

RESULTS_PATH = Path("experiments/exp57_results.json")


# === Geometric median ===

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


# === Decoder norm helpers ===

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


# === SAE Architectures ===

class BatchTopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)
            self.b_enc.zero_()

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_active=False):
        pre_acts = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)
        if return_active:
            active_indices = encoded.sum(0) > 0
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


class AdaptiveCosineBatchTopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)
            self.b_enc.zero_()

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def encode(self, x, return_active=False):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        pre_acts = scale * cos_sim + self.b_enc
        post_relu = F.relu(pre_acts)
        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)
        if return_active:
            active_indices = encoded.sum(0) > 0
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


# === Auxiliary k-loss ===

def get_auxiliary_loss(residual, post_relu_acts, num_tokens_since_fired, d_model, d_sae):
    dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return residual.new_zeros(()), n_dead
    top_k_aux = d_model // 2
    k_aux = min(top_k_aux, n_dead)
    auxk_latents = torch.where(
        dead_mask[None], post_relu_acts,
        torch.tensor(-torch.inf, device=post_relu_acts.device)
    )
    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
    auxk_buffer = torch.zeros_like(post_relu_acts)
    auxk_acts_BF = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
    return auxk_acts_BF, n_dead


# === LR Schedule ===

def make_lr_schedule(total_steps, warmup_steps, decay_start):
    def schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step >= decay_start:
            return (total_steps - step) / max(total_steps - decay_start, 1)
        return 1.0
    return schedule


# === Streaming Activation Collection ===

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


class ActivationStream:
    def __init__(self, model, tokenizer, layer_idx, seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.seed = seed
        self.buffer = None
        self._text_iter = None
        self._init_dataset()

    def _init_dataset(self):
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT",
            split="train", streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        self._text_iter = iter(ds)

    def fill_buffer(self):
        all_acts = []
        tokens_collected = 0
        while tokens_collected < BUFFER_TOKENS:
            batch_texts = []
            for _ in range(COLLECTION_BATCH_SIZE):
                try:
                    row = next(self._text_iter)
                    if len(row["text"]) > 50:
                        batch_texts.append(row["text"][:8192])
                except StopIteration:
                    self._init_dataset()
                    row = next(self._text_iter)
                    if len(row["text"]) > 50:
                        batch_texts.append(row["text"][:8192])
            if not batch_texts:
                continue
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=CTX_LEN,
            ).to(DEVICE)
            acts = _collect_layer_acts(self.model, self.layer_idx, inputs)
            flat = acts[inputs["attention_mask"].bool()]
            norms = flat.float().norm(dim=-1)
            median = norms.median()
            if median > 0:
                flat = flat[norms < median * OUTLIER_MULTIPLIER]
            all_acts.append(flat.to("cpu", dtype=DTYPE))
            tokens_collected += flat.shape[0]
        self.buffer = torch.cat(all_acts, dim=0)[:BUFFER_TOKENS]
        perm = torch.randperm(self.buffer.shape[0])
        self.buffer = self.buffer[perm]
        return self.buffer.shape[0]

    def get_batch(self, batch_idx):
        start = (batch_idx * BATCH_SIZE) % self.buffer.shape[0]
        end = start + BATCH_SIZE
        if end > self.buffer.shape[0]:
            idx = torch.cat([
                torch.arange(start, self.buffer.shape[0]),
                torch.arange(0, end - self.buffer.shape[0]),
            ])
        else:
            idx = torch.arange(start, end)
        return self.buffer[idx].to(DEVICE, dtype=torch.float32)


# === Training ===

def train_sae(name, sae, stream, save_dir, d_model, d_sae, layer):
    tag = f"{name}/L{layer}"
    print(f"\n  Training {tag} | d_sae={d_sae}, k={K}, lr={LR}, "
          f"{N_TRAIN_TOKENS:,} tokens ({N_STEPS} steps)")

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    schedule_fn = make_lr_schedule(N_STEPS, WARMUP_STEPS, DECAY_START)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    num_tokens_since_fired = torch.zeros(d_sae, dtype=torch.long, device=DEVICE)
    sae.train()
    log = []
    b_dec_initialized = False
    t0 = time.time()
    global_step = 0

    while global_step < N_STEPS:
        stream.fill_buffer()
        steps_in_buffer = min(BUFFER_BATCHES, N_STEPS - global_step)

        for buf_step in range(steps_in_buffer):
            batch = stream.get_batch(buf_step)

            if not b_dec_initialized:
                with torch.no_grad():
                    median = geometric_median(batch)
                    sae.b_dec.data.copy_(median.to(sae.b_dec.dtype))
                b_dec_initialized = True
                print(f"    [{tag}] b_dec init (norm={median.norm():.1f})")

            x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            did_fire = torch.zeros(d_sae, dtype=torch.bool, device=DEVICE)
            did_fire[active_indices] = True
            num_tokens_since_fired += batch.shape[0]
            num_tokens_since_fired[did_fire] = 0

            residual = (batch - x_hat).detach()
            auxk_acts, n_dead = get_auxiliary_loss(
                residual, post_relu_acts, num_tokens_since_fired, d_model, d_sae
            )

            if n_dead > 0:
                x_reconstruct_aux = auxk_acts @ sae.W_dec
                auxk_l2 = (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
                residual_mu = residual.mean(dim=0, keepdim=True)
                loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
                auxk_loss = (auxk_l2 / loss_denom.clamp(min=1e-8)).nan_to_num(0.0)
            else:
                auxk_loss = torch.tensor(0.0, device=DEVICE)

            loss = recon_loss + AUXK_ALPHA * auxk_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if sae.W_dec.grad is not None:
                sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                    sae.W_dec.data, sae.W_dec.grad.data
                )

            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Decoder unit-norm constraint
            set_decoder_norm_to_unit_norm(sae.W_dec.data)

            if global_step >= THRESHOLD_START_STEP:
                with torch.no_grad():
                    active_vals = features[features > 0]
                    if active_vals.numel() > 0:
                        min_active = active_vals.min().float()
                        if sae.threshold < 0:
                            sae.threshold.fill_(min_active)
                        else:
                            sae.threshold.mul_(THRESHOLD_BETA).add_(
                                (1 - THRESHOLD_BETA) * min_active
                            )

            global_step += 1

            if global_step % LOG_EVERY == 0 or global_step == N_STEPS:
                with torch.no_grad():
                    l0 = (features != 0).float().sum(dim=-1).mean().item()
                    total_var = torch.var(batch, dim=0, unbiased=False).sum()
                    resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                    fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                    cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                    dead_frac = (num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD).float().mean().item()

                entry = {
                    "step": global_step, "recon_loss": recon_loss.item(),
                    "auxk_loss": auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                    "total_loss": loss.item(), "l0": l0, "fve": fve,
                    "cos_recon": cos_r, "dead_frac": dead_frac, "n_dead": n_dead,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": global_step * BATCH_SIZE,
                    "threshold": sae.threshold.item(),
                }
                if hasattr(sae, "scale_a"):
                    entry["scale_a"] = sae.scale_a.item()

                log.append(entry)
                elapsed = time.time() - t0
                tok = global_step * BATCH_SIZE
                tok_per_sec = tok / elapsed if elapsed > 0 else 0
                eta = (N_STEPS - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                scale_str = ""
                if hasattr(sae, "scale_a"):
                    scale_str = f" a={sae.scale_a.item():.4f}"
                print(f"    [{tag}] {global_step:>5d}/{N_STEPS} | "
                      f"loss={loss.item():.1f} recon={recon_loss.item():.1f} "
                      f"auxk={auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else 0:.3f} | "
                      f"FVE={fve:.4f} L0={l0:.0f} dead={dead_frac:.3f}({n_dead})"
                      f"{scale_str} | {tok/1e6:.1f}M ETA {eta/3600:.1f}h")

    sae.eval()
    elapsed = time.time() - t0
    print(f"    [{tag}] Done in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

    final_path = save_dir / f"{name}_final.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "num_tokens_since_fired": num_tokens_since_fired,
        "step": global_step,
    }, final_path)
    print(f"    [{tag}] Saved: {final_path}")

    return log


# === Evaluation: Reconstruction ===

@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_data, layer, d_sae):
    tag = f"{name}/L{layer}"
    sae.eval()
    n = eval_data.shape[0]
    recon_losses, cos_sims, l0s = [], [], []
    total_var_sum, resid_var_sum = 0.0, 0.0
    dead_counts = None

    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        recon_losses.append((batch - x_hat).pow(2).sum(dim=-1).mean().item())
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        alive = features.sum(dim=0) != 0
        if dead_counts is None:
            dead_counts = ~alive
        else:
            dead_counts &= ~alive

    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0
    fve = 1 - resid_var_sum / total_var_sum if total_var_sum > 0 else 0

    results = {
        "fve": fve, "cos_recon": float(np.mean(cos_sims)), "mean_l0": float(np.mean(l0s)),
        "dead_frac": dead_frac, "alive_count": alive_count,
    }
    print(f"    [{tag}] FVE={fve:.4f} dead={dead_frac:.3f} "
          f"alive={alive_count:,} L0={np.mean(l0s):.1f}")
    return results


# === Evaluation: SAEBench Sparse Probing ===

def wrap_for_saebench(name, sae, model_name, layer, device, dtype):
    from benchmarks.adapter import BenchSAE

    W_dec = F.normalize(sae.W_dec.detach(), dim=1)
    b_enc = sae.b_enc.detach() if hasattr(sae, "b_enc") else torch.zeros(
        sae.d_sae, device=device, dtype=sae.W_enc.dtype
    )
    return BenchSAE(
        W_enc=sae.W_enc.detach().T,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=sae.b_dec.detach(),
        encode_fn=sae.encode,
        decode_fn=sae.decode,
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
    )


def run_sparse_probing(name, sae, model_name, layer, model_tag, expansion, device, dtype):
    from benchmarks.run_saebench import run_saebench

    out_dir = f"benchmarks/eval_results/exp57/{model_tag}_{expansion}x"
    os.makedirs(out_dir, exist_ok=True)

    bench_sae = wrap_for_saebench(name, sae, model_name, layer, device, dtype)
    sae_name = f"exp57-{name}-{model_tag}-{expansion}x"

    print(f"    Running SAEBench sparse probing for {sae_name}...")
    results = run_saebench(
        bench_sae,
        sae_name=sae_name,
        eval_types=["sparse_probing"],
        output_dir=out_dir,
        llm_batch_size=4,
        device=device,
    )
    return results


# === Eval data collection ===

def collect_eval_data(model, tokenizer, layer_idx, n_tokens):
    print(f"  Collecting eval activations for L{layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    # Skip past training region
    for i, _ in enumerate(text_iter):
        if i >= 500_000:
            break

    all_acts = []
    tokens_collected = 0
    while tokens_collected < n_tokens:
        batch_texts = []
        for _ in range(COLLECTION_BATCH_SIZE):
            try:
                row = next(text_iter)
                if len(row["text"]) > 50:
                    batch_texts.append(row["text"][:8192])
            except StopIteration:
                break
        if not batch_texts:
            break
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=CTX_LEN,
        ).to(DEVICE)
        acts = _collect_layer_acts(model, layer_idx, inputs)
        flat = acts[inputs["attention_mask"].bool()]
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            flat = flat[norms < median * OUTLIER_MULTIPLIER]
        all_acts.append(flat.to("cpu", dtype=DTYPE))
        tokens_collected += flat.shape[0]
    result = torch.cat(all_acts, dim=0)[:n_tokens]
    norms = result.float().norm(dim=-1)
    mean_norm = norms.mean().item()
    print(f"    L{layer_idx}: {result.shape[0]:,} tokens in {time.time()-t0:.1f}s "
          f"(norm: mean={mean_norm:.1f}, std={norms.std():.1f})")
    return result, mean_norm


# === Results persistence ===

def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


# === Main: run one cell (model_tag, expansion) ===

def run_cell(model_tag, expansion, model, tokenizer, eval_only=False, train_only=False):
    cfg = MODEL_CONFIGS[model_tag]
    model_name = cfg["model_name"]
    d_model = cfg["d_model"]
    layer = cfg["hook_layer"]
    d_sae = d_model * expansion

    cell_key = f"{model_tag}_{expansion}x"
    save_dir = Path(f"checkpoints/exp57/{cell_key}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Cell: {cell_key} | {model_name} L{layer} | d_sae={d_sae}")
    print(f"{'='*70}")

    variants = [
        ("standard", BatchTopKSAE),
        ("adaptive_l2", AdaptiveCosineBatchTopKSAE),
    ]

    cell_results = {}

    for name, cls in variants:
        print(f"\n  --- {name} ({cell_key}) ---")

        final_path = save_dir / f"{name}_final.pt"

        if eval_only:
            if not final_path.exists():
                print(f"    SKIP: no checkpoint at {final_path}")
                continue
            print(f"    Loading checkpoint: {final_path}")
            sae = cls(d_model, d_sae, K).to(DEVICE)
            ckpt = torch.load(final_path, map_location=DEVICE, weights_only=False)
            sae.load_state_dict(ckpt["state_dict"])
            sae.eval()
        else:
            # Collect eval data (reuse across variants if possible)
            if name == variants[0][0]:
                eval_data, mean_norm = collect_eval_data(model, tokenizer, layer, N_EVAL_TOKENS)
                stream = ActivationStream(model, tokenizer, layer, seed=SEED)

            # Skip training if checkpoint already exists
            if final_path.exists():
                print(f"    Checkpoint exists, loading: {final_path}")
                sae = cls(d_model, d_sae, K).to(DEVICE)
                ckpt = torch.load(final_path, map_location=DEVICE, weights_only=False)
                sae.load_state_dict(ckpt["state_dict"])
                sae.eval()
            else:
                # Train
                torch.manual_seed(SEED)
                sae = cls(d_model, d_sae, K).to(DEVICE)
                training_log = train_sae(name, sae, stream, save_dir, d_model, d_sae, layer)

            # Reconstruction eval
            recon = evaluate_reconstruction(name, sae, eval_data, layer, d_sae)
            cell_results.setdefault(name, {})["reconstruction"] = recon

        # SAEBench sparse probing
        if not train_only:
            try:
                sp_results = run_sparse_probing(
                    name, sae, model_name, layer, model_tag, expansion, DEVICE, DTYPE
                )
                cell_results.setdefault(name, {})["sparse_probing"] = sp_results.get("sparse_probing", {})
            except Exception as e:
                print(f"    WARNING: sparse probing failed: {e}")
                cell_results.setdefault(name, {})["sparse_probing"] = {"error": str(e)}

        if hasattr(sae, "scale_a"):
            cell_results[name]["scale_a"] = sae.scale_a.item()

        # Clean up SAE
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    return cell_results


# === CLI + Main ===

def main():
    parser = argparse.ArgumentParser(
        description="Exp57: Dictionary Size x Model Size scaling matrix"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()),
        help="Model size tag: 1.7b, 4b, or 8b"
    )
    parser.add_argument(
        "--expansion", type=int, choices=EXPANSION_RATIOS, default=None,
        help="Expansion ratio (4, 8, or 16)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all expansion ratios for the given model"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, load existing checkpoints, re-run SAEBench"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Train SAEs and run reconstruction eval, skip SAEBench (avoids CUDA abort)"
    )
    args = parser.parse_args()

    if not args.all and args.expansion is None:
        parser.error("Must specify --expansion or --all")

    expansions = EXPANSION_RATIOS if args.all else [args.expansion]
    model_tag = args.model
    cfg = MODEL_CONFIGS[model_tag]

    print(f"Experiment 57: Scaling Matrix")
    print(f"  Model: {cfg['model_name']} (tag={model_tag})")
    print(f"  Layer: {cfg['hook_layer']}, d_model: {cfg['d_model']}")
    print(f"  Expansions: {expansions}")
    print(f"  Eval-only: {args.eval_only}")
    print(f"  Tokens: {N_TRAIN_TOKENS:,} ({N_STEPS} steps)")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"  Model loaded: {cfg['model_name']}")

    # Load existing results
    all_results = load_results()

    # Run each expansion
    for expansion in expansions:
        cell_key = f"{model_tag}_{expansion}x"
        cell_results = run_cell(model_tag, expansion, model, tokenizer, args.eval_only, args.train_only)

        # Accumulate into results
        all_results[cell_key] = {
            "model": cfg["model_name"],
            "model_tag": model_tag,
            "d_model": cfg["d_model"],
            "layer": cfg["hook_layer"],
            "expansion": expansion,
            "d_sae": cfg["d_model"] * expansion,
            "k": K,
            "n_train_tokens": N_TRAIN_TOKENS,
            "architectures": cell_results,
        }
        save_results(all_results)
        print(f"\n  Results saved for {cell_key}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary — Exp57 Scaling Matrix")
    print(f"{'='*70}")
    print(f"  {'Cell':<15} {'Arch':<15} {'FVE':<8} {'Dead%':<8} {'SP top-1'}")
    print(f"  {'-'*60}")

    for expansion in expansions:
        cell_key = f"{model_tag}_{expansion}x"
        if cell_key not in all_results:
            continue
        archs = all_results[cell_key].get("architectures", {})
        for arch_name, arch_data in archs.items():
            recon = arch_data.get("reconstruction", {})
            sp = arch_data.get("sparse_probing", {})
            fve = recon.get("fve", "?")
            dead = recon.get("dead_frac", "?")
            # Extract sparse probing top-1 accuracy
            sp_top1 = "?"
            if isinstance(sp, dict):
                # SAEBench sparse probing result format varies; try common keys
                for key in ["eval_result_metrics", "metrics", "results"]:
                    if key in sp:
                        inner = sp[key]
                        if isinstance(inner, dict):
                            for k2, v2 in inner.items():
                                if "top1" in k2.lower() or "accuracy" in k2.lower():
                                    sp_top1 = v2
                                    break
                            if sp_top1 != "?":
                                break
                if sp_top1 == "?" and "error" not in sp:
                    sp_top1 = sp  # show raw if can't parse

            fve_str = f"{fve:.4f}" if isinstance(fve, float) else str(fve)
            dead_str = f"{dead:.3f}" if isinstance(dead, float) else str(dead)
            sp_str = f"{sp_top1:.4f}" if isinstance(sp_top1, float) else str(sp_top1)[:40]
            print(f"  {cell_key:<15} {arch_name:<15} {fve_str:<8} {dead_str:<8} {sp_str}")

    print(f"\n  Full results: {RESULTS_PATH}")
    print(f"  Checkpoints: checkpoints/exp57/")


if __name__ == "__main__":
    main()
