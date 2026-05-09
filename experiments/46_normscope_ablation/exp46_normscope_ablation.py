"""
Experiment 46: NoC Normalization-Scope Ablation (12 variants from cache)
========================================================================

Tests three orthogonal design choices in the NoC architecture:
  1. Encoder weight unit-norm constraint
  2. Decoder weight unit-norm constraint
  3. Post-decode norm restoration

...crossed with aux-k loss {on, off}. 6 architecture cells × 2 aux conditions
= 12 variants. All trained from a single 10M-token activation cache for fast
iteration (~2-3 hours wall time on A100 vs ~16 days serial).

Variants:
    noc_baseline           (enc=T, dec=T, restore=T) — current NoC
    noc_dec_free_restore   (enc=T, dec=F, restore=T) — free decoder, output rescaled
    noc_dec_free_no_restore(enc=T, dec=F, restore=F) — free decoder + free output norm
    noc_enc_free           (enc=F, dec=T, restore=T) — free encoder rows
    noc_input_only_restore (enc=F, dec=F, restore=T) — both free, output rescaled
    noc_input_only_no_restore (enc=F, dec=F, restore=F) — input-norm-only, no restore
    + each above with _aux (saprmarks aux-k) and _noaux (aux-k disabled).

Run on <gpu-server>:
    ssh <server>     cd ~/MechInter--RNH
    nohup uv run python experiments/exp46_normscope_ablation.py \
        --layer 27 --tokens 10000000 \
        > experiments/exp46_l27.log 2>&1 &
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

# Disable cuDNN SDPA backend if running on H100 with broken driver. Harmless on A100.
torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.activation_cache import (
    CachedActivationStream,
    build_activation_cache,
    cache_exists_and_valid,
    cache_paths,
)


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_SLUG = "qwen3_8b"
D_MODEL = 4096

D_SAE = 65536
K = 80

CTX_LEN = 2048
COLLECTION_BATCH_SIZE = 4
OUTLIER_MULTIPLIER = 10.0
N_EVAL_TOKENS = 500_000  # smaller eval set than exp43 to keep wall time tight

# saprmarks recipe. Throughput is GPU-compute-bound (~0.8k tok/s for a parallel
# group of 6 SAEs at d_sae=65k, k=80 on A100); larger batch sizes don't help
# since per-step compute scales proportionally with batch. Stick with batch=2048
# for memory headroom (batch=8192 brought peak VRAM to 79.6/80 GB at smoke).
LR = 5e-5
BATCH_SIZE = 2048
WARMUP_STEPS = 200  # scaled to ~10% of total steps for 5-10M-token runs
AUXK_ALPHA = 1 / 32
DEAD_FEATURE_THRESHOLD = 10_000_000
TOP_K_AUX = D_MODEL // 2
THRESHOLD_BETA = 0.999
THRESHOLD_START_STEP = 200
SEED = 42
LOG_EVERY = 200
NORM_EPS = 1e-8

BUFFER_TOKENS = 500_000

# Ablation eval scope: 12 SAEs × N_FEAT × N_SAMP × 1 ablated forward each.
# At ~1s per forward, 30×20=600 forwards per SAE × 12 SAEs = ~2h.
# We share the clean forward across SAEs so it's ~N_SAMP forwards once + ablation forwards per SAE.
N_ABLATION_FEATURES = 30
N_ABLATION_SAMPLES = 20

DEFAULT_CACHE_DIR = Path("~/MechInter--RNH/cache").expanduser()
DEFAULT_CHECKPOINT_ROOT = Path("~/MechInter--RNH/checkpoints").expanduser()
RESULTS_PATH = Path("experiments/exp46_results.json")

# Default: what the dead-feature threshold scales to under a smaller token
# budget. Exp43's threshold is 10M tokens (set at 50M training). At 10M we
# need a proportionally smaller threshold to detect dead features.
DEFAULT_DEAD_FEATURE_THRESHOLD_FRACTION = 0.2  # 20% of training horizon


# Sentinel for early-exit forward hooks. Cannot use StopIteration since that has
# special meaning in Python iterators and can mask real bugs.
class _EarlyStop(Exception):
    pass


# =============================================================================
# Decoder norm helpers (reused from exp43)
# =============================================================================

@torch.no_grad()
def set_decoder_norm_to_unit_norm(W_dec):
    norms = W_dec.norm(dim=1, keepdim=True).clamp(min=NORM_EPS)
    W_dec.div_(norms)
    return W_dec


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(W_dec, W_dec_grad):
    normed = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=NORM_EPS)
    parallel = (W_dec_grad * normed).sum(dim=1, keepdim=True)
    W_dec_grad -= parallel * normed
    return W_dec_grad


@torch.no_grad()
def geometric_median(points, max_iter=100, tol=1e-5):
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess.clone()
        dists = torch.norm(points - guess, dim=1).clamp(min=NORM_EPS)
        weights = 1.0 / dists
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


# =============================================================================
# NoC SAE with three normalization knobs
# =============================================================================

class NoCFlexSAE(nn.Module):
    """
    Norm-preserving cosine SAE with three independent boolean knobs:
      - normalize_encoder: unit-normalize W_enc rows in the forward pass
                           AND in post_step (renorm after every optimizer step)
      - normalize_decoder: unit-normalize W_dec rows in the forward pass
                           (training loop also conditionally applies gradient
                            projection + per-step renorm to W_dec)
      - restore_output_norm: rescale the decoded vector to match the input's
                             centered norm before adding b_dec back

    All three default to True (canonical NoC).
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        k: int = 80,
        *,
        normalize_encoder: bool = True,
        normalize_decoder: bool = True,
        restore_output_norm: bool = True,
    ):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.normalize_encoder = bool(normalize_encoder)
        self.normalize_decoder = bool(normalize_decoder)
        self.restore_output_norm = bool(restore_output_norm)

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec)

    def _batch_topk(self, acts):
        batch_size = max(acts.shape[0], 1)
        total_k = min(self.k * batch_size, acts.numel())
        flat = acts.reshape(-1)
        values, indices = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[indices] = values
        return sparse.view_as(acts)

    def _enc_weight(self):
        if self.normalize_encoder:
            return F.normalize(self.W_enc, dim=-1, eps=NORM_EPS)
        return self.W_enc

    def _dec_weight(self):
        if self.normalize_decoder:
            return F.normalize(self.W_dec, dim=-1, eps=NORM_EPS)
        return self.W_dec

    def encode(self, x, return_active=False):
        x_c = x - self.b_dec
        x_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        x_u = x_c / x_norm
        w = self._enc_weight()
        post_relu = F.relu(x_u @ w.T)
        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)
        self._cached_x_norm = x_norm
        if return_active:
            active_indices = encoded.sum(0) > 0
            return encoded, active_indices, post_relu
        return encoded

    def decode(self, f, x_norm=None):
        w = self._dec_weight()
        x_raw = f @ w
        if self.restore_output_norm:
            raw_norm = x_raw.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
            if x_norm is None:
                x_norm = getattr(self, "_cached_x_norm", None)
            if x_norm is not None:
                x_raw = x_raw * (x_norm / raw_norm)
        return x_raw + self.b_dec

    def forward(self, x, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def post_step(self):
        """Apply per-step weight constraints. Called after every optimizer.step()."""
        if self.normalize_encoder:
            self.W_enc.div_(
                self.W_enc.norm(dim=1, keepdim=True).clamp(min=NORM_EPS)
            )
        if self.normalize_decoder:
            set_decoder_norm_to_unit_norm(self.W_dec.data)


# =============================================================================
# Variant matrix
# =============================================================================

ARCH_CELLS = [
    # (name, kwargs to NoCFlexSAE)
    ("noc_baseline",              dict(normalize_encoder=True,  normalize_decoder=True,  restore_output_norm=True)),
    ("noc_dec_free_restore",      dict(normalize_encoder=True,  normalize_decoder=False, restore_output_norm=True)),
    ("noc_dec_free_no_restore",   dict(normalize_encoder=True,  normalize_decoder=False, restore_output_norm=False)),
    ("noc_enc_free",              dict(normalize_encoder=False, normalize_decoder=True,  restore_output_norm=True)),
    ("noc_input_only_restore",    dict(normalize_encoder=False, normalize_decoder=False, restore_output_norm=True)),
    ("noc_input_only_no_restore", dict(normalize_encoder=False, normalize_decoder=False, restore_output_norm=False)),
]


def build_variants(aux_modes: list[bool]) -> list[tuple[str, dict, bool]]:
    """Returns list of (name, sae_kwargs, aux_k_enabled)."""
    variants = []
    for aux in aux_modes:
        suffix = "_aux" if aux else "_noaux"
        for name, kw in ARCH_CELLS:
            variants.append((f"{name}{suffix}", kw, aux))
    return variants


# =============================================================================
# Auxiliary k-loss
# =============================================================================

def get_auxiliary_loss(post_relu_acts, num_tokens_since_fired, dead_threshold):
    dead_mask = num_tokens_since_fired >= dead_threshold
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return None, n_dead
    k_aux = min(TOP_K_AUX, n_dead)
    auxk_latents = torch.where(
        dead_mask[None],
        post_relu_acts,
        torch.tensor(-torch.inf, device=post_relu_acts.device),
    )
    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
    auxk_buffer = torch.zeros_like(post_relu_acts)
    auxk_acts_BF = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
    return auxk_acts_BF, n_dead


# =============================================================================
# LR schedule
# =============================================================================

def make_lr_schedule(total_steps, warmup_steps, decay_start):
    def schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step >= decay_start:
            return (total_steps - step) / max(total_steps - decay_start, 1)
        return 1.0
    return schedule


# =============================================================================
# Per-SAE training state
# =============================================================================

class SAEState:
    """All per-SAE state needed to run one training step in the parallel loop."""

    def __init__(
        self,
        name: str,
        sae: NoCFlexSAE,
        aux_k_enabled: bool,
        n_steps: int,
        save_dir: Path,
        layer: int,
        dead_threshold: int,
    ):
        self.name = name
        self.sae = sae
        self.aux_k_enabled = aux_k_enabled
        self.layer = layer
        self.save_dir = save_dir
        self.dead_threshold = dead_threshold

        self.optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
        schedule_fn = make_lr_schedule(n_steps, WARMUP_STEPS, int(0.8 * n_steps))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule_fn)

        self.num_tokens_since_fired = torch.zeros(
            D_SAE, dtype=torch.long, device=DEVICE
        )
        self.b_dec_initialized = False
        self.log: list[dict] = []
        self.t0 = time.time()


def step_one_sae(state: SAEState, batch: torch.Tensor, global_step: int):
    """Forward+backward+step for one SAE on one shared batch. Returns log entry or None."""
    sae = state.sae

    if not state.b_dec_initialized:
        with torch.no_grad():
            median = geometric_median(batch)
            sae.b_dec.data.copy_(median.to(sae.b_dec.dtype))
        state.b_dec_initialized = True

    x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)
    recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

    did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
    did_fire[active_indices] = True
    state.num_tokens_since_fired += batch.shape[0]
    state.num_tokens_since_fired[did_fire] = 0

    n_dead = 0
    auxk_loss_val = 0.0
    if state.aux_k_enabled:
        residual = (batch - x_hat).detach()
        auxk_acts, n_dead = get_auxiliary_loss(
            post_relu_acts, state.num_tokens_since_fired, state.dead_threshold
        )
        if n_dead > 0 and auxk_acts is not None:
            # Use the same decoder weight policy as the main path so the aux
            # loss reflects the architecture under test (free vs unit-norm dec).
            w_dec_eff = sae._dec_weight()
            x_reconstruct_aux = auxk_acts @ w_dec_eff
            auxk_l2 = (residual.float() - x_reconstruct_aux.float()).pow(2).sum(-1).mean()
            residual_mu = residual.mean(dim=0, keepdim=True)
            loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(-1).mean()
            auxk_loss = (auxk_l2 / loss_denom.clamp(min=1e-8)).nan_to_num(0.0)
            loss = recon_loss + AUXK_ALPHA * auxk_loss
            auxk_loss_val = float(auxk_loss.item())
        else:
            loss = recon_loss
    else:
        loss = recon_loss
        # Still track dead features for diagnostics even when aux-k is off.
        n_dead = int((state.num_tokens_since_fired >= state.dead_threshold).sum())

    state.optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if sae.normalize_decoder and sae.W_dec.grad is not None:
        sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
            sae.W_dec.data, sae.W_dec.grad.data
        )

    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    state.optimizer.step()
    state.scheduler.step()
    sae.post_step()

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

    if global_step % LOG_EVERY == 0:
        with torch.no_grad():
            l0 = (features != 0).float().sum(dim=-1).mean().item()
            total_var = torch.var(batch, dim=0, unbiased=False).sum()
            resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
            fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
            cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
            dead_frac = (
                state.num_tokens_since_fired >= state.dead_threshold
            ).float().mean().item()
            enc_norms = sae.W_enc.norm(dim=1)
            dec_norms = sae.W_dec.norm(dim=1)

        return {
            "step": global_step,
            "recon_loss": float(recon_loss.item()),
            "auxk_loss": auxk_loss_val,
            "total_loss": float(loss.item()),
            "l0": l0,
            "fve": fve,
            "cos_recon": cos_r,
            "dead_frac": dead_frac,
            "n_dead": n_dead,
            "lr": state.scheduler.get_last_lr()[0],
            "enc_norm_mean": float(enc_norms.mean().item()),
            "enc_norm_p90": float(enc_norms.float().quantile(0.9).item()),
            "dec_norm_mean": float(dec_norms.mean().item()),
            "dec_norm_p90": float(dec_norms.float().quantile(0.9).item()),
            "threshold": float(sae.threshold.item()),
        }
    return None


def train_parallel_group(
    states: list[SAEState],
    stream: CachedActivationStream,
    n_steps: int,
    layer: int,
    print_every: int = LOG_EVERY,
):
    """Run all SAEs in `states` in parallel sharing one batch per step from the cache."""
    print(f"\n{'=' * 70}")
    print(f"  Parallel group of {len(states)} SAEs at L{layer}, {n_steps} steps")
    print(f"  Variants: {[s.name for s in states]}")
    print(f"{'=' * 70}\n")

    for s in states:
        s.sae.train()

    global_step = 0
    t_group = time.time()
    while global_step < n_steps:
        stream.fill_buffer()
        steps_in_buffer = min(stream.buffer_batches, n_steps - global_step)

        for buf_step in range(steps_in_buffer):
            batch = stream.get_batch(buf_step)

            for s in states:
                entry = step_one_sae(s, batch, global_step)
                if entry is not None:
                    s.log.append(entry)

            global_step += 1

            if global_step % print_every == 0 or global_step == n_steps:
                elapsed = time.time() - t_group
                tok = global_step * BATCH_SIZE
                tok_per_sec = tok / elapsed if elapsed > 0 else 0
                eta = (n_steps - global_step) * BATCH_SIZE / tok_per_sec if tok_per_sec > 0 else 0
                print(f"\n  --- step {global_step}/{n_steps} ({tok / 1e6:.1f}M tok, "
                      f"{tok_per_sec / 1e3:.1f}k tok/s, ETA {eta / 60:.1f}m) ---")
                for s in states:
                    if s.log:
                        e = s.log[-1]
                        print(
                            f"    [{s.name}] FVE={e['fve']:.3f} L0={e['l0']:.0f} "
                            f"dead={e['dead_frac']:.3f}({e['n_dead']}) "
                            f"recon={e['recon_loss']:.1f} auxk={e['auxk_loss']:.3f} "
                            f"|enc|={e['enc_norm_mean']:.2f} |dec|={e['dec_norm_mean']:.2f}"
                        )
            if global_step >= n_steps:
                break

    for s in states:
        s.sae.eval()
        final_path = s.save_dir / f"{s.name}_L{layer}_final.pt"
        torch.save(
            {
                "state_dict": s.sae.state_dict(),
                "num_tokens_since_fired": s.num_tokens_since_fired,
                "step": global_step,
                "config": {
                    "normalize_encoder": s.sae.normalize_encoder,
                    "normalize_decoder": s.sae.normalize_decoder,
                    "restore_output_norm": s.sae.restore_output_norm,
                    "aux_k_enabled": s.aux_k_enabled,
                },
            },
            final_path,
        )
        print(f"  [{s.name}] Saved {final_path}")

    print(f"\n  Group done in {(time.time() - t_group) / 60:.1f} min")


# =============================================================================
# Eval (FVE / dead% / cos>inner)
# =============================================================================

@torch.no_grad()
def evaluate_reconstruction(name: str, sae: NoCFlexSAE, eval_data: torch.Tensor) -> dict:
    sae.eval()
    n = eval_data.shape[0]
    cos_sims, l0s = [], []
    total_var_sum, resid_var_sum = 0.0, 0.0
    dead_counts = None
    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i : i + BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        total_var_sum += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var_sum += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        alive = features.sum(dim=0) != 0
        if dead_counts is None:
            dead_counts = ~alive
        else:
            dead_counts &= ~alive
    fve = 1 - resid_var_sum / total_var_sum if total_var_sum > 0 else 0
    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0
    alive_count = int((~dead_counts).sum().item()) if dead_counts is not None else 0
    return {
        "fve": fve,
        "cos_recon": float(np.mean(cos_sims)),
        "mean_l0": float(np.mean(l0s)),
        "dead_frac": dead_frac,
        "alive_count": alive_count,
    }


@torch.no_grad()
def collect_ablation_corpus(model, tokenizer, layer, n_samples, ctx_len_eval=1024):
    """
    Collect a small text corpus and pre-compute the (inputs, clean_probs, last_token_act)
    triple for each — these are *shared* across all SAEs in the ablation eval, eliminating
    a 12x redundant model forward pass.
    Returns list of dicts.
    """
    print(f"    [ablation] Collecting {n_samples} texts + clean forwards...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= 600_000:
            break
    corpus = []
    while len(corpus) < n_samples:
        try:
            row = next(text_iter)
        except StopIteration:
            break
        if len(row["text"]) <= 100:
            continue
        text = row["text"][:4096]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=ctx_len_eval
        ).to(DEVICE)

        # Clean forward + capture layer activation at last token in one pass.
        captured = {}

        def hook(module, inp, out):
            captured["act"] = (out[0] if isinstance(out, tuple) else out).detach()

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            outputs = model(**inputs)
        finally:
            handle.remove()
        clean_logits = outputs.logits[0, -1].float()
        clean_probs = F.softmax(clean_logits, dim=-1).clone()
        last_act = captured["act"][0, -1].float().clone()
        full_act = captured["act"][0].float().clone()  # for top-feature aggregation

        corpus.append(
            {
                "inputs": inputs,
                "clean_probs": clean_probs,
                "last_act": last_act,
                "full_act": full_act,
            }
        )
    print(f"    [ablation] Collected {len(corpus)} texts")
    return corpus


@torch.no_grad()
def evaluate_ablation_shared(name, model, sae, layer, corpus, n_features=N_ABLATION_FEATURES):
    """
    Ablation eval reusing pre-computed clean forwards from `corpus`.
    Only the per-feature ablated forward is per-SAE (unavoidable — each
    feature direction is different).
    """
    sae.eval()

    # Find top features by aggregate activation across corpus.
    act_sums = torch.zeros(D_SAE, device=DEVICE)
    for sample in corpus:
        flat = sample["full_act"].reshape(-1, D_MODEL)
        features = sae.encode(flat)
        act_sums += features.sum(dim=0)
    top_features = act_sums.topk(n_features).indices.tolist()

    cos_wins = 0
    cos_kl_corrs, inner_kl_corrs = [], []
    for feat_idx in top_features:
        feat_dir = sae.W_dec[feat_idx]
        feat_dir_unit = feat_dir / feat_dir.norm().clamp(min=NORM_EPS)
        cos_vals, inner_vals, kl_vals = [], [], []
        for sample in corpus:
            act_flat = sample["last_act"]

            cos_sim = F.cosine_similarity(
                act_flat.unsqueeze(0), feat_dir_unit.unsqueeze(0)
            ).item()
            inner_prod = (act_flat * feat_dir).sum().item()

            proj = (act_flat @ feat_dir_unit) * feat_dir_unit
            ablated_act = act_flat - proj

            def ablation_hook(module, inp, out):
                result = out[0] if isinstance(out, tuple) else out
                result = result.clone()
                result[0, -1] = ablated_act.to(result.dtype)
                if isinstance(out, tuple):
                    return (result,) + out[1:]
                return result

            handle = model.model.layers[layer].register_forward_hook(ablation_hook)
            try:
                outputs_abl = model(**sample["inputs"])
            finally:
                handle.remove()
            abl_probs = F.softmax(outputs_abl.logits[0, -1], dim=-1)

            kl = F.kl_div(abl_probs.log(), sample["clean_probs"], reduction="sum").item()
            cos_vals.append(abs(cos_sim))
            inner_vals.append(abs(inner_prod))
            kl_vals.append(kl)
        if len(kl_vals) > 2:
            ca, ia, ka = np.array(cos_vals), np.array(inner_vals), np.array(kl_vals)
            cos_corr = np.corrcoef(ca, ka)[0, 1] if ca.std() > 0 else 0
            inner_corr = np.corrcoef(ia, ka)[0, 1] if ia.std() > 0 else 0
            cos_kl_corrs.append(cos_corr)
            inner_kl_corrs.append(inner_corr)
            if cos_corr > inner_corr:
                cos_wins += 1
    return {
        "n_features": len(cos_kl_corrs),
        "cos_wins_inner": f"{cos_wins}/{len(cos_kl_corrs)}",
        "cos_wins_pct": float(cos_wins / max(len(cos_kl_corrs), 1)),
        "cos_kl_mean": float(np.mean(cos_kl_corrs)) if cos_kl_corrs else 0,
        "inner_kl_mean": float(np.mean(inner_kl_corrs)) if inner_kl_corrs else 0,
    }


@torch.no_grad()
def evaluate_ablation(name, model, tokenizer, sae, layer, n_features=N_ABLATION_FEATURES,
                      n_samples=N_ABLATION_SAMPLES):
    """RNH diagnostic: cos vs inner product as causal predictor."""
    sae.eval()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= 600_000:
            break
    texts = []
    while len(texts) < n_samples:
        try:
            row = next(text_iter)
            if len(row["text"]) > 100:
                texts.append(row["text"][:4096])
        except StopIteration:
            break

    # Find top features by aggregate activation
    act_sums = torch.zeros(D_SAE, device=DEVICE)
    for text in texts[:30]:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=CTX_LEN
        ).to(DEVICE)
        captured = {}

        def hook(module, inp, out):
            captured["act"] = out[0] if isinstance(out, tuple) else out
            raise _EarlyStop

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(**inputs)
        except _EarlyStop:
            pass
        handle.remove()
        flat = captured["act"].reshape(-1, D_MODEL).float()
        features = sae.encode(flat)
        act_sums += features.sum(dim=0)
    top_features = act_sums.topk(n_features).indices.tolist()

    cos_wins = 0
    cos_kl_corrs, inner_kl_corrs = [], []
    for feat_idx in top_features:
        feat_dir = sae.W_dec[feat_idx]
        feat_dir_unit = feat_dir / feat_dir.norm().clamp(min=NORM_EPS)
        cos_vals, inner_vals, kl_vals = [], [], []
        for text in texts[:n_samples]:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=CTX_LEN
            ).to(DEVICE)
            with torch.no_grad():
                outputs_clean = model(**inputs)
                clean_logits = outputs_clean.logits[0, -1]
                clean_probs = F.softmax(clean_logits, dim=-1)
            captured = {}

            def hook(module, inp, out):
                captured["act"] = out[0] if isinstance(out, tuple) else out
                raise StopIteration

            handle = model.model.layers[layer].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    model(**inputs)
            except StopIteration:
                pass
            handle.remove()
            act_flat = captured["act"][0, -1].float()

            cos_sim = F.cosine_similarity(
                act_flat.unsqueeze(0), feat_dir_unit.unsqueeze(0)
            ).item()
            inner_prod = (act_flat * feat_dir).sum().item()

            proj = (act_flat @ feat_dir_unit) * feat_dir_unit
            ablated_act = act_flat - proj

            def ablation_hook(module, inp, out):
                result = out[0] if isinstance(out, tuple) else out
                result = result.clone()
                result[0, -1] = ablated_act.to(result.dtype)
                if isinstance(out, tuple):
                    return (result,) + out[1:]
                return result

            handle = model.model.layers[layer].register_forward_hook(ablation_hook)
            with torch.no_grad():
                outputs_abl = model(**inputs)
                abl_probs = F.softmax(outputs_abl.logits[0, -1], dim=-1)
            handle.remove()

            kl = F.kl_div(abl_probs.log(), clean_probs, reduction="sum").item()
            cos_vals.append(abs(cos_sim))
            inner_vals.append(abs(inner_prod))
            kl_vals.append(kl)
        if len(kl_vals) > 2:
            ca, ia, ka = np.array(cos_vals), np.array(inner_vals), np.array(kl_vals)
            cos_corr = np.corrcoef(ca, ka)[0, 1] if ca.std() > 0 else 0
            inner_corr = np.corrcoef(ia, ka)[0, 1] if ia.std() > 0 else 0
            cos_kl_corrs.append(cos_corr)
            inner_kl_corrs.append(inner_corr)
            if cos_corr > inner_corr:
                cos_wins += 1
    return {
        "n_features": len(cos_kl_corrs),
        "cos_wins_inner": f"{cos_wins}/{len(cos_kl_corrs)}",
        "cos_wins_pct": float(cos_wins / max(len(cos_kl_corrs), 1)),
        "cos_kl_mean": float(np.mean(cos_kl_corrs)) if cos_kl_corrs else 0,
        "inner_kl_mean": float(np.mean(inner_kl_corrs)) if inner_kl_corrs else 0,
    }


# =============================================================================
# Eval data collection (small float32 tensor held in CPU RAM)
# =============================================================================


def collect_eval_data(model, tokenizer, layer_idx, n_tokens):
    print(f"  Collecting eval activations at L{layer_idx} ({n_tokens:,} tokens)...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )
    text_iter = iter(ds)
    for i, _ in enumerate(text_iter):
        if i >= 500_000:
            break
    all_acts = []
    tokens = 0
    while tokens < n_tokens:
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
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CTX_LEN,
        ).to(DEVICE)
        captured = {}

        def hook(module, inp, out):
            captured["act"] = (out[0] if isinstance(out, tuple) else out).detach()
            raise _EarlyStop

        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(**inputs)
        except _EarlyStop:
            pass
        handle.remove()
        flat = captured["act"][inputs["attention_mask"].bool()]
        norms = flat.float().norm(dim=-1)
        median = norms.median()
        if median > 0:
            flat = flat[norms < median * OUTLIER_MULTIPLIER]
        all_acts.append(flat.to("cpu", dtype=DTYPE))
        tokens += flat.shape[0]
    result = torch.cat(all_acts, dim=0)[:n_tokens]
    norms = result.float().norm(dim=-1)
    print(
        f"    {result.shape[0]:,} tokens in {time.time() - t0:.1f}s "
        f"(norm mean={norms.mean():.1f}, std={norms.std():.1f})"
    )
    return result, float(norms.mean().item())


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--tokens", type=int, default=10_000_000,
                        help="Token budget for the cache (and training)")
    parser.add_argument("--aux", choices=["on", "off", "both"], default="both",
                        help="Which aux-k condition(s) to run")
    parser.add_argument("--group-size", type=int, default=6,
                        help="Number of SAEs trained in parallel per group")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: 1M tokens, 1 variant per cell, ~15 min")
    parser.add_argument("--cache-only", action="store_true",
                        help="Build the activation cache and exit (no training)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Where to store the activation cache (default: ~/MechInter--RNH/cache)")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT,
                        help="Where to store SAE checkpoints")
    parser.add_argument("--skip-ablation-eval", action="store_true",
                        help="Skip the cos>inner ablation eval entirely (just FVE/dead/alive)")
    parser.add_argument("--ablation-features", type=int, default=N_ABLATION_FEATURES)
    parser.add_argument("--ablation-samples", type=int, default=N_ABLATION_SAMPLES)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    checkpoint_root = Path(args.checkpoint_root).expanduser()

    if args.smoke:
        args.tokens = 1_000_000

    layer = args.layer
    n_train_tokens = args.tokens
    n_steps = n_train_tokens // BATCH_SIZE
    dead_threshold = max(int(n_train_tokens * DEFAULT_DEAD_FEATURE_THRESHOLD_FRACTION),
                         100_000)

    save_dir = checkpoint_root / f"exp46_L{layer}_{n_train_tokens // 1_000_000}M"
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment 46: NoC normalization-scope ablation")
    print(f"  Layer: L{layer}")
    print(f"  Tokens: {n_train_tokens:,} ({n_steps} steps)")
    print(f"  Aux-k mode: {args.aux}")
    print(f"  Group size: {args.group_size}")
    print(f"  Dead-feature threshold: {dead_threshold:,} tokens")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Save dir: {save_dir}")
    print()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()

    bin_path, _ = cache_paths(cache_dir, MODEL_SLUG, layer, n_train_tokens)
    if not cache_exists_and_valid(cache_dir, MODEL_SLUG, layer, n_train_tokens, D_MODEL):
        print(f"\n[cache] Building activation cache (one-time per layer/budget combo)...")
        bin_path = build_activation_cache(
            model,
            tokenizer,
            layer=layer,
            n_tokens=n_train_tokens,
            cache_dir=cache_dir,
            model_slug=MODEL_SLUG,
            d_model=D_MODEL,
            seed=SEED,
            ctx_len=CTX_LEN,
            collection_batch_size=COLLECTION_BATCH_SIZE,
            outlier_multiplier=OUTLIER_MULTIPLIER,
            chunk_tokens=BUFFER_TOKENS,
            text_skip=0,
            device=DEVICE,
        )
    else:
        print(f"\n[cache] Reusing cache at {bin_path}")

    if args.cache_only:
        print("--cache-only: exiting after cache build.")
        return

    print("\nCollecting eval data (separate from training cache)...")
    eval_data, mean_norm = collect_eval_data(model, tokenizer, layer, N_EVAL_TOKENS)

    # Decide variant set
    aux_modes = {
        "on": [True],
        "off": [False],
        "both": [True, False],
    }[args.aux]
    variants = build_variants(aux_modes)
    if args.smoke:
        # Smoke: one variant per cell of one aux mode
        variants = [v for v in variants if "noc_baseline" in v[0]
                    or "noc_dec_free_no_restore" in v[0]
                    or "noc_enc_free" in v[0]
                    or "noc_input_only_no_restore" in v[0]][:6]

    print(f"\nVariants ({len(variants)}):")
    for name, kw, aux in variants:
        print(f"  {name:32s} | enc={kw['normalize_encoder']!s:5s} "
              f"dec={kw['normalize_decoder']!s:5s} restore={kw['restore_output_norm']!s:5s} "
              f"aux={aux}")

    # Group variants for parallel training
    groups: list[list[tuple[str, dict, bool]]] = []
    for i in range(0, len(variants), args.group_size):
        groups.append(variants[i : i + args.group_size])
    print(f"\n{len(groups)} parallel group(s) of up to {args.group_size}.\n")

    results: dict = {
        "config": {
            "experiment": "exp46_normscope_ablation",
            "model": MODEL_NAME,
            "layer": layer,
            "d_sae": D_SAE,
            "k": K,
            "lr": LR,
            "n_train_tokens": n_train_tokens,
            "n_steps": n_steps,
            "dead_feature_threshold": dead_threshold,
            "aux_k_alpha": AUXK_ALPHA,
            "mean_norm": mean_norm,
            "smoke": args.smoke,
        },
        "runs": {},
    }

    stream = CachedActivationStream(
        bin_path,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        chunk_tokens=BUFFER_TOKENS,
        shuffle_seed=SEED,
    )

    for group_idx, group in enumerate(groups):
        print(f"\n{'#' * 70}")
        print(f"  Group {group_idx + 1}/{len(groups)}: {[v[0] for v in group]}")
        print(f"{'#' * 70}")

        # Reset stream state so each group sees the same activation order.
        stream._cursor = 0
        stream._chunk_idx = 0

        # Build SAEs and states.
        states: list[SAEState] = []
        for name, kw, aux in group:
            torch.manual_seed(SEED)
            sae = NoCFlexSAE(D_MODEL, D_SAE, K, **kw).to(DEVICE)
            states.append(
                SAEState(
                    name=name,
                    sae=sae,
                    aux_k_enabled=aux,
                    n_steps=n_steps,
                    save_dir=save_dir,
                    layer=layer,
                    dead_threshold=dead_threshold,
                )
            )

        train_parallel_group(states, stream, n_steps=n_steps, layer=layer)

        # FVE eval (cheap, per-SAE).
        for s in states:
            recon = evaluate_reconstruction(s.name, s.sae, eval_data)
            print(f"    [{s.name}] FVE={recon['fve']:.4f} dead={recon['dead_frac']:.3f} "
                  f"alive={recon['alive_count']:,} L0={recon['mean_l0']:.1f}")
            results["runs"][s.name] = {
                "name": s.name,
                "config": {
                    "normalize_encoder": s.sae.normalize_encoder,
                    "normalize_decoder": s.sae.normalize_decoder,
                    "restore_output_norm": s.sae.restore_output_norm,
                    "aux_k_enabled": s.aux_k_enabled,
                },
                "reconstruction": recon,
                "training_log": s.log,
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2, default=str)

        # Ablation eval — share clean forwards across all SAEs in the group.
        if not args.skip_ablation_eval:
            print(f"\n  Ablation eval (group of {len(states)}, "
                  f"{args.ablation_features}f × {args.ablation_samples}s)...")
            t_abl = time.time()
            corpus = collect_ablation_corpus(
                model, tokenizer, layer, args.ablation_samples
            )
            for s in states:
                print(f"    [{s.name}] ablation eval...")
                ablation = evaluate_ablation_shared(
                    s.name, model, s.sae, layer, corpus,
                    n_features=args.ablation_features,
                )
                print(f"    [{s.name}] cos>inner={ablation['cos_wins_inner']} "
                      f"(cos_corr={ablation['cos_kl_mean']:.3f}, "
                      f"inner_corr={ablation['inner_kl_mean']:.3f})")
                results["runs"][s.name]["ablation"] = ablation
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2, default=str)
            print(f"  Ablation eval done in {(time.time() - t_abl) / 60:.1f} min")

        # Free GPU memory before next group.
        del states
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Summary L{layer}, {n_train_tokens // 1_000_000}M tokens")
    print(f"{'=' * 70}")
    print(f"  {'variant':32s} | {'FVE':>6} | {'dead':>6} | {'alive':>7} | cos>inner")
    print(f"  {'-' * 32} | {'-' * 6} | {'-' * 6} | {'-' * 7} | {'-' * 9}")
    for name, data in results["runs"].items():
        r = data["reconstruction"]
        a = data.get("ablation", {})
        cos_wins = a.get("cos_wins_inner", "—")
        print(
            f"  {name:32s} | {r['fve']:.4f} | {r['dead_frac']:.3f} | "
            f"{r['alive_count']:>7,} | {cos_wins}"
        )
    print(f"\nResults: {RESULTS_PATH}")
    print(f"Checkpoints: {save_dir}/")


if __name__ == "__main__":
    main()
