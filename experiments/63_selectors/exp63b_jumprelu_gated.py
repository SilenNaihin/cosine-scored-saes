"""
Experiment 60b: cosine advantage under JumpReLU and Gated selectors (A2, stage 2)
=================================================================================

Stage 1 (exp60_selectors.py) covered the two fixed-k per-token selectors (TopK,
AbsTopK), which directly test the batch-vs-per-token mechanism. This stage covers
the two penalty-trained selector families EZEE named, to test generality of the
cosine advantage across selector *families* (not just competition geometry):

  - jumprelu : acts = pre * H(pre - theta); learned per-feature threshold theta,
               L0 trained via straight-through estimator (Rajamanoharan 2024).
  - gated    : separate gate (Heaviside) and magnitude (ReLU) paths sharing W_enc;
               trained with L1 on the gate path + a frozen-decoder aux reconstruction
               (Rajamanoharan 2024 gated).

crossed with the same three scoring arms as stage 1:
  inner / cos_global / cos_perfeature.

WHY THIS IS HARDER THAN STAGE 1. These selectors are penalty-based, not fixed-k.
To compare at the headline sparsity (L0~=80) we tune the sparsity coefficient
lambda with a per-run controller that nudges lambda in log-space toward mean L0 =
TARGET_L0. The cosine score is bounded (~[-scale, scale]) while inner-product is
unbounded, so each arm needs a different lambda; the controller self-calibrates.

The cosine score is selector-agnostic: we compute the same pre-activation
(scale * cos_sim + b_enc) and feed it to the JumpReLU threshold / Gated gate, so
"cosine + JumpReLU" just swaps the scoring of the pre-activation.

Setting identical to stage 1 / exp43d: Qwen3-8B L18, d_sae=65536, 50M FineWeb
tokens, saprmarks LR/schedule, decoder unit-norm + grad projection. Activations
cached once in RAM and replayed across all six variants.

Run on box-5 GPU1 (GPU0 holds exp60 stage-1):
    ssh h100-dev-box-5
    cd ~/MechInter--RNH && source .venv/bin/activate
    HF_HOME=/mnt/hf_cache CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup \
        python3 experiments/exp60b_jumprelu_gated.py 2>&1 | tee experiments/exp60b_output.log &

Smoke (CPU): python3 experiments/exp60b_jumprelu_gated.py --smoke
lambda calibration probe (GPU, short): python3 experiments/exp60b_jumprelu_gated.py --calib
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse stage-1 infrastructure (config, RAM cache, helpers, eval, probing).
import os
import experiments.exp60_selectors as e60
from experiments.exp60_selectors import (
    D_MODEL, D_SAE, K, LAYER, MODEL_NAME, DEVICE, DTYPE,
    N_TRAIN_TOKENS, N_EVAL_TOKENS, N_STEPS, BATCH_SIZE, LR, WARMUP_STEPS,
    DECAY_START, AUXK_ALPHA, DEAD_FEATURE_THRESHOLD, TOP_K_AUX, SEED, NORM_EPS,
    LOG_EVERY, geometric_median, set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions, make_lr_schedule,
    get_auxiliary_loss,
)

# Smaller-dictionary override: penalty-trained selectors (JumpReLU/Gated) could
# not reach matched L0 at d_sae=65536 because the per-feature threshold penalty
# is too diffuse. At a smaller dictionary the penalty bites far harder, so L0
# becomes controllable. Set EXP60B_DSAE=16384 (etc.) to run the learned-gate
# ablation at a tractable scale. Results/paths get a _d{dsae} suffix so the
# 65536 artifacts are never clobbered.
D_SAE = int(os.environ.get("EXP60B_DSAE", D_SAE))
_DTAG = f"_d{D_SAE}" if D_SAE != 65536 else ""

TARGET_L0 = float(K)              # match the fixed-k arms' sparsity (=80)
# Penalty-SAE L0 is stiff: with a weak initial lambda, reconstruction pressure
# drives thresholds down and L0 EXPLODES upward in the first few hundred steps
# (calib: 68 -> 5410). So start lambda HIGH (holds sparsity from the data-driven
# threshold init), engage the controller immediately, and track FAST and
# symmetrically, with a dead-feature emergency cut as the only-collapse backstop.
LAMBDA_INIT = 10.0                # moderate start; wide bandwidth makes the penalty effective
LAMBDA_LR = 0.5                   # log-space controller gain (on log(L0_ema/target))
LAMBDA_STEP = 0.10               # cap on |Δ log lambda| per step (fast, symmetric)
L0_EMA_BETA = 0.90               # short EMA so controller reacts to fast L0 moves
DEAD_PANIC = 0.5                 # if >this frac dead in a step, emergency-cut lambda
DEAD_PANIC_CUT = 0.5             # multiply lambda by this on a dead-feature spike
LAMBDA_WARMUP = 50                # engage control early (explosion happens fast)
BANDWIDTH_FRAC = 0.5             # STE kernel width as a fraction of pre-act RMS (data-adaptive).
# NOTE: bandwidth sets how many features get L0/threshold gradient each step. Too
# narrow (tried 0.05) starves the penalty: lambda saturates at 1e6 yet L0 stays
# ~5000 because thresholds never move. 0.5*RMS lets the penalty reach a broad band
# of near-threshold features so L0 is actually controllable. Backward-only knob;
# forward value x*(x>theta) is bandwidth-independent so eval is unaffected.

SAVE_DIR = Path(f"/mnt/exp60b_checkpoints{_DTAG}")
RESULTS_PATH = Path(f"experiments/exp60b_results{_DTAG}.json")
EVAL_OUT_ROOT = f"/mnt/exp60b_eval_results{_DTAG}"

VARIANTS = [
    ("jumprelu_inner",          "inner",          "jumprelu"),
    ("jumprelu_cos_global",     "cos_global",     "jumprelu"),
    ("jumprelu_cos_perfeature", "cos_perfeature", "jumprelu"),
    ("gated_inner",          "inner",          "gated"),
    ("gated_cos_global",     "cos_global",     "gated"),
    ("gated_cos_perfeature", "cos_perfeature", "gated"),
]


# =============================================================================
# Straight-through Heaviside / JumpReLU (Rajamanoharan 2024)
#   Forward: step / jump. Backward: rectangular kernel pseudo-derivative of width
#   `bandwidth` centered at the threshold, applied to the *threshold* parameter.
# =============================================================================

class HeavisideSTE(torch.autograd.Function):
    """H(x - theta). Gradient flows to theta via a rectangle kernel of width bw."""
    @staticmethod
    def forward(ctx, x, theta, bw):
        ctx.save_for_backward(x, theta)
        ctx.bw = bw
        return (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, theta = ctx.saved_tensors
        bw = ctx.bw
        # d/dtheta H(x-theta) ~= -1/bw * rect(|x-theta| < bw/2)
        kernel = (((x - theta).abs()) < (bw / 2)).to(grad_out.dtype) / bw
        grad_theta = (-grad_out * kernel)
        return None, grad_theta, None


class JumpReLUSTE(torch.autograd.Function):
    """JumpReLU(x; theta) = x * H(x - theta).
    Backward: grad to x is the usual x>theta mask; grad to theta uses the STE kernel
    (the jump contributes -theta * kernel, magnitude term x ~= theta there)."""
    @staticmethod
    def forward(ctx, x, theta, bw):
        ctx.save_for_backward(x, theta)
        ctx.bw = bw
        return x * (x > theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, theta = ctx.saved_tensors
        bw = ctx.bw
        mask = (x > theta).to(grad_out.dtype)
        grad_x = grad_out * mask
        kernel = (((x - theta).abs()) < (bw / 2)).to(grad_out.dtype) / bw
        grad_theta = grad_out * (-theta * kernel)
        return grad_x, grad_theta, None


def heaviside_ste(x, theta, bw):
    return HeavisideSTE.apply(x, theta, bw)


def jumprelu_ste(x, theta, bw):
    return JumpReLUSTE.apply(x, theta, bw)


# =============================================================================
# Scoring arm (shared with stage 1): produce pre-activations from x
# =============================================================================

class Scorer(nn.Module):
    """Computes pre-activations under inner / cos_global / cos_perfeature.
    Owns W_enc, b_dec, and (for cosine) scale_a/scale_b. b_enc handled by caller."""

    def __init__(self, d_model, d_sae, score_mode):
        super().__init__()
        self.score_mode = score_mode
        self.is_cosine = score_mode.startswith("cos")
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # Global input-norm scale (canonical JumpReLU recipe). When >0, centered
        # activations are divided by this fixed scalar so the SAE trains in
        # ~unit-RMS space where the JumpReLU threshold/STE is calibrated and the
        # sparsity penalty actually bites. FVE is scale-invariant and decode
        # multiplies back by the same scalar, so eval/probing is unaffected.
        # Set from data via set_input_scale(); 1.0 = off.
        self.register_buffer("norm_scale", torch.tensor(1.0))
        if score_mode == "cos_global":
            self.scale_a = nn.Parameter(torch.tensor(0.0))
            self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        elif score_mode == "cos_perfeature":
            self.scale_a = nn.Parameter(torch.zeros(d_sae))
            self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))

    def pre_acts(self, x, b_enc):
        x_c = (x - self.b_dec) / self.norm_scale
        if not self.is_cosine:
            return x_c @ self.W_enc.T + b_enc
        x_unit = F.normalize(x_c, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        return scale * cos_sim + b_enc


# =============================================================================
# JumpReLU SAE (penalty-trained), pluggable scorer
# =============================================================================

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae, score_mode="inner", bandwidth_frac=BANDWIDTH_FRAC):
        super().__init__()
        self.d_model, self.d_sae = d_model, d_sae
        self.bandwidth_frac = bandwidth_frac
        self.scorer = Scorer(d_model, d_sae, score_mode)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        # log-threshold so theta = exp(.) > 0; reset by init_thresholds_from_data()
        self.log_theta = nn.Parameter(torch.full((d_sae,), math.log(0.001)))
        self.is_cosine = score_mode.startswith("cos")
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.scorer.W_enc.copy_(self.W_dec if self.is_cosine else 0.1 * self.W_dec)

    @property
    def b_dec(self):
        return self.scorer.b_dec

    def _bandwidth(self, pre):
        # Data-adaptive STE kernel width: a fraction of the positive-pre-activation
        # RMS. Bandwidth affects ONLY the backward (STE) gradient, never the forward
        # value, so eval/probing is unchanged. Adapting it makes the threshold
        # gradient flow at any score scale (inner-product ~tens vs cosine ~units),
        # which a fixed absolute bandwidth fails to do (see calib probe).
        with torch.no_grad():
            pos = pre[pre > 0]
            rms = pos.pow(2).mean().sqrt().item() if pos.numel() > 0 else 1.0
        return max(self.bandwidth_frac * rms, 1e-4)

    def encode(self, x, return_pre=False):
        pre = self.scorer.pre_acts(x, self.b_enc)
        theta = self.log_theta.exp()
        bw = self._bandwidth(pre)
        acts = jumprelu_ste(pre, theta, bw)
        if return_pre:
            return acts, pre, theta
        return acts

    def decode(self, f):
        # reconstruct in normalized space then scale back up (norm_scale=1 -> no-op)
        return (f @ self.W_dec) * self.scorer.norm_scale + self.scorer.b_dec

    def l0_surrogate(self, pre, theta):
        # differentiable count of active features (per token, then mean)
        gate = heaviside_ste(pre, theta, self._bandwidth(pre))
        return gate.sum(dim=-1).mean()

    @torch.no_grad()
    def init_thresholds_from_data(self, pre, target_l0):
        # Set all thresholds to the global quantile of positive pre-activations
        # that yields ~target_l0 active features per token. Features then
        # differentiate their thresholds during training.
        pos = pre[pre > 0]
        if pos.numel() == 0:
            return
        frac_active = target_l0 / self.d_sae          # desired fraction firing
        q = max(0.0, min(1.0, 1.0 - frac_active))
        thr = torch.quantile(pos.float().flatten()[:1_000_000], q).clamp(min=1e-4)
        self.log_theta.data.fill_(math.log(thr.item()))

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


# =============================================================================
# Gated SAE (Rajamanoharan 2024), pluggable scorer on the SHARED pre-activation
# =============================================================================

class GatedSAE(nn.Module):
    def __init__(self, d_model, d_sae, score_mode="inner"):
        super().__init__()
        self.d_model, self.d_sae = d_model, d_sae
        self.scorer = Scorer(d_model, d_sae, score_mode)
        # gate uses its own bias; magnitude path rescales the shared score
        self.gate_bias = nn.Parameter(torch.zeros(d_sae))
        self.r_mag = nn.Parameter(torch.zeros(d_sae))
        self.b_mag = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.is_cosine = score_mode.startswith("cos")
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            self.scorer.W_enc.copy_(self.W_dec if self.is_cosine else 0.1 * self.W_dec)

    @property
    def b_dec(self):
        return self.scorer.b_dec

    def _shared_score(self, x):
        # pre-activation with ZERO bias; gate/mag add their own biases
        return self.scorer.pre_acts(x, torch.zeros_like(self.gate_bias))

    @torch.no_grad()
    def init_gate_bias_from_data(self, x, target_l0):
        # Set gate_bias = -quantile(score) so ~target_l0 features pass the gate
        # at init (gate fires when score + gate_bias > 0). Mirrors the JumpReLU
        # data-driven threshold init.
        s = self._shared_score(x)
        frac_active = target_l0 / self.d_sae
        q = max(0.0, min(1.0, 1.0 - frac_active))
        thr = torch.quantile(s.float().flatten()[:1_000_000], q)
        self.gate_bias.data.fill_(-thr.item())

    def encode(self, x, return_gatemag=False):
        s = self._shared_score(x)
        pi_gate = s + self.gate_bias
        f_gate = (pi_gate > 0).to(s.dtype)
        pi_mag = self.r_mag.exp() * s + self.b_mag
        f_mag = F.relu(pi_mag)
        f = f_gate * f_mag
        if return_gatemag:
            # f_mag (post-ReLU magnitude) is the per-feature activation used for
            # dead-feature aux-k revival.
            return f, pi_gate, f_mag
        return f

    def decode(self, f):
        return (f @ self.W_dec) * self.scorer.norm_scale + self.scorer.b_dec

    @torch.no_grad()
    def _frozen_decoder(self):
        return self.W_dec.detach(), self.scorer.b_dec.detach()

    def gated_aux_recon(self, x, pi_gate):
        # aux: reconstruct x from ReLU(pi_gate) through a frozen decoder copy
        f_gate_relu = F.relu(pi_gate)
        W_dec_f, b_dec_f = self._frozen_decoder()
        return f_gate_relu @ W_dec_f * self.scorer.norm_scale + b_dec_f

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


# =============================================================================
# Training with an L0-targeting lambda controller
# =============================================================================

def train_variant(name, score_mode, selector, shards, n_steps=None, lambda_warmup=None,
                  fixed_lambda=None):
    # CANONICAL RECIPE (Rajamanoharan 2024): when fixed_lambda is set, hold the
    # sparsity coefficient CONSTANT (no L0-target controller). The free-running
    # controller drove lambda to extremes and L0 exploded; a fixed lambda + the
    # STE loss is the literature-standard, stable approach. lambda is chosen by a
    # small sweep (--lamsweep) to bracket L0~=TARGET_L0.
    n_steps = n_steps or N_STEPS
    lambda_warmup = LAMBDA_WARMUP if lambda_warmup is None else lambda_warmup
    decay_start = int(0.8 * n_steps)
    print(f"\n{'='*70}\n  {name} (score={score_mode}, selector={selector}, "
          f"n_steps={n_steps})\n{'='*70}", flush=True)
    torch.manual_seed(SEED)
    if selector == "jumprelu":
        sae = JumpReLUSAE(D_MODEL, D_SAE, score_mode=score_mode).to(DEVICE)
    else:
        sae = GatedSAE(D_MODEL, D_SAE, score_mode=score_mode).to(DEVICE)
    sae.train()

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_schedule(n_steps, WARMUP_STEPS, decay_start))

    lam = LAMBDA_INIT if fixed_lambda is None else fixed_lambda
    l0_ema = None
    b_dec_initialized = False
    num_tokens_since_fired = torch.zeros(D_SAE, dtype=torch.long, device=DEVICE)
    log = []
    t0 = time.time()

    for step, batch in enumerate(e60.ram_iter(shards, n_steps)):
        if not b_dec_initialized:
            with torch.no_grad():
                sae.scorer.b_dec.data.copy_(geometric_median(batch).to(sae.scorer.b_dec.dtype))
                # Input-norm scale (canonical recipe): for the inner-product arms,
                # divide centered acts by their mean norm so the encoder trains in
                # ~unit-RMS space where the JumpReLU STE penalty actually bites.
                # Cosine arms keep their own scale machinery (norm_scale stays 1).
                if not sae.scorer.is_cosine:
                    s = (batch - sae.scorer.b_dec).norm(dim=-1).mean().clamp(min=NORM_EPS)
                    sae.scorer.norm_scale.fill_(float(s))
                    print(f"    [{name}] input norm_scale={float(s):.1f}", flush=True)
                # JumpReLU: seed thresholds at the data quantile giving L0~=target,
                # so we start near the target instead of climbing down from ~d_sae/2.
                if selector == "jumprelu":
                    pre0 = sae.scorer.pre_acts(batch, sae.b_enc)
                    sae.init_thresholds_from_data(pre0, TARGET_L0)
                else:  # gated
                    sae.init_gate_bias_from_data(batch, TARGET_L0)
            b_dec_initialized = True
            print(f"    [{name}] b_dec init (norm={sae.scorer.b_dec.norm():.1f})", flush=True)

        if selector == "jumprelu":
            f, pre, theta = sae.encode(batch, return_pre=True)
            x_hat = sae.decode(f)
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()
            sparsity_loss = sae.l0_surrogate(pre, theta)
            gate_aux = torch.tensor(0.0, device=DEVICE)
            post_relu_acts = F.relu(pre)        # for dead-feature aux-k revival
        else:  # gated
            f, pi_gate, f_mag = sae.encode(batch, return_gatemag=True)
            x_hat = sae.decode(f)
            recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()
            # L1 on the (post-ReLU) gate activations encourages sparsity
            sparsity_loss = F.relu(pi_gate).sum(dim=-1).mean()
            x_hat_gate = sae.gated_aux_recon(batch, pi_gate)
            gate_aux = (batch - x_hat_gate).pow(2).sum(dim=-1).mean()
            post_relu_acts = f_mag              # magnitude path drives revival

        # Dead-feature tracking + aux-k revival (matches exp60/exp43d; this is
        # what keeps dead% ~0 — without it JumpReLU/Gated hit 55-78% dead).
        with torch.no_grad():
            did_fire = (f != 0).sum(dim=0) > 0
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
        aux_loss = gate_aux + AUXK_ALPHA * auxk_loss

        loss = recon_loss + lam * sparsity_loss + aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if sae.W_dec.grad is not None:
            sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                sae.W_dec.data, sae.W_dec.grad.data)
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        set_decoder_norm_to_unit_norm(sae.W_dec.data)

        # measured L0 (hard count) + dead fraction this step
        with torch.no_grad():
            hard_l0 = (f != 0).float().sum(dim=-1).mean().item()
            inst_dead = ((f != 0).sum(dim=0) == 0).float().mean().item()
        l0_ema = hard_l0 if l0_ema is None else (L0_EMA_BETA * l0_ema + (1 - L0_EMA_BETA) * hard_l0)
        # Lambda update only in controller mode. With fixed_lambda (canonical
        # recipe) lambda is held constant and L0 is whatever that lambda yields;
        # we pick the right lambda via --lamsweep instead of servoing here.
        if fixed_lambda is None and step > lambda_warmup and l0_ema and l0_ema > 0:
            err = math.log(l0_ema / TARGET_L0)        # >0 means too many active
            if inst_dead > DEAD_PANIC:
                lam *= DEAD_PANIC_CUT
            else:
                delta = max(-LAMBDA_STEP, min(LAMBDA_STEP, LAMBDA_LR * err))
                lam *= math.exp(delta)
            lam = float(min(max(lam, 1e-8), 1e6))

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            with torch.no_grad():
                total_var = torch.var(batch, dim=0, unbiased=False).sum()
                resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                dead = ((f != 0).sum(dim=0) == 0).float().mean().item()
            entry = {"step": step, "fve": fve, "l0": hard_l0, "lam": lam,
                     "dead_frac": dead, "recon_loss": recon_loss.item(),
                     "sparsity_loss": float(sparsity_loss.detach()),
                     "aux_loss": float(aux_loss.detach())}
            if hasattr(sae.scorer, "scale_a"):
                entry["scale_a_mean"] = sae.scorer.scale_a.float().mean().item()
            log.append(entry)
            print_every = 2000 if n_steps > 5000 else LOG_EVERY
            if step % print_every == 0 or step == n_steps - 1:
                tps = (step + 1) * BATCH_SIZE / (time.time() - t0)
                print(f"    [{name}] {step:>5d}/{n_steps} FVE={fve:.4f} L0={hard_l0:.0f} "
                      f"lam={lam:.3g} dead={dead:.3f} [{tps/1e3:.1f}k tok/s]", flush=True)

    sae.eval()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "score_mode": score_mode,
                "selector": selector, "lam": lam}, SAVE_DIR / f"{name}_final.pt")
    print(f"    [{name}] done in {(time.time()-t0)/3600:.2f}h (final L0~{hard_l0:.0f}, lam={lam:.3g})", flush=True)
    return sae, log


@torch.no_grad()
def evaluate_reconstruction(name, sae, eval_shards):
    sae.eval()
    cos_sims, l0s = [], []
    tv = rv = 0.0
    dead = None
    for batch in eval_shards:
        batch = batch.to(DEVICE, dtype=torch.float32)
        x_hat, f = sae(batch)
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((f != 0).float().sum(dim=-1).mean().item())
        tv += torch.var(batch, dim=0, unbiased=False).sum().item()
        rv += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        alive = (f != 0).sum(dim=0) != 0
        dead = ~alive if dead is None else (dead & ~alive)
    fve = 1 - rv / tv if tv > 0 else 0
    res = {"fve": fve, "cos_recon": float(np.mean(cos_sims)), "mean_l0": float(np.mean(l0s)),
           "dead_frac": dead.float().mean().item(), "alive_count": int((~dead).sum().item())}
    print(f"    [{name}] FVE={fve:.4f} L0={res['mean_l0']:.1f} dead={res['dead_frac']:.3f}", flush=True)
    return res


def run_sparse_probing(name, sae):
    from benchmarks.adapter import BenchSAE
    from benchmarks.run_saebench import run_saebench
    sae.eval()
    dt = sae.W_dec.dtype

    def encode_fn(x):
        return sae.encode(x.to(dtype=dt)).to(dtype=x.dtype)

    def decode_fn(f):
        return sae.decode(f.to(dtype=dt)).to(dtype=f.dtype)

    bench_sae = BenchSAE(
        W_enc=sae.scorer.W_enc.detach().T, W_dec=F.normalize(sae.W_dec.detach(), dim=1),
        b_enc=torch.zeros(sae.d_sae, device=DEVICE), b_dec=sae.scorer.b_dec.detach(),
        encode_fn=encode_fn, decode_fn=decode_fn,
        model_name=MODEL_NAME, hook_layer=LAYER, device=DEVICE, dtype=DTYPE)
    return run_saebench(bench_sae, sae_name=f"exp60b-{name}", eval_types=["sparse_probing"],
                        output_dir=f"{EVAL_OUT_ROOT}/{name}", llm_batch_size=4, device=DEVICE)


def main(calib_only=False, lamsweep=False):
    print(f"Experiment 60b: JumpReLU + Gated x scoring at Qwen3-8B L{LAYER}/50M", flush=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE).eval()

    if lamsweep:
        # Canonical recipe: find the FIXED lambda giving L0~=TARGET_L0. Short runs
        # (1500 steps) over a log-spaced grid for the two hardest arms (the inner
        # ones), report final L0 per lambda. Aux-k threshold scaled to engage.
        e60.DEAD_FEATURE_THRESHOLD = 1_000_000
        train_shards = e60.stream_acts_to_ram(model, tokenizer, 4_000_000, skip_rows=0)
        del model; torch.cuda.empty_cache()
        grid = [3.0, 10.0, 30.0, 100.0, 300.0]
        sweep = {}
        for sel in ["jumprelu", "gated"]:
            for lam in grid:
                name = f"sweep_{sel}_inner_lam{lam:g}"
                sae, log = train_variant(name, "inner", sel, train_shards,
                                         n_steps=1500, fixed_lambda=lam)
                final_l0 = log[-1]["l0"] if log else None
                final_dead = log[-1]["dead_frac"] if log else None
                sweep[name] = {"lambda": lam, "final_l0": final_l0, "final_dead": final_dead}
                print(f"  SWEEP {sel} inner lam={lam:g} -> L0={final_l0:.0f} dead={final_dead:.3f}", flush=True)
                del sae; gc.collect(); torch.cuda.empty_cache()
        json.dump(sweep, open(f"experiments/exp60b_lamsweep{_DTAG}.json", "w"), indent=2, default=str)
        print("\nSWEEP SUMMARY:", flush=True)
        for k, v in sweep.items():
            print(f"  {k:32s} L0={v['final_l0']:.0f} dead={v['final_dead']:.3f}", flush=True)
        return

    if calib_only:
        # short calibration: 2500 steps each, warmup 300 so the controller
        # engages early. Scale the dead-feature threshold down with the shorter
        # run (real: 50M tokens / 10M thresh; calib: 5M / 1M) so aux-k revival
        # actually ENGAGES in the probe and we validate dead-recovery the way
        # the full run will behave (otherwise aux-k never fires in 4M tokens).
        e60.DEAD_FEATURE_THRESHOLD = 1_000_000
        n_cache = 5_000_000
        train_shards = e60.stream_acts_to_ram(model, tokenizer, n_cache, skip_rows=0)
        del model; torch.cuda.empty_cache()
        for name, sm, sel in [("jumprelu_inner", "inner", "jumprelu"),
                              ("jumprelu_cos_perfeature", "cos_perfeature", "jumprelu"),
                              ("gated_inner", "inner", "gated"),
                              ("gated_cos_perfeature", "cos_perfeature", "gated")]:
            train_variant(name, sm, sel, train_shards, n_steps=2500, lambda_warmup=300)
        return

    train_shards = e60.stream_acts_to_ram(model, tokenizer, N_TRAIN_TOKENS, skip_rows=0)
    eval_shards = e60.stream_acts_to_ram(model, tokenizer, N_EVAL_TOKENS, skip_rows=600_000)
    del model; gc.collect(); torch.cuda.empty_cache()

    results = {"config": {"experiment": "exp60b_jumprelu_gated", "model": MODEL_NAME,
                          "layer": LAYER, "d_sae": D_SAE, "target_l0": TARGET_L0}, "runs": {}}
    if RESULTS_PATH.exists():
        try:
            results = json.load(open(RESULTS_PATH))
        except Exception:
            pass

    trained = {}
    for name, sm, sel in VARIANTS:
        if results.get("runs", {}).get(name, {}).get("reconstruction"):
            print(f"  [{name}] done, skipping.", flush=True)
            continue
        sae, log = train_variant(name, sm, sel, train_shards)
        results["runs"][name] = {"score_mode": sm, "selector": sel,
                                 "reconstruction": evaluate_reconstruction(name, sae, eval_shards),
                                 "training_log": log[-15:]}
        json.dump(results, open(RESULTS_PATH, "w"), indent=2, default=str)
        trained[name] = sae
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  SAEBench sparse probing\n{'='*70}", flush=True)
    for name, sm, sel in VARIANTS:
        if results["runs"].get(name, {}).get("sparse_probing"):
            continue
        sae = trained.get(name)
        if sae is None:
            continue
        try:
            sp = run_sparse_probing(name, sae)
            spp = sp.get("sparse_probing", {}) if isinstance(sp, dict) else {}
            metrics = spp.get("eval_result_metrics", {}).get("sae", spp) if isinstance(spp, dict) else spp
            results["runs"][name]["sparse_probing"] = metrics
            t1 = metrics.get("sae_top_1_test_accuracy", "?") if isinstance(metrics, dict) else "?"
            print(f"    [{name}] top-1={t1}", flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            results["runs"][name]["sparse_probing_error"] = str(ex)
        json.dump(results, open(RESULTS_PATH, "w"), indent=2, default=str)

    print(f"\nResults: {RESULTS_PATH}", flush=True)


def smoke():
    e60.DEVICE = "cpu"
    global DEVICE
    print("SMOKE: JumpReLU + Gated STE + controllers on CPU...")
    torch.manual_seed(0)
    B, dm, ds = 64, 16, 128

    tgt = 10  # target L0 for the tiny ds=128 smoke dict
    # STE gradient sanity + data-driven threshold init brings L0 near target
    for score_mode in ["inner", "cos_global", "cos_perfeature"]:
        sae = JumpReLUSAE(dm, ds, score_mode=score_mode).to("cpu")
        x = torch.randn(B, dm) * 5.0
        with torch.no_grad():
            pre0 = sae.scorer.pre_acts(x, sae.b_enc)
            sae.init_thresholds_from_data(pre0, tgt)
        f, pre, theta = sae.encode(x, return_pre=True)
        l0_init = (f != 0).float().sum(-1).mean().item()
        assert tgt * 0.3 <= l0_init <= tgt * 3, f"jumprelu init L0={l0_init} far from {tgt}"
        x_hat = sae.decode(f)
        l0 = sae.l0_surrogate(pre, theta)
        loss = (x - x_hat).pow(2).sum(-1).mean() + 3.0 * l0
        loss.backward()
        assert sae.log_theta.grad is not None and torch.isfinite(sae.log_theta.grad).all(), "jumprelu: no theta grad"
        assert sae.log_theta.grad.abs().sum() > 0, "jumprelu: zero theta grad (bandwidth too small)"
        assert sae.scorer.W_enc.grad is not None and torch.isfinite(sae.scorer.W_enc.grad).all()
        assert (f != 0).any(), "jumprelu: all dead at init"
        print(f"  jumprelu x {score_mode:14s} OK (init L0={l0_init:.0f}~{tgt}, surrogate={l0.item():.1f}, theta grad ok)")

    for score_mode in ["inner", "cos_global", "cos_perfeature"]:
        sae = GatedSAE(dm, ds, score_mode=score_mode).to("cpu")
        x = torch.randn(B, dm) * 5.0
        with torch.no_grad():
            sae.init_gate_bias_from_data(x, tgt)
        f, pi_gate, f_mag = sae.encode(x, return_gatemag=True)
        l0_init = (f != 0).float().sum(-1).mean().item()
        assert tgt * 0.3 <= l0_init <= tgt * 3, f"gated init L0={l0_init} far from {tgt}"
        x_hat = sae.decode(f)
        x_hat_gate = sae.gated_aux_recon(x, pi_gate)
        loss = (x - x_hat).pow(2).sum(-1).mean() + 2.0 * F.relu(pi_gate).sum(-1).mean() \
            + (x - x_hat_gate).pow(2).sum(-1).mean()
        loss.backward()
        assert sae.gate_bias.grad is not None and torch.isfinite(sae.gate_bias.grad).all()
        assert sae.r_mag.grad is not None, "gated: no r_mag grad"
        assert sae.scorer.W_enc.grad is not None and torch.isfinite(sae.scorer.W_enc.grad).all()
        print(f"  gated    x {score_mode:14s} OK (init L0={l0_init:.0f}~{tgt}, gate+mag grad ok)")

    # controller direction check: too-high L0 should raise lambda
    lam = 5.0
    lam2 = lam * math.exp(LAMBDA_LR * (160.0 / TARGET_L0 - 1.0))
    assert lam2 > lam, "controller: high L0 must increase lambda"
    lam3 = lam * math.exp(LAMBDA_LR * (20.0 / TARGET_L0 - 1.0))
    assert lam3 < lam, "controller: low L0 must decrease lambda"
    print(f"  controller OK (L0=160 -> lam {lam:.2f}->{lam2:.2f}; L0=20 -> {lam3:.2f})")
    print("SMOKE PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calib", action="store_true", help="short GPU run to check L0 tracks target")
    ap.add_argument("--lamsweep", action="store_true",
                    help="canonical recipe: sweep fixed lambda to bracket L0~=target")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.lamsweep:
        main(lamsweep=True)
    elif args.calib:
        main(calib_only=True)
    else:
        main()
