"""
Experiment 64: Direct score-swap mechanism test (camera-ready A5)
=================================================================

Reviewer 762k: "Add a direct score swap experiment. Train a standard SAE and a
cosine SAE, then evaluate what happens when the score geometry is swapped or
partially swapped at inference and during continued training. This would test the
mechanism more directly than gradient reweighting alone."

exp29 did ONLY the forward direction (post-hoc cosine on a trained STANDARD SAE
destroys FVE: 0.638 -> 0.306; the advantage is training-time, not inference-time).
This experiment completes the picture, SYMMETRICALLY:

  PART 1 (inference-time swap, near-free):
    - cosine checkpoint scored with INNER-PRODUCT at inference  (reverse; NEW)
    - standard checkpoint scored with COSINE at inference       (forward; confirms exp29)
    Measure FVE + SAEBench sparse-probing top-1 degradation in both directions.

  PART 2 (continued-training swap, moderate):
    - take each trained checkpoint, SWAP the score, continue training ~10M tokens
    - measure FVE recovery and feature-set drift (Jaccard of alive features, decoder
      cosine) vs the original. Tests whether the dictionary REORGANIZES toward the
      new score's solution -- i.e. whether the score geometry, not just the weights,
      determines the learned features.

The architecture makes the swap exact: the adaptive-cosine encoder recovers inner
product at a=1 and pure cosine scoring at a=0; the standard encoder can be scored
post-hoc with cosine by normalizing x and W. We hot-swap the score module at eval.

Setting: Qwen3-8B L18, d_sae=65536, k=80 (BatchTopK), 50M FineWeb tokens, saprmarks
recipe (matches exp43d/exp29/exp59). Activations cached once in RAM and replayed.

Run on box-5:
    ssh h100-dev-box-5
    cd ~/MechInter--RNH && source .venv/bin/activate
    HF_HOME=/mnt/hf_cache CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup \
        python3 experiments/exp64_score_swap.py 2>&1 | tee experiments/exp64_output.log &

CPU smoke: python3 experiments/exp64_score_swap.py --smoke
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
torch.backends.cuda.enable_cudnn_sdp(False)

# Reuse exp60's streaming RAM cache, recipe constants, helpers, and probing.
import experiments.exp60_selectors as e60
from experiments.exp60_selectors import (
    D_MODEL, D_SAE, K, LAYER, MODEL_NAME, DEVICE, DTYPE,
    N_TRAIN_TOKENS, N_EVAL_TOKENS, BATCH_SIZE, LR, WARMUP_STEPS,
    AUXK_ALPHA, DEAD_FEATURE_THRESHOLD, TOP_K_AUX, SEED, NORM_EPS, LOG_EVERY,
    geometric_median, set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions, make_lr_schedule,
    get_auxiliary_loss,
)

N_STEPS = N_TRAIN_TOKENS // BATCH_SIZE
DECAY_START = int(0.8 * N_STEPS)
N_FINETUNE_TOKENS = 10_000_000          # continued-training swap budget
N_FT_STEPS = N_FINETUNE_TOKENS // BATCH_SIZE

SAVE_DIR = Path("/mnt/exp64_checkpoints")
RESULTS_PATH = Path("experiments/exp64_results.json")
EVAL_OUT_ROOT = "/mnt/exp64_eval_results"

THRESHOLD_BETA = 0.999
THRESHOLD_START_STEP = 1000


# =============================================================================
# BatchTopK SAE with a SWAPPABLE score (score_mode: "inner" | "cosine")
#   inner : s_i = (x - b_dec) . W_enc_i + b_enc                 (standard)
#   cosine: s_i = exp(a log||x_c|| + b) * cos(x_c, W_enc_i) + b_enc   (adaptive)
# The same module trains in one mode and can be evaluated in the other; weights
# are shared, only the scoring function changes. This is the score swap.
# =============================================================================

class SwappableBatchTopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80, score_mode="inner"):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.score_mode = score_mode
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # adaptive-cosine scale params (used only when score_mode == "cosine")
        self.scale_a = nn.Parameter(torch.tensor(0.0))
        self.scale_b = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec)
        set_decoder_norm_to_unit_norm(self.W_dec)
        with torch.no_grad():
            # Match exp43d exactly: BOTH standard and adaptive-cosine init W_enc =
            # W_dec (the 0.1x init used in exp59/exp60 caused ~79% dead features for
            # the standard arm here; exp43d's plain copy gives ~0% dead).
            self.W_enc.copy_(self.W_dec)
            self.b_enc.zero_()

    def pre_acts(self, x, score_mode=None):
        mode = score_mode or self.score_mode
        x_c = x - self.b_dec
        if mode == "inner":
            return x_c @ self.W_enc.T + self.b_enc
        # cosine
        x_unit = F.normalize(x_c, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_c.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        scale = torch.exp(self.scale_a * torch.log(input_norm) + self.scale_b)
        return scale * cos_sim + self.b_enc

    def _batch_topk(self, acts):
        bsz = max(acts.shape[0], 1)
        total_k = min(self.k * bsz, acts.numel())
        flat = acts.reshape(-1)
        vals, idx = torch.topk(flat, total_k)
        sparse = torch.zeros_like(flat)
        sparse[idx] = vals
        return sparse.view_as(acts)

    def encode(self, x, score_mode=None, return_active=False):
        post_relu = F.relu(self.pre_acts(x, score_mode))
        if self.training:
            encoded = self._batch_topk(post_relu)
        else:
            if self.threshold < 0:
                encoded = self._batch_topk(post_relu)
            else:
                encoded = post_relu * (post_relu > self.threshold)
        if return_active:
            return encoded, (encoded.sum(0) > 0), post_relu
        return encoded

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, score_mode=None, return_active=False):
        if return_active:
            f, active, post_relu = self.encode(x, score_mode, return_active=True)
            return self.decode(f), f, active, post_relu
        f = self.encode(x, score_mode)
        return self.decode(f), f


# =============================================================================
# Training (BatchTopK + aux-k, matches exp43d). reset_threshold for finetune.
# =============================================================================

def train(sae, shards, n_steps, decay_start, label, swap_score_to=None,
          reset_threshold=False):
    """Train sae for n_steps. If swap_score_to is set, training uses that score
    mode (the continued-training swap); otherwise sae.score_mode."""
    mode = swap_score_to or sae.score_mode
    print(f"\n{'='*70}\n  TRAIN {label} (score={mode}, steps={n_steps})\n{'='*70}", flush=True)
    sae.train()
    if reset_threshold:
        sae.threshold.fill_(-1.0)
    opt = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_schedule(n_steps, WARMUP_STEPS, decay_start))
    ntsf = torch.zeros(D_SAE, dtype=torch.long, device=DEVICE)
    b_dec_init = (sae.b_dec.abs().sum().item() > 0)   # already init'd if finetuning
    log = []
    t0 = time.time()
    for step, batch in enumerate(e60.ram_iter(shards, n_steps)):
        if not b_dec_init:
            with torch.no_grad():
                sae.b_dec.data.copy_(geometric_median(batch).to(sae.b_dec.dtype))
            b_dec_init = True
        x_hat, f, active, post_relu = sae(batch, score_mode=mode, return_active=True)
        recon = (batch - x_hat).pow(2).sum(-1).mean()
        did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
        did_fire[active] = True
        ntsf += batch.shape[0]
        ntsf[did_fire] = 0
        residual = (batch - x_hat).detach()
        auxk_buf, n_dead = get_auxiliary_loss(post_relu, ntsf)
        if n_dead > 0:
            x_aux = auxk_buf @ sae.W_dec
            auxk_l2 = (residual.float() - x_aux.float()).pow(2).sum(-1).mean()
            mu = residual.mean(0, keepdim=True)
            denom = (residual.float() - mu.float()).pow(2).sum(-1).mean()
            auxk = (auxk_l2 / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            auxk = torch.tensor(0.0, device=DEVICE)
        loss = recon + AUXK_ALPHA * auxk
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if sae.W_dec.grad is not None:
            sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                sae.W_dec.data, sae.W_dec.grad.data)
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sched.step()
        set_decoder_norm_to_unit_norm(sae.W_dec.data)
        if step > THRESHOLD_START_STEP:
            with torch.no_grad():
                av = f[f > 0]
                if av.numel() > 0:
                    m = av.min().float()
                    sae.threshold.fill_(m) if sae.threshold < 0 else \
                        sae.threshold.mul_(THRESHOLD_BETA).add_((1 - THRESHOLD_BETA) * m)
        if step % LOG_EVERY == 0 or step == n_steps - 1:
            with torch.no_grad():
                l0 = (f != 0).float().sum(-1).mean().item()
                tv = torch.var(batch, 0, unbiased=False).sum()
                rv = torch.var(batch - x_hat, 0, unbiased=False).sum()
                fve = (1 - rv / tv).item() if tv > 0 else 0
                dead = (ntsf >= DEAD_FEATURE_THRESHOLD).float().mean().item()
            log.append({"step": step, "fve": fve, "l0": l0, "dead_frac": dead, "n_dead": n_dead})
            if step % 2000 == 0 or step == n_steps - 1:
                tps = (step + 1) * BATCH_SIZE / (time.time() - t0)
                print(f"    [{label}] {step}/{n_steps} FVE={fve:.4f} L0={l0:.0f} "
                      f"dead={dead:.3f} a={sae.scale_a.item():.4f} [{tps/1e3:.1f}k tok/s]", flush=True)
    sae.eval()
    return log


# =============================================================================
# Eval: reconstruction under a chosen score mode (the inference-time swap)
# =============================================================================

@torch.no_grad()
def eval_recon(sae, eval_shards, score_mode, label):
    sae.eval()
    cos, l0s = [], []
    tv = rv = 0.0
    dead = None
    for batch in eval_shards:
        batch = batch.to(DEVICE, dtype=torch.float32)
        x_hat, f = sae(batch, score_mode=score_mode)
        cos.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((f != 0).float().sum(-1).mean().item())
        tv += torch.var(batch, 0, unbiased=False).sum().item()
        rv += torch.var(batch - x_hat, 0, unbiased=False).sum().item()
        alive = (f != 0).sum(0) != 0
        dead = ~alive if dead is None else (dead & ~alive)
    fve = 1 - rv / tv if tv > 0 else 0
    res = {"fve": fve, "cos_recon": float(np.mean(cos)), "mean_l0": float(np.mean(l0s)),
           "dead_frac": dead.float().mean().item(), "alive": int((~dead).sum().item())}
    print(f"    [{label}] score={score_mode}: FVE={fve:.4f} L0={res['mean_l0']:.0f} "
          f"dead={res['dead_frac']:.3f}", flush=True)
    return res


@torch.no_grad()
def alive_set(sae, eval_shards, score_mode):
    sae.eval()
    alive = torch.zeros(sae.d_sae, dtype=torch.bool, device=next(sae.parameters()).device)
    for batch in eval_shards:
        _, f = sae(batch.to(DEVICE, dtype=torch.float32), score_mode=score_mode)
        alive |= (f != 0).sum(0) > 0
    return alive


def jaccard(a, b):
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 0.0


def run_probing(sae, score_mode, name):
    from benchmarks.adapter import BenchSAE
    from benchmarks.run_saebench import run_saebench
    sae.eval()
    dt = sae.W_enc.dtype

    def enc(x): return sae.encode(x.to(dtype=dt), score_mode=score_mode).to(dtype=x.dtype)
    def dec(f): return sae.decode(f.to(dtype=dt)).to(dtype=f.dtype)

    bench = BenchSAE(W_enc=sae.W_enc.detach().T, W_dec=F.normalize(sae.W_dec.detach(), dim=1),
                     b_enc=sae.b_enc.detach(), b_dec=sae.b_dec.detach(),
                     encode_fn=enc, decode_fn=dec, model_name=MODEL_NAME,
                     hook_layer=LAYER, device=DEVICE, dtype=DTYPE)
    return run_saebench(bench, sae_name=f"exp64-{name}", eval_types=["sparse_probing"],
                        output_dir=f"{EVAL_OUT_ROOT}/{name}", llm_batch_size=4, device=DEVICE)


def _probe_top1(sp):
    if isinstance(sp, dict):
        m = sp.get("sparse_probing", {})
        m = m.get("eval_result_metrics", {}).get("sae", m) if isinstance(m, dict) else m
        if isinstance(m, dict):
            return m.get("sae_top_1_test_accuracy")
    return None


# =============================================================================
# Main
# =============================================================================

def main(probe=True):
    print(f"Exp 64: score-swap at Qwen3-8B L{LAYER}/{N_TRAIN_TOKENS/1e6:.0f}M", flush=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    results = {"config": {"model": MODEL_NAME, "layer": LAYER, "d_sae": D_SAE, "k": K,
                          "n_train_tokens": N_TRAIN_TOKENS, "n_finetune_tokens": N_FINETUNE_TOKENS}}
    if RESULTS_PATH.exists():
        try: results = json.load(open(RESULTS_PATH))
        except Exception: pass
    results.setdefault("runs", {})

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE).eval()
    print("Caching activations to RAM...", flush=True)
    train_shards = e60.stream_acts_to_ram(model, tok, N_TRAIN_TOKENS, skip_rows=0)
    eval_shards = e60.stream_acts_to_ram(model, tok, N_EVAL_TOKENS, skip_rows=600_000)
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Train the pair ----
    def get_or_train(name, score_mode):
        ckpt = SAVE_DIR / f"{name}_final.pt"
        sae = SwappableBatchTopKSAE(D_MODEL, D_SAE, K, score_mode=score_mode).to(DEVICE)
        if ckpt.exists():
            sae.load_state_dict(torch.load(ckpt, weights_only=True)["state_dict"])
            sae.eval()
            print(f"  [{name}] loaded checkpoint", flush=True)
            return sae, None
        torch.manual_seed(SEED)
        sae = SwappableBatchTopKSAE(D_MODEL, D_SAE, K, score_mode=score_mode).to(DEVICE)
        log = train(sae, train_shards, N_STEPS, DECAY_START, name)
        torch.save({"state_dict": sae.state_dict(), "score_mode": score_mode}, ckpt)
        return sae, log

    std, std_log = get_or_train("standard", "inner")
    cos, cos_log = get_or_train("cosine", "cosine")
    results["runs"].setdefault("standard", {})["train_log"] = (std_log or [])[-5:]
    results["runs"].setdefault("cosine", {})["train_log"] = (cos_log or [])[-5:]

    # ---- PART 1: inference-time swap (both directions) ----
    print(f"\n{'#'*70}\n# PART 1: inference-time score swap\n{'#'*70}", flush=True)
    p1 = {}
    p1["standard_native_inner"] = eval_recon(std, eval_shards, "inner", "std@inner")
    p1["standard_swap_cosine"]  = eval_recon(std, eval_shards, "cosine", "std@cosine(exp29 fwd)")
    p1["cosine_native_cosine"]  = eval_recon(cos, eval_shards, "cosine", "cos@cosine")
    p1["cosine_swap_inner"]     = eval_recon(cos, eval_shards, "inner", "cos@inner(REVERSE,new)")
    results["part1_inference_swap"] = p1
    json.dump(results, open(RESULTS_PATH, "w"), indent=2, default=str)

    # ---- PART 2: continued-training swap ----
    print(f"\n{'#'*70}\n# PART 2: continued-training swap ({N_FINETUNE_TOKENS/1e6:.0f}M)\n{'#'*70}", flush=True)
    std_alive0 = alive_set(std, eval_shards, "inner")
    cos_alive0 = alive_set(cos, eval_shards, "cosine")
    W_std0 = F.normalize(std.W_dec.detach(), dim=1).clone()
    W_cos0 = F.normalize(cos.W_dec.detach(), dim=1).clone()

    # standard checkpoint, continue training under COSINE score
    std_ft = SwappableBatchTopKSAE(D_MODEL, D_SAE, K, score_mode="cosine").to(DEVICE)
    std_ft.load_state_dict(std.state_dict(), strict=False)
    ft1_log = train(std_ft, train_shards, N_FT_STEPS, int(0.8*N_FT_STEPS),
                    "std->cosine-ft", swap_score_to="cosine", reset_threshold=True)
    # cosine checkpoint, continue training under INNER score
    cos_ft = SwappableBatchTopKSAE(D_MODEL, D_SAE, K, score_mode="inner").to(DEVICE)
    cos_ft.load_state_dict(cos.state_dict(), strict=False)
    ft2_log = train(cos_ft, train_shards, N_FT_STEPS, int(0.8*N_FT_STEPS),
                    "cos->inner-ft", swap_score_to="inner", reset_threshold=True)

    p2 = {
        "std_then_cosine_ft": {
            "recon": eval_recon(std_ft, eval_shards, "cosine", "std->cos-ft"),
            "jaccard_vs_std_orig": jaccard(alive_set(std_ft, eval_shards, "cosine"), std_alive0),
            "jaccard_vs_cos_orig": jaccard(alive_set(std_ft, eval_shards, "cosine"), cos_alive0),
            "dec_cos_vs_std_orig": float((F.normalize(std_ft.W_dec.detach(), dim=1) * W_std0).sum(-1).mean()),
            "train_log": ft1_log[-5:],
        },
        "cos_then_inner_ft": {
            "recon": eval_recon(cos_ft, eval_shards, "inner", "cos->inner-ft"),
            "jaccard_vs_cos_orig": jaccard(alive_set(cos_ft, eval_shards, "inner"), cos_alive0),
            "jaccard_vs_std_orig": jaccard(alive_set(cos_ft, eval_shards, "inner"), std_alive0),
            "dec_cos_vs_cos_orig": float((F.normalize(cos_ft.W_dec.detach(), dim=1) * W_cos0).sum(-1).mean()),
            "train_log": ft2_log[-5:],
        },
    }
    results["part2_continued_training_swap"] = p2
    for nm, sae in [("std_then_cosine_ft", std_ft), ("cos_then_inner_ft", cos_ft)]:
        torch.save({"state_dict": sae.state_dict()}, SAVE_DIR / f"{nm}_final.pt")
    json.dump(results, open(RESULTS_PATH, "w"), indent=2, default=str)

    # ---- PART 1 probing (headline metric; reload model internally) ----
    if probe:
        print(f"\n{'#'*70}\n# PART 1 probing\n{'#'*70}", flush=True)
        for name, sae, mode in [
            ("standard_native_inner", std, "inner"),
            ("standard_swap_cosine", std, "cosine"),
            ("cosine_native_cosine", cos, "cosine"),
            ("cosine_swap_inner", cos, "inner"),
        ]:
            try:
                sp = run_probing(sae, mode, name)
                results["part1_inference_swap"][name]["probe_top1"] = _probe_top1(sp)
                print(f"    [{name}] top1={_probe_top1(sp)}", flush=True)
            except Exception as ex:
                import traceback; traceback.print_exc()
                results["part1_inference_swap"][name]["probe_error"] = str(ex)
            json.dump(results, open(RESULTS_PATH, "w"), indent=2, default=str)

    print(f"\nResults: {RESULTS_PATH}", flush=True)


def smoke():
    global DEVICE
    DEVICE = "cpu"; e60.DEVICE = "cpu"
    print("SMOKE: score-swap SAE on CPU...")
    torch.manual_seed(0)
    B, dm, ds, k = 64, 16, 128, 8
    x = torch.randn(B, dm) * 5.0
    for mode in ["inner", "cosine"]:
        sae = SwappableBatchTopKSAE(dm, ds, k, score_mode=mode)
        # TRAIN mode = batch-wide budget: total nonzeros ~ k*B (not per-row).
        sae.train()
        for sm in ["inner", "cosine"]:
            xh, f = sae(x, score_mode=sm)
            assert xh.shape == x.shape
            assert (f != 0).sum().item() <= k * B + 1, f"{mode}->{sm}: batch budget exceeded"
        # grads flow under the native score
        loss = (x - sae(x, score_mode=mode)[0]).pow(2).sum(-1).mean()
        loss.backward()
        assert sae.W_enc.grad is not None and torch.isfinite(sae.W_enc.grad).all()
        # EVAL mode with a set threshold = per-row sparsity cap respected
        sae.eval()
        with torch.no_grad():
            sae.threshold.fill_(0.0)  # nonneg threshold -> per-row gating active
            for sm in ["inner", "cosine"]:
                _, fe = sae(x, score_mode=sm)
                assert torch.isfinite(fe).all()
        print(f"  trained-as {mode}: native+swap forward OK (train batch-budget, eval thresh), grads finite")
    # alive_set + jaccard sanity
    sae = SwappableBatchTopKSAE(dm, ds, k, score_mode="inner").eval()
    a = alive_set(sae, [x], "inner"); b = alive_set(sae, [x], "inner")
    assert jaccard(a, b) == 1.0, "jaccard(self)!=1"
    print("  alive_set/jaccard OK")
    print("SMOKE PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        main(probe=not args.no_probe)
