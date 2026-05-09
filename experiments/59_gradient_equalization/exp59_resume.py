"""
Exp59 resume: train perfeature_l2 + run SAEBench eval on all 4 variants.
Picks up from checkpoints saved by the main exp59 script.
"""

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

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"
D_MODEL = 4096
D_SAE = 65536
K = 80
LAYER = 18

N_TOKENS = 50_000_000
BATCH_SIZE = 2048
LR = 5e-5
AUXK_ALPHA = 1 / 32
TOP_K_AUX = D_MODEL // 2
DEAD_FEATURE_THRESHOLD = 10_000_000
SEED = 42
NORM_EPS = 1e-8

N_STEPS = N_TOKENS // BATCH_SIZE
WARMUP_STEPS = 1000
DECAY_START = int(0.8 * N_STEPS)

CACHE_DIR = Path("/mnt/nvme0/activations_cache/exp59")
CHECKPOINT_DIR = Path("/mnt/nvme0/checkpoints/exp59")
RESULTS_PATH = "experiments/exp59_results.json"

CHECKPOINT_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
CHECKPOINT_STEPS = sorted(set(int(f * N_STEPS) for f in CHECKPOINT_FRACS))
THRESHOLD_BETA = 0.999
THRESHOLD_START_STEP = 1000
LOG_EVERY = 500


# ─── Utilities (same as main script) ─────────────────────────────────────

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


# ─── SAE Architectures ────────────────────────────────────────────────────

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
        pre_acts = x_centered @ self.W_enc.T + self.b_enc
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

    def project_decoder_norms(self):
        with torch.no_grad():
            set_decoder_norm_to_unit_norm(self.W_dec.data)


class PerFeatureAdaptiveCosineSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=80):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.scale_a = nn.Parameter(torch.zeros(d_sae))
        self.scale_b = nn.Parameter(torch.full((d_sae,), math.log(math.sqrt(d_model))))
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.normalize_decoder = False
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

    def encode(self, x, return_active=False):
        x_centered = x - self.b_dec
        x_unit = F.normalize(x_centered, dim=-1)
        w_unit = F.normalize(self.W_enc, dim=-1)
        cos_sim = x_unit @ w_unit.T
        input_norm = x_centered.norm(dim=-1, keepdim=True).clamp(min=NORM_EPS)
        log_norm = torch.log(input_norm)
        scale = torch.exp(self.scale_a * log_norm + self.scale_b)
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

    def project_decoder_norms(self):
        with torch.no_grad():
            set_decoder_norm_to_unit_norm(self.W_dec.data)


# ─── Aux-k Loss ──────────────────────────────────────────────────────────

def get_auxiliary_loss(residual, post_relu_acts, num_tokens_since_fired):
    dead_mask = num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return torch.tensor(0.0, device=residual.device), n_dead

    k_aux = min(TOP_K_AUX, n_dead)
    auxk_latents = torch.where(
        dead_mask[None], post_relu_acts,
        torch.tensor(-torch.inf, device=post_relu_acts.device)
    )
    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
    auxk_buffer = torch.zeros_like(post_relu_acts)
    auxk_acts_BF = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
    return auxk_acts_BF, n_dead


# ─── Training ────────────────────────────────────────────────────────────

def train_variant(variant_name, loss_weight_mode, sae_class, data_iter):
    print(f"\n{'='*60}")
    print(f"Training: {variant_name} (loss_weight={loss_weight_mode})")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    sae = sae_class(D_MODEL, D_SAE, K).to(device=DEVICE, dtype=DTYPE)
    sae.train()

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    schedule_fn = make_lr_schedule(N_STEPS, WARMUP_STEPS, DECAY_START)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_fn)

    ckpt_dir = CHECKPOINT_DIR / variant_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_tokens_since_fired = torch.zeros(D_SAE, dtype=torch.long, device=DEVICE)
    b_dec_initialized = False

    log = []
    t0 = time.time()

    for step, batch in enumerate(data_iter):
        if step >= N_STEPS:
            break

        batch = batch.to(device=DEVICE, dtype=DTYPE, non_blocking=True)

        if not b_dec_initialized:
            with torch.no_grad():
                median = geometric_median(batch.float())
                sae.b_dec.data.copy_(median.to(sae.b_dec.dtype))
            b_dec_initialized = True
            print(f"  [{variant_name}] b_dec initialized (norm={median.norm():.1f})")

        x_hat, features, active_indices, post_relu_acts = sae(batch, return_active=True)
        per_token_loss = (batch - x_hat).pow(2).sum(dim=-1)

        if loss_weight_mode == "none":
            recon_loss = per_token_loss.mean()
        elif loss_weight_mode == "inv_norm":
            x_centered = batch - sae.b_dec.detach()
            input_norms = x_centered.norm(dim=-1).clamp(min=NORM_EPS)
            weights = 1.0 / input_norms
            weights = weights / weights.mean()
            recon_loss = (per_token_loss * weights).mean()
        elif loss_weight_mode == "inv_norm_sq":
            x_centered = batch - sae.b_dec.detach()
            input_norms = x_centered.norm(dim=-1).clamp(min=NORM_EPS)
            weights = 1.0 / (input_norms ** 2)
            weights = weights / weights.mean()
            recon_loss = (per_token_loss * weights).mean()
        else:
            raise ValueError(f"Unknown: {loss_weight_mode}")

        did_fire = torch.zeros(D_SAE, dtype=torch.bool, device=DEVICE)
        did_fire[active_indices] = True
        num_tokens_since_fired += batch.shape[0]
        num_tokens_since_fired[did_fire] = 0

        residual = (batch - x_hat).detach()
        auxk_result, n_dead = get_auxiliary_loss(residual, post_relu_acts, num_tokens_since_fired)

        if n_dead > 0:
            x_reconstruct_aux = auxk_result @ sae.W_dec
            auxk_l2 = (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            residual_mu = residual.mean(dim=0, keepdim=True)
            loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            auxk_loss = (auxk_l2 / loss_denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            auxk_loss = torch.tensor(0.0, device=DEVICE)

        loss = recon_loss + AUXK_ALPHA * auxk_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        normalize_dec = getattr(sae, "normalize_decoder", True)
        if normalize_dec and sae.W_dec.grad is not None:
            sae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                sae.W_dec.data, sae.W_dec.grad.data
            )

        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if normalize_dec:
            set_decoder_norm_to_unit_norm(sae.W_dec.data)

        if step > THRESHOLD_START_STEP:
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

        tokens_seen = (step + 1) * BATCH_SIZE
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                l0 = (features != 0).float().sum(dim=-1).mean().item()
                total_var = torch.var(batch, dim=0, unbiased=False).sum()
                resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                dead_frac = (num_tokens_since_fired >= DEAD_FEATURE_THRESHOLD).float().mean().item()

            entry = {
                "step": step, "tokens": tokens_seen,
                "recon_loss": recon_loss.item(),
                "auxk_loss": auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                "total_loss": loss.item(), "l0": l0, "fve": fve,
                "dead_frac": dead_frac, "n_dead": n_dead,
                "lr": scheduler.get_last_lr()[0],
            }
            log.append(entry)

            if step % 2000 == 0:
                elapsed = time.time() - t0
                tps = tokens_seen / elapsed
                print(f"  [{variant_name}] step {step}/{N_STEPS} "
                      f"({tokens_seen/1e6:.1f}M) recon={recon_loss.item():.1f} "
                      f"auxk={auxk_loss.item():.4f} FVE={fve:.4f} "
                      f"dead={dead_frac:.4f}({n_dead}) L0={l0:.0f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"[{tps/1e3:.1f}k tok/s]")

        if step in set(CHECKPOINT_STEPS):
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            torch.save({"state_dict": sae.state_dict(), "step": step, "log": log[-20:]}, ckpt_path)
            print(f"  [{variant_name}] Checkpoint at step {step}")

    final_path = ckpt_dir / "final.pt"
    torch.save({"state_dict": sae.state_dict(), "step": N_STEPS, "log": log[-20:]}, final_path)
    print(f"  [{variant_name}] Final checkpoint saved (threshold={sae.threshold.item():.4f})")

    return sae, log


# ─── Data ─────────────────────────────────────────────────────────────────

def shard_data_iter():
    shard_paths = sorted(CACHE_DIR.glob("shard_*.pt"))
    for shard_path in shard_paths:
        shard = torch.load(shard_path, weights_only=True)
        shard = shard.to(device=DEVICE, dtype=DTYPE)
        for i in range(shard.shape[0]):
            yield shard[i]
        del shard


# ─── Evaluation ───────────────────────────────────────────────────────────

def run_sparse_probing(sae, variant_name):
    from benchmarks.adapter import BenchSAE
    from benchmarks.run_saebench import run_saebench

    sae.eval()
    sae_dtype = sae.W_enc.dtype

    def encode_fn(x):
        return sae.encode(x.to(dtype=sae_dtype)).to(dtype=x.dtype)
    def decode_fn(f):
        return sae.decode(f.to(dtype=sae_dtype)).to(dtype=f.dtype)

    bench_sae = BenchSAE(
        W_enc=sae.W_enc.detach().T,
        W_dec=F.normalize(sae.W_dec.detach(), dim=1),
        b_enc=sae.b_enc.detach(),
        b_dec=sae.b_dec.detach(),
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        model_name=MODEL_NAME,
        hook_layer=LAYER,
        device=DEVICE,
        dtype=DTYPE,
    )

    out_dir = f"/mnt/nvme0/eval_results/exp59/{variant_name}"
    results = run_saebench(
        bench_sae,
        sae_name=f"exp59-{variant_name}",
        eval_types=["sparse_probing"],
        output_dir=out_dir,
        llm_batch_size=4,
        device=DEVICE,
    )
    return results


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Exp 59 RESUME: Train perfeature_l2 + eval all variants")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Phase 1: Train perfeature_l2 (the one that didn't get to run)
    print("\nPhase 1: Train perfeature_l2")
    data_iter = shard_data_iter()
    perfeature_sae, perfeature_log = train_variant(
        "perfeature_l2", "none", PerFeatureAdaptiveCosineSAE, data_iter
    )

    # Phase 2: Load other 3 from checkpoints + eval all 4
    print(f"\n{'='*70}")
    print("Phase 2: Load checkpoints and run SAEBench sparse probing")
    print(f"{'='*70}")

    all_saes = {}

    # Load standard, grad_eq_strong, grad_eq_mild from checkpoints
    for vname, sae_class in [
        ("standard", BatchTopKSAE),
        ("grad_eq_strong", BatchTopKSAE),
        ("grad_eq_mild", BatchTopKSAE),
    ]:
        ckpt_path = CHECKPOINT_DIR / vname / "final.pt"
        print(f"\n  Loading {vname} from {ckpt_path}")
        sae = sae_class(D_MODEL, D_SAE, K).to(device=DEVICE, dtype=DTYPE)
        ckpt = torch.load(ckpt_path, weights_only=True)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()
        all_saes[vname] = sae

    all_saes["perfeature_l2"] = perfeature_sae

    # Run eval
    all_results = {"config": {
        "model": MODEL_NAME, "layer": LAYER, "d_sae": D_SAE, "k": K,
        "n_tokens": N_TOKENS, "batch_size": BATCH_SIZE, "lr": LR,
        "warmup_steps": WARMUP_STEPS, "decay_start": DECAY_START,
        "seed": SEED, "ctx_len": 2048,
    }, "variants": {}}

    for variant_name, sae in all_saes.items():
        print(f"\n  Evaluating {variant_name}...")
        try:
            sp_results = run_sparse_probing(sae, variant_name)
            if "sparse_probing" in sp_results:
                sp = sp_results["sparse_probing"]
                if isinstance(sp, dict) and "eval_result_metrics" in sp:
                    metrics = sp["eval_result_metrics"]["sae"]
                    all_results["variants"][variant_name] = {"sparse_probing": metrics}
                    t1 = metrics.get('sae_top_1_test_accuracy', 0)
                    t5 = metrics.get('sae_top_5_test_accuracy', 0)
                    print(f"    top-1={t1:.4f} top-5={t5:.4f}")
                else:
                    all_results["variants"][variant_name] = {"sparse_probing": sp}
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            all_results["variants"][variant_name] = {"sparse_probing_error": str(e)}

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':25s} {'Top-1':>8s} {'Top-2':>8s} {'Top-5':>8s}")
    print("-" * 55)
    for vname, vdata in all_results["variants"].items():
        sp = vdata.get("sparse_probing", {})
        if isinstance(sp, dict):
            t1 = sp.get("sae_top_1_test_accuracy", 0) or 0
            t2 = sp.get("sae_top_2_test_accuracy", 0) or 0
            t5 = sp.get("sae_top_5_test_accuracy", 0) or 0
            print(f"{vname:25s} {t1:8.4f} {t2:8.4f} {t5:8.4f}")
        else:
            print(f"{vname:25s} ERROR")

    # Gap analysis
    print(f"\n{'='*70}")
    print("GAP ANALYSIS")
    print(f"{'='*70}")
    std = all_results["variants"].get("standard", {}).get("sparse_probing", {})
    mild = all_results["variants"].get("grad_eq_mild", {}).get("sparse_probing", {})
    strong = all_results["variants"].get("grad_eq_strong", {}).get("sparse_probing", {})
    cos = all_results["variants"].get("perfeature_l2", {}).get("sparse_probing", {})

    for k_name in ["sae_top_1_test_accuracy", "sae_top_2_test_accuracy", "sae_top_5_test_accuracy"]:
        s = std.get(k_name, 0) or 0
        m = mild.get(k_name, 0) or 0
        st = strong.get(k_name, 0) or 0
        c = cos.get(k_name, 0) or 0
        gap = c - s
        mild_closure = (m - s) / gap * 100 if gap > 0 else 0
        strong_closure = (st - s) / gap * 100 if gap > 0 else 0
        k_short = k_name.replace("sae_top_", "top-").replace("_test_accuracy", "")
        print(f"  {k_short}: std={s:.4f} mild={m:.4f} strong={st:.4f} cos={c:.4f}")
        print(f"    Gap (cos-std): {gap:.4f}")
        print(f"    Mild closure: {mild_closure:.1f}%")
        print(f"    Strong closure: {strong_closure:.1f}%")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
