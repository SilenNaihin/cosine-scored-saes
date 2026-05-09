"""
exp42c-long: Train no_C at 16k on Gemma L13 for 150M tokens.

Tests how the SAEBench gap to the published 16k 500M BatchTopK closes
with more training. If 50M → 150M shows clear trajectory toward parity,
the full exp42b (500M) is worth the ~12 hr.

Streaming: 6 chunks of 25M each.
"""

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp39_norm_preserving_sae import NORM_EPS
from exp39d_leave_one_out import NoC_SAE


DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
STORAGE_DTYPE = torch.float16
CKPT_DTYPE = torch.float16
MODEL_NAME = "google/gemma-2-2b"

LAYER = 13
D_MODEL = 2304
D_SAE = 16384
K = 80
N_TRAIN_TOKENS = 150_000_000
CHUNK_TOKENS = 25_000_000
N_EVAL_TOKENS = 400_000
N_DIAG_TOKENS = 200_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
OUTLIER_MULTIPLIER = 10.0
BATCH_SIZE = 4096
LR = 3e-4
WARMUP_FRAC = 0.05
SEED = 42
LOG_EVERY = 250

SAVE_DIR = Path("checkpoints/exp42c_long")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = Path("experiments/exp42c_long_results.json")


class _EarlyStop(Exception):
    pass


def _collect_layer_acts(model, layer_idx, inputs):
    captured = {}
    def hook(m, inp, out):
        captured["a"] = (out[0] if isinstance(out, tuple) else out).detach()
        raise _EarlyStop
    h = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    except _EarlyStop:
        pass
    h.remove()
    return captured["a"]


def collect_chunk(model, tokenizer, layer_idx, n_tokens, doc_skip):
    print(f"  Collecting {n_tokens:,} tokens at L{layer_idx} (skip={doc_skip})...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    text_iter = iter(ds)
    if doc_skip > 0:
        for i, _ in enumerate(text_iter):
            if i >= doc_skip:
                break

    result = torch.empty((n_tokens, D_MODEL), dtype=STORAGE_DTYPE)
    cursor = 0
    docs_read = 0
    batch_count = 0
    while cursor < n_tokens:
        batch_texts = []
        for _ in range(COLLECTION_BATCH_SIZE):
            try:
                row = next(text_iter)
                docs_read += 1
                if len(row["text"]) > 50:
                    batch_texts.append(row["text"][:2048])
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
        take = min(flat.shape[0], n_tokens - cursor)
        result[cursor:cursor+take] = flat[:take].to("cpu", dtype=STORAGE_DTYPE)
        cursor += take
        del acts, flat, norms, inputs
        batch_count += 1
        if batch_count % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    result = result[:cursor]
    print(f"    {cursor:,} tokens, {docs_read} docs in {time.time()-t0:.1f}s")
    return result, doc_skip + docs_read


def train_streaming(name, sae, layer_idx, model, tokenizer, log,
                     save_at_steps=None):
    n_steps_total = N_TRAIN_TOKENS // BATCH_SIZE
    warmup = max(int(n_steps_total * WARMUP_FRAC), 1)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=LR)
    def lr_sched(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(n_steps_total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    sae.train()
    global_step = 0
    doc_cursor = 0
    t_start = time.time()
    chunks = (N_TRAIN_TOKENS + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    save_at_steps = save_at_steps or set()

    for chunk_i in range(chunks):
        target = min(CHUNK_TOKENS, N_TRAIN_TOKENS - chunk_i * CHUNK_TOKENS)
        print(f"\n  [{name}] chunk {chunk_i+1}/{chunks}: {target:,} tokens")
        chunk, doc_cursor = collect_chunk(model, tokenizer, layer_idx, target, doc_cursor)
        n_chunk_steps = chunk.shape[0] // BATCH_SIZE
        perm = torch.randperm(chunk.shape[0])
        chunk_t0 = time.time()
        for cs in range(1, n_chunk_steps + 1):
            global_step += 1
            start = (cs - 1) * BATCH_SIZE
            end = start + BATCH_SIZE
            idx = perm[start:end]
            batch = chunk[idx].to(DEVICE, dtype=torch.float32)
            x_hat, features = sae(batch)
            loss = (batch - x_hat).pow(2).sum(dim=-1).mean()  # MSE
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            sae.post_step()
            if global_step % LOG_EVERY == 0 or global_step == n_steps_total:
                with torch.no_grad():
                    l0 = (features != 0).float().sum(dim=-1).mean().item()
                    total_var = torch.var(batch, dim=0, unbiased=False).sum()
                    resid_var = torch.var(batch - x_hat, dim=0, unbiased=False).sum()
                    fve = (1 - resid_var / total_var).item() if total_var > 0 else 0
                    dead = (features.sum(dim=0) == 0).float().mean().item()
                    cos_r = F.cosine_similarity(batch, x_hat, dim=-1).mean().item()
                    norm_ratio = (x_hat.norm(dim=-1) / batch.norm(dim=-1).clamp_min(NORM_EPS)).mean().item()
                entry = {"step": global_step, "chunk": chunk_i+1, "loss": loss.item(),
                         "l0": l0, "fve": fve, "cos_recon": cos_r, "dead_frac": dead,
                         "norm_ratio": norm_ratio, "lr": scheduler.get_last_lr()[0],
                         "tokens_seen": global_step * BATCH_SIZE}
                log.append(entry)
                if global_step % (LOG_EVERY * 10) == 0 or global_step == n_steps_total:
                    print(f"    step {global_step:>5d}/{n_steps_total} chunk {chunk_i+1} | "
                          f"loss={loss.item():.4f} L0={l0:.0f} FVE={fve:.4f} "
                          f"dead={dead:.3f} nr={norm_ratio:.3f} "
                          f"tok={global_step*BATCH_SIZE/1e6:.1f}M")
            # Save intermediate checkpoint at requested steps
            if global_step in save_at_steps:
                ck = {k: v.to(CKPT_DTYPE) if torch.is_tensor(v) and v.is_floating_point() else v
                      for k, v in sae.state_dict().items()}
                torch.save(ck, SAVE_DIR / f"{name}_L{LAYER}_step{global_step}_tok{global_step*BATCH_SIZE//1_000_000}M.pt")
                print(f"    [chk] saved at step {global_step} ({global_step*BATCH_SIZE/1e6:.0f}M tokens)")
        print(f"    [{name}] chunk {chunk_i+1} trained in {time.time()-chunk_t0:.1f}s")
        del chunk
        gc.collect()
        torch.cuda.empty_cache()
    sae.eval()
    print(f"  [{name}] L{layer_idx} {N_TRAIN_TOKENS:,}-token training done in {time.time()-t_start:.1f}s")


@torch.no_grad()
def evaluate(name, sae, eval_data):
    sae.eval()
    n = eval_data.shape[0]
    total_var, resid_var, cos_sims, l0s, norm_ratios = 0.0, 0.0, [], [], []
    dead_counts = None
    for i in range(0, n, BATCH_SIZE):
        batch = eval_data[i:i+BATCH_SIZE].to(DEVICE, dtype=torch.float32)
        x_hat, features = sae(batch)
        total_var += torch.var(batch, dim=0, unbiased=False).sum().item()
        resid_var += torch.var(batch - x_hat, dim=0, unbiased=False).sum().item()
        cos_sims.append(F.cosine_similarity(batch, x_hat, dim=-1).mean().item())
        l0s.append((features != 0).float().sum(dim=-1).mean().item())
        norm_ratios.append((x_hat.norm(dim=-1) / batch.norm(dim=-1).clamp_min(NORM_EPS)).mean().item())
        alive = (features > 0).any(dim=0)
        dead_counts = (~alive) if dead_counts is None else dead_counts & ~alive
    dead_frac = dead_counts.float().mean().item() if dead_counts is not None else 1.0
    out = {
        "fve": float(1 - resid_var / total_var) if total_var > 0 else 0,
        "cos_recon": float(np.mean(cos_sims)),
        "l0": float(np.mean(l0s)),
        "norm_ratio": float(np.mean(norm_ratios)),
        "dead_frac": dead_frac,
        "alive_count": int((~dead_counts).sum().item()) if dead_counts is not None else 0,
    }
    print(f"    [{name}] FVE={out['fve']:.4f} cos={out['cos_recon']:.4f} "
          f"L0={out['l0']:.0f} dead={dead_frac*100:.1f}% alive={out['alive_count']}")
    return out


def main():
    print(f"exp42c-long — no_C 16k Gemma L{LAYER}, {N_TRAIN_TOKENS:,} tokens")
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=MODEL_DTYPE, device_map=DEVICE,
    )
    model.eval()

    eval_data, doc_eval = collect_chunk(model, tokenizer, LAYER, N_EVAL_TOKENS, 0)

    n_steps_total = N_TRAIN_TOKENS // BATCH_SIZE
    save_at = {n_steps_total // 3, 2 * n_steps_total // 3}  # save at 50M and 100M

    torch.manual_seed(SEED)
    sae = NoC_SAE(D_MODEL, D_SAE, K).to(DEVICE)
    log = []
    train_streaming("no_C", sae, LAYER, model, tokenizer, log, save_at_steps=save_at)
    recon = evaluate("no_C-150M", sae, eval_data)

    ckpt = {k: v.to(CKPT_DTYPE) if torch.is_tensor(v) and v.is_floating_point() else v
            for k, v in sae.state_dict().items()}
    torch.save(ckpt, SAVE_DIR / f"no_C_L{LAYER}_150M.pt")

    results = {"config": {"layer": LAYER, "n_train": N_TRAIN_TOKENS, "d_sae": D_SAE},
               "reconstruction_150M": recon, "training_log": log,
               "intermediate_steps": list(save_at)}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDone. Results: {RESULTS_PATH}")
    print(f"Checkpoints: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
