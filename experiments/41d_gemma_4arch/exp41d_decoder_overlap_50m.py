"""
Experiment 41d — Decoder direction overlap on 50M Gemma checkpoints
====================================================================

Mirror of exp40h pointing at the exp41/ checkpoints (50M tokens, L7/L13/L19
on Gemma-2-2B). Tests whether the 5M overlap pattern — `no_C` mean
overlap ~45% higher than `standard` but no pairs above cos=0.5 — holds
at 10× scale.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp39_norm_preserving_sae import BatchTopKSAE, NORM_EPS
from exp39d_leave_one_out import NoC_SAE


DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
MODEL_NAME = "google/gemma-2-2b"
LAYERS = [7, 13, 19]
D_MODEL = 2304
D_SAE = 9216
K = 80
N_DIAG_TOKENS = 100_000
CTX_LEN = 256
COLLECTION_BATCH_SIZE = 16
ALIVE_THRESHOLD = 1e-4
CKPT_DIR = Path("checkpoints/exp41")
OUT_PATH = Path("experiments/exp41d_decoder_overlap_50m.json")

VARIANTS = [("standard", BatchTopKSAE), ("no_C", NoC_SAE)]


def load_sae(variant, cls, layer):
    p = CKPT_DIR / f"{variant}_L{layer}.pt"
    if not p.exists():
        return None
    sae = cls(D_MODEL, D_SAE, K).to(DEVICE)
    state = torch.load(p, map_location=DEVICE)
    sae.load_state_dict({k: v.float() for k, v in state.items()})
    sae.eval()
    return sae


@torch.no_grad()
def collect_diag(model, tokenizer, layer, n_tokens):
    print(f"  collecting {n_tokens:,} tokens at L{layer}...")
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True,
    )
    text_iter = iter(ds)
    captured = {}
    def hook(m, inp, out):
        captured["a"] = (out[0] if isinstance(out, tuple) else out).detach()
    h = model.model.layers[layer].register_forward_hook(hook)
    acts = torch.empty((n_tokens, D_MODEL), dtype=torch.float16)
    cursor = 0
    while cursor < n_tokens:
        batch_texts = []
        for _ in range(COLLECTION_BATCH_SIZE):
            try:
                row = next(text_iter)
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
        try:
            model(**inputs)
        except Exception:
            pass
        a = captured["a"]
        flat = a[inputs["attention_mask"].bool()]
        take = min(flat.shape[0], n_tokens - cursor)
        acts[cursor:cursor+take] = flat[:take].to("cpu", dtype=torch.float16)
        cursor += take
        del a, flat, inputs
    h.remove()
    print(f"  got {cursor:,} tokens in {time.time()-t0:.1f}s")
    return acts[:cursor]


@torch.no_grad()
def alive_mask(sae, acts, batch_size=4096):
    fire = torch.zeros(D_SAE, dtype=torch.float64, device=DEVICE)
    for i in range(0, acts.shape[0], batch_size):
        b = acts[i:i+batch_size].to(DEVICE, dtype=torch.float32)
        _, f = sae(b)
        fire += (f > 0).float().sum(dim=0).double()
    freq = (fire / acts.shape[0]).cpu().numpy()
    return freq > ALIVE_THRESHOLD


@torch.no_grad()
def overlap_histogram(W, alive):
    W = W[alive].float()
    n = W.shape[0]
    if n < 2:
        return {"n_alive": int(n), "summary": "too_few_alive"}
    W = F.normalize(W, dim=-1, eps=NORM_EPS).to(DEVICE)
    bins = np.linspace(-0.2, 1.0, 121)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    pair_count = 0
    chunk = 2048
    cols = torch.arange(n, device=DEVICE)
    top_vals_chunks = []
    for i in range(0, n, chunk):
        Wi = W[i:i+chunk]
        sim = Wi @ W.T
        ci = sim.shape[0]
        rows = torch.arange(i, i+ci, device=DEVICE).unsqueeze(1)
        upper = cols.unsqueeze(0) > rows
        vals = sim[upper].cpu().numpy().astype(np.float32)
        h, _ = np.histogram(vals, bins=bins)
        counts += h
        pair_count += vals.size
        if vals.size > 0:
            k_top = min(2000, vals.size)
            top_vals_chunks.append(np.partition(vals, -k_top)[-k_top:])
    top_vals = np.concatenate(top_vals_chunks) if top_vals_chunks else np.array([0.0])
    return {
        "n_alive": int(n),
        "n_pairs": int(pair_count),
        "p50": float(np.percentile(top_vals, 50)),
        "p90": float(np.percentile(top_vals, 90)),
        "p99": float(np.percentile(top_vals, 99)),
        "p999": float(np.percentile(top_vals, 99.9)),
        "max": float(top_vals.max()),
        "mean_overlap": float(top_vals.mean()),
        "frac_above_0_5": float((top_vals > 0.5).mean()),
        "frac_above_0_8": float((top_vals > 0.8).mean()),
    }


def main():
    print("Experiment 41d — decoder overlap on 50M Gemma SAEs")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=MODEL_DTYPE, device_map=DEVICE,
    )
    model.eval()

    results = {"layers": {}}
    for layer in LAYERS:
        print(f"\n=== L{layer} ===")
        diag = collect_diag(model, tokenizer, layer, N_DIAG_TOKENS)
        for vname, cls in VARIANTS:
            sae = load_sae(vname, cls, layer)
            if sae is None:
                print(f"  [{vname}] L{layer} missing, skip")
                continue
            print(f"  --- {vname} ---")
            alive = alive_mask(sae, diag)
            print(f"    alive: {alive.sum()} / {D_SAE}")
            ov = overlap_histogram(sae.W_dec.detach(), alive)
            ov["alive_count"] = int(alive.sum())
            results["layers"].setdefault(str(layer), {})[vname] = ov
            if "p99" in ov:
                print(f"    mean={ov['mean_overlap']:.3f} p99={ov['p99']:.3f} "
                      f">0.5={ov['frac_above_0_5']*100:.2f}% "
                      f">0.8={ov['frac_above_0_8']*100:.3f}%")
            del sae
            gc.collect()
            torch.cuda.empty_cache()
        del diag
        gc.collect()
        torch.cuda.empty_cache()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  {'L':>3s} {'Variant':<10s} {'alive':>7s} {'mean':>7s} {'p99':>7s} {'>0.5%':>8s}")
    for layer in LAYERS:
        for v, _ in VARIANTS:
            r = results["layers"].get(str(layer), {}).get(v, {})
            if "p99" not in r:
                continue
            print(f"  {layer:>3d} {v:<10s} {r['alive_count']:>7d} "
                  f"{r['mean_overlap']:>7.3f} {r['p99']:>7.3f} "
                  f"{r['frac_above_0_5']*100:>7.2f}%")
    print(f"\nResults: {OUT_PATH}")


if __name__ == "__main__":
    main()
