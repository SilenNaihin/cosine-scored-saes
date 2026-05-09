"""
Experiment 36 — Phase 0: Eval Collection + Pipeline Validation
================================================================

Pre-collecting 500M training tokens is infeasible (4.1 TB per layer).
Training will stream activations inline instead.

This script does two things:
1. Collects 2M EVAL tokens per layer (~16 MB each) for consistent
   evaluation across checkpoints and architectures.
2. Validates the data pipeline: FineWeb/LMSYS mixing, tokenization,
   attention sink filtering, and norm statistics on ~100K tokens.

Output:
  data/qwen3_8b_500M_eval_L{9,18,27}.bin  (2M eval tokens each, ~16 MB)
  data/qwen3_8b_500M_meta.json  (norm stats, pipeline config)

Usage:
    ssh <server>     cd ~/MechInter--RNH
    CUDA_VISIBLE_DEVICES=1 python -u experiments/exp36_collect.py
"""

import json
import os
import time
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

DEVICE = "cuda"
MODEL_DTYPE = torch.bfloat16
STORAGE_DTYPE = np.float16

MODEL_NAME = "Qwen/Qwen3-8B"
LAYERS = [9, 18, 27]
D_MODEL = 4096

# --- Match adamkarvonen ---
CTX_LEN = 2048
FINEWEB_FRAC = 0.9
LMSYS_FRAC = 0.1
OUTLIER_MULTIPLIER = 10.0

# --- Collection ---
N_EVAL_TOKENS = 2_000_000
N_VALIDATION_TOKENS = 100_000  # quick pipeline check
BATCH_SIZE = 8  # sequences per forward pass

# --- Output ---
DATA_DIR = Path("data")
META_PATH = DATA_DIR / "qwen3_8b_500M_meta.json"

SEED = 42


# =============================================================================
# Multi-layer hook collection
# =============================================================================

class MultiLayerCollector:
    """Capture residual stream activations at multiple layers in one forward pass."""

    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = layer_indices
        self.captured = {}
        self.handles = []

    def _make_hook(self, layer_idx):
        def hook(module, inp, out):
            act = out[0].detach() if isinstance(out, tuple) else out.detach()
            self.captured[layer_idx] = act
        return hook

    def register(self):
        for layer_idx in self.layer_indices:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @torch.no_grad()
    def collect(self, inputs):
        self.captured = {}
        self.model(**inputs)
        return self.captured


# =============================================================================
# Data mixing: 90% FineWeb + 10% LMSYS-Chat
# =============================================================================

def make_text_iterator(seed=42, skip_docs=0):
    """Interleave FineWeb (90%) and LMSYS-Chat (10%) to match adamkarvonen."""
    rng = random.Random(seed)

    fineweb = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT",
        split="train", streaming=True,
    )
    fineweb_iter = iter(fineweb)

    # Skip docs if needed (for eval separation)
    if skip_docs > 0:
        print(f"  Skipping {skip_docs:,} FineWeb docs...")
        for i, _ in enumerate(fineweb_iter):
            if i >= skip_docs:
                break

    lmsys = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train", streaming=True,
    )
    lmsys_iter = iter(lmsys)

    if skip_docs > 0:
        lmsys_skip = int(skip_docs * LMSYS_FRAC / FINEWEB_FRAC)
        print(f"  Skipping {lmsys_skip:,} LMSYS docs...")
        for i, _ in enumerate(lmsys_iter):
            if i >= lmsys_skip:
                break

    while True:
        if rng.random() < FINEWEB_FRAC:
            try:
                row = next(fineweb_iter)
                text = row.get("text", "")
                if len(text) > 100:
                    yield text
            except StopIteration:
                break
        else:
            try:
                row = next(lmsys_iter)
                conversation = row.get("conversation", [])
                text = "\n".join(
                    turn.get("content", "")
                    for turn in conversation
                    if turn.get("content")
                )
                if len(text) > 100:
                    yield text
            except StopIteration:
                try:
                    row = next(fineweb_iter)
                    text = row.get("text", "")
                    if len(text) > 100:
                        yield text
                except StopIteration:
                    break


# =============================================================================
# Collection loop
# =============================================================================

def collect_activations(model, tokenizer, collector, text_iter, n_tokens, label="eval"):
    """Collect activations from all layers simultaneously."""
    print(f"\n  Collecting {label} activations ({n_tokens:,} tokens, ctx_len={CTX_LEN})...")
    t0 = time.time()

    # Pre-allocate memory-mapped files (small: ~16MB per layer for eval)
    storage_bytes = n_tokens * D_MODEL * 2
    print(f"  Storage per layer: {storage_bytes / 1e6:.1f} MB")

    paths = {}
    writers = {}
    for layer_idx in LAYERS:
        path = DATA_DIR / f"qwen3_8b_500M_{label}_L{layer_idx}.bin"
        paths[layer_idx] = path
        fp = np.memmap(str(path), dtype=STORAGE_DTYPE, mode="w+", shape=(n_tokens, D_MODEL))
        writers[layer_idx] = fp

    tokens_collected = 0
    norm_stats = {l: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "max": 0.0} for l in LAYERS}
    tokens_filtered = 0
    last_report = 0

    while tokens_collected < n_tokens:
        batch_texts = []
        for _ in range(BATCH_SIZE):
            try:
                text = next(text_iter)
                batch_texts.append(text[:CTX_LEN * 6])
            except StopIteration:
                break
        if not batch_texts:
            print(f"  WARNING: Text iterator exhausted at {tokens_collected:,} tokens")
            break

        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=CTX_LEN,
        ).to(DEVICE)

        layer_acts = collector.collect(inputs)
        mask = inputs["attention_mask"].bool()

        # Flatten all layers and compute unified outlier mask
        flat_per_layer = {}
        norms_per_layer = {}
        for layer_idx in LAYERS:
            acts = layer_acts[layer_idx]
            flat_per_layer[layer_idx] = acts[mask]
            norms_per_layer[layer_idx] = flat_per_layer[layer_idx].float().norm(dim=-1)

        # Unified outlier filter: keep only tokens clean at every layer
        n_flat = flat_per_layer[LAYERS[0]].shape[0]
        keep = torch.ones(n_flat, dtype=torch.bool, device=DEVICE)
        for layer_idx in LAYERS:
            median = norms_per_layer[layer_idx].median()
            if median > 0:
                keep &= norms_per_layer[layer_idx] < median * OUTLIER_MULTIPLIER

        n_filtered = (~keep).sum().item()
        tokens_filtered += n_filtered

        n_valid = keep.sum().item()
        if n_valid == 0:
            continue

        remaining = n_tokens - tokens_collected
        if n_valid > remaining:
            indices = keep.nonzero(as_tuple=False).squeeze(-1)[:remaining]
            keep = torch.zeros_like(keep)
            keep[indices] = True
            n_valid = remaining

        end = tokens_collected + n_valid
        for layer_idx in LAYERS:
            flat = flat_per_layer[layer_idx][keep]
            norms = norms_per_layer[layer_idx][keep]

            writers[layer_idx][tokens_collected:end] = flat.cpu().to(torch.float16).numpy()

            norm_stats[layer_idx]["sum"] += norms.sum().item()
            norm_stats[layer_idx]["sum_sq"] += norms.pow(2).sum().item()
            norm_stats[layer_idx]["count"] += n_valid
            norm_stats[layer_idx]["max"] = max(norm_stats[layer_idx]["max"], norms.max().item())

        tokens_collected = end

        if tokens_collected - last_report >= 200_000:
            elapsed = time.time() - t0
            rate = tokens_collected / elapsed if elapsed > 0 else 0
            eta = (n_tokens - tokens_collected) / rate if rate > 0 else 0
            pct = tokens_collected / n_tokens * 100
            s = norm_stats[LAYERS[0]]
            mean_norm = s["sum"] / s["count"] if s["count"] > 0 else 0
            print(f"  [{label}] {tokens_collected/1e6:.1f}M / {n_tokens/1e6:.0f}M "
                  f"({pct:.1f}%) | {rate/1e3:.1f}K tok/s | "
                  f"ETA {eta/60:.0f}m | filtered {tokens_filtered:,} | "
                  f"L{LAYERS[0]} norm mean={mean_norm:.1f}")
            last_report = tokens_collected

    # Final flush
    for fp in writers.values():
        fp.flush()

    elapsed = time.time() - t0

    final_stats = {}
    for layer_idx in LAYERS:
        s = norm_stats[layer_idx]
        n = s["count"]
        if n > 0:
            mean = s["sum"] / n
            var = s["sum_sq"] / n - mean ** 2
            std = var ** 0.5 if var > 0 else 0
        else:
            mean = std = 0
        final_stats[layer_idx] = {
            "mean_norm": mean, "std_norm": std, "max_norm": s["max"], "count": n,
        }
        print(f"  L{layer_idx}: mean_norm={mean:.1f}, std={std:.1f}, "
              f"max={s['max']:.1f}, count={n:,}")

    print(f"  {label} done: {tokens_collected:,} tokens in {elapsed:.0f}s "
          f"({tokens_collected/elapsed/1e3:.1f}K tok/s)")
    print(f"  Filtered {tokens_filtered:,} outlier tokens total")

    return tokens_collected, final_stats, paths


# =============================================================================
# Main
# =============================================================================

def main():
    print("Experiment 36 — Phase 0: Eval Collection + Pipeline Validation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Data: {FINEWEB_FRAC*100:.0f}% FineWeb + {LMSYS_FRAC*100:.0f}% LMSYS-Chat")
    print(f"Context length: {CTX_LEN}")
    print(f"Outlier filter: >{OUTLIER_MULTIPLIER}x median norm")
    print(f"Eval tokens: {N_EVAL_TOKENS:,}")
    print(f"Validation tokens: {N_VALIDATION_TOKENS:,}")
    eval_bytes = N_EVAL_TOKENS * D_MODEL * 2
    print(f"Eval storage: {eval_bytes / 1e6:.1f} MB per layer "
          f"({eval_bytes * len(LAYERS) / 1e6:.1f} MB total)")

    random.seed(SEED)
    torch.manual_seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=MODEL_DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    collector = MultiLayerCollector(model, LAYERS)
    collector.register()

    meta = {
        "model_name": MODEL_NAME,
        "layers": LAYERS,
        "d_model": D_MODEL,
        "ctx_len": CTX_LEN,
        "storage_dtype": "float16",
        "outlier_multiplier": OUTLIER_MULTIPLIER,
        "data_mixture": f"{FINEWEB_FRAC*100:.0f}% FineWeb + {LMSYS_FRAC*100:.0f}% LMSYS-Chat",
        "seed": SEED,
        "note": "Training activations streamed inline (4.1 TB/layer too large for disk)",
    }

    # --- Step 1: Pipeline validation (100K tokens, no save) ---
    print(f"\n{'='*70}")
    print(f"  PIPELINE VALIDATION: {N_VALIDATION_TOKENS:,} tokens")
    print(f"{'='*70}")
    val_text_iter = make_text_iterator(seed=SEED)
    n_val, val_stats, _ = collect_activations(
        model, tokenizer, collector, val_text_iter, N_VALIDATION_TOKENS, label="validation"
    )
    meta["validation"] = {
        "n_tokens": n_val,
        "norm_stats": {str(k): v for k, v in val_stats.items()},
    }

    # Clean up validation files (we only wanted stats)
    for layer_idx in LAYERS:
        vpath = DATA_DIR / f"qwen3_8b_500M_validation_L{layer_idx}.bin"
        if vpath.exists():
            vpath.unlink()

    # --- Step 2: Eval set (2M tokens, saved) ---
    print(f"\n{'='*70}")
    print(f"  EVAL SET: {N_EVAL_TOKENS:,} tokens")
    print(f"{'='*70}")
    eval_text_iter = make_text_iterator(seed=SEED + 1, skip_docs=500_000)
    n_eval, eval_stats, eval_paths = collect_activations(
        model, tokenizer, collector, eval_text_iter, N_EVAL_TOKENS, label="eval"
    )
    meta["eval"] = {
        "n_tokens": n_eval,
        "norm_stats": {str(k): v for k, v in eval_stats.items()},
        "files": {str(k): str(v) for k, v in eval_paths.items()},
    }

    collector.remove()

    # Save meta
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Validation: {n_val:,} tokens (stats only, files deleted)")
    print(f"  Eval: {n_eval:,} tokens (saved)")
    print(f"\n  Eval files:")
    for layer_idx in LAYERS:
        size = os.path.getsize(eval_paths[layer_idx]) / 1e6
        print(f"    L{layer_idx}: {eval_paths[layer_idx]} ({size:.1f} MB)")
    print(f"  Meta: {META_PATH}")

    print(f"\n  Norm statistics (validation, representative of training data):")
    for layer_idx in LAYERS:
        vs = val_stats[layer_idx]
        print(f"    L{layer_idx}: mean={vs['mean_norm']:.1f}, "
              f"std={vs['std_norm']:.1f}, max={vs['max_norm']:.1f}")

    total_mb = sum(os.path.getsize(p) for p in eval_paths.values()) / 1e6
    print(f"\n  Total disk usage: {total_mb:.1f} MB")
    print("\nPipeline validated. Ready for Phase 1 training (inline streaming).")


if __name__ == "__main__":
    main()
