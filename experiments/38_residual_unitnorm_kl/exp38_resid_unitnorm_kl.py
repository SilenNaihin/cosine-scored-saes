"""
Experiment 38: Residual Unit-Norm KL Sweep
==========================================

Normalize the residual stream to unit norm at one layer at a time, then measure
the final-output KL divergence versus the unmodified forward pass.

This is an eval-only experiment on Qwen3-8B intended to test whether residual
stream norm matters more at later layers, possibly up to a peak.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
      uv run python experiments/exp38_resid_unitnorm_kl.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_LAYERS = list(range(36))
DEFAULT_TOKEN_BUDGET = 50_000
DEFAULT_CTX_LEN = 256
DEFAULT_BATCH_SEQS = 4
DEFAULT_SEED = 42
DEFAULT_SKIP_DOCS = 500_000
DEFAULT_WRITE_EVERY = 5
DEFAULT_EPS = 1e-8
DEFAULT_JSON_PATH = "experiments/exp38_resid_unitnorm_kl_results.json"
DEFAULT_CSV_PATH = "experiments/exp38_resid_unitnorm_kl_summary.csv"
DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_NAME = "Qwen/Qwen3-8B"


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def count_valid_tokens(attention_mask: torch.Tensor) -> int:
    return int(attention_mask.sum().item())


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def flatten_valid_positions(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if values.ndim < 3:
        raise ValueError("values must have shape [batch, seq, ...]")
    if attention_mask.shape != values.shape[:2]:
        raise ValueError("attention_mask must match the first two value dims")
    return values[attention_mask.bool()]


def scatter_valid_positions(
    base: torch.Tensor,
    replacement: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if base.ndim < 3:
        raise ValueError("base must have shape [batch, seq, ...]")
    if attention_mask.shape != base.shape[:2]:
        raise ValueError("attention_mask must match the first two base dims")

    mask = attention_mask.bool()
    n_valid = int(mask.sum().item())
    if replacement.shape[0] != n_valid:
        raise ValueError("replacement rows must equal the number of valid positions")

    result = base.clone()
    result[mask] = replacement.to(device=base.device, dtype=base.dtype)
    return result


def masked_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    valid_student = flatten_valid_positions(student_logits, attention_mask)
    valid_teacher = flatten_valid_positions(teacher_logits, attention_mask)
    if valid_student.numel() == 0:
        raise ValueError("attention_mask selects no valid positions")

    return F.kl_div(
        torch.log_softmax(valid_student.float(), dim=-1),
        torch.log_softmax(valid_teacher.float(), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def unit_normalize_last_dim_f32(
    values: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("values must have at least 2 dims")
    if eps <= 0:
        raise ValueError("eps must be positive")

    values_f32 = values.float()
    norms = values_f32.norm(dim=-1, keepdim=True)
    return values_f32 / norms.clamp_min(eps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in DEFAULT_LAYERS),
        help="Comma-separated layer indices to sweep.",
    )
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--batch-seqs", type=int, default=DEFAULT_BATCH_SEQS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-docs", type=int, default=DEFAULT_SKIP_DOCS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--write-every", type=int, default=DEFAULT_WRITE_EVERY)
    parser.add_argument("--output-json", default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-csv", default=DEFAULT_CSV_PATH)
    return parser


@torch.no_grad()
def teacher_forward_all_layers(
    model: AutoModelForCausalLM,
    layer_indices: list[int],
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, hook_inputs, hook_output):
            act = hook_output[0] if isinstance(hook_output, tuple) else hook_output
            captured[layer_idx] = act.detach()

        return hook

    for layer_idx in layer_indices:
        handle = model.model.layers[layer_idx].register_forward_hook(
            make_hook(layer_idx)
        )
        handles.append(handle)

    try:
        outputs = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer_idx for layer_idx in layer_indices if layer_idx not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture activations for layers: {missing}")

    return captured, outputs.logits.detach()


def student_forward_with_replacement(
    model: AutoModelForCausalLM,
    layer_idx: int,
    inputs: dict[str, torch.Tensor],
    replacement: torch.Tensor,
) -> torch.Tensor:
    def hook(module, hook_inputs, hook_output):
        if isinstance(hook_output, tuple):
            return (replacement,) + hook_output[1:]
        return replacement

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    return outputs.logits


class SequenceStream:
    def __init__(
        self,
        tokenizer,
        *,
        ctx_len: int,
        collection_batch_size: int,
        seed: int,
        skip_docs: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.collection_batch_size = collection_batch_size
        self.seed = seed
        self.skip_docs = skip_docs
        self._text_iter = None
        self._init_dataset()

    def _init_dataset(self) -> None:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        self._text_iter = iter(dataset)
        for _ in range(self.skip_docs):
            try:
                next(self._text_iter)
            except StopIteration:
                break

    def _collect_texts(self, n_sequences: int) -> list[str]:
        texts: list[str] = []
        while len(texts) < n_sequences:
            try:
                row = next(self._text_iter)
            except StopIteration:
                self._init_dataset()
                row = next(self._text_iter)

            text = row["text"]
            if len(text) <= 50:
                continue
            texts.append(text[:8192])
        return texts

    def get_batch(self, n_sequences: int) -> dict[str, torch.Tensor]:
        texts = self._collect_texts(n_sequences)
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.ctx_len,
        )


def collect_eval_batches(
    tokenizer,
    *,
    token_budget: int,
    ctx_len: int,
    batch_seqs: int,
    seed: int,
    skip_docs: int,
) -> list[dict[str, torch.Tensor]]:
    print(f"Collecting eval batches ({token_budget:,} tokens target)...")
    stream = SequenceStream(
        tokenizer,
        ctx_len=ctx_len,
        collection_batch_size=batch_seqs,
        seed=seed,
        skip_docs=skip_docs,
    )
    batches: list[dict[str, torch.Tensor]] = []
    tokens = 0
    while tokens < token_budget:
        batch = stream.get_batch(batch_seqs)
        tokens += count_valid_tokens(batch["attention_mask"])
        batches.append({key: value.cpu() for key, value in batch.items()})
    print(f"  Collected {tokens:,} eval tokens across {len(batches)} batches")
    return batches


def summarize_layer_batches(
    batch_kls: list[float],
    batch_tokens: list[int],
) -> dict[str, float | int]:
    if not batch_kls:
        return {
            "weighted_mean_kl": 0.0,
            "batch_mean_kl": 0.0,
            "std_kl": 0.0,
            "sem_kl": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n_batches": 0,
            "n_tokens": 0,
        }
    if len(batch_kls) != len(batch_tokens):
        raise ValueError("batch_kls and batch_tokens must have matching lengths")

    kl_array = np.asarray(batch_kls, dtype=np.float64)
    token_array = np.asarray(batch_tokens, dtype=np.float64)
    if np.any(token_array <= 0):
        raise ValueError("batch_tokens must all be positive")

    n_batches = len(batch_kls)
    batch_mean = float(kl_array.mean())
    std = float(kl_array.std(ddof=1)) if n_batches > 1 else 0.0
    sem = std / np.sqrt(n_batches) if n_batches > 1 else 0.0
    ci_half = 1.96 * sem
    return {
        "weighted_mean_kl": float(np.average(kl_array, weights=token_array)),
        "batch_mean_kl": batch_mean,
        "std_kl": std,
        "sem_kl": sem,
        "ci95_low": batch_mean - ci_half,
        "ci95_high": batch_mean + ci_half,
        "n_batches": n_batches,
        "n_tokens": int(token_array.sum()),
    }


def build_results_payload(
    *,
    config: dict[str, Any],
    per_layer: dict[int, dict[str, list[float] | list[int]]],
    status: str,
    started_at: str,
    finished_at: str | None,
) -> dict[str, Any]:
    layers_payload: dict[str, Any] = {}
    for layer_idx in sorted(per_layer):
        batch_kls = [float(value) for value in per_layer[layer_idx]["batch_kls"]]
        batch_tokens = [int(value) for value in per_layer[layer_idx]["batch_tokens"]]
        layers_payload[str(layer_idx)] = {
            **summarize_layer_batches(batch_kls, batch_tokens),
            "batch_kls": batch_kls,
            "batch_tokens": batch_tokens,
        }
    return {
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "config": config,
        "layers": layers_payload,
    }


def write_summary_csv(payload: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "weighted_mean_kl",
        "batch_mean_kl",
        "std_kl",
        "sem_kl",
        "ci95_low",
        "ci95_high",
        "n_batches",
        "n_tokens",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for layer_str in sorted(payload["layers"], key=lambda value: int(value)):
            summary = payload["layers"][layer_str]
            writer.writerow(
                {
                    "layer": int(layer_str),
                    **{key: summary[key] for key in fieldnames if key != "layer"},
                }
            )


def write_outputs(payload: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    write_summary_csv(payload, csv_path)


@torch.no_grad()
def run_sweep(
    model: AutoModelForCausalLM,
    eval_batches: list[dict[str, torch.Tensor]],
    layer_indices: list[int],
    *,
    eps: float,
    write_every: int,
    json_path: Path,
    csv_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    per_layer: dict[int, dict[str, list[float] | list[int]]] = {
        layer_idx: {"batch_kls": [], "batch_tokens": []} for layer_idx in layer_indices
    }
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_batches = len(eval_batches)
    t0 = time.time()

    for batch_idx, cpu_batch in enumerate(eval_batches, start=1):
        batch = move_batch_to_device(cpu_batch, DEVICE)
        batch_tokens = count_valid_tokens(batch["attention_mask"])
        teacher_acts_by_layer, teacher_logits = teacher_forward_all_layers(
            model, layer_indices, batch
        )

        for layer_idx in layer_indices:
            teacher_acts = teacher_acts_by_layer[layer_idx]
            valid_teacher_acts = flatten_valid_positions(
                teacher_acts, batch["attention_mask"]
            )
            normalized_valid_acts = unit_normalize_last_dim_f32(
                valid_teacher_acts,
                eps=eps,
            )
            replacement = scatter_valid_positions(
                teacher_acts,
                normalized_valid_acts.to(dtype=teacher_acts.dtype),
                batch["attention_mask"],
            )
            student_logits = student_forward_with_replacement(
                model,
                layer_idx,
                batch,
                replacement,
            )
            kl_value = masked_token_kl(
                student_logits,
                teacher_logits,
                batch["attention_mask"],
            ).item()
            per_layer[layer_idx]["batch_kls"].append(float(kl_value))
            per_layer[layer_idx]["batch_tokens"].append(batch_tokens)

        if batch_idx % max(write_every, 1) == 0 or batch_idx == total_batches:
            elapsed = time.time() - t0
            print(
                f"  Batches {batch_idx}/{total_batches} complete "
                f"({elapsed / 60:.1f} min elapsed)"
            )
            partial_payload = build_results_payload(
                config=config,
                per_layer=per_layer,
                status="running",
                started_at=started_at,
                finished_at=None,
            )
            write_outputs(partial_payload, json_path, csv_path)

        del batch, teacher_acts_by_layer, teacher_logits

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = build_results_payload(
        config=config,
        per_layer=per_layer,
        status="complete",
        started_at=started_at,
        finished_at=finished_at,
    )
    write_outputs(payload, json_path, csv_path)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    layer_indices = parse_int_csv(args.layers)
    if not layer_indices:
        raise ValueError("At least one layer must be provided")
    if args.eps <= 0:
        raise ValueError("--eps must be positive")

    print("Experiment 38: Residual Unit-Norm KL Sweep")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {layer_indices}")
    print(f"Token budget: {args.token_budget:,}")
    print(f"ctx_len={args.ctx_len}, batch_seqs={args.batch_seqs}, eps={args.eps}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    n_model_layers = len(model.model.layers)
    invalid_layers = [
        layer for layer in layer_indices if layer < 0 or layer >= n_model_layers
    ]
    if invalid_layers:
        raise ValueError(
            f"Requested layers out of range for model with {n_model_layers} layers: {invalid_layers}"
        )

    eval_batches = collect_eval_batches(
        tokenizer,
        token_budget=args.token_budget,
        ctx_len=args.ctx_len,
        batch_seqs=args.batch_seqs,
        seed=args.seed,
        skip_docs=args.skip_docs,
    )
    actual_eval_tokens = sum(
        count_valid_tokens(batch["attention_mask"]) for batch in eval_batches
    )
    print(f"Actual eval tokens: {actual_eval_tokens:,}")

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    config = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "layers": layer_indices,
        "n_model_layers": n_model_layers,
        "token_budget": args.token_budget,
        "actual_eval_tokens": actual_eval_tokens,
        "ctx_len": args.ctx_len,
        "batch_seqs": args.batch_seqs,
        "seed": args.seed,
        "skip_docs": args.skip_docs,
        "eps": args.eps,
        "hostname": os.uname().nodename,
        "output_json": str(json_path),
        "output_csv": str(csv_path),
    }

    payload = run_sweep(
        model,
        eval_batches,
        layer_indices,
        eps=args.eps,
        write_every=args.write_every,
        json_path=json_path,
        csv_path=csv_path,
        config=config,
    )

    print("\nTop layers by weighted mean KL:")
    ranked = sorted(
        (
            (int(layer_str), summary["weighted_mean_kl"])
            for layer_str, summary in payload["layers"].items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    for layer_idx, mean_kl in ranked[:5]:
        print(f"  L{layer_idx:02d}: weighted_mean_KL={mean_kl:.6f}")

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
