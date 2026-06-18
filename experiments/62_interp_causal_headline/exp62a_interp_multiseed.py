"""
Experiment 62a: Headline auto-interpretability, multi-seed (camera-ready B5 Tier 1).

Re-runs exp53's describe-then-predict auto-interp at 1000 features/arm across the
3 SAE-training seeds {42, 123, 456} on the recommended variants only
(standard, adaptive_l2 = global-a, perfeature_l2 = per-feature + delta), at the
true headline recipe (Qwen3-8B L18, 500M tokens, d_sae=65,536, k=80). One unified
protocol replacing exp33 (50M/L27/200) and exp40 (100M/no_C/>=4). Produces
mean +/- SD interp rate per arm + frequency-stratified breakdown.

BLOCKED on exp61's saved 500M L18 checkpoints (A1 multi-seed, box-8, ETA ~2026-06-19).
This is a thin wrapper over exp53 (same pattern as exp61 over exp40): it imports the
exp53 module, overrides config, and loops seeds. No logic is duplicated.

Checkpoint layout expected (one subdir per seed, matching exp61's SAVE_DIR convention):
    $EXP62_CKPT_DIR/seed{S}/{standard,adaptive_l2,perfeature_l2}_L18_final.pt

Run on box-8 (after exp61 finishes), GPU 1:
    ssh h100-dev-box-8
    cd ~/cosine-scored-saes && source .venv/bin/activate
    export EXP62_CKPT_DIR=/mnt/checkpoints/exp61     # <- set to exp61's SAVE_DIR
    # Phase 1 (GPU): collect activating contexts for all seeds
    CUDA_VISIBLE_DEVICES=1 python3 experiments/62_interp_causal_headline/exp62a_interp_multiseed.py --collect
    # Phase 2 (LLM API): score (no GPU; needs AWS Bedrock creds)
    python3 experiments/62_interp_causal_headline/exp62a_interp_multiseed.py --score
    # Phase 3: aggregate seeds -> mean +/- SD
    python3 experiments/62_interp_causal_headline/exp62a_interp_multiseed.py --aggregate
"""

import argparse
import importlib.util
import json
import os
import statistics
from pathlib import Path

# =============================================================================
# Configuration (overrides exp53)
# =============================================================================

HERE = Path(__file__).resolve().parent
EXP53 = HERE.parent / "53_llm_interp_500m" / "exp53_llm_interp_500m.py"

SEEDS = [42, 123, 456]                 # matched to exp61 SAE-training seeds
ARMS = ["standard", "adaptive_l2", "perfeature_l2"]   # recommended variants only; no_C/ref dropped
N_FEATURES_PER_SAE = 1000              # exp53 used 200; tighten CIs
N_HIGH_FREQ = 250
N_MED_FREQ = 500
N_LOW_FREQ = 250

# exp61 saves one checkpoint dir per seed. Set EXP62_CKPT_DIR to exp61's SAVE_DIR root.
CKPT_DIR = os.environ.get("EXP62_CKPT_DIR", "/mnt/checkpoints/exp61")

RESULTS_DIR = HERE
AGG_PATH = HERE / "exp62a_results.json"


def _load_exp53_module():
    """Import the exp53 script as a module so we reuse its harness verbatim."""
    spec = importlib.util.spec_from_file_location("exp53", EXP53)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _configure(mod, seed):
    """Override exp53 module-level config for a given seed."""
    mod.SEED = seed
    mod.N_FEATURES_PER_SAE = N_FEATURES_PER_SAE
    mod.N_HIGH_FREQ = N_HIGH_FREQ
    mod.N_MED_FREQ = N_MED_FREQ
    mod.N_LOW_FREQ = N_LOW_FREQ
    mod.VARIANT_NAMES = list(ARMS)
    # Per-seed checkpoint paths (exp61 SAVE_DIR/seed{S}/{arm}_L18_final.pt).
    seed_dir = Path(CKPT_DIR) / f"seed{seed}"
    mod.CKPTS = {arm: str(seed_dir / f"{arm}_L18_final.pt") for arm in ARMS}
    # Per-seed artifact paths so seeds don't clobber each other.
    mod.CONTEXTS_PATH = str(RESULTS_DIR / f"exp62a_contexts_seed{seed}.json")
    mod.RESULTS_PATH = str(RESULTS_DIR / f"exp62a_results_seed{seed}.json")
    # Restrict SAE_CLASSES to the arms we load (exp53 keys it by name).
    if hasattr(mod, "SAE_CLASSES"):
        mod.SAE_CLASSES = {k: v for k, v in mod.SAE_CLASSES.items() if k in ARMS}
    return mod


def _missing_checkpoints():
    missing = []
    for seed in SEEDS:
        seed_dir = Path(CKPT_DIR) / f"seed{seed}"
        for arm in ARMS:
            p = seed_dir / f"{arm}_L18_final.pt"
            if not p.exists():
                missing.append(str(p))
    return missing


def run_collect():
    missing = _missing_checkpoints()
    if missing:
        raise SystemExit(
            "BLOCKED: exp61 checkpoints not found. Set EXP62_CKPT_DIR to exp61's "
            f"SAVE_DIR. Missing {len(missing)} files, e.g.:\n  " + "\n  ".join(missing[:6])
        )
    for seed in SEEDS:
        print(f"\n{'='*70}\nExp 62a COLLECT — seed {seed}\n{'='*70}")
        mod = _configure(_load_exp53_module(), seed)
        mod.collect_contexts()


def run_score():
    for seed in SEEDS:
        print(f"\n{'='*70}\nExp 62a SCORE — seed {seed}\n{'='*70}")
        mod = _configure(_load_exp53_module(), seed)
        if not Path(mod.CONTEXTS_PATH).exists():
            print(f"  skip seed {seed}: no contexts at {mod.CONTEXTS_PATH} (run --collect)")
            continue
        mod.score_features()


def run_aggregate():
    """Aggregate per-seed results into mean +/- SD per arm + frequency breakdown."""
    per_seed = {}
    for seed in SEEDS:
        p = RESULTS_DIR / f"exp62a_results_seed{seed}.json"
        if p.exists():
            per_seed[seed] = json.loads(p.read_text())
    if not per_seed:
        raise SystemExit("No per-seed results found; run --score first.")

    def collect(metric_key):
        out = {}
        for arm in ARMS:
            vals = []
            for seed, res in per_seed.items():
                block = res.get(arm, {})
                v = block.get(metric_key)
                if v is not None:
                    vals.append(v)
            if vals:
                out[arm] = {
                    "n_seeds": len(vals),
                    "mean": statistics.mean(vals),
                    "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "values": vals,
                }
        return out

    agg = {
        "config": {
            "seeds": list(per_seed.keys()),
            "arms": ARMS,
            "n_features_per_sae": N_FEATURES_PER_SAE,
            "model": "Qwen/Qwen3-8B", "layer": 18, "tokens": "500M",
            "d_sae": 65536, "k": 80,
            "protocol": "describe-then-predict (>=50% accuracy)",
            "judge": "bedrock/us.anthropic.claude-sonnet-4-6",
            "replaces": ["exp33 (50M/L27/200)", "exp40 (100M/no_C/>=4)"],
        },
        "interpretability_rate": collect("interpretability_rate"),
        "low_freq_rate": collect("low_freq_rate"),
        "alive_count": collect("alive_count"),
    }
    AGG_PATH.write_text(json.dumps(agg, indent=2))
    print(f"Wrote {AGG_PATH}")
    for arm in ARMS:
        r = agg["interpretability_rate"].get(arm)
        if r:
            print(f"  {arm:14s} interp rate = {r['mean']:.3f} +/- {r['sd']:.3f} "
                  f"(n={r['n_seeds']})")


def main():
    ap = argparse.ArgumentParser(description="Exp 62a multi-seed headline auto-interp")
    ap.add_argument("--collect", action="store_true", help="Phase 1: GPU context collection")
    ap.add_argument("--score", action="store_true", help="Phase 2: LLM-judge scoring")
    ap.add_argument("--aggregate", action="store_true", help="Phase 3: mean +/- SD across seeds")
    args = ap.parse_args()
    if args.collect:
        run_collect()
    if args.score:
        run_score()
    if args.aggregate:
        run_aggregate()
    if not (args.collect or args.score or args.aggregate):
        ap.error("Specify --collect, --score, and/or --aggregate")


if __name__ == "__main__":
    main()
