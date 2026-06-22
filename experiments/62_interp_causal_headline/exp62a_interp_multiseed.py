"""
Experiment 62a: Headline auto-interpretability (camera-ready B5 Tier 1).

Re-runs exp53's describe-then-predict auto-interp at 1000 features/arm on the
recommended variants only (standard, global-a, per-feature + delta) at the true
headline recipe (Qwen3-8B L18, 500M tokens, d_sae=65,536, k=80). One unified
protocol replacing exp33 (50M/L27/200) and exp40 (100M/no_C/>=4) as the headline
interpretability source.

Two checkpoint sources (auto-detected; override with --source):
  - hf   (DEFAULT, runs now): the published 500M headline checkpoints at
         https://huggingface.co/Silen/cosine-scored-saes-qwen3-8b
         ONE checkpoint per variant (best of seeds 123/456 by top-1) -> single-seed
         auto-interp. Sufficient to fix the mislabeled exp40/exp33 numbers.
  - box8 (optional, 3-seed): per-seed checkpoints from exp61's SAVE_DIR, for
         mean +/- SD auto-interp. Set EXP62_CKPT_DIR=<exp61 SAVE_DIR root>.

The published multi-seed FVE + sparse-probing table already lives on the HF card
(A1 deliverable); only AUTO-INTERP is still missing, which is what this produces.

Thin wrapper over exp53 (same pattern as exp61 over exp40): imports the exp53
module, overrides config, loads checkpoints. No harness logic is duplicated.

Run (HF source, single seed):
    cd ~/cosine-scored-saes && source .venv/bin/activate
    # Phase 1 (GPU): collect activating contexts
    CUDA_VISIBLE_DEVICES=0 python3 experiments/62_interp_causal_headline/exp62a_interp_multiseed.py --collect
    # Phase 2 (LLM API; no GPU; needs AWS Bedrock creds)
    python3 experiments/62_interp_causal_headline/exp62a_interp_multiseed.py --score
    # Phase 3: aggregate -> mean +/- SD (single seed => SD=0)
    python3 experiments/62_interp_causal_headline/exp62a_interp_multiseed.py --aggregate

Run (box8 source, 3 seeds): add `--source box8` and `export EXP62_CKPT_DIR=...`.
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

# Recommended variants only (no_C / Karvonen ref dropped). Names map:
#   exp53 arm key  ->  HF repo folder
ARMS = ["standard", "adaptive_l2", "perfeature_l2"]
HF_FOLDER = {"standard": "standard", "adaptive_l2": "global-a", "perfeature_l2": "perfeature"}

N_FEATURES_PER_SAE = 1000              # exp53 used 200; tighten CIs
N_HIGH_FREQ = 250
N_MED_FREQ = 500
N_LOW_FREQ = 250

HF_REPO = "Silen/cosine-scored-saes-qwen3-8b"
HF_SEEDS = [-1]                        # single published checkpoint per variant
BOX8_SEEDS = [42, 123, 456]            # exp61 SAE-training seeds
# box8 layout: $EXP62_CKPT_DIR/seed{S}/{arm}_L18_final.pt
CKPT_DIR = os.environ.get("EXP62_CKPT_DIR", "/mnt/checkpoints/exp61")

RESULTS_DIR = HERE
AGG_PATH = HERE / "exp62a_results.json"


# =============================================================================
# Checkpoint resolution
# =============================================================================

def _hf_ckpts():
    """Download the published checkpoints; return {arm: local_path}. One seed."""
    from huggingface_hub import hf_hub_download
    out = {}
    for arm in ARMS:
        out[arm] = hf_hub_download(HF_REPO, f"{HF_FOLDER[arm]}/sae_final.pt")
    return out


def _box8_ckpts(seed):
    seed_dir = Path(CKPT_DIR) / f"seed{seed}"
    return {arm: str(seed_dir / f"{arm}_L18_final.pt") for arm in ARMS}


def _resolve(source, seed):
    """Return {arm: path} for a given source/seed, raising if anything is missing."""
    if source == "hf":
        ckpts = _hf_ckpts()
    else:
        ckpts = _box8_ckpts(seed)
        missing = [p for p in ckpts.values() if not Path(p).exists()]
        if missing:
            raise SystemExit(
                f"BLOCKED (box8, seed {seed}): missing checkpoints. Set EXP62_CKPT_DIR "
                f"to exp61's SAVE_DIR. Missing:\n  " + "\n  ".join(missing)
            )
    return ckpts


def _seeds_for(source):
    return HF_SEEDS if source == "hf" else BOX8_SEEDS


def _tag(source, seed):
    return source if source == "hf" else f"box8_seed{seed}"


# =============================================================================
# exp53 reuse
# =============================================================================

def _load_exp53_module():
    spec = importlib.util.spec_from_file_location("exp53", EXP53)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _configure(mod, source, seed):
    """Override exp53 module-level config + checkpoint paths for one run."""
    if seed >= 0:
        mod.SEED = seed
    mod.N_FEATURES_PER_SAE = N_FEATURES_PER_SAE
    mod.N_HIGH_FREQ = N_HIGH_FREQ
    mod.N_MED_FREQ = N_MED_FREQ
    mod.N_LOW_FREQ = N_LOW_FREQ
    mod.VARIANT_NAMES = list(ARMS)
    mod.CKPTS = _resolve(source, seed)
    tag = _tag(source, seed)
    mod.CONTEXTS_PATH = str(RESULTS_DIR / f"exp62a_contexts_{tag}.json")
    mod.RESULTS_PATH = str(RESULTS_DIR / f"exp62a_results_{tag}.json")
    if hasattr(mod, "SAE_CLASSES"):
        mod.SAE_CLASSES = {k: v for k, v in mod.SAE_CLASSES.items() if k in ARMS}
    return mod


def run_collect(source):
    for seed in _seeds_for(source):
        print(f"\n{'='*70}\nExp 62a COLLECT — {_tag(source, seed)}\n{'='*70}")
        mod = _configure(_load_exp53_module(), source, seed)
        mod.collect_contexts()


def run_score(source):
    for seed in _seeds_for(source):
        print(f"\n{'='*70}\nExp 62a SCORE — {_tag(source, seed)}\n{'='*70}")
        mod = _configure(_load_exp53_module(), source, seed)
        if not Path(mod.CONTEXTS_PATH).exists():
            print(f"  skip {_tag(source, seed)}: no contexts at {mod.CONTEXTS_PATH} "
                  f"(run --collect)")
            continue
        mod.score_features()


def run_aggregate(source):
    """Aggregate per-run results into mean +/- SD per arm + frequency breakdown."""
    runs = {}
    for seed in _seeds_for(source):
        tag = _tag(source, seed)
        p = RESULTS_DIR / f"exp62a_results_{tag}.json"
        if p.exists():
            runs[tag] = json.loads(p.read_text())
    if not runs:
        raise SystemExit("No per-run results found; run --score first.")

    def collect(metric_key):
        out = {}
        for arm in ARMS:
            vals = [res[arm][metric_key] for res in runs.values()
                    if arm in res and res[arm].get(metric_key) is not None]
            if vals:
                out[arm] = {
                    "n_runs": len(vals),
                    "mean": statistics.mean(vals),
                    "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "values": vals,
                }
        return out

    agg = {
        "config": {
            "source": source,
            "runs": list(runs.keys()),
            "arms": ARMS,
            "n_features_per_sae": N_FEATURES_PER_SAE,
            "model": "Qwen/Qwen3-8B", "layer": 18, "tokens": "500M",
            "d_sae": 65536, "k": 80,
            "protocol": "describe-then-predict (>=50% accuracy)",
            "judge": "bedrock/us.anthropic.claude-sonnet-4-6",
            "checkpoints": HF_REPO if source == "hf" else CKPT_DIR,
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
                  f"(n={r['n_runs']})")


def main():
    ap = argparse.ArgumentParser(description="Exp 62a headline auto-interp")
    ap.add_argument("--source", choices=["hf", "box8"], default="hf",
                    help="hf = published single-seed checkpoints (default); "
                         "box8 = exp61 per-seed checkpoints (3 seeds)")
    ap.add_argument("--collect", action="store_true", help="Phase 1: GPU context collection")
    ap.add_argument("--score", action="store_true", help="Phase 2: LLM-judge scoring")
    ap.add_argument("--aggregate", action="store_true", help="Phase 3: mean +/- SD")
    args = ap.parse_args()
    if args.collect:
        run_collect(args.source)
    if args.score:
        run_score(args.source)
    if args.aggregate:
        run_aggregate(args.source)
    if not (args.collect or args.score or args.aggregate):
        ap.error("Specify --collect, --score, and/or --aggregate")


if __name__ == "__main__":
    main()
