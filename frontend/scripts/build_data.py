#!/usr/bin/env python3
"""Export experiment results into compact, frontend-ready JSON.

Run from repo root:
    uv run --no-project --with transformers,tokenizers \
        python frontend/scripts/build_data.py

Outputs into frontend/src/data/:
    features_l18.json   - Qwen3-8B L18 feature dashboards (standard vs adaptive_cosine), tokens decoded
    features_l27.json   - Qwen3-8B L27 feature dashboards (standard vs adaptive_l2), already text
    isolation.json      - exp2b: false positives / low-norm misses / outlier pairs with KL ground truth
    vignettes.json      - exp65: python/french concept-ablation K-sweeps
    autointerp.json     - exp42f: Claude auto-interp concept + coherence per feature
    headline.json       - curated headline numbers (modelcard + exp33)
    charts.json         - figures/_paper_data.py series + palette

If the Qwen tokenizer can't be loaded (offline), features_l18.json is written
with raw token ids and a `tokenizerMissing` flag; the L27 set still works.
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "experiments")
FIG = os.path.join(REPO, "figures")
OUT = os.path.join(REPO, "frontend", "src", "data")
os.makedirs(OUT, exist_ok=True)


def load(path):
    with open(path) as f:
        return json.load(f)


def clean(obj):
    """Recursively replace non-finite floats (Infinity/NaN) with None for valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def write(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(clean(obj), f, separators=(",", ":"))
    print(f"  wrote {name:22s} ({os.path.getsize(path)/1024:.0f} KB)")


# ---------------------------------------------------------------------------
# 1. L18 feature dashboards (decode token ids)
# ---------------------------------------------------------------------------
def build_l18():
    print("L18 feature dashboards (exp40d)...")
    # Prefer the LITERAL HF-checkpoint dashboards (exp68); else fall back to exp40d.
    hf_dir = os.path.join(EXP, "exp68_contexts")
    if os.path.isdir(hf_dir):
        ctx_dir = hf_dir
        variants = {"standard": "Standard", "perfeature_l2": "Cosine",
                    "global_a": "Cosine-global"}
        print("  source: published HF checkpoints (exp68_contexts)")
    else:
        ctx_dir = os.path.join(EXP, "exp40d_contexts")
        variants = {"standard": "Standard", "adaptive_cosine": "Cosine"}
        print("  source: exp40d dev checkpoints (fallback)")

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        print("  loaded Qwen3-8B tokenizer")
    except Exception as e:  # offline / missing
        print(f"  WARNING: tokenizer unavailable ({e}); emitting raw ids")

    def decode_one(tid):
        if tok is None:
            return f"[{tid}]"
        s = tok.decode([tid])
        return s

    out = {"layer": 18, "tokenizerMissing": tok is None,
           "source": "hf" if ctx_dir == hf_dir else "dev", "variants": {}}
    for vkey, vlabel in variants.items():
        fpath = os.path.join(ctx_dir, f"{vkey}_L18.json")
        if not os.path.exists(fpath):
            print(f"  (skip {vlabel}: {os.path.basename(fpath)} not found)")
            continue
        data = load(fpath)
        feats = []
        for fid, f in data["features"].items():
            windows = []
            # keep the strongest 8 windows for payload size
            for w in sorted(f["windows"], key=lambda x: -x["activation"])[:8]:
                pos = w["activating_pos_in_window"]
                toks = [{"t": decode_one(t)} for t in w["token_ids"]]
                windows.append({"act": round(w["activation"], 3), "pos": pos, "tokens": toks})
            feats.append({
                "id": int(fid),
                "freq": f["freq"],
                "maxAct": round(f["max_act"], 3),
                "windows": windows,
            })
        feats.sort(key=lambda x: -x["maxAct"])
        out["variants"][vlabel] = {
            "key": vkey,
            "nAlive": data["n_alive"],
            "features": feats,
        }
    write("features_l18.json", out)


# ---------------------------------------------------------------------------
# 2. L27 feature dashboards (already decoded text)
# ---------------------------------------------------------------------------
def build_l27():
    print("L27 feature dashboards (exp33)...")
    d = load(os.path.join(EXP, "exp33_contexts.json"))
    variants = {"standard": "Standard", "adaptive_l2": "Cosine"}
    out = {"layer": 27, "variants": {}}
    for vkey, vlabel in variants.items():
        v = d[vkey]
        feats = []
        for f in v["features"]:
            ctxs = []
            for c in f["contexts"][:8]:
                ctxs.append({
                    "act": round(c["activation"], 3),
                    "prefix": c["prefix"][-160:],
                    "target": c["target_token"],
                    "suffix": c["suffix"][:160],
                })
            feats.append({
                "id": f["feature_idx"],
                "freq": round(f["frequency"], 5),
                "band": f["frequency_band"],
                "maxAct": round(f["contexts"][0]["activation"], 3) if f["contexts"] else 0,
                "contexts": ctxs,
            })
        feats.sort(key=lambda x: -x["maxAct"])
        out["variants"][vlabel] = {
            "key": vkey,
            "nAlive": v["alive_count"],
            "features": feats,
        }
    write("features_l27.json", out)


# ---------------------------------------------------------------------------
# 3. Isolation demo (exp2b)
# ---------------------------------------------------------------------------
def build_isolation():
    hf = os.path.join(EXP, "exp68_isolation.json")
    src = hf if os.path.exists(hf) else os.path.join(EXP, "exp2b_results.json")
    print(f"Isolation demo ({'HF checkpoint' if src == hf else 'exp2b'})...")
    d = load(src)
    false_pos, low_norm, pairs = [], [], []
    for f in d["features"]:
        fid = f["feature_idx"]
        for x in f["false_positives"]:
            false_pos.append({"feature": fid, "token": x["token"], "cos": x["cos"],
                              "saeAct": x["sae_act"], "norm": x["norm"], "kl": x["kl_divergence"]})
        for x in f["low_norm_misses"]:
            low_norm.append({"feature": fid, "token": x["token"], "cos": x["cos"],
                             "saeAct": x["sae_act"], "norm": x["norm"], "kl": x["kl_divergence"]})
        for p in f["outlier_pairs"]:
            pairs.append({"feature": fid,
                          "highCosToken": p["high_cos_token"], "lowCosToken": p["low_cos_token"],
                          "highCos": p["high_cos"], "lowCos": p["low_cos"],
                          "highSaeAct": p["high_sae_act"], "lowSaeAct": p["low_sae_act"],
                          "highNorm": p["high_norm"], "lowNorm": p["low_norm"],
                          "highKl": p["high_kl"], "lowKl": p["low_kl"],
                          "rnhCorrect": p["rnh_correct"]})
    # Curate clean, on-narrative examples (keep all as fallback if too few):
    #  false positive  = standard fires loud, direction is wrong, no causal effect.
    #  low-norm miss    = standard silent, direction is right, real causal effect.
    fp_clean = [x for x in false_pos if x["saeAct"] > 4 and x["cos"] < 0.25 and x["kl"] < 0.02]
    lm_clean = [x for x in low_norm if x["saeAct"] < 1.0 and x["cos"] > 0.30 and x["kl"] > 0.05]
    if len(fp_clean) >= 3:
        false_pos = fp_clean
    if len(lm_clean) >= 3:
        low_norm = lm_clean
    # sort to surface the most dramatic examples first
    false_pos.sort(key=lambda x: -x["saeAct"])          # loudest false alarms
    low_norm.sort(key=lambda x: -x["kl"])               # biggest missed causal impact
    write("isolation.json", {
        "model": d.get("model"), "layer": d.get("layer"),
        "falsePositives": false_pos, "lowNormMisses": low_norm, "pairs": pairs,
    })


# ---------------------------------------------------------------------------
# 4. Causal vignettes (exp65)
# ---------------------------------------------------------------------------
def build_vignettes():
    print("Causal vignettes (exp65)...")
    d = load(os.path.join(EXP, "exp65_results_v2.json"))
    arm_map = {"standard": "Standard", "perfeature_l2": "Cosine"}
    concepts = {}
    for cname, c in d["concepts"].items():
        arms = {}
        for akey, alabel in arm_map.items():
            if akey not in c:
                continue
            sweep = [{"k": s["K"], "onKl": s["on_kl"], "offKl": s["off_kl"],
                      "selectivityGap": s.get("mean_selectivity_gap"),
                      "featureIds": s.get("feature_ids", [])} for s in c[akey]["k_sweep"]]
            arms[alabel] = sweep
        concepts[cname] = arms
    write("vignettes.json", {"config": d.get("config"), "concepts": concepts})


# ---------------------------------------------------------------------------
# 5. Auto-interp (exp42f)
# ---------------------------------------------------------------------------
def build_autointerp():
    print("Auto-interp (exp42f)...")
    d = load(os.path.join(EXP, "exp42f_results", "scores.json"))
    # map the 100M files: no_C == ARES-style cosine scoring, standard == baseline
    file_map = {"no_C_100M.json": "Cosine", "our_standard_100M.json": "Standard"}
    variants = {}
    for fname, vlabel in file_map.items():
        pf = d["per_file"].get(fname)
        if not pf:
            continue
        feats = []
        for fid, f in pf["features"].items():
            sc = f.get("score") or {}
            if sc.get("concept", "none") == "none":
                continue
            feats.append({"id": int(fid), "concept": sc.get("concept"),
                          "coherence": sc.get("coherence_score"),
                          "examplesConsistent": sc.get("examples_consistent_with_concept"),
                          "maxAct": round(f.get("max_act", 0), 3)})
        feats.sort(key=lambda x: (-(x["coherence"] or 0), -(x["examplesConsistent"] or 0)))
        variants[vlabel] = {"agg": d["aggregates"].get(fname), "features": feats}
    write("autointerp.json", {"modelId": d.get("model_id"), "variants": variants})


# ---------------------------------------------------------------------------
# 6. Headline numbers (curated from modelcard + exp33)
# ---------------------------------------------------------------------------
def build_headline():
    print("Headline numbers...")
    # values from experiments/exp61_modelcard.md and exp33_results.json
    out = {
        "variants": [
            {"name": "Standard", "a": None, "fve": 0.7702, "fveSd": 0.0002,
             "probing": 0.6669, "probingSd": 0.0026},
            {"name": "Global a", "a": 0.2577, "fve": 0.7690, "fveSd": 0.0000,
             "probing": 0.8081, "probingSd": 0.0059},
            {"name": "Per-feature", "a": 0.0759, "fve": 0.7707, "fveSd": 0.0002,
             "probing": 0.8128, "probingSd": 0.0126},
        ],
        "headline": {
            "probingDeltaPct": 14.9,            # per-feature cosine vs standard (paper headline)
            "fve": 0.77,
            "globalA": 0.26,
            # interpretability: per-feature rates MATCHED; the larger TOTAL count is a
            # no-auxiliary-loss result (more features stay alive), ~4.5x, not a per-feature win.
            "interpPerFeatureMatched": True,
            "interpTotalRatioNoAux": 4.5,
            "firingRatioStd": 22,               # std unmatched features: Q4/Q1 firing share
            "firingRatioCos": 4.7,              # cosine unmatched features
            "reconNormStdQ4": 9.5,              # standard recon-norm bias on high-norm tokens
            "reconNormCosQ4": 0.55,
            "dirVsNormMin": 87,                 # direction-vs-norm patching range across depths
            "dirVsNormMax": 2560,
        },
        "hf": "https://huggingface.co/Silen/cosine-scored-saes-qwen3-8b",
        "code": "https://github.com/SilenNaihin/cosine-scored-saes",
        "citation": (
            "@inproceedings{naihin2026cosine,\n"
            "  title={Size Doesn't Matter: Cosine-Scored Sparse Autoencoders},\n"
            "  author={Naihin, Silen and Stambler, Lev},\n"
            "  booktitle={ICML 2026 Workshop on Mechanistic Interpretability},\n"
            "  year={2026}\n}"
        ),
    }
    write("headline.json", out)


# ---------------------------------------------------------------------------
# 6b. Per-dataset sparse probing (exp71 Modal run on the published HF SAEs)
# ---------------------------------------------------------------------------
# raw SAEBench dataset id -> (short name, task label), in display order.
DATASET_META = [
    ("Helsinki-NLP/europarl", "europarl", "language"),
    ("codeparrot/github-code", "github-code", "code"),
    ("LabHC/bias_in_bios_class_set1", "bias_in_bios 1", "occupation A"),
    ("LabHC/bias_in_bios_class_set2", "bias_in_bios 2", "occupation B"),
    ("LabHC/bias_in_bios_class_set3", "bias_in_bios 3", "occupation C"),
    ("fancyzhx/ag_news", "ag-news", "topic"),
    ("canrager/amazon_reviews_mcauley_1and5", "amazon-reviews", "category"),
    ("canrager/amazon_reviews_mcauley_1and5_sentiment", "amazon-sentiment", "sentiment"),
]


def build_probing():
    """Read the Modal SAEBench sparse-probing run (experiments/exp71_probing.json,
    produced by `modal run experiments/modal_app.py::build_probing`) and emit the
    per-dataset Standard-vs-Cosine breakdown for the Probe Win Explorer."""
    print("Per-dataset sparse probing (exp71 Modal run)...")
    src = os.path.join(EXP, "exp71_probing.json")
    if not os.path.exists(src):
        print("  (skip: exp71_probing.json not found — run `modal run "
              "experiments/modal_app.py::build_probing` first)")
        return
    d = load(src)
    arms = d.get("arms", {})
    std, cos = arms.get("standard"), arms.get("perfeature_l2")
    if not std or not cos:
        print("  (skip: missing standard/perfeature_l2 arm in exp71_probing.json)")
        return

    def by_dataset(arm):
        return {r["dataset"]: r for r in (arm.get("per_dataset") or [])}

    si, ci = by_dataset(std), by_dataset(cos)
    trip = lambda r: {"top_1": r.get("sae_top_1"), "top_2": r.get("sae_top_2"),
                      "top_5": r.get("sae_top_5")}
    datasets = []
    for raw, name, label in DATASET_META:
        if raw not in si or raw not in ci:
            print(f"  (warn: dataset {raw} missing in run output)")
            continue
        datasets.append({"name": name, "label": label,
                         "standard": trip(si[raw]), "cosine": trip(ci[raw]),
                         "llm": si[raw].get("llm_top_1")})
    out = {
        "model": d.get("model"), "layer": d.get("layer"), "seed": d.get("seed"),
        "source": "exp71_modal", "hfRepo": d.get("hf_repo"),
        "aggregate": {"standard": trip(std), "cosine": trip(cos)},
        "datasets": datasets,
    }
    write("probing_datasets.json", out)


# ---------------------------------------------------------------------------
# 7. Chart series (figures/_paper_data.py)
# ---------------------------------------------------------------------------
def build_charts():
    print("Chart series (_paper_data)...")
    sys.path.insert(0, FIG)
    import _paper_data as p
    keep = ["PROBING_HEADLINE", "PARASITISM", "DISCOVERY_SEP", "GRAD_EQ",
            "DIRECTION_PATCH", "ARCH_OVERVIEW", "NO_AUX_QWEN", "Q4_PATHOLOGY"]
    out = {"palette": {"standard": p.COLOR_STD, "ares": p.COLOR_PF, "globalA": p.COLOR_GA,
                       "ink": p.COLOR_INK, "muted": p.COLOR_MUTED, "border": p.COLOR_BORDER,
                       "grid": p.COLOR_GRID}}
    for name in keep:
        if hasattr(p, name):
            out[name] = getattr(p, name)
    write("charts.json", out)


if __name__ == "__main__":
    print(f"Exporting to {OUT}\n")
    build_l27()       # no tokenizer needed
    build_isolation()
    build_vignettes()
    build_autointerp()
    build_headline()
    build_probing()
    build_charts()
    build_l18()       # last (may download tokenizer)
    print("\nDone.")
