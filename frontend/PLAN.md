# Frontend Plan: Neuronpedia Display for Cosine-Scored SAEs (Qwen3-8B)

## Goal
Build a frontend that uses [Neuronpedia](https://www.neuronpedia.org) to display all three SAE variants from [`Silen/cosine-scored-saes-qwen3-8b`](https://huggingface.co/Silen/cosine-scored-saes-qwen3-8b) and show how the adaptive cosine (ARES) variants beat the standard inner-product SAE.

## Source Data (HuggingFace)
- **Repo:** `Silen/cosine-scored-saes-qwen3-8b`
- **Base model:** Qwen3-8B, layer 18 residual stream (`blocks.18.hook_resid_post`)
- **Config:** `d_sae = 65,536`, BatchTopK `k = 80`, 500M tokens FineWeb, SAEBench/saprmarks recipe
- **Checkpoint format:** each `*_final.pt` = `{"state_dict", "num_tokens_since_fired", "step"}`; state_dict keys: `W_enc [65536,4096]`, `b_enc [65536]`, `W_dec [65536,4096]`, `b_dec [4096]`, plus `scale_a`, `scale_b`, `threshold` for cosine variants.
- **SAE class defs:** `experiments/40_saprmarks_recipe/exp40_karvonen_recipe.py` in [code repo](https://github.com/SilenNaihin/cosine-scored-saes)

### Three Variants
| Folder | Variant | `a` | Top-1 probing |
|--------|---------|-----|---------------|
| `standard/` | Standard BatchTopK (inner product) | — | 0.667 |
| `global-a/` | Adaptive Cosine (single shared `a`) — **ARES-global** | 0.258 | 0.808 |
| `perfeature/` | Per-Feature Adaptive Cosine (`a_i = a_base + delta_i`) — **ARES-per-feature** | mean 0.076 | 0.813 |

### ARES Advantage Evidence (from HF README + repo)
- At matched reconstruction (FVE within 0.4%), cosine variants improve single-feature sparse-probing top-1 by **+14.1%** (global) and **+14.6%** (per-feature) over standard.
- Multi-seed (n=3): Standard 0.6669±0.0026 vs Global 0.8081±0.0059 vs Per-feature 0.8128±0.0126.
- FVE: Standard 0.7702±0.0002, Global 0.7690±0.0000, Per-feature 0.7707±0.0002.
- Learned `a` stays far below inner-product limit (a=1): 0.2577±0.0007 (global), 0.0759±0.0001 (per-feature).
- Repo evidence: cosine SAEs win at every layer on every SAEBench metric; standard degrades with depth, cosine doesn't; adaptive cosine = best all-round architecture (0% dead at all layers).

## Neuronpedia Integration
- **Docs:** https://docs.neuronpedia.org — SAE Release > SAE Set > SAE hierarchy.
- **Feature pages:** per-feature dashboards with activations, explanations, test activations, lists.
- **JSON API:** documented at `/features` (Example - JSON API section).
- **Embeddable dashboards:** Neuronpedia hosts feature dashboards; SAEDashboard notebook generates Neuronpedia outputs.
- **Python client:** `neuronpedia` PyPI package (requests + python-dotenv).
- **Model support:** need to verify Qwen3-8B support; if absent, plan an upload/onboarding path.

## Frontend Architecture

### Stack
- **Framework:** Next.js (App Router) + TypeScript — matches Neuronpedia's own stack and enables SSR/embedding.
- **Styling:** Tailwind CSS.
- **Data layer:**
  - Static variant metadata + ARES results table bundled from HF README.
  - Server-side fetch from Neuronpedia JSON API for live feature dashboards.
  - Fallback: embed Neuronpedia dashboard iframes when API lacks direct Qwen3-8B support.

### Pages / Routes
1. `/` — Landing: hero explaining cosine-scored SAEs, 3-variant comparison cards, ARES advantage headline.
2. `/variants` — Side-by-side comparison table (FVE, probing top-1, learned `a`, dead features) with charts.
3. `/variants/[id]` — Per-variant detail: checkpoint info, config, Neuronpedia feature browser/embed.
4. `/ares` — Evidence page: charts and figures showing ARES advantage, links to repo experiments.
5. `/features` — Neuronpedia-powered feature explorer for the uploaded SAE sets.

### Key Components
- `VariantCard` — shows folder, variant name, `a`, top-1 score, FVE.
- `ComparisonTable` — sortable table of all three variants with multi-seed stats.
- `AresAdvantageChart` — bar chart of probing top-1: standard vs global vs per-feature with error bars.
- `NeuronpediaEmbed` — iframe or API-driven feature dashboard for a selected SAE set.
- `CheckpointMeta` — displays state_dict keys, shapes, and cosine-specific params (`scale_a`, `scale_b`, `threshold`).

### Data Files
- `frontend/data/variants.json` — static metadata for the 3 variants (folder, name, a, scores, FVE, checkpoint path).
- `frontend/data/ares-evidence.json` — multi-seed results, SAEBench metrics, dead-feature rates from repo.
- `frontend/data/hf-readme.md` — cached HF README for reference.

## ARES Narrative (for UI copy)
- **Standard** uses inner-product pre-activation `s_i(x) = <w_i, x_c> + b_i`.
- **ARES** (Adaptive coRrelation Encoder Scoring) replaces it with `s_i(x) = exp(b) * ||x_c||^a * cos(x_c, w_i) + b_enc,i`, where `a` interpolates pure cosine (a=0) ↔ inner product (a=1).
- The learned `a` stays far below 1, showing the model prefers cosine similarity over magnitude for feature detection.
- Result: matched reconstruction, +14% sparse-probing top-1 — better causal/interpretable features without sacrificing FVE.

## Implementation Steps
1. Scaffold Next.js app in `frontend/`, Tailwind, TypeScript.
2. Add `frontend/data/variants.json` + `ares-evidence.json` from HF README and repo figures.
3. Build `VariantCard`, `ComparisonTable`, `AresAdvantageChart`.
4. Build landing + variants + ares pages.
5. Integrate Neuronpedia: probe API for Qwen3-8B; if available, fetch feature dashboards; else embed iframes or document upload path.
6. Add per-variant detail pages with checkpoint metadata and Neuronpedia feature browser.
7. Verify build (`npm run build`) and run (`npm run dev`) on system Node.

## Open Questions
- Does Neuronpedia currently list Qwen3-8B? If not, need upload/onboarding via `neuronpedia` client or SAEDashboard pipeline.
- Auth: Neuronpedia API key for upload (env var `NEURONPEDIA_API_KEY`).
- Whether to generate Neuronpedia dashboard outputs locally from checkpoints or rely on hosted dashboards.
