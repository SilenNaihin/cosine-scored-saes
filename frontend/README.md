# ARES · Cosine-Scored SAEs — interactive explorer

A self-hosted, Neuronpedia-style frontend for **"Size Doesn't Matter: Cosine-Scored Sparse
Autoencoders"** (Qwen3-8B). It contrasts the standard inner-product SAE against ARES
(cosine-scored) and foregrounds where ARES *isolates a clean concept* and the standard SAE
fails.

Pages: **Home** · **Isolation** (interactive ground-truth demo) · **Features**
(Neuronpedia-style browser, L18 & L27, standard vs ARES) · **Causal** (concept-ablation
vignettes) · **Results** (animated paper charts).

## Stack
Vite + React + TypeScript + Tailwind v4, Framer Motion, Recharts. Static SPA (hash router) —
deploy the `dist/` to Vercel / HF Spaces / GitHub Pages with no server.

## Develop
```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # -> dist/
npm run preview
```

## Data
All UI is driven by JSON in `src/data/`, generated from the repo's experiments by:
```bash
# from the repo root
uv run --no-project --with transformers,tokenizers,numpy \
    python frontend/scripts/build_data.py
```
This decodes the Qwen3-8B tokenizer for L18 contexts and dumps feature dashboards, the
isolation examples (exp2b), causal vignettes (exp65), auto-interp (exp42f), headline numbers,
and the paper's chart series (`figures/_paper_data.py`). Re-run it to refresh.

See `docs/neuronpedia-upload.md` for the path to also host these SAEs on neuronpedia.org.
