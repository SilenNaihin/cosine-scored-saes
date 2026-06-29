# Uploading ARES SAEs to Neuronpedia (future path)

This site self-hosts feature dashboards from local experiment JSON. To *also* surface
these SAEs on the hosted [Neuronpedia](https://neuronpedia.org) platform, follow this
path (not yet implemented).

## Prerequisites
- Confirm Qwen3-8B (`Qwen/Qwen3-8B`) is onboarded as a Neuronpedia model. If not, open a
  model-onboarding request — Neuronpedia is open source: https://github.com/hijohnnylin/neuronpedia
- A Neuronpedia API key in `NEURONPEDIA_API_KEY`.

## Steps
1. **Generate dashboards.** Use [SAEDashboard](https://github.com/jbloomAus/SAEDashboard) to
   produce per-feature activation dashboards from each checkpoint in
   [`Silen/cosine-scored-saes-qwen3-8b`](https://huggingface.co/Silen/cosine-scored-saes-qwen3-8b)
   (`standard/`, `global-a/`, `perfeature/`). The cosine variants need the
   `AdaptiveCosineBatchTopKSAE` encoder (`s = exp(b)·‖x‖^a·cos(x,w)`); load weights with the
   SAE class defs from `exp40_karvonen_recipe.py`.
2. **Create an SAE Set** per variant (release → set → SAE hierarchy) via the `neuronpedia`
   PyPI client.
3. **Upload features** (activations, top contexts, auto-interp explanations).
4. **Link back** from this site's `/results` "Coming to Neuronpedia" note to the live sets,
   and optionally embed Neuronpedia dashboards alongside the self-hosted ones.

## Why self-host first
Hosted Neuronpedia shows one SAE at a time and does not render our **standard-vs-ARES
side-by-side** comparison, which is the paper's whole point. The self-hosted explorer here
covers that; Neuronpedia hosting is complementary reach.
