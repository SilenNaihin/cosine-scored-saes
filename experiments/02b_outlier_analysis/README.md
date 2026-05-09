# Experiment 2b: Adversarial pairs where cosine and inner product disagree

This experiment identifies three categories of adversarial tokens at layer 18 of Qwen3-8B where cosine similarity and SAE activation diverge, then uses logit-level KL divergence to determine which metric correctly predicts causal importance. Type A: matched SAE activation but differing cosine. Type B: high SAE activation but low cosine (false positives). Type C: high cosine but zero SAE activation (misses). Dataset: 1060 tokens from 24 prompts.

## Results

Type A (matched SAE activation, different cosine): Higher-cosine token has greater KL divergence on 39/45 pairs (86.7%).

| Category | SAE activation | Cosine | KL divergence |
|---|---|---|---|
| Type B (false positives) | 9.32 | ~0 | 0.0276 |
| Type C (misses) | 0.00 | 0.122 | 0.2473 |

- Type C / Type B KL ratio: ~9x
