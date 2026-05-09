# Experiment 3: LLM-judged interpretability of cosine vs inner product tokens

This experiment tests whether tokens detected by cosine similarity (but missed by the SAE) are more interpretable than tokens detected by the SAE (but with low cosine similarity). For each of 10 features at layer 18 of Qwen3-8B, a feature interpretation is generated from cosine-top tokens, then an LLM judge evaluates whether cosine-detected/SAE-missed tokens and SAE-detected/low-cosine tokens match that interpretation.

## Results

- Cosine-detected (SAE misses) match rate: 31.4%
- SAE false-positives (low cosine) match rate: 8.7%
- Cosine wins on 6/8 features with data in both categories

Per-feature match rates:

| Feature | Cosine-detected match | SAE false-positive match |
|---|---|---|
| 59854 | 71.4% | 0.0% |
| 3459 | 60.0% | 13.3% |
| 34587 | 33.0% | 0.0% |
| 14202 | 26.7% | 0.0% |
| 63114 | 26.7% | 27.0% |
| 10935 | 26.7% | 0.0% |
| 33257 | 6.7% | 13.0% |
| 64644 | 0.0% | 27.0% |
