# Experiments Index

This index lists experiment evidence by pipeline phase.

## Ad eligibility

- [Ad Eligibility Experiments](./ad-eligibility-experiments.md)
- Evidence sources:
  - `experiments/k-means-for-appropriateness/*.py`
  - `experiments/k-means-for-appropriateness/findings*.md`
  - `experiments/toxicity/*.json`
  - `generate-ads-or-not/models/toxicity/*.json`

## Category extraction

- [Category Extraction Experiments](./category-extraction-experiments.md)
- Evidence sources:
  - `data/category_index.faiss`, `data/category_meta.json`
  - `data/generated_queries.json`, `data/generated_vertical_queries.json`
  - `app/services/categories.py`

## Retrieval

- [Retrieval Experiments](./retrieval-experiments.md)
- Evidence sources:
  - `scripts/build_index.py`, `scripts/build_index_v4.py`
  - `experiments/vector-db-test/*`
  - `model-test/embedding-model-comparison.md`

## Ranking

- [Ranking Experiments](./ranking-experiments.md)
- Evidence sources:
  - `experiments/relevance-scoring/*`
  - `tests/test_ranking.py`
  - `scripts/fine_tuning_runs.py`

## Latency

- [Latency Experiments](./latency-experiments.md)
- Evidence sources:
  - `scripts/benchmark_latency.py`
  - `data/test_results/*/output.json`
  - `app/main.py` stage timings

## Summary outcomes

- Eligibility architecture converged to layered safety plus commercial affinity, with hard fail-safe gating.
- Category extraction moved toward query-index FAISS matching to reduce vocabulary mismatch.
- Retrieval is stable on exact FAISS search and aligned side artifacts.
- Ranking uses interpretable additive components with coverage-ratio scaling.
- Observed local latency remains well below the 100ms target in current saved artifacts.
