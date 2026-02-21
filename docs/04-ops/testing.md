# Testing

## Test layers

### Unit and integration tests

Run all:

```bash
pytest -q
```

Focused quick pass from `TESTING.md`:

```bash
pytest -q tests/test_eligibility.py tests/test_ranking.py tests/test_retrieval_pipeline.py tests/test_api_integration.py
```

### What each suite covers

- `tests/test_eligibility.py`
  - score bounds, sensitive suppression, deterministic behavior, metadata fields.
- `tests/test_ranking.py`
  - ordering, deterministic ties, coverage-ratio math, top-1000 trimming.
- `tests/test_retrieval_pipeline.py`
  - category shape/order behavior, FAISS id handling, top-k behavior.
- `tests/test_campaign_eligibility.py`
  - filter reasons and drop logic for geo/age/gender/price/negative keywords.
- `tests/test_api_integration.py`
  - end-to-end contract shape and gated behavior with stubs.
- `tests/test_api.py`
  - data-backed endpoint checks when real artifacts exist.

## Golden query and regression workflows

- Replay request corpus:

```bash
python3 scripts/run_test_queries.py
```

- Benchmark latency:

```bash
python3 scripts/benchmark_latency.py
```

- Score-tuning introspection:

```bash
python3 scripts/fine_tuning_runs.py
```

## Load and stress testing

- `scripts/benchmark_latency.py` performs concurrent request rounds and percentile summaries.
- For broader load tests, inference: integrate with external load tools (for example k6/Locust) against `/api/retrieve` using `data/test_queries` payloads.

## Coverage gaps

- Not found in repo: formal CI gate for load/perf regressions.
- Not found in repo: production-like soak tests with sustained concurrency.

## Related docs

- [Local Development](./local-dev.md)
- [Latency Experiments](../03-experiments/latency-experiments.md)
