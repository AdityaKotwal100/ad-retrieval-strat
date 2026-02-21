# 07 Observability and Metadata

## Purpose

Expose enough per-request diagnostics to explain why a query was gated, why campaigns were dropped, and why top results ranked where they did.

## Inputs and outputs

### Inputs

- Timing checkpoints from `app/main.py`.
- Filter drop counters from `CampaignEligibilityFilter`.
- Eligibility detail from `EligibilityScorer.score_with_metadata`.
- Ranking breakdown from `ContextReranker.rerank`.

### Output

`metadata` field in API response, including:

- `timing_ms`
- `eligibility_detail`
- `filter_drops`
- `parser_output`
- `categories`
- `score_breakdown`

## Key files, classes, and functions

- `app/main.py`
  - metadata assembly.
- `app/services/campaign_eligibility.py`
  - `FilterResult.drop_counts`.
- `app/services/eligibility.py`
  - `score_with_metadata` details.
- `app/services/ranking.py`
  - debug breakdown generation.

## Algorithm details and edge cases

1. Track phase timers (`embed`, `parallel_stages`, `parse`, `filter_and_rerank`, `total`).
2. Collect candidate counts before and after filtering.
3. Include category scores and parser extraction details.
4. Emit top-N ranking delta explanations for interpretability.

Edge cases:

- Blocked path has reduced timing breakdown (total only).
- Gated path has empty filter drops and score breakdown.

## Performance notes

- Metadata collection has low CPU cost but non-trivial payload size.
- Keeping `debug_top_n` small controls response bloat.

## Failure modes and fallbacks

- Missing keys in scorer/ranker metadata can break downstream dashboards.
- Inconsistent metadata shape across gated vs non-gated can break brittle consumers.

## How to change this safely

- Run:
  - `pytest -q tests/test_api.py tests/test_api_integration.py`
- Validate with saved queries:
  - `python3 scripts/run_test_queries.py`
- Watch:
  - metadata parsing errors in downstream systems
  - drift in `filter_drops` reason names

## Related docs

- [Request and Response Contract](../00-overview/request-response-contract.md)
- [Latency Experiments](../03-experiments/latency-experiments.md)
