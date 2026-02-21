# 06 Post Processing and Response

## Purpose

Assemble a stable API response contract from ranked campaigns and diagnostics.

## Inputs and outputs

### Inputs

- Eligibility result and gating state.
- Category extraction output.
- Ranked campaigns and score breakdown.
- Per-phase timing measurements.

### Output

`RetrieveResponse` object:

```json
{
  "ad_eligibility": 0.9224,
  "extracted_categories": ["running shoes"],
  "campaigns": [{"campaign_id": "camp_0001", "relevance_score": 1.07, "internal_score": 0.48}],
  "latency_ms": 14.2,
  "metadata": {"n_results": 128, "gated": false}
}
```

## Key files, classes, and functions

- `app/main.py`
  - response assembly in `retrieve`.
- `app/models.py`
  - `CampaignResult`, `RetrieveResponse`.

## Algorithm details and edge cases

1. If query is hard-blocked, return immediately with empty campaigns. Metadata is still fully populated (eligibility detail, parser output, filter counts, etc.) but `timing_ms` contains only `total` — embed and parallel stage timers are absent since those stages never ran.
2. If query is gated after semantic scoring, skip filter/rerank and return empty campaigns.
3. Convert ranked dicts to typed `CampaignResult` instances.
4. Build metadata payload including counts, timing, parser output, categories, and score breakdown.
5. Round latency values for response readability.

Edge cases:

- Gated queries suppress category output to avoid insensitive messaging.
- Blocklist and semantic-gate paths still include shape-consistent metadata keys.

## Performance notes

- Post-processing is mostly serialization and lightweight dict/list transformations.
- Debug payload size can become large if breakdown depth is increased.

## Failure modes and fallbacks

- `CampaignResult` construction can fail if ranker output keys drift from schema.
- Large metadata payload can increase response size and client parsing time.

## How to change this safely

- Run:
  - `pytest -q tests/test_api.py tests/test_api_integration.py`
- Verify:
  - blocked and gated responses still satisfy schema
  - non-gated responses include expected metadata fields
- Watch:
  - response size
  - client parsing issues

## Related docs

- [Request and Response Contract](../00-overview/request-response-contract.md)
- [07 Observability and Metadata](./07-observability-and-metadata.md)
