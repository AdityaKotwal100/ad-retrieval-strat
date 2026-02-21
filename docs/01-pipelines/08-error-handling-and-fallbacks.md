# 08 Error Handling and Fallbacks

## Purpose

Define how the system fails safely and what fallback behavior keeps user-facing responses valid under degraded conditions.

## Inputs and outputs

### Inputs

- Invalid requests.
- Missing artifacts or model failures.
- Runtime exceptions in scoring sub-systems.

### Outputs

- Deterministic HTTP errors for contract violations.
- Fail-closed ad suppression for safety-critical failures.
- Startup failure for missing critical artifacts.

## Key files, classes, and functions

- `app/main.py`
  - query validation and short-circuit responses.
- `app/services/eligibility.py`
  - conservative handling in `_score_internal`.
- `app/services/categories.py`
  - FAISS-to-matrix fallback.
- `app/services/retrieval.py`
  - artifact validation and startup failure behavior.

## Algorithm details and edge cases

- Invalid/malformed payloads return FastAPI validation errors.
- Empty query after strip returns `HTTP 400`.
- Hard blocklist or semantic hard gate returns safe empty-campaign response.
- If insensitive model inference throws, scorer sets probability to `1.0` and gates.
- Category index load failure falls back to matrix extraction mode.
- Retrieval artifact mismatch causes startup exception rather than partial unsafe serving.

## Performance notes

- Fast-fail behavior for blocked queries reduces latency and compute.
- Startup validation avoids per-request defensive checks on artifact alignment.

## Failure modes and fallbacks

- Category extraction downgrade can reduce relevance quality.
- Over-conservative fail-closed gating can reduce monetization on benign traffic.
- Startup hard failure on artifact issues causes downtime until artifacts are fixed.

## How to change this safely

- Run:
  - `pytest -q tests/test_api_integration.py tests/test_retrieval_pipeline.py tests/test_eligibility.py`
- Chaos test manually:
  - temporarily hide category index artifacts and verify matrix fallback path
  - validate blocked and gated response shapes
- Watch:
  - startup failure rates
  - gated-rate spikes after model updates

## Related docs

- [System Flow](../00-overview/system-flow.md)
- [Safety and Sensitivity](../05-design-decisions/safety-and-sensitivity.md)
