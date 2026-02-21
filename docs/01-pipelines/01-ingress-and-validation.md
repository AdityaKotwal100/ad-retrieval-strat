# 01 Ingress and Validation

## Purpose

Accept HTTP requests, validate contract shape, enforce basic query sanity, and initialize per-request context before heavy compute.

## Inputs and outputs

### Input

Example request:

```json
{
  "query": "best running shoes for marathon training",
  "context": {
    "age": 28,
    "gender": "male",
    "location": "San Francisco, CA",
    "interests": ["fitness", "outdoor activities"]
  }
}
```

### Output

- Validated `RetrieveRequest` object.
- `query` normalized by `strip()`.
- `context_dict` produced via `model_dump(exclude_none=True)`.

## Key files, classes, and functions

- `app/main.py`
  - `retrieve(request: RetrieveRequest)`.
- `app/models.py`
  - `UserContext`, `RetrieveRequest`, `RetrieveResponse`.

## Algorithm details and edge cases

1. FastAPI + Pydantic validates body against `RetrieveRequest`.
2. Runtime strips query whitespace and rejects empty strings with `HTTP 400`.
3. Context is optional; missing context produces `{}` and still proceeds.
4. Extra keys in context are allowed by `UserContext.model_config.extra="allow"`.

Edge cases:

- Missing `query` field: `422` (schema-level).
- Empty `query`: `422` from min length or `400` after strip if whitespace-only.
- Malformed JSON: `400` or `422` depending on parser path.

## Performance notes

- Work here is minimal and mostly CPU-inexpensive.
- Keeping this stage thin protects tail latency for all requests.

## Failure modes and fallbacks

- Invalid request schema: FastAPI validation response.
- Whitespace query: explicit runtime rejection.
- No context: parser/filter/ranker handle missing values safely.

## How to change this safely

- Run:
  - `pytest -q tests/test_api_integration.py tests/test_api.py`
- Watch:
  - request 4xx rate
  - malformed body error distribution
  - average ingress parse time (if instrumented)
- Do not add expensive operations before embedding or parallel stage.

## Related docs

- [Request and Response Contract](../00-overview/request-response-contract.md)
- [08 Error Handling and Fallbacks](./08-error-handling-and-fallbacks.md)
