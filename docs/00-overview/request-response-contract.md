# Request and Response Contract

## Endpoint

- `POST /api/retrieve`
- Request schema: `RetrieveRequest` in `app/models.py`.
- Response schema: `RetrieveResponse` in `app/models.py`.

## Request contract

### Request body

- `query` (string, required, min length 1)
  - User free-text query.
  - Empty strings are rejected by schema and by runtime strip check in `app/main.py`.
- `context` (object, optional)
  - Typed fields in `UserContext` (`gender`, `age`, `location`, `interests`, `device`).
  - Extra keys are accepted (`extra="allow"`) and passed through to parser context.

### Context normalization behavior

- Location is normalized in `QueryParser._parse_user_location` to `US-XX` where possible.
- Age is enforced as integer by the Pydantic `UserContext` schema before reaching the parser.
- Gender is kept as provided string and interpreted by filters/ranker.
- Interests are used for category context boost and interest scoring.

## Response contract

### Top-level fields

- `ad_eligibility` (float, `[0,1]`)
  - Final eligibility score used for gating.
- `extracted_categories` (list of strings)
  - Category names only.
  - Category scores are preserved in `metadata.categories`.
- `campaigns` (list of `CampaignResult`)
  - Scored campaign rows.
- `latency_ms` (float)
  - End-to-end request latency measured in `app/main.py`.
- `metadata` (object)
  - Observability and debugging payload.

### CampaignResult fields

From `app/models.py` and `app/services/ranking.py`:

- `campaign_id` string
- `relevance_score` float
- `internal_score` float
- `title` string
- `advertiser` string
- `category` string
- `vertical` string
- `landing_url` string
- `creative_format` string
- `budget_remaining` float

### Metadata philosophy

Metadata is intentionally verbose because ranking and safety failures are hard to debug from final rankings alone. `app/main.py` includes:

- retrieval counts (`n_candidates_retrieved`, `n_candidates_after_filter`, `n_results`)
- per-filter drops (`filter_drops`)
- per-phase timing (`timing_ms`) — shape differs by path: normal requests include `embed`, `parallel_stages`, `parse`, `filter_and_rerank`, and `total`; hard-blocked requests include only `total`
- eligibility breakdown (`eligibility_detail`) — includes `insensitive_model_probability`, `insensitive_model_threshold`, `cluster_penalty`, `top_cluster`, `top_cluster_sim`, `hard_block_cluster`
- category breakdown with scores (`categories`)
- parser outputs (`parser_output`)
- ranking explanations for top rows (`score_breakdown`)

This enables fast diagnosis without replaying requests in a debugger.

## Clamping and normalization rules

- `ad_eligibility` is clamped in `EligibilityScorer._score_internal` using `np.clip(..., 0.0, 1.0)`.
- `ad_eligibility` is also schema-constrained by Pydantic `Field(ge=0.0, le=1.0)`.
- `internal_score` is normalized by query-specific potential max in `ContextReranker.rerank`.
- `relevance_score` currently stores raw rerank score in code, not clamped.
  - Inference: docstring in `app/services/ranking.py` says clamped, but implementation assigns raw `final_score`. This mismatch should be treated as a contract risk.

## Why this schema looks like this

- `ad_eligibility` in `[0,1]` keeps a probability-like interpretation for product and policy consumers.
- `extracted_categories` is flat strings for simple clients, with full scored objects retained in metadata for debug clients.
- Returning up to 1,000 campaigns gives downstream systems enough depth for secondary business logic while still bounded for latency.
- Metadata exists to make safety and ranking behavior auditable per request.
- Separate `relevance_score` and `internal_score` allows exposing raw ordering signal while keeping one normalized field for UI/analytics.

## Related docs

- [System Flow](./system-flow.md)
- [Output Design Rationale](../05-design-decisions/output-design-rationale.md)
