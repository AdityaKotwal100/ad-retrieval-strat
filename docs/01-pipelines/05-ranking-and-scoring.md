# 05 Ranking and Scoring

## Purpose

Convert retrieved candidates into final ordered campaigns using interpretable component deltas and query-aware normalization.

## Inputs and outputs

### Inputs

- Candidate ids and FAISS scores (up to 1,000 from FAISS retrieval).
- Parsed query (`ParsedQuery`) — structured intent and constraints.
- Extracted categories — semantic category matches with scores.
- Score weight config (`score_weights.yaml`).

### Outputs

- Ranked campaign list with `relevance_score` (raw) and `internal_score` (normalized `[0,1]`).
- Top-N per-campaign score breakdown for metadata.

## Key files, classes, and functions

- `app/services/ranking.py`
  - `ContextReranker.rerank`
  - `_active_spec_count`
  - `_query_potential_max`
- `app/services/scoring/builder.py`
  - score pipeline construction order.
- `app/services/scoring/components.py`
  - location, demographic, interest, taxonomy, category, brand, attribute, keyword, price, contradiction components.
- `app/config/score_weights.yaml`
  - scoring weights and penalties.

## Algorithm details

### Phase 1: Component delta computation

For each of the (up to 1,000) candidate campaigns, each scoring component runs and returns a `ScoreDelta` with:
- `component`: name (e.g., `"location"`, `"taxonomy"`, `"contradiction"`)
- `delta`: signed float score contribution
- `reason`: human-readable explanation string

Components are split into two groups:

**Spec-gated** (active only when query explicitly states the dimension):
- `location` — geo match/mismatch
- `demographic` — age and gender alignment
- `interest` — user interest overlap
- `brand_alignment` — explicit brand mention in query
- `attribute_alignment` — structured attribute intent match
- `price_alignment` — price constraint satisfaction

**Always-on** (apply regardless of query specificity):
- `taxonomy` — category/subcategory/leaf match
- `category_alignment` — extracted-category to campaign alignment
- `keywords` — five-bucket keyword overlap (brand, model, category, feature, benefit terms)
- `contradiction` — negative keyword penalty

Batch mode (`apply_batch`) is used where available to reduce per-component Python overhead.

### Phase 2: Coverage-ratio scoring

```python
active_params = _active_spec_count(parsed_query)  # how many spec dimensions were stated

for each candidate i:
    spec_delta   = sum(d.delta for d in deltas if d.component in _SPEC_GATED)
    always_delta = sum(d.delta for d in deltas if d.component not in _SPEC_GATED)

    if active_params > 0:
        matched = count of spec-gated components where delta > 0
        ratio = matched / active_params
    else:
        ratio = 1.0

    scores[i] = faiss_scores[i] + spec_delta * ratio + always_delta
```

### Phase 3: Sort, normalize, build response

1. `argsort(scores)[::-1][:top_k]` — descending sort, keep top 1000.
2. `q_max = _query_potential_max(parsed_query, config, faiss_scores.max())` — sum of best possible FAISS base score + all positive always-on deltas + positive spec-gated deltas for each dimension the query stated.
3. `internal_score = clamp(final_score / q_max, 0.0, 1.0)` — bounded [0,1] normalized score.
4. `relevance_score = round(final_score, 6)` — raw score (can exceed 1.0).
5. Score breakdowns emitted for top `debug_top_n` (default 5) candidates.

### Edge cases

- No candidates: returns empty outputs.
- Tie ordering is deterministic due to numpy `argsort` stability and consistent candidate order.
- Missing optional campaign metadata does not crash ranking — components return 0 delta for missing fields.
- `active_params == 0` (broad query with no user context): `ratio = 1.0`, spec deltas apply at full weight.

## Design rationale

### Why not cross-encoder reranking

Cross-encoders (e.g., `ms-marco-MiniLM-L6-reranker-v1`) take (query, document) pairs and run a full forward pass for each pair. At 1,000 candidates, this means 1,000 model forward passes:
- On CPU: ~10-100ms per inference × 1000 = 10-100 seconds. Completely incompatible with a 100ms budget.
- On GPU: ~0.5-2ms per inference × 1000 = 500ms-2s. Still far too slow.
- Even with batching, the compute budget for 1,000 pairs is 200-800ms at realistic hardware.

Bi-encoder (FAISS) recall + lightweight additive scoring is the standard latency-safe pattern for this retrieval scale. This is how production ad retrieval systems (Google, Meta) work at much larger scale.

### Why not re-embedding or LLM reranking

Any approach that requires running the embedding model again per candidate (pairwise re-embedding) or calling an LLM per candidate has the same 1,000× scaling problem as cross-encoders. The query embedding is computed exactly once and reused across all stages.

### Why componentized additive deltas over a learned reranker

A learned reranker (trained on click/conversion labels) would be ideal in production, but:
1. No real click/conversion labels exist for this synthetic dataset.
2. Learned rerankers are black boxes — they provide no explanation for why a campaign ranked where it did.
3. The componentized approach returns `score_breakdown` in metadata for every top result, making ranking decisions fully auditable per request.
4. Weights are centralized in `score_weights.yaml` and can be tuned without code changes.

The tradeoff is that hand-tuned weights are suboptimal relative to a properly trained model. With real business outcome labels, this framework could be used as feature engineering for a trained ranker without architectural changes.

### Why coverage-ratio scaling for spec-gated components

Consider two queries:
- **Broad:** "running shoes" (no location, age, gender, brand, price stated) — `active_params = 0`, `ratio = 1.0`
- **Specific:** "Nike running shoes for 30-year-old men in San Francisco under $150" — `active_params = 5`

Without coverage ratio, a campaign satisfying 2 of the 5 stated constraints would receive the same spec delta as a campaign satisfying all 5 — the partial-match campaign would be over-ranked relative to the exact-match campaign.

With coverage ratio, the spec delta is scaled by `matched / active_params`. A campaign satisfying 5/5 constraints gets `ratio = 1.0` and full spec delta. A campaign satisfying 2/5 gets `ratio = 0.4` — its spec delta is 40% as large, correctly reducing its rank relative to better-matching campaigns.

Always-on components (taxonomy, category, keyword, contradiction) are not gated because they are meaningful signal for both broad and specific queries — taxonomy match matters whether or not the user specified a location.

### Why `faiss_scores` as the base, not zero

FAISS cosine similarity is the primary semantic relevance signal. Starting from it means campaigns that are semantically far from the query cannot be rescued by targeting-spec matches alone. A campaign about "retirement financial planning" should not outrank a campaign about "running shoes" for a marathon query just because it happens to target 30-year-old males in San Francisco.

The final score is `faiss_score + spec_delta * coverage_ratio + always_delta` — targeting and intent signals add to semantic similarity, never replace it.

### Why two output scores (`relevance_score` and `internal_score`)

- `relevance_score` is the raw final score (can exceed 1.0 when component deltas are large). It preserves the actual ordering signal and relative magnitude of scores — useful for downstream systems that want to preserve ranking distance.
- `internal_score` is `clamp(final_score / q_max, 0, 1)` — bounded, interpretable as a fraction of the theoretical best possible score for this query. Appropriate for UI display, analytics, and A/B metric comparison.

Known contract risk: the `ranking.py` module docstring states `relevance_score` is normalized and clamped. The code assigns the raw score. Consumers should use `internal_score` for bounded interpretation.

### Experiment history

- **Experiment 1 (componentized scoring pipeline):** Unit tests validate deterministic ordering, coverage-ratio math, 1000 trim, and missing-metadata resilience. The framework was retained for interpretability and tunability.
- **Experiment 2 (sigmoid display-score normalization):** Z-score + sigmoid normalization was explored on a 512-campaign response sample (mean final score 0.4478, std 0.4490). Not adopted — the current `internal_score` (query-adaptive normalization by potential max) provides a cleaner bounded score without requiring the population distribution of a specific response batch.
- **Experiment 3 (weight tuning loop):** `score_weights.yaml` weights were tuned iteratively using the `fine_tuning_runs.py` script and test query replay. No versioned before/after metrics were saved — a known documentation gap.

## Performance notes

- Batch component evaluation reduces Python-loop overhead.
- Candidate count cap keeps sorting and component evaluation bounded.
- Lightweight arithmetic and preloaded arrays keep stage fast.
- This stage runs after filtering and is not parallelized with other stages (it depends on filter output).

## Failure modes and fallbacks

- Weight misconfiguration can over-penalize or over-boost specific features — use `score_breakdown` metadata to diagnose.
- Contract mismatch: `relevance_score` stores raw score, comments describe clamped semantics. Use `internal_score` for bounded interpretation.
- If parser extracts weak signals (broad query with no constraints), ranker relies mostly on FAISS base similarity — expected behavior for broad queries.

## How to change this safely

- Run:
  - `pytest -q tests/test_ranking.py tests/test_scoring_components.py`
- Watch:
  - top result precision on golden queries
  - score component drift in `metadata.score_breakdown`
  - output size and latency for heavy-query scenarios

## Related docs

- [Output Design Rationale](../05-design-decisions/output-design-rationale.md)
- [Ranking Experiments](../03-experiments/ranking-experiments.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
