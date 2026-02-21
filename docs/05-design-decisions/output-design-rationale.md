# Output Design Rationale

## Why `ad_eligibility` is 0 to 1

- `RetrieveResponse` enforces `Field(ge=0.0, le=1.0)` in `app/models.py` and the score is clamped in `EligibilityScorer._score_internal` via `np.clip`.
- A probability-like bounded range is interpretable by any downstream consumer without needing to understand the internal scoring formula.
- The problem statement explicitly defines this range and gives target values for reference queries (e.g., "best running shoes" → 0.95, "I'm feeling stressed" → 0.5, "self-harm" → 0.0). Matching this makes evaluation straightforward.

## Why `relevance_score` and `internal_score` are separate fields

- `relevance_score` is the raw final score from the ranker (can exceed 1.0 when component deltas are large). It preserves the actual ordering signal and relative magnitude — useful for downstream systems that want to measure ranking distance between candidates.
- `internal_score = clamp(final_score / q_max, 0, 1)` is normalized by the theoretical maximum achievable score for this specific query. It gives a bounded, query-comparable score suitable for UI display, analytics, and A/B testing.
- The separation allows exposing both signals without forcing a choice. A consumer showing a relevance bar to a user would use `internal_score`; a system computing nDCG would use `relevance_score` for its raw ranking signal.

**Known contract risk:** The `ranking.py` module docstring describes `relevance_score` as normalized and clamped. The implementation assigns the raw score. Until this is resolved, use `internal_score` for bounded interpretation.

## Why categories are shaped as they are

- Top-level `extracted_categories` is a flat list of strings — the simplest possible interface for a downstream system that just needs to know what the query is about.
- Full scored category objects `[{"category": "...", "score": 0.80}]` are preserved in `metadata.categories` for debugging, ranking explainability, and threshold tuning.
- Separating the two means: simple clients need no schema changes when category confidence values change; diagnostic clients get the full picture.

## Why categories are suppressed for gated queries

Returning categories like "baby food", "financial planning", or "life insurance" for a bereavement or crisis query is insensitive even if no campaigns are returned. The category list echoes what the system understood about the query intent — surfacing this for a crisis query signals the system saw the query in a commercial context, which can feel dehumanizing. Categories are explicitly cleared in `app/main.py` after the eligibility gate check.

## Why up to 1,000 campaigns are returned

- The problem statement requires returning the top 1,000 campaigns per request.
- Downstream systems (ad auctions, delivery logic, A/B ranking experiments) may want more candidates than the system will actually show — 1,000 gives enough depth for secondary business logic.
- `FAISS_TOP_K=1000` and `ContextReranker.rerank(top_k=1000)` both enforce this cap.
- Response size at 1,000 campaigns is ~500KB JSON — acceptable payload.

## Why metadata exists

The problem statement explicitly includes `metadata: {}` in the response contract and calls it "any debugging info, model names, intermediate scores." The implementation makes this verbose by design:

- `n_candidates_retrieved`, `n_candidates_after_filter`, `n_results`: lets callers see filtering loss at a glance without replaying the request.
- `filter_drops`: per-filter drop counts show exactly which targeting constraint is eliminating campaigns.
- `timing_ms`: per-phase timing lets callers identify which stage is slow without needing server-side profiling access.
- `eligibility_detail`: full eligibility scoring breakdown for safety audit — which layer triggered, what the probability was, what the commercial sim was.
- `categories` with scores: debugging category extraction and ranking boost behavior.
- `parser_output`: shows what the query parser extracted — useful for diagnosing why a constraint wasn't applied.
- `score_breakdown`: per-component deltas for top-N results — explains ranking decisions without a debugger.

A missed ad is hard to diagnose from final rankings alone. Verbose metadata makes safety and ranking behavior auditable per request.

## Why `timing_ms` shape differs between blocked and normal paths

Hard-blocked requests (blocklist score = 0.0) return immediately before embedding. Their `timing_ms` contains only `{"total": ...}` — the embed, parallel_stages, parse, and filter_and_rerank phases never ran. Normal requests include all five keys.

This is intentional: returning zeros for phases that never ran would be misleading. Absent keys are a clear signal that the request was short-circuited. Consumers should treat absent `timing_ms` sub-keys as "phase did not execute."

## Why thresholds exist

- `ELIGIBILITY_GATE` blocks low-appropriateness queries before any downstream compute.
- Category thresholds and fallback floors prevent low-confidence category noise from reaching ranking.
- Hard price, geo, age, gender, and negative-keyword thresholds keep serveability deterministic — these are binary targeting constraints, not soft signals.
- Separating hard filters (deterministic, before ranking) from soft signals (additive component deltas in ranking) makes the system predictable: a campaign either can be served or cannot, before relevance is computed.

## Why caution bias is explicit

- On sensitive-model inference failure, scorer defaults to max insensitive probability and gates.
- Hard blocklist and semantic gates short-circuit before ranking.
- In ambiguous or failure cases, it is safer to suppress ads than to risk harmful ad serving. This matches the problem statement's explicit guidance: "when in doubt, err on the side of caution."

## Contract caveat

`relevance_score` in current runtime is raw final score and can exceed 1.0, while comments describe clamped behavior. Consumers should rely on `internal_score` for bounded interpretation until this mismatch is resolved.

## Related docs

- [Request and Response Contract](../00-overview/request-response-contract.md)
- [Safety and Sensitivity](./safety-and-sensitivity.md)
- [Ranking and Scoring](../01-pipelines/05-ranking-and-scoring.md)
