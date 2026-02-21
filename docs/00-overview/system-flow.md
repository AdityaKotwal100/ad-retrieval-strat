# System Flow

## Architecture

The service is a single FastAPI app (`app/main.py`) with one retrieval endpoint (`POST /api/retrieve`) and one health endpoint (`GET /health`). Startup preloads all heavy dependencies into `app.state` in the lifespan hook in three sequential phases:

- **Phase 1** (embedding backend alone, required by later phases): `app/services/__init__.py`.
- **Phase 2** (independent services, run in parallel): eligibility scorer (`app/services/eligibility.py`), category extractor (`app/services/categories.py`), campaign index (`app/services/retrieval.py`), score config (`app/services/scoring/score_config.py`).
- **Phase 3** (dependent services, run in parallel after Phase 2): ranker (`app/services/ranking.py`), query parser (`app/services/query_parser.py`), eligibility filter (`app/services/campaign_eligibility.py`).

This startup pattern removes model and index load cost from request latency. Every request serves from fully initialized, warm objects.

### Why a single FastAPI service, not microservices

Each logical component (eligibility, categories, FAISS retrieval) could be a separate microservice. This is rejected for this system because:

- **Network cost:** Each service call adds 5-30ms of socket, serialization, and queuing overhead. With a 100ms budget and 3 parallel stages, splitting into microservices would consume 15-90ms in network overhead alone.
- **Shared state:** The embedding model is shared between eligibility scoring, category extraction, and FAISS indexing. In a single process, this is a single loaded artifact. In microservices, each would need its own copy or an additional embedding microservice call.
- **Operational simplicity:** One process, one deploy, one health check. Microservice decomposition adds orchestration complexity (service discovery, circuit breakers, distributed tracing) that is disproportionate for the current scale.

At 10× QPS or if components need independent scaling, microservice decomposition becomes worthwhile.

## End-to-end request sequence

1. FastAPI validates request body against `RetrieveRequest` in `app/models.py`.
2. `retrieve` in `app/main.py` trims `query` and rejects empty strings (`HTTP 400`).
3. Hard blocklist pre-check runs (`EligibilityScorer.blocklist_match`). Two outcomes:
   - Score is `0.0` (hard block): endpoint returns immediately with `campaigns=[]` and `gated: true` in `metadata`. No embedding or downstream work is done.
   - Score is between `0.0` and `1.0` exclusive (soft cap): request falls through. `score_with_metadata` applies the cap after semantic scoring in Phase 3.
4. Query is embedded once (`svc.embed`) and reused by all downstream stages.
5. Three expensive stages execute in parallel (`asyncio.gather` in `app/main.py`):
   - `EligibilityScorer.score_with_metadata`.
   - `CategoryExtractor.extract`.
   - `CampaignIndex.search` with `FAISS_TOP_K`.
6. Eligibility gate applies (`eligibility_score < ELIGIBILITY_GATE`). Gated queries skip filtering and ranking, and categories are suppressed.
7. Query parsing runs (`QueryParser.parse`) to extract structured constraints.
8. Hard campaign filtering runs (`CampaignEligibilityFilter.filter_with_refill`) for geo, age, gender, negative keyword, and price-hard constraints.
9. Ranking runs (`ContextReranker.rerank`) using composable score components and coverage-ratio scaling.
10. Response model (`RetrieveResponse`) is assembled with campaign results, timing metrics, and debug metadata.

### Why embed exactly once (step 4)

The embedding model is the single largest latency contributor (8-20ms). All three parallel stages in step 5 need the same query representation — eligibility scoring uses it for commercial centroid similarity, category extraction uses it for FAISS neighbor search, and retrieval uses it for campaign search. Computing it once and sharing the vector costs nothing compared to the alternative of computing it 3 times.

### Why steps 5 and 7 are ordered the way they are

- Embedding (step 4) must complete before parallel stages (step 5) can start — all three need the vector.
- Query parsing (step 7) runs after the parallel stages because it is fast (<2ms regex + dict) and its output (`ParsedQuery`) is only needed by filtering and ranking (steps 8-9), which come after.
- Running query parsing in parallel with step 5 would save ~1-2ms in theory but adds coordination complexity. Given it runs in the main event loop thread (no I/O), keeping it sequential after the parallel gather is simpler with negligible cost.

### Why the eligibility gate (step 6) comes before filtering and ranking (steps 8-9)

If a query is gated (eligibility below threshold), no campaign output is returned regardless. Running the filter and rerank pipeline for a gated query would waste 5-35ms of compute for no output. The gate early-exits the expensive downstream pipeline for unsafe queries.

## Main component interactions

- `EligibilityScorer` decides whether ads are allowed and emits detailed safety metadata.
- `CategoryExtractor` adds semantic category hints used by ranking (`CategoryAlignmentComponent`).
- `CampaignIndex` provides candidate ids and aligned artifact arrays used by filters and ranker.
- `QueryParser` normalizes context and query intents into `ParsedQuery`.
- `CampaignEligibilityFilter` removes non-serveable campaigns before scoring.
- `ContextReranker` re-scores survivors and limits output size.

## Critical invariants

- `ad_eligibility` must stay in `[0,1]`.
  - Enforced by scoring clamp in `app/services/eligibility.py` and schema bounds in `app/models.py`.
- Service target is sub-100ms p95.
  - Declared in `app/main.py` module header and supported by startup warmup and parallel request stages.
- Return at most 1,000 campaigns.
  - Controlled by `FAISS_TOP_K` (`app/config.py`) and `ContextReranker.rerank(top_k=1000)`.
- Never serve campaigns for hard-blocked or gated sensitive queries.
  - Hard-block exit is in `app/main.py` via the `blocklist_match` result check (score == 0.0). Gate check (eligibility score below threshold) is also in `app/main.py`. `EligibilityScorer._score_internal` handles semantic scoring but does not short-circuit the request itself.
- Keep response shape stable even for blocked/gated flows.
  - `RetrieveResponse` model is always returned with full top-level fields.
- Suppress extracted categories when query is gated.
  - Explicitly set in `app/main.py` to avoid insensitive category echoing.

## Scalability notes

- **10× campaign corpus (100k):** Switch FAISS index from `IndexFlatIP` to `IndexHNSWFlat`. Filter and rerank cost stays bounded by `FAISS_TOP_K=1000`. No other changes needed.
- **100× QPS:** FastAPI + async I/O handles concurrent requests on a single process. ONNX and FAISS hold the GIL only briefly during C-extension calls, allowing real parallelism via ThreadPoolExecutor. Horizontal scaling (multiple processes behind a load balancer) is straightforward since all state is read-only after startup.
- **100× campaign corpus (1M):** `IndexIVFPQ` with product quantization. Microservice decomposition for retrieval becomes worthwhile at this scale to allow independent scaling of the FAISS shard layer.

## Related docs

- [Request and Response Contract](./request-response-contract.md)
- [Latency Budget](./latency-budget.md)
- [Ingress Pipeline](../01-pipelines/01-ingress-and-validation.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
