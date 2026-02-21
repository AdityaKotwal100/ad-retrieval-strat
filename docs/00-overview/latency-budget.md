# Latency Budget

## Budget table

Target from `app/main.py` header: p95 under 100ms.

| Phase | Code path | Expected budget (ms) | Notes |
|---|---|---:|---|
| Ingress and validation | `app/main.py` + Pydantic | 0.5 to 2 | Cheap parsing and guard checks |
| Hard blocklist pre-check | `EligibilityScorer.blocklist_match` | less than 1 | Short-circuit for obvious blocked text |
| Query embedding | `EmbeddingService.embed` | 8 to 20 | Dominant single-stage compute on CPU |
| Parallel stage (eligibility + categories + search) | `asyncio.gather` in `app/main.py` | 5 to 20 | Runs in executor threads; wall time is max of sub-stages |
| Query parse | `QueryParser.parse` | less than 2 | Regex and dictionary extraction only |
| Filter + rerank | `CampaignEligibilityFilter.filter_with_refill` + `ContextReranker.rerank` | 5 to 35 | Both stages share one timer (`timing_ms.filter_and_rerank`); no per-stage breakout |
| Response assembly | `CampaignResult` conversion + metadata | 1 to 8 | No dedicated timer; captured implicitly as `total_ms` minus all named phases |
| Total | `latency_ms` in response | p95 less than 100 | Measured per request |

## Evidence from saved outputs

From `data/test_results/*/output.json` (56 cases in this repo snapshot):

- min: 0.45ms
- p50: 9.12ms
- p95: 30.84ms
- max: 32.3ms
- mean: 10.55ms

These numbers are **optimistic** relative to production:
- Local runs have no network overhead.
- Many cases in the saved set are hard-blocked or gated, which short-circuit at ~1ms (blocklist check) or ~25ms (embed + eligibility only), pulling down the distribution.
- Production hardware may be noisier than a development laptop.

Real p95 for non-gated requests on noisy hardware is likely in the 50-80ms range, still within the 100ms target.

## How the service stays under 100ms

### Embedding is the bottleneck — minimize it

- ONNX backend reduces embedding time by 20-40% vs raw SentenceTransformer.
- Query is embedded exactly once and the vector is shared across eligibility, categories, and retrieval.
- Startup warmup pre-initializes executor thread pools and JIT paths, eliminating a ~90ms cold-start penalty on the first real request.

### Parallelism eliminates stacking costs

- Eligibility scoring, category extraction, and FAISS search are independent given the query vector.
- Running sequentially: ~5ms + ~3ms + ~3ms = ~11ms.
- Running in parallel via `asyncio.gather` + `ThreadPoolExecutor`: wall time is max(~5ms, ~3ms, ~3ms) ≈ 5ms.
- This saves ~6ms per request — meaningful against a 100ms budget.

### FAISS in-memory exact search is fast at this corpus size

- 10,783 campaigns × 384d × float32 = 16.6MB, fits in L2/L3 cache.
- Flat exact search takes 1-3ms measured.
- No network hop (in-process), no serialization overhead.

### Candidates are bounded throughout

- `FAISS_TOP_K=1000` caps retrieval output.
- Filter and rerank cost scales linearly with candidate count — this cap keeps them predictable.
- Refill loops in `filter_with_refill` are bounded by `max_refill_rounds` and +500 increments to prevent runaway latency in sparse campaign sets.

### Category extraction is fast because the index is preloaded

- FAISS category index is loaded at startup and held in memory.
- No runtime taxonomy embedding calls during extraction (unlike the matrix fallback).
- 50-neighbor search over the category index takes ~0.5ms.

### Score computation is lightweight arithmetic

- Ranking components use preloaded numpy arrays and dictionary lookups, not model inference.
- Batch mode reduces Python-loop overhead across 1,000 candidates.

## Top latency risks and mitigations

### Embedding backend regression

- **Risk:** A model update or ONNX configuration change causes embed time to spike from 10ms to 40ms.
- **Mitigation:** Monitor `timing_ms.embed` per request. Keep `EMBEDDING_BACKEND=onnx` as default. Profile on the same hardware class as production.

### Candidate explosion in filter and rerank

- **Risk:** `FAISS_TOP_K` is increased without measuring downstream latency impact. At 5,000 candidates, rerank cost grows 5× to potentially 50-175ms.
- **Mitigation:** Hard cap at `FAISS_TOP_K=1000`. Load-test any change to this value.

### Refill loops adding latency for sparse campaign sets

- **Risk:** A query with strict targeting (narrow geo + age + gender) exhausts the initial candidate pool and triggers multiple refill rounds, each adding a FAISS search.
- **Mitigation:** Refill bounded by `max_refill_rounds` and +500 candidate increments. Monitor `n_candidates_after_filter` for low-survival-rate queries.

### Heavy metadata payload for top results

- **Risk:** Increasing `debug_top_n` from 5 to 50 grows the score breakdown payload and JSON serialization time.
- **Mitigation:** `debug_top_n` is already limited in the ranker. Keep it small in production; increase only for debugging sessions.

### Cold starts after deploy or process restart

- **Risk:** First request after deploy pays embedding cold-start costs (~90ms extra) plus potential ONNX JIT compilation.
- **Mitigation:** Startup warmup runs 10 embed calls through executor threads. Use `/health` checks before routing traffic to a freshly started instance.

## Related docs

- [System Flow](./system-flow.md)
- [Latency Experiments](../03-experiments/latency-experiments.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
