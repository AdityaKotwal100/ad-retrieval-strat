# Tradeoffs

## Embedding model: `all-MiniLM-L6-v2` vs alternatives

**Choice:** `all-MiniLM-L6-v2` (384d) as the default embedding model.

**Alternatives considered:**

| Model | Avg cosine gap | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 0.6099 | Chosen |
| `multi-qa-MiniLM-L6-cos-v1` | 0.6003 | Marginally worse gap |
| `msmarco-MiniLM-L6-cos-v5` | 0.4837 | Worst — asymmetric training hurts |
| `all-mpnet-base-v2` | Untested | 768d, ~2× FAISS memory, slower embed |
| OpenAI `text-embedding-3-small` | Untested | API call adds ~30-80ms, destroys latency budget |
| E5-large / BGE-large | Untested | 768d+, >3× inference time on CPU |

**Why `all-MiniLM-L6-v2`:** It produced the highest average cosine gap between relevant and irrelevant query-document pairs in the embedding comparison experiment (`model-test/embedding-model-comparison.md`). The msmarco variant scored worst because it was fine-tuned for asymmetric query-to-passage retrieval (short queries vs long passages), while ads are short-form text — the symmetric-trained model matches better. Larger models (MPNet, E5) would require 768d vectors, doubling FAISS index memory and adding FAISS search time. Any API-based embedding model is incompatible with the 100ms budget (network round-trip alone costs 30-80ms even without queuing).

**Cost:** No multi-lingual support. May underperform on very long or highly technical queries.

---

## Embedding backend: ONNX vs SentenceTransformer

**Choice:** ONNX runtime (`EMBEDDING_BACKEND=onnx`) as default; SentenceTransformer as fallback.

**Why ONNX:** The ONNX export eliminates PyTorch autograd graph overhead and Python-level dispatch. On CPU, this reduces per-query embed time by roughly 20-40% compared to the raw SentenceTransformer path. The embedding step is the single largest latency contributor (8-20ms in the budget), so any reduction here has proportional p95 impact. The ONNX model is loaded once at startup and held in process — no interpreter overhead per call.

**Cost:** Platform-specific ONNX export. Requires regenerating the ONNX file if the base model changes. Thread-count settings (`OMP_NUM_THREADS`) need to be tuned per deployment environment to avoid over-subscription.

---

## FAISS `IndexFlatIP` vs approximate ANN and external vector DBs

**Choice:** In-process `faiss.IndexFlatIP` over all external vector DB options and approximate index types.

**Why not Pinecone / Qdrant / Weaviate / pgvector:**
- Every query to an external vector DB adds a network round-trip. Even local Qdrant adds 5-15ms of socket and serialization overhead; Pinecone (cloud) adds 30-100ms. With a 100ms p95 target and ~10-20ms already spent on embedding, there is no room for a remote index hop.
- External services require separate deployment, auth, and ops. For a 10k-campaign corpus this is operationally disproportionate.
- In-process FAISS keeps the entire retrieval path in a single Python process with no serialization cost.

**Why not approximate ANN (`IndexHNSWFlat`, `IndexIVFFlat`):**
- At 10,783 campaigns × 384 dimensions, `IndexFlatIP` holds approximately 16MB of float32 data. This fits easily in L2/L3 cache. A flat exact search over this takes ~1-3ms — already within the 5-20ms parallel-stage budget.
- Approximate indexes trade recall for speed. At 10k campaigns, the recall loss from approximate search is not worth the latency gain — you would lose relevant campaigns without measurable latency benefit.
- HNSW adds significant build-time and memory overhead for a corpus this small. The break-even point for approximate search is roughly 100k+ vectors at 384d on modern hardware.
- `IndexFlatIP` search quality is exact and deterministic, which simplifies debugging ranking anomalies.

**Why `IndexFlatIP` not `IndexFlatL2`:** Both embedding backends return L2-normalized vectors. For unit-norm vectors, inner product equals cosine similarity exactly. `IndexFlatIP` + normalized vectors gives cosine ANN semantics with no extra normalization step per query.

**Scalability path:** At 10× corpus (100k campaigns), switch to `IndexHNSWFlat` with M=32, ef=64. At 100× (1M campaigns), switch to `IndexIVFPQ` with product quantization, accepting ~5-10% recall loss. No code change to the retrieval interface is required — only the index artifact changes.

---

## Eligibility scoring: layered ML vs single-model vs LLM

**Choice:** Three-layer approach — regex/ML blocklist → lightweight v5 joblib model → continuous penalty and commercial boost.

**Why not LLM-based classification:**
- GPT-4o-mini via API: ~200-500ms per call. With a 100ms p95 budget, a single LLM call consumes the entire budget.
- Even a local quantized LLM (Llama 3B): ~50-150ms on CPU, still too slow.
- LLMs also add non-determinism — the same query can get different scores across runs, making policy audits unreliable.

**Why not a single heavy BERT toxicity classifier:**
- `unitary/toxic-bert` inference on CPU: ~50-100ms per query. Kills the budget.
- These models are trained on social media toxicity (hate speech, profanity), not the nuanced ad-appropriateness task (grief, medical crises, financial hardship — which are not toxic but should not show ads).
- The toxicity model prototype (`model-test/harmful-content/`) was evaluated and confirmed too slow for inline use.

**Why not pure k-means centroid scoring:**
- Experiment 1 (bipartite k-means) achieved MAE 0.195 with Spearman +0.779 — insufficient for a safety-critical gate.
- The fundamental problem is that a single centroid metric conflates the safety signal with commercial affinity, causing dilution when medium-risk phrases appear in commercial contexts.
- Rejected as standalone approach; the core insight (separate safety and commercial signals) was kept.

**Why the current layered approach:**
- Regex/ML blocklist: microsecond-level, catches known exact patterns deterministically. Hard block (score=0.0) exits before any model inference runs.
- Joblib v5 model: lightweight TF-IDF + logistic regression artifact (~100KB), sub-millisecond inference. Trained specifically on the ad-appropriateness task with custom sensitive + commercial exemplar sets. Fail-closed on any exception.
- Continuous penalty × commercial boost: allows nuanced scores for borderline queries (0.3-0.7 range) where the right answer is "show fewer/cheaper ads" rather than hard block.
- Layers are independent — each can be tuned or replaced without changing the others.

**Why v5 model specifically (not v10):** v5 was chosen for stability and verified behavior compatibility with the saved test suite. v10 showed higher AUC offline (`micro ROC AUC 0.9716`) but the model-version upgrade rationale was not formally documented; conservative deployment preference kept v5 in production. The threshold is loaded from the model artifact, not hardcoded, allowing threshold updates without code changes.

**Why the formula `min((1 - sensitivity_penalty) * commercial_boost, fuzzy_cap)` → clamp to [0,1]:**
- `(1 - sensitivity_penalty)` converts the "how insensitive is this" probability into an "how appropriate is this" score. Sensitivity=0 → score ceiling is 1.0. Sensitivity=1 → score is 0.0.
- Multiplying by `commercial_boost` amplifies scores for clearly commercial queries and dampens them for informational queries — matching the problem statement's nuance that "informational queries can surface ads."
- `fuzzy_cap` from a soft-cap blocklist match applies an upper bound without hard-blocking.
- Neutralizing boost when `sensitivity_penalty > 0.70` prevents commercial affinity from rescuing genuinely sensitive queries.

---

## Query-index category extraction vs taxonomy-description matrix

**Choice:** FAISS query-index mode as primary; description-matrix as automatic fallback.

**Why not taxonomy-description matrix as primary:**
- The description matrix embeds taxonomy description strings (e.g., "Athletic Footwear: shoes and footwear for sports activities"). Users do not query in this language.
- A user asks "best marathon running shoes" — the description matrix must bridge vocabulary from "marathon running shoes" to "Athletic Footwear." This mismatch degrades cosine similarity and produces lower-confidence or wrong categories.

**Why query-index:** The query-index is built by embedding hundreds of representative user queries per category (generated via LLM). Each query is stored in the index with its category label. At runtime, the user's query embedding is searched against these representative queries — matching like-for-like query language. This gives a more accurate category match for the vocabulary users actually use.

**Why not LLM-based category extraction inline:**
- Same argument as eligibility: any LLM API call adds 100-500ms to request latency.
- Categories are used to boost ranking, not gate the response — a missed or slightly wrong category degrades ranking marginally, not catastrophically. A lightweight FAISS query is proportionate to this role.

**Why coherence ratio filter:** Without it, the extractor might return categories with scores spread from 0.41 to 0.80 — returning "children's clothing" for a fitness query just because it crossed the 0.40 threshold. The coherence ratio (score / top_score) filters out categories whose similarity is far below the best match, improving precision at the cost of recall.

---

## Hybrid ranking vs pure vector similarity

**Choice:** FAISS base score + componentized additive deltas + coverage-ratio scaling.

**Why not pure FAISS cosine similarity for ranking:**
- FAISS retrieves by semantic embedding similarity, which does not know about geo targeting, age ranges, price constraints, or brand intent.
- A campaign for "children's shoes" in Texas might outscore a campaign for adult marathon shoes globally in embedding space for the query "running shoes for adults in California." Hard targeting constraints and intent signals must override pure semantic ordering.

**Why not a cross-encoder reranker (e.g., `ms-marco-MiniLM-L6-reranker-v1`):**
- Cross-encoders take (query, document) pairs and must run inference for each candidate. At 1000 candidates, this is 1000 model forward passes — easily 10-100 seconds on CPU.
- Even on GPU, cross-encoder reranking at this scale is 200-800ms, which blows the budget completely.
- Bi-encoder (FAISS) + lightweight additive scoring is the standard latency-safe pattern for this retrieval scale.

**Why componentized additive deltas:**
- Each component (geo, demographic, interest, brand, taxonomy, keyword, price, contradiction) is an independent, auditable score contribution.
- The breakdown is returned in `metadata.score_breakdown` for every top result, making ranking fully explainable without replaying the request.
- Components can be individually weighted via `score_weights.yaml` without touching code.
- Components that return 0 delta have no effect — missing optional metadata does not crash ranking.

**Why coverage-ratio scaling for spec-gated components:**
- Without coverage ratio, a query specifying geo + age + brand + price would penalize campaigns that satisfy 3 out of 4 constraints equally with campaigns that satisfy 0 out of 4.
- Coverage ratio (`matched_specs / active_params`) scales the spec delta so campaigns are rewarded proportionally to how many stated constraints they satisfy.
- Always-on components (taxonomy, category, keyword, contradiction) apply at full weight regardless of coverage — they are relevant for both broad and specific queries.

---

## LLM generation vs deterministic data authoring

**Choice:** LLM-assisted synthetic campaign generation via `google/gemini-2.0-flash-001` plus deterministic post-processing.

**Why LLM generation:**
- Building 10k+ realistic campaigns manually across 20 verticals is impractical in 72 hours.
- LLMs can produce coherent brand names, descriptions, attributes, and keywords at scale with structured JSON output.
- Deterministic post-processing (taxonomy normalization, keyword extraction, targeting distribution sampling) enforces schema integrity after generation.

**Cost:** LLM-generated text has style artifacts and repeated phrasing. No ground-truth conversion or business outcome labels exist. Distribution realism is hand-tuned, not calibrated against real ad logs. This limits the ability to claim ranking accuracy beyond plausibility.

---

## Caching strategy: preload-heavy vs per-query cache

**Choice:** Heavy startup preloading and warmup; no per-query cache layer.

**Why no query cache:**
- Queries are natural language — exact-match caching has very low hit rate in practice.
- Semantic cache (cache by embedding similarity) requires infrastructure (Redis + vector search) and adds cache consistency complexity.
- The latency target is p95 < 100ms. With ONNX + FAISS + lightweight scoring, this is achievable on every query without a cache.
- Adding a cache would introduce a warm/cold distribution that makes latency harder to reason about.

**Why warmup at startup:**
- The first query pays PyTorch/ONNX thread initialization and JIT costs (~90ms extra). Warmup runs 10 embed calls through the executor threads at startup, pre-initializing thread pools so every real request runs warm.

---

## Safety-first gating vs aggressive monetization

**Choice:** Fail closed on model errors; hard gate at `ELIGIBILITY_GATE=0.3` before any campaign output.

**Why fail-closed:** A missed ad opportunity has no user-facing harm. An insensitive ad shown to someone expressing grief or self-harm has significant harm. The asymmetry of outcomes justifies conservative defaults. This is also the explicit guidance in the problem statement: "a missed ad opportunity is better than an insensitive one."

**Cost:** False-positive gating on borderline benign queries (e.g., informational health queries) reduces ad yield. Monitoring both leakage rate and over-suppression rate is necessary to tune the threshold.

---

## Async parallel execution: asyncio + ThreadPoolExecutor vs sequential

**Choice:** `asyncio.gather` with `loop.run_in_executor` for embedding, eligibility, categories, and FAISS.

**Why not sequential:** Eligibility scoring (~5ms), category extraction (~3ms), and FAISS search (~3ms) are independent given the query vector. Running them sequentially would cost ~11ms; running in parallel costs ~5ms (wall time is the max, not the sum). At p95, this saves ~6ms — meaningful against a 100ms budget.

**Why ThreadPoolExecutor not ProcessPoolExecutor:** All three tasks are CPU-bound but short. Subprocess spawn overhead for ProcessPoolExecutor (~50ms) would negate any parallelism benefit. Threads share the embedding model and FAISS index in-process with no copy overhead; the GIL is released during ONNX/FAISS C extension calls, allowing true parallelism.

---

## Notes on rejected or unadopted variants

- **Single-pass k-means appropriateness scorer:** MAE 0.195, insufficient calibration for safety-critical gating.
- **Two-pass orthogonal scoring (hard threshold variant):** Achieved MAE 0.146 but hard-block accuracy dropped to 78% — unacceptable for a safety gate.
- **Sigmoid display-score normalization:** Explored in `experiments/relevance-scoring/`; not adopted because raw `relevance_score` and clamped `internal_score` already provide two interpretability levels.
- **External vector DB (Pinecone/Qdrant prototype):** Network latency incompatible with 100ms budget at the current corpus size.
- **Cross-encoder reranker:** Forward-pass cost at 1000 candidates eliminates this as an option on CPU within latency budget.

## Related docs

- [Ad Eligibility Experiments](../03-experiments/ad-eligibility-experiments.md)
- [Retrieval Experiments](../03-experiments/retrieval-experiments.md)
- [Ranking Experiments](../03-experiments/ranking-experiments.md)
- [Output Design Rationale](./output-design-rationale.md)
- [Indexing and Embeddings](../02-data/indexing-and-embeddings.md)
