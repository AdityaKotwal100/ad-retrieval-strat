# 04 Retrieval

## Purpose

Retrieve high-recall candidate campaigns from FAISS using the query embedding.

## Inputs and outputs

### Inputs

- Query embedding vector (384d, L2-normalized).
- `top_k` (default `FAISS_TOP_K=1000`).

### Outputs

- `ids`: FAISS row ids.
- `scores`: raw inner-product similarities (== cosine similarities on normalized vectors).

Example:

```python
ids, scores = index.search(query_vec, top_k=1000)
```

## Key files, classes, and functions

- `app/services/retrieval.py`
  - `CampaignIndex.__init__`
  - `CampaignIndex.search`
- `app/config.py`
  - `DATA_DIR`, `FAISS_TOP_K`
- Build scripts (see repo history; run whichever current build script populates `data/`)

## Algorithm details

### Startup

`CampaignIndex.__init__` loads and validates all artifacts:

1. `faiss.IndexFlatIP` from `data/faiss.index`.
2. Response surface metadata from `data/campaigns_meta.json` (one dict per campaign, indexed by FAISS row id).
3. Numpy filter arrays: `ages_min.npy`, `ages_max.npy`, `genders.npy`, `bids.npy`, `budgets.npy`.
4. Targeting lists: `geo_include.json`, `geo_exclude.json`.
5. Ranking sidecar data: `campaign_taxonomy.json`, `brands.json`, `keyword_sets.json`, `negative_keywords.json`, `attributes.json`, `interests.json`.

Each sidecar array length is validated against FAISS row count. Mismatch raises `ValueError` at startup rather than silently serving incorrect results.

### Per-request search

1. Query vector is reshaped to `(1, d)` and passed to `index.search(query_vec, k)`.
2. `k` is bounded by actual index size to avoid FAISS returning padding `-1` ids unnecessarily.
3. Any `-1` ids returned by FAISS are removed (can appear if k > index size despite the clamp).
4. Returned ids directly index all sidecar arrays and response metadata 1:1.

Current corpus size: **10,783 campaigns** at **384 dimensions** (`data/index_meta.json`).

### Edge cases

- Empty index: returns empty arrays.
- Artifact length mismatch with FAISS row count: startup raises `ValueError`.
- Missing index file: startup fails with explicit build instruction.

## Design rationale

### Why `IndexFlatIP` not approximate ANN

The corpus is 10,783 campaigns × 384 dimensions. Float32 storage is `10,783 × 384 × 4 bytes ≈ 16.6 MB`. This fits entirely in L2/L3 cache. An exact flat search over this takes 1-3ms measured, which fits comfortably within the 5-20ms parallel-stage budget.

Approximate ANN indexes (`IndexHNSWFlat`, `IndexIVFFlat`, `IndexIVFPQ`) trade recall for speed. At this corpus size, the recall loss is not worth the latency savings — you save <1ms while potentially losing relevant campaigns from the top-1000 set. The break-even point for approximate search is roughly 100k+ vectors at 384d on modern hardware.

`IndexHNSWFlat` also carries significant build-time overhead and requires graph tuning (M, efConstruction, efSearch) that adds operational complexity for no benefit at this scale.

**Scalability path:**
- 10× corpus (~100k campaigns): `IndexHNSWFlat` with M=32, ef_search=64. ~5% recall loss, 3-5× search speedup.
- 100× corpus (~1M campaigns): `IndexIVFPQ` with product quantization. ~5-10% recall loss, 10-20× speedup and 4-8× memory reduction.
- Neither requires changes to `CampaignIndex.search` — only the artifact changes.

### Why not an external vector database (Pinecone, Qdrant, Weaviate, pgvector)

Every query to an external vector DB adds a network round-trip:
- Local Qdrant (same machine, Docker): 5-15ms socket + serialization overhead.
- Remote Pinecone (cloud): 30-100ms including TLS + queuing.

The total latency budget is 100ms. Embedding already costs 8-20ms. A remote vector DB call would consume 30-100ms of the remaining ~80ms. This leaves no room for eligibility scoring, ranking, and response assembly.

For a corpus of 10k campaigns, operating a separate vector DB service is also operationally disproportionate. In-process FAISS requires no network config, no auth, no cluster management, and has zero serialization overhead — the index is a direct memory-mapped array.

At 10× QPS or 10× corpus size, the scalability trade-off may shift toward a managed vector DB. At current scale, in-process is the right choice.

### Why inner product (`IndexFlatIP`) not L2 (`IndexFlatL2`)

Both embedding backends (`ONNXBackend`, `SentenceTransformerBackend`) return L2-normalized vectors — they explicitly divide by the L2 norm after inference. For unit-norm vectors, inner product equals cosine similarity exactly:

```
cosine(a, b) = (a · b) / (|a| × |b|) = a · b  when |a| = |b| = 1
```

`IndexFlatIP` gives cosine similarity semantics with no extra computation. `IndexFlatL2` would require computing L2 distance and converting, which is an unnecessary step when vectors are already normalized.

### Why 1,000 candidates (`FAISS_TOP_K=1000`)

The problem statement requires returning the top 1,000 campaigns per request. Retrieving exactly 1,000 from FAISS and passing them through filtering and ranking means:

- High recall: even after filtering drops some candidates (geo, age, gender, negative keyword conflicts), enough survivors remain for a meaningful ranked output.
- Bounded latency: downstream filter and rerank cost scales linearly with candidate count. Capping at 1,000 keeps this predictable.
- Bounded response size: at ~500 bytes per campaign in JSON, 1,000 campaigns is ~500KB — acceptable payload size.

If `FAISS_TOP_K` were reduced (e.g., 200), filtering losses could leave too few ranked results for the downstream system. If increased (e.g., 5,000), filter and rerank latency would grow proportionally.

### Why in-process, not a microservice

A microservice retrieval architecture would add: HTTP client overhead (~2ms), JSON serialization (~5ms at 1000 results), and an additional deployment unit to manage. All three are unnecessary overhead for a single-service system. The FAISS index and all sidecar data are loaded once at startup and held in `app.state` — any request can access them directly without IPC.

### Embedding text design

Campaign text is embedded as natural language sentences rather than keyword bags:

```
Adidas Ultraboost 22 — Athletic Footwear, Running Shoes. Lightweight, responsive running shoe with...
```

This design was adopted because embedding models trained on natural language produce better semantic representations for sentence-like input. Keyword bags generate embeddings that cluster around individual terms rather than the holistic meaning of the campaign. Experiment 2 in the retrieval experiments (`03-experiments/retrieval-experiments.md`) established this as foundational. All current index builders (`build_index.py`, `build_index_v4.py`) use the natural-language `build_embed_text` format.

v4 indexing further appends representative category and vertical query phrases from generated query files. This increases query-term overlap by embedding campaigns alongside the vocabulary users actually use, improving retrieval recall for semantically distant but conceptually related queries.

## Performance notes

- In-memory `IndexFlatIP` gives predictable low-latency cosine search on normalized vectors.
- Candidate cap at 1000 bounds downstream filter and rerank cost.
- Retrieval runs in parallel with eligibility and category extraction.
- Startup artifact validation avoids per-request defensive checks on alignment.

## Failure modes and fallbacks

- Missing or inconsistent artifacts block service startup.
- Overly large `FAISS_TOP_K` increases filter/rerank latency.
- Too small `FAISS_TOP_K` can reduce relevance recall and diversity.

## How to change this safely

- Run:
  - `pytest -q tests/test_retrieval_pipeline.py tests/test_api.py`
  - `python3 scripts/test_retrieval.py`
- Rebuild and validate artifacts when schema changes using the current build script.
- Watch:
  - `metadata.n_candidates_retrieved`
  - p95 for `timing_ms.parallel_stages`
  - top-k recall quality on golden queries

## Related docs

- [Campaign Schema](../02-data/campaign-schema.md)
- [Indexing and Embeddings](../02-data/indexing-and-embeddings.md)
- [Retrieval Experiments](../03-experiments/retrieval-experiments.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
