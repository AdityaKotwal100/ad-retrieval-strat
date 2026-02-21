# Indexing and Embeddings

## What text is embedded

Retrieval embedding text is built in `scripts/build_index.py` via `build_embed_text` using:

- taxonomy phrases (category, subcategory, leaf)
- brand name and product line
- key structured attributes
- selected feature/benefit terms
- price-language phrase (e.g., "affordable", "premium")

`build_index_v4.py` additionally appends representative category and vertical query words from:

- `data/generated_queries.json`
- `data/generated_vertical_queries.json`

Purpose: bridge the vocabulary gap between user query language and structured campaign fields.

### Why natural-language embedding text, not keyword bags

Early iterations embedded campaigns as structured keyword bags:

```
running shoes athletic footwear Adidas Ultraboost lightweight responsive
```

The embedding model (`all-MiniLM-L6-v2`) was trained on sentence pairs, not keyword bags. It produces better semantic representations for coherent natural-language input. A sentence-like format:

```
Adidas Ultraboost 22 — Running Shoes, Athletic Footwear. Lightweight, responsive running shoe for marathon training. Affordable performance.
```

produces an embedding closer to the user query "best marathon running shoes" than the keyword bag version. This is because the model's attention mechanism can use syntactic context and phrase boundaries — information that disappears in a bag-of-words format.

This rewrite is considered foundational and is retained in all current index builders. Experiment 2 in the retrieval experiments established this as the baseline format.

### Why append representative query phrases (v3/v4)

Even with natural-language embedding text, campaigns use product-marketing language ("Ultraboost 22 — lightweight performance") while users use intent-driven language ("best running shoe for flat feet"). There is a persistent vocabulary gap even when both are sentences.

The v3/v4 approach appends LLM-generated representative user queries for each category and vertical directly into campaign embedding text:

```
... marathon training shoe | user queries: best shoes for marathon, long-distance running footwear, comfortable running shoes for road races
```

This forces the campaign embedding to be close to actual user queries in embedding space, improving retrieval recall for semantically distant but conceptually related queries. The indexed corpus (`data/index_meta.json`) uses this enriched format.

## Embedding backends

From `app/services/embedding.py` and `app/services/__init__.py`:

- `ONNXBackend` (default via `EMBEDDING_BACKEND=onnx`)
- `SentenceTransformerBackend` (fallback)

Both return L2-normalized vectors. Inner product in FAISS therefore equals cosine similarity.

### Why ONNX as default

The ONNX export of `all-MiniLM-L6-v2` eliminates PyTorch autograd graph overhead and Python-level dispatch. On CPU, this reduces per-query embed time by approximately 20-40% compared to the raw SentenceTransformer path. Since embedding is the single largest latency contributor (8-20ms in budget), any reduction here has proportional p95 impact.

The ONNX runtime (`onnxruntime`) also exposes fine-grained thread control (`OMP_NUM_THREADS`), which prevents thread over-subscription on multi-tenant servers.

**Cost:** Platform-specific ONNX file. Requires regenerating if the base model changes. Thread settings need tuning per deployment environment.

### Why `all-MiniLM-L6-v2` (384d)

Embedding model selection was driven by a direct experiment (`model-test/embedding-model-comparison.md`):

| Model | Avg cosine gap (relevant vs irrelevant) |
|---|---|
| `all-MiniLM-L6-v2` | **0.6099** ← chosen |
| `multi-qa-MiniLM-L6-cos-v1` | 0.6003 |
| `msmarco-MiniLM-L6-cos-v5` | 0.4837 |

The cosine gap measures how well the model separates relevant query-campaign pairs from irrelevant pairs. `all-MiniLM-L6-v2` produced the best gap with comparable latency.

**Why `msmarco` scored worst:** It was fine-tuned for asymmetric retrieval — short queries against long passages. Ad campaigns are short-form text, making the symmetric-pairing assumption of `all-MiniLM-L6-v2` more appropriate.

**Why not larger models (MPNet, E5-large, BGE-large):**
- 768d vectors double FAISS index memory (~33MB → ~66MB) and proportionally increase search time.
- Inference time on CPU: `all-MiniLM-L6-v2` ~8-15ms; `all-mpnet-base-v2` ~20-35ms; E5-large ~40-80ms. These differences are significant against a 100ms budget.
- Marginal quality gains were not evaluated experimentally, but the latency cost alone is disqualifying.

**Why not OpenAI / Cohere API embeddings:**
- Network latency alone (30-100ms) consumes most of the budget.
- Cost per query is non-trivial at scale.
- Creates a hard dependency on external service availability.
- Cannot be preloaded and warmed in-process.

### Why 384 dimensions

384d is the native output size of `all-MiniLM-L6-v2`. No dimensionality reduction (PCA, random projection) was applied because:
1. At 10k campaigns × 384d × 4 bytes = 16.6MB — well within RAM budget.
2. Dimension reduction introduces reconstruction error and requires fitting a projection matrix, adding complexity for no current benefit.
3. PCA would only become relevant at 100k+ campaigns or on memory-constrained hardware.

## Index type and storage

- Retrieval index: `faiss.IndexFlatIP` stored in `data/faiss.index`.
- Category index: separate `faiss.IndexFlatIP` in `data/category_index.faiss` with sidecar `data/category_meta.json`.
- Current retrieval corpus size from `data/index_meta.json`: 10,783 campaigns at 384 dimensions.

### Why separate category index

The category FAISS index contains embeddings of representative user queries per category — fundamentally different content from campaign embeddings. Mixing them into one index would confuse retrieval (campaign embeddings would appear as category matches and vice versa). Separate indexes with separate search paths keep the two use cases clean and independently tunable.

## Update strategy

- Rebuild retrieval index after campaign schema changes, embedding-text changes, taxonomy normalization changes, or model changes.
- Rebuild category index after taxonomy/category changes or representative query generation updates.
- Keep all sidecar arrays aligned by FAISS row id. `CampaignIndex.__init__` validates alignment on every startup.

## Tradeoffs

- `IndexFlatIP` gives exact search quality and simple operations, but memory grows linearly with corpus size.
- Query-style embedding text improves semantic match but can blur strict field-level intent.
- ONNX backend improves latency, but platform-specific ONNX files and threading settings require care.
- Separate category index improves extraction quality and runtime speed, but adds artifact management overhead.

## Known artifact integrity checks

- `CampaignIndex.__init__` validates each artifact length against FAISS row count at startup.
- `scripts/check_faiss_embedding_texts.py` checks index-to-text consistency.
- Taxonomy sync warning emitted if campaign taxonomy categories are missing in `taxonomy.json`.

## Related docs

- [Synthetic Campaign Generation](./synthetic-campaign-generation.md)
- [Retrieval Experiments](../03-experiments/retrieval-experiments.md)
- [03 Category Extraction](../01-pipelines/03-category-extraction.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
