# 03 Category Extraction

## Purpose

Extract semantic query categories to support downstream ranking and metadata explainability.

## Inputs and outputs

### Inputs

- Query embedding.
- Optional context (`interests`, `gender`).
- Configurable `top_k` and threshold.

### Output

List of category objects sorted by score:

```json
[
  {"category": "running shoes", "score": 0.802},
  {"category": "shoes", "score": 0.7693}
]
```

`app/main.py` exposes category names at top-level response and full scored categories in metadata.

## Key files, classes, and functions

- `app/services/categories.py`
  - `CategoryExtractor.__init__`
  - `CategoryExtractor.extract`
  - `CategoryExtractor._extract_faiss`
  - `CategoryExtractor._extract_matrix` (legacy fallback)
- `app/config.py`
  - `CATEGORY_TOP_K`, `CATEGORY_THRESHOLD`, `CATEGORY_INDEX_DIR`
- Artifacts
  - `data/category_index.faiss` and `data/category_meta.json`
  - `data/taxonomy.json`
  - `data/taxonomy_overrides.json`

## Algorithm details

### FAISS query-index mode (preferred)

1. **Search candidate queries:** `_FAISS_SEARCH_K = 50` nearest neighbors from the category query index. Each neighbor is a stored representative user query with a category label.
2. **Aggregate by category:** For each unique category among the 50 neighbors, take the max similarity across all its representative queries. This gives a per-category score.
3. **Context boosts:** If the user provided `interests`, boost categories whose names overlap with stated interests. If `gender` is present, apply suppression for tokens strongly associated with the opposite gender.
4. **Coherence-ratio filter:** Compute `score / top_score` for each candidate category. Categories below the coherence ratio floor are dropped — this prevents low-confidence fringe categories from appearing alongside a strong primary match.
5. **Threshold filter:** Remove any category with score below `CATEGORY_THRESHOLD` (default 0.40).
6. **Fallback to top category:** If no category passes threshold but the top score exceeds a minimum floor, return just the top category.
7. **Top-k limit:** Return at most `CATEGORY_TOP_K` (default 5) categories, sorted by score descending.

### Description-matrix fallback mode

Used automatically when FAISS index artifacts are unavailable. Embeds taxonomy description strings (e.g., "Athletic Footwear: shoes and footwear for sports") at startup, then computes dot product between query embedding and each description vector. Applies the same threshold and top-k logic. Slower to set up (requires runtime description embedding if not precomputed) and less accurate due to vocabulary mismatch.

### Edge cases

- Missing category index artifacts falls back automatically to matrix mode.
- If no category passes threshold and top score < fallback floor, returns empty list.
- Gated queries have categories cleared in `app/main.py` even if extracted — avoids insensitive category echoing for crisis queries.

## Design rationale

### Why FAISS query-index over description-matrix matching

The core vocabulary problem: taxonomy descriptions are written in formal product-category language. Users do not query in this language.

| User query | Taxonomy description |
|---|---|
| "best marathon running shoes" | "Athletic Footwear: shoes designed for sports" |
| "cheap laptop for college" | "Consumer Electronics: personal computing devices" |
| "protein powder for muscle gain" | "Sports Nutrition: dietary supplements" |

When a user says "best marathon running shoes," the embedding of their query is close to other marathon-shoe queries in embedding space — but may not be as close to the formal description "Athletic Footwear: shoes designed for sports." The query-index contains hundreds of representative user phrasings per category, so the user's natural query finds close neighbors and gets correct category labels even when the formal description is distant.

The description-matrix approach performed noticeably worse in offline evaluation due to this vocabulary gap. It is retained only as an automatic fallback for robustness.

### Why not LLM-based category extraction

Any LLM API call (GPT-4o-mini, Gemini Flash) adds 100-500ms. Categories are used for ranking boosts, not for safety gating — a missed or wrong category degrades ranking slightly but does not cause a harmful outcome. The role of category extraction does not justify LLM latency cost. A FAISS search over 50 neighbors takes ~0.5ms.

### Why `_FAISS_SEARCH_K = 50` neighbors

50 is enough to cover multiple representative queries per category while staying fast. The aggregation step takes max per category, so more neighbors improve robustness to query phrasing variation but with diminishing returns. At 50 neighbors, category resolution is stable for queries within the taxonomy vocabulary. Fewer neighbors (e.g., 10) miss rare phrasings. More neighbors (e.g., 200) add marginal cost with no quality gain at this taxonomy scale.

### Why the coherence-ratio filter

Without it, a query like "best running shoes" might return:

```
running shoes: 0.82
athletic footwear: 0.74
children's clothing: 0.41
```

The last category crossed the 0.40 threshold but is clearly irrelevant — its score is only 50% of the top match. The coherence ratio (score / top_score) drops it because it falls below the relative confidence floor. This improves precision for downstream ranking without losing recall on the primary categories.

### Why gender suppression is token-based

Gender suppression removes categories with tokens strongly associated with the opposite gender (e.g., "men's clothing" for a female user). Token-based heuristics are used because the user's gender string is free-form and building an embedding-space gender classifier adds unnecessary complexity for a boost/suppression signal. Known limitation: ambiguous category names can be mis-handled.

### Context boost design

User `interests` from context boost categories that share vocabulary with stated interests. This is additive to the cosine similarity score, capped to avoid completely overriding semantic relevance. The intent is to make extraction query-aware when context is available, not to force categories based on demographics alone.

### Experiment history

- **Embedding model comparison** confirmed `all-MiniLM-L6-v2` as the best model for query-to-category similarity (see retrieval experiments). The same model is used for both campaign retrieval and category extraction, so a single ONNX artifact serves both use cases.
- **Query-index design** was chosen over description-matrix after observing that taxonomy descriptions use formal language that mismatches user query vocabulary. The query-index was built by generating hundreds of representative user queries per category via LLM (`model-test/category/generate_data.py`), embedding them, and indexing them with their category labels.

## Performance notes

- FAISS mode avoids runtime taxonomy embedding and is low latency (~0.5-3ms).
- Category extraction runs in parallel with eligibility and retrieval.
- Query-index vocabulary match reduces mismatch between user language and taxonomy descriptions.

## Failure modes and fallbacks

- FAISS category index load failure prints warning and falls back to matrix mode.
- Poor threshold tuning can over-return noisy categories or under-return useful ones.
- Gender suppression heuristic is token-based and can mis-handle ambiguous naming.

## How to change this safely

- Run:
  - `pytest -q tests/test_retrieval_pipeline.py`
  - `python3 scripts/test_categories.py --demo`
- Validate artifacts:
  - `python3 scripts/check_faiss_embedding_texts.py`
- Watch:
  - `metadata.n_categories`
  - category confidence distribution
  - ranking lift from `CategoryAlignmentComponent`

## Related docs

- [Indexing and Embeddings](../02-data/indexing-and-embeddings.md)
- [Category Extraction Experiments](../03-experiments/category-extraction-experiments.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
