# Synthetic Campaign Generation

## Scope

This system uses synthetic campaign generation to build retrieval corpora and sidecar targeting artifacts. Primary generation scripts are `scripts/generate_campaigns.py` and `scripts/generate_campaigns_v3.py`.

## Why synthetic data

The problem statement requires at least 10,000 campaigns across diverse verticals. Building a realistic dataset of this scale manually is impractical in 72 hours — it would require product copywriters, category taxonomy designers, and targeting strategy inputs. LLM-assisted generation produces structured, coherent campaign content across many verticals quickly, while deterministic post-processing enforces schema integrity and targeting realism.

The tradeoff: without real advertiser data, ranking weights are tuned for plausibility rather than business outcome optimization. This is acceptable for a prototype; a production system would use real campaign delivery logs.

## Generation pipeline

1. **Base campaign generation** (`scripts/generate_campaigns.py`)
   - Uses `google/gemini-2.0-flash-001` via OpenRouter for structured campaign content.
   - Prompts the model to generate campaigns with coherent taxonomy, brand, targeting, keywords, and pricing.
   - Produces `data/campaigns_v2.json`.
   - Why Gemini Flash: fast (~500ms per batch), cheap, structured JSON output with good schema adherence at this task. Latency acceptable for offline generation.

2. **Query-style enrichment** (`scripts/generate_campaigns_v3.py`)
   - Appends category representative queries from `data/generated_queries.json` into each campaign's `embedding_text`.
   - Produces `data/campaigns_v3.json`.
   - Purpose: close the vocabulary gap between user query language and campaign marketing language.

3. **Taxonomy normalization and index artifact extraction** (current build scripts)
   - Normalizes taxonomy fields, extracts per-campaign keyword sets and targeting arrays.
   - Produces `faiss.index` and all sidecar `.npy`, `.json` artifacts in `data/`.

## Distributions and realism assumptions

Evidence from `scripts/generate_campaigns.py`:

- **20 verticals** with vertical-specific subcategories, brands, and attribute schemas.
- **Weighted demographic sampling** by vertical and category overrides (e.g., sporting goods campaigns skew younger, financial services campaigns skew 35+).
- **Price bucket controls** by vertical and category overrides (e.g., luxury goods cluster in higher price buckets).
- **Geographic and targeting fields** are synthetic but structured to mimic ad-ops constraints — geo include/exclude lists use real US state codes and major city names.

### Why 20 verticals

Enough vertical diversity to test category extraction accuracy and ranking generalization without becoming intractably large for LLM generation. Fewer verticals would produce a corpus too easy for retrieval (low ambiguity); more would require category taxonomy work disproportionate to the prototype scope.

### Why 10,783 campaigns

The problem statement requires at least 10,000. The final corpus is 10,783 after taxonomy normalization and deduplication. This is sufficient to:
- Exercise FAISS retrieval at meaningful scale.
- Produce diverse enough candidate sets that filter and ranking steps are non-trivial.
- Keep index build time reasonable (~minutes, not hours).

At `IndexFlatIP` with 384d, 10,783 campaigns use 16.6MB of memory — trivially small.

## Keyword schema design

Each campaign carries five keyword buckets:

- `brand_terms`: brand and product line names.
- `model_terms`: specific product model or variant names.
- `category_terms`: category and subcategory vocabulary.
- `feature_terms`: product feature descriptors (lightweight, waterproof, wireless).
- `benefit_terms`: user benefit descriptors (save time, reduce pain, increase performance).

This five-bucket structure directly feeds the `KeywordComponent` in ranking, which awards different weights per bucket. Brand terms and model terms get higher weight (stronger purchase intent signal) than generic benefit terms. Separating buckets makes the keyword signal auditable and individually tunable.

## Current dataset evidence

- `data/campaigns_v2.json`: 10,783 rows.
- `data/campaigns_v3.json`: 10,783 rows (+ query enrichment in embedding_text).
- `data/campaigns_v4.json`: 10,783 rows (+ additional vertical query enrichment).
- `data/index_meta.json`: `n_campaigns=10783`, `embedding_dim=384`.

## Known limitations

- LLM-generated campaign text can contain style artifacts and repeated phrasing patterns.
- No evidence of real advertiser spend curves, conversion outcomes, or click-through rates.
- Distribution realism is hand-tuned, not calibrated against production ad logs.
- Synthetic data may encode unintended lexical shortcuts that inflate offline retrieval quality metrics (the model may have learned patterns specific to the generation prompt rather than genuine user intent).

Without real traffic labels and campaign delivery logs, ranking weights are tuned for retrieval plausibility, not business outcome optimality.

## Operational notes

- Generation requires `OPENROUTER_API_KEY`.
- Reproducibility is partial because LLM outputs can vary across API calls.
- Index rebuild is required after any schema or embedding text format change.
- Run `python3 scripts/check_faiss_embedding_texts.py` after any rebuild to verify index-to-text alignment.

## Related docs

- [Campaign Schema](./campaign-schema.md)
- [Indexing and Embeddings](./indexing-and-embeddings.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
