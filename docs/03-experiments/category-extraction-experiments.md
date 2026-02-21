# Category Extraction Experiments

## Experiment 1: Description embeddings vs query-index matching

- Hypothesis
- Category extraction improves when matching user query embeddings against representative user queries rather than taxonomy descriptions.

- Change made
- Added query-index build workflow in `model-test/category/build_index.py`.
- Runtime `CategoryExtractor` now prefers FAISS query-index mode when files exist (`app/services/categories.py`).

- Evaluation method
- Code-level comparison and runtime tests in `tests/test_retrieval_pipeline.py`.
- Script-level smoke checks in `scripts/test_categories.py`.

- Results
- Code comments in `app/services/categories.py` state expected score separation improvements (query-index similarity often higher than description mode).
- Runtime artifacts exist: `category_meta.json` has 975 rows, 133 unique categories across 20 verticals.

- Decision and rationale
- Kept FAISS query-index as preferred mode.
- Description matrix retained as fallback when index artifacts are missing.

## Experiment 2: Context-boost and coherence-gate heuristics

- Hypothesis
- Small context boosts and coherence filtering increase relevance and reduce noisy secondary categories.

- Change made
- Added interest boost, gender suppression, thresholding, fallback floor, and coherence ratios in `CategoryExtractor`.

- Evaluation method
- Unit tests: `tests/test_retrieval_pipeline.py` verifies context can alter ordering and output shape constraints.

- Results
- Deterministic test confirms context changes top category ordering under controlled setup.
- Confidence-weighted and ratio-based pruning logic reduces long noisy category tails.

- Decision and rationale
- Kept heuristics with conservative thresholds and fallback floor.

## Experiment 3: Category query generation quality filtering

- Hypothesis
- Restricting generated representative queries to recommendation-style intent patterns yields cleaner category index content.

- Change made
- Added intent-pattern filters in `model-test/category/build_index.py` (`RECOMMENDATION_PATTERNS`, `is_recommendation_query`).

- Evaluation method
- Manual review workflow implied by `generated_queries.json` output and comments.

- Results
- Not found in repo: no explicit numeric quality report comparing filtered vs unfiltered category retrieval outcomes.

- Decision and rationale
- Kept filtering logic in build pipeline.

## Evidence gaps

- Not found in repo: a formal before/after metric table for category precision/recall on a labeled category-query benchmark.
- Not found in repo: ablation report quantifying each heuristic's impact (interest boost, coherence, fallback floor).

## Proposed replacement experiments

1. Build labeled benchmark: at least 1,000 query-category pairs with primary and secondary labels.
2. Compare description-matrix vs FAISS query-index on top-1 and top-3 accuracy.
3. Run ablations for context boost, gender suppression, coherence ratio, threshold, and fallback floor.
4. Save results in `experiments/category-extraction/<date>/report.json` and document in markdown.
