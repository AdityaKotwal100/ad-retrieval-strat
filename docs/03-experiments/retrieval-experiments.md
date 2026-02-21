# Retrieval Experiments

## Experiment 1: Embedding model comparison

- Hypothesis
- Asymmetric retrieval-focused MiniLM variants may outperform current embedding model for query-to-campaign matching.

- Change made
- Compared `all-MiniLM-L6-v2`, `msmarco-MiniLM-L6-cos-v5`, and `multi-qa-MiniLM-L6-cos-v1` in `model-test/embedding-model-comparison.md`.

- Evaluation method
- Measured relevant/irrelevant cosine gaps on curated query-document pairs.
- Measured CPU latency percentiles across batch sizes.

- Results
- Average gap:
  - `all-MiniLM-L6-v2`: 0.6099
  - `multi-qa`: 0.6003
  - `msmarco`: 0.4837
- Latency differences were small.

- Decision and rationale
- Kept `all-MiniLM-L6-v2` as default.
- Re-indexing cost not justified by small or negative quality deltas.

## Experiment 2: Embedding text rewrite to natural query-like format

- Hypothesis
- Campaign embedding text written as natural language should improve retrieval vs keyword bags.

- Change made
- `build_embed_text` in `scripts/build_index.py` constructs sentence-like text from structured campaign fields.

- Evaluation method
- Not found in repo: formal benchmark artifact comparing old vs new embedding text.
- Evidence exists in code comments and follow-on retrieval quality scripts.

- Results
- Inference from code and downstream behavior: this rewrite is considered foundational and retained in all current index builders.

- Decision and rationale
- Kept natural-language embedding text generation as standard path.

## Experiment 3: Query enrichment for campaign text (v3/v4)

- Hypothesis
- Appending generated category and vertical query phrases to campaign embedding text increases query-term overlap and retrieval recall.

- Change made
- `scripts/generate_campaigns_v3.py` appends category queries.
- `scripts/build_index_v4.py` supports category and vertical query word append.

- Evaluation method
- Not found in repo: explicit A/B recall metrics for v2 vs v3/v4 index text.
- Indirect evidence from test query outputs and retention of v3/v4 pipeline files.

- Results
- Current indexed corpus uses v4 metadata and has 10,783 campaigns (`data/index_meta.json`).

- Decision and rationale
- Kept enrichment path; explicit quantitative gain documentation is missing.

## Experiment 4: Lightweight vector DB prototype

- Hypothesis
- A simple FAISS + hashing vector prototype can validate minimal vector retrieval behavior before full production pipeline complexity.

- Change made
- Implemented in `experiments/vector-db-test/vector_db.py` and `main.py`.

- Evaluation method
- Interactive/manual search harness.

- Results
- Demonstrates functional FAISS retrieval with normalized vectors and top-k ranking.
- Not used in production runtime.

- Decision and rationale
- Rejected as production architecture; retained as sandbox reference.


## Proposed replacement experiments

1. Build a labeled retrieval benchmark with at least 200 queries and graded relevance labels.
2. Track Recall@50, Recall@1000, MRR, and nDCG for each index build variant.
3. Run model + embedding text + enrichment ablations on same benchmark.
4. Persist results and chosen index metadata checksum together.
