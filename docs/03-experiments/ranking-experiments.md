# Ranking Experiments

## Experiment 1: Componentized scoring pipeline and coverage-ratio scaling

- Hypothesis
- Explicit component deltas plus coverage-ratio scaling should improve ranking stability when queries specify different amounts of constraints.

- Change made
- Implemented `ContextReranker` and component framework in:
  - `app/services/ranking.py`
  - `app/services/scoring/builder.py`
  - `app/services/scoring/components.py`

- Evaluation method
- Unit tests in `tests/test_ranking.py` and `tests/test_scoring_components.py`.
- In-process sweep script `scripts/fine_tuning_runs.py` for full pipeline behavior.

- Results
- Tests validate:
  - deterministic ordering behavior
  - coverage-ratio worked example math
  - trimming to top 1000
  - resilience to missing optional metadata

- Decision and rationale
- Kept componentized architecture for interpretability and tunable behavior.

## Experiment 2: Output score normalization exploration

- Hypothesis
- Sigmoid-normalized display scores might produce a cleaner user-facing relevance scale than raw final scores.

- Change made
- Prototype script in `experiments/relevance-scoring/test.py` computes z-score then sigmoid.
- Inputs from `experiments/relevance-scoring/response.json`, outputs to `normalized_scores.json`.

- Evaluation method
- Statistical transformation over 512-campaign response sample.

- Results
- Sample stats from artifact:
  - mean final score: 0.4478
  - std: 0.4490
  - n=512
- Normalized `display_score` values generated and sorted.

- Decision and rationale
- Not adopted into runtime response contract.
- Current runtime exposes raw `relevance_score` and clamped `internal_score`.

## Experiment 3: Weight tuning loop for score config

- Hypothesis
- Iterative score-weight tuning using test query suite and score breakdowns can improve relevance quality without large architectural changes.

- Change made
- Added `scripts/fine_tuning_runs.py` for in-process query replay and low-score diagnosis.
- Weights centralized in `app/config/score_weights.yaml`.

- Evaluation method
- Replay `data/test_queries/*.json`, inspect top scores and top component deltas.

- Results
- Not found in repo: versioned experiment reports with before/after relevance metrics per weight change.

- Decision and rationale
- Kept tooling and config-driven weight model.

## Evidence gaps

- Not found in repo: a canonical ranked relevance benchmark with human labels and historical run logs.
- Not found in repo: tracked quality delta per individual weight edit.

## Proposed replacement experiments

1. Build a judged ranking dataset (query + top-50 candidates with graded labels).
2. Report nDCG@10, Precision@10, and calibration of `internal_score`.
3. Add automated regression checks for score-distribution drift.
4. Persist each weight file hash with evaluation results.
