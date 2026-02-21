# Safety and Sensitivity

## Policy logic in code

Safety control is layered in `app/services/eligibility.py` and enforced in `app/main.py`:

1. Hard blocklist strategy match (regex/ML, microseconds).
2a. Insensitive-query model hard threshold gate (sub-millisecond joblib inference).
2b. Sensitivity cluster centroid matching — k=3 sub-centroids per cluster, eight topic areas, OR-combined with 2a.
3. Continuous sensitivity penalty = `max(model_prob, cluster_penalty)` and commercial boost interaction (produces 0.0–1.0 score).
4. Endpoint-level eligibility gate (`ELIGIBILITY_GATE`) before ranking.

## Why five layers instead of one

No single mechanism covers all failure modes:

| Mechanism | Catches | Misses |
|---|---|---|
| Blocklist only | Exact and near-exact known patterns | Novel phrasings, paraphrases, multilingual variants |
| ML model only | Semantic sensitivity patterns | New vocabulary outside training distribution; near-threshold cases |
| Cluster centroids only | Broad concept coverage for known topic areas | Novel sensitive topics outside the eight defined clusters |
| Threshold gate only | Clearly low-eligibility queries | Borderline queries that need calibrated scores (0.4–0.7) |
| Single k-means score | Broad concept coverage | Calibration — MAE 0.195 in Experiment 1, unacceptable for safety |

The layered design uses each mechanism where it is strongest. Critically, every layer is fast — total eligibility scoring cost is <5ms even with all layers active.

The model (2a) and cluster centroids (2b) run in parallel and are OR-combined: a query only needs one signal to fire for a hard block. This conservatism reflects the "missed ad opportunity is better than insensitive one" policy directive — higher sensitivity recall is preferred over avoiding false positives.

## Blocklists and thresholds

- Blocklist strategy defaults from `app/config/eligibility_config.yaml`.
- Insensitive model threshold loaded from model artifact by `InsensitiveQueryModel`.
- Endpoint gate defaults to `0.3` (`app/config.py`).

### Why 0.3 as the eligibility gate threshold

The gate at `ELIGIBILITY_GATE=0.3` means any query scoring below 30% appropriateness triggers ad suppression. This was calibrated against the saved test-suite outputs:

- Hard-crisis queries (self-harm, bomb-making, grief) score 0.0 — well below the gate.
- Ambiguous queries (financial hardship, medical stress) score 0.2–0.45 — some fall below, some above. The 0.3 threshold catches the clearly sensitive ones while letting marginally informational queries through.
- Commercial queries score 0.7–1.0 — well above the gate.

A lower threshold (0.1) would let more borderline queries through; a higher threshold (0.5) would over-suppress informational queries. 0.3 is the best available balance given the current test set. This threshold requires recalibration if the v5 model or its training data changes.

## Sensitive query handling examples

Based on saved outputs in `data/test_results/*/output.json`:

- Hard-blocked examples (blocklist layer, score = 0.0):
  - self-harm ideation
  - bomb-making queries
  - poisoning queries
- Semantic hard-gated examples (model layer, score = 0.0):
  - grief and bereavement
  - medical emergency and child crisis
  - severe financial hardship

Based on saved outputs, hard-block and semantic-gate coverage is consistent across self-harm, crisis, grief, and harmful-intent queries.

## Err-on-caution rationale

The problem statement is explicit: "a missed ad opportunity is better than an insensitive one." This is encoded in three places:

1. **Fail-closed on model errors:** If the joblib model raises any exception, `insensitive_prob` is set to `1.0`, which triggers a hard semantic block. The alternative (treating model errors as benign and serving ads) risks unsafe serving on corrupted inputs or model failures.
2. **Category suppression for gated queries:** Even if categories were extracted before gating, they are cleared in `app/main.py` before the response is assembled. Returning "baby food" or "life insurance" categories for a bereavement query is insensitive even without returning campaigns.
3. **No soft serving for high-penalty queries:** When `sensitivity_penalty > 0.70`, commercial boost is neutralized (`commercial_boost = 1.0` neutral). Commercial affinity cannot rescue a genuinely sensitive query.

## Safety risks still present

- **Borderline language overlap:** Queries like "running shoes for my late father's marathon" touch grief vocabulary but are commercial. The model may flag the grief signal and over-gate. Known false-positive category.
- **Runtime/documented behavior drift:** Threshold changes or model updates without full test-suite replay can shift safety behavior without detection.
- **Blocklist-only misses:** The regex blocklist cannot catch novel paraphrases. Without the ML model layer, users could bypass safety controls with minor rewording.
- **Borderline language overlap:** Queries that touch sensitive vocabulary in a clearly commercial context (e.g., fitness queries mentioning grief or loss) may be over-gated due to sensitivity signal overlap.

## Experiment evidence

- **k-means Experiment 1:** Single-centroid model had 100% hard-block accuracy but MAE 0.195 overall. Insufficient calibration for the 0.4–0.7 range.
- **Two-pass Experiment 3:** Orthogonal two-pass model achieved MAE 0.146 but hard-block fell to 78%. Hard-block misses are categorically unacceptable — this was the decisive rejection criterion.
- **Option E Experiment 4:** Layered approach achieved hard-block 100%, MAE 0.087, Spearman 0.905, 19/20 within ±0.20. This experiment directly determined the production architecture.
- **Model iteration Experiment 6 (v1–v10):** v10 achieved micro ROC AUC 0.9716 with sub-millisecond inference. v5 deployed in production for stability; v10 upgrade is a pending improvement.

## How to operate safely

- Re-run full sensitive query suite before threshold/model updates.
- Treat any increase in ungated sensitive cases as release blocker.
- Track both leakage (sensitive queries receiving campaigns) and overblocking (commercial queries being gated) with explicit reviewed datasets.
- For any model update: run the saved test suite, compare gated rates, confirm hard-block cases remain gated.

## Related docs

- [Ad Eligibility Pipeline](../01-pipelines/02-ad-eligibility.md)
- [Ad Eligibility Experiments](../03-experiments/ad-eligibility-experiments.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
