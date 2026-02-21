# 02 Ad Eligibility

## Purpose

Score query safety and ad appropriateness, then gate unsafe or highly insensitive queries before retrieval results are returned.

## Inputs and outputs

### Inputs

- Raw query text.
- Unit-normalized query embedding.

### Outputs

- Eligibility score and metadata from `EligibilityScorer.score_with_metadata`.
- Gating decision in `app/main.py`: `eligibility_score < ELIGIBILITY_GATE`.

Example metadata fields (hard block via model):

```json
{
  "eligibility": 0.0,
  "blocklist_triggered": false,
  "hard_block_cluster": "insensitive_model_v5",
  "sensitivity_penalty": 1.0,
  "commercial_boost": 0.0,
  "top_cluster": "insensitive_model_v5",
  "top_cluster_sim": 0.97,
  "insensitive_model_probability": 0.97,
  "insensitive_model_threshold": 0.5,
  "cluster_penalty": 0.63
}
```

Example metadata fields (hard block via cluster centroid):

```json
{
  "eligibility": 0.0,
  "blocklist_triggered": false,
  "hard_block_cluster": "self_harm_crisis",
  "sensitivity_penalty": 1.0,
  "commercial_boost": 0.0,
  "top_cluster": "self_harm_crisis",
  "top_cluster_sim": 0.68,
  "insensitive_model_probability": 0.42,
  "insensitive_model_threshold": 0.5,
  "cluster_penalty": 1.0
}
```

Example metadata fields (pass-through commercial query):

```json
{
  "eligibility": 0.94,
  "blocklist_triggered": false,
  "hard_block_cluster": null,
  "sensitivity_penalty": 0.04,
  "commercial_boost": 0.99,
  "top_cluster": "insensitive_model_v5",
  "top_cluster_sim": 0.04,
  "commercial_sim": 0.38,
  "insensitive_model_probability": 0.04,
  "insensitive_model_threshold": 0.5,
  "cluster_penalty": 0.0
}
```

## Key files, classes, and functions

- `app/services/eligibility.py`
  - `EligibilityScorer`
  - `EligibilityScorer.blocklist_match`
  - `EligibilityScorer.score_with_metadata`
  - `EligibilityScorer._score_internal`
  - `InsensitiveQueryModel`
  - `_build_sub_centroids` — k-means sub-centroid construction
- `app/config.py`
  - `ELIGIBILITY_GATE`, `BLOCKLIST_STRATEGY`, `INSENSITIVE_MODEL_PATH`
- `app/config/eligibility_config.yaml`
  - blocklist and gate defaults; `scoring.cluster_hard_thresholds`, `scoring.cluster_sensitivity_scale`.
- `data/sensitivity_clusters.json`
  - phrase lists for each sensitivity cluster (loaded at startup, embedded into sub-centroids).
- `app/main.py`
  - hard blocklist short-circuit and eligibility gate application.

## Algorithm details and edge cases

### Layer 1: Blocklist strategy (microseconds)

Runs before embedding. `EligibilityScorer.blocklist_match` delegates to the configured `BlocklistStrategy`, which matches the raw query against a list of regex patterns, exact phrases, or ML-classified terms from `data/blocklist.txt`.

Two outcomes:
- Score `== 0.0` (hard block): `app/main.py` returns immediately. No embedding, no model inference, no FAISS search. Total latency is <1ms.
- Score `> 0.0` and `< 1.0` (soft cap): request continues. The cap is stored as `fuzzy_cap` and applied as an upper bound on final eligibility after semantic scoring.

### Layer 2a: v5 insensitive-query model (sub-millisecond)

`InsensitiveQueryModel.predict_insensitive_probability` runs a TF-IDF + logistic regression artifact (joblib) against the query text. The model was trained specifically on the ad-appropriateness task — distinguishing crisis/sensitive language from commercial language — not generic toxicity.

- If `probability >= threshold` (loaded from the model artifact): model hard-blocks; `hard_block_cluster` set to `"insensitive_model_v5"`.
- Otherwise: probability becomes one input to the combined continuous `sensitivity_penalty`.

Query text is lightly normalized before the model (strip intent scaffolding, filter known v5 false-positive terms).

### Layer 2b: Sensitivity cluster centroids (sub-millisecond)

At startup `EligibilityScorer.__init__` embeds all phrases in each of the eight sensitivity clusters from `data/sensitivity_clusters.json` and reduces them to **k=3 sub-centroids** per cluster via pure-numpy k-means (`_build_sub_centroids`). Using three sub-centroids rather than a single mean captures semantically diverse sub-groups within a cluster (e.g. `grief_loss` spans death, divorce, and pregnancy loss — a single mean misses the fringes).

For each query, `_score_internal` computes `max(query_embedding @ sub_centroid)` across all sub-centroids for every cluster. Two outcomes per cluster:

- **Hard gate**: if `cluster_sim >= _CLUSTER_HARD_THRESHOLDS[cluster_name]`, the cluster fires and `hard_block_cluster` is set.
- **Continuous penalty**: if `cluster_sim >= SENSITIVITY_MIN_SIM`, `cluster_penalty = max(cluster_sim * scale)` accumulates.

Per-cluster thresholds and sensitivity scales (from `eligibility_config.yaml`):

| Cluster | Hard gate | Sensitivity scale |
|---|---|---|
| `self_harm_crisis` | 0.60 | 1.5× |
| `medical_emergency` | 0.60 | 1.5× |
| `weapons_violence` | 0.62 | 1.5× |
| `immediate_danger` | 0.55 | 1.5× |
| `abuse_trauma` | 0.62 | 1.4× |
| `addiction_recovery` | 0.65 | 1.4× |
| `grief_loss` | 0.90 | 1.3× |
| `financial_hardship` | 0.95 | 1.1× |

### Combining Layer 2a and Layer 2b (OR logic)

Both signals are consulted independently, then combined conservatively:

- **Hard block**: if *either* the model probability ≥ threshold *or* any cluster sim ≥ its hard gate, `eligibility = 0.0`. `hard_block_cluster` names the triggering source.
- **Continuous penalty**: `sensitivity_penalty = max(insensitive_prob, cluster_penalty)`. Neither signal is discarded — the more conservative reading wins.

This OR combination means the cluster centroids can catch queries the model under-scores (e.g., novel crisis phrasing outside model training distribution), and the model can catch queries that fall below all centroid hard gates but still have high insensitivity probability.

### Layer 3: Commercial affinity boost

`commercial_boost` is computed from cosine similarity between the query embedding and a precomputed centroid of commercial exemplar phrases (e.g., "buy", "shop", "best deal", "discount"). This centroid is computed once at startup.

- If `sensitivity_penalty > 0.70`: `commercial_boost = 1.0` (neutral). High-sensitivity queries must not have commercial signals override the penalty.
- Otherwise: `commercial_boost = COMMERCIAL_MIN + COMMERCIAL_RANGE * clamp(commercial_sim * COMMERCIAL_SCALE, 0, 1)`. Purely commercial queries receive a boost that increases final eligibility; purely informational queries receive a smaller boost.

### Final score

```
sensitivity_penalty = max(insensitive_prob, cluster_penalty)
eligibility = clamp(min((1 - sensitivity_penalty) * commercial_boost, fuzzy_cap), 0.0, 1.0)
```

- `sensitivity_penalty` is the conservative OR of the model probability and the cluster centroid penalty.
- `(1 - sensitivity_penalty)` inverts the combined insensitiveness signal into an appropriateness score.
- Multiplying by `commercial_boost` amplifies scores for clearly transactional queries and dampens scores for informational queries.
- `fuzzy_cap` enforces a soft-cap upper bound from blocklist soft matches.
- Final `np.clip` ensures the output is always in `[0,1]`.

### Gating at the endpoint

`app/main.py` checks `eligibility_score < ELIGIBILITY_GATE` (default `0.3`). Gated queries return `campaigns=[]` and suppressed categories.

Edge cases:

- Model inference exception: code sets probability to `1.0` as conservative fail-safe. Never allows ads through on model failure.
- Blocklisted but nonzero soft-cap cases stay eligible for downstream scoring, bounded by cap.

## Design rationale

### Why not LLM-based classification

Any LLM API call (OpenAI, Anthropic, OpenRouter) adds 100-500ms per query. The entire latency budget is 100ms. A local quantized LLM (e.g., Llama 3B on CPU) costs 50-150ms — still fatal. Beyond latency, LLM classification introduces non-determinism (the same query may score differently across calls), which makes policy auditing and threshold setting unreliable. The problem requires reproducible safety behavior.

### Why not `unitary/toxic-bert` or similar BERT toxicity classifier

BERT-based inference on CPU costs 50-100ms per call — enough to consume the entire p95 budget before FAISS even runs. Additionally, models trained on social media toxicity (hate speech, profanity) are poorly calibrated for the ad-appropriateness task. Grief expressions, financial hardship queries, and medical emergencies are not toxic in the social-media sense, but should receive zero or low ad eligibility. A purpose-trained lightweight model is more accurate and three orders of magnitude faster.

### Why the layered approach

Each layer catches a different failure mode:

1. **Blocklist** catches known literal patterns (bomb-making instructions, self-harm prompts) deterministically, with zero compute cost.
2. **ML model (Layer 2a)** catches semantic paraphrases and novel phrasing the blocklist misses, via statistical generalization from labeled training examples.
3. **Cluster centroids (Layer 2b)** provide an explicit semantic safety net for the eight known sensitive topic areas. They catch queries that fall just below the model's threshold but are geometrically close to known crisis language — particularly useful for novel phrasing outside the model's training distribution. Using three sub-centroids per cluster covers the within-cluster diversity that a single mean centroid misses.
4. **Commercial boost** provides continuous calibration for borderline queries — the problem statement explicitly asks for scores in the 0.4–0.7 range for ambiguous cases, not just binary gating.

No single mechanism covers all four failure modes at acceptable latency. The OR combination of 2a and 2b is conservative by design: a query only needs to trigger one signal to be blocked, so the combined system has higher recall on sensitive content than either signal alone.

### Why the v5 joblib model (not v10)

Six iterations of model training were run (`generate-ads-or-not/models/toxicity/v1` through `v10`). v10 achieved `micro ROC AUC 0.9716` on the offline test set. v5 was selected for production because it had verified behavior compatibility with the saved test suite, and the v10 → v5 promotion rationale at deployment time favored stability. The `threshold` field is loaded from the model artifact itself, so threshold changes can be deployed by replacing only the model file.

### Experiment history

- **Experiment 1 (bipartite k-means):** Best MAE 0.195, Spearman 0.779. Hard-block 100%. Only 11/20 test queries within ±0.20 of expected score. Rejected as insufficient for safety-critical gating.
- **Experiment 2 (3-tier corpus):** MAE improved to 0.182. Within ±0.20 improved to 13/20. But regressions appeared on pure commercial queries because the medium-tier phrases diluted commercial signal. Partially kept as evidence that medium-tier examples matter.
- **Experiment 3 (two-pass anchored centroid):** Best two-pass: MAE 0.146, but hard-block accuracy dropped to 78%. Unacceptable — a safety gate that misses 22% of hard-block cases fails the primary requirement.
- **Experiment 4 (Option E layered design):** MAE 0.087, Spearman 0.905, hard-block 100%, 19/20 within ±0.20. One persistent miss: "why do runners get blisters" (scored 0.55 vs expected 0.80) — a known false positive. This experiment directly informed the current `EligibilityScorer` structure.
- **Experiment 5 (production test-suite):** MAE 0.134, Spearman 0.700, hard-block 100%, 27/36 within ±0.20. Errors concentrated on high-eligibility commercial queries with incidental sensitivity overlap.
- **Experiment 6 (toxicity model iterations v1-v10):** v10 objective 1.135, micro ROC AUC 0.9716, sub-millisecond p95 inference confirmed. Runtime uses v5 for stability.

## Performance notes

- Blocklist match is microsecond-level and can short-circuit the request.
- Insensitive model uses lightweight artifact inference (joblib model), not transformer per request.
- Commercial centroid is precomputed once at startup.
- This entire stage runs in parallel with category extraction and FAISS search via `asyncio.gather`.


## How to change this safely

- Run:
  - `pytest -q tests/test_eligibility.py tests/test_api_integration.py`
  - `python3 scripts/test_eligibility.py`
- Replay test suites:
  - `python3 scripts/run_test_queries.py`
- Watch:
  - `metadata.gated` rate
  - sensitive-query leakage (nonzero campaigns on crisis prompts)
  - false-positive gating on commercial queries

## Related docs

- [Safety and Sensitivity](../05-design-decisions/safety-and-sensitivity.md)
- [Ad Eligibility Experiments](../03-experiments/ad-eligibility-experiments.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
