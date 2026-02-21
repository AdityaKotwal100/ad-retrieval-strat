# Configuration

Runtime configuration is resolved in `app/config.py` from environment variables plus YAML defaults (`app/config/eligibility_config.yaml`).

## Core environment variables

### Embedding

- `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`)
- `EMBEDDING_BACKEND` (`onnx` or `sentence_transformer`, default `onnx`)

### Eligibility and safety

- `BLOCKLIST_STRATEGY` (default from YAML: `all`)
- `TOXIC_THRESHOLD` (default from YAML: `0.5`, toxic_bert only)
- `ELIGIBILITY_GATE` (default from YAML: `0.3`)
- `INSENSITIVE_MODEL_PATH` (default `data/insensitive_model_v5.joblib`)
- `LR_MODEL_PATH` (LR blocklist model path, required only for `lr` strategy)

### Retrieval and categories

- `FAISS_TOP_K` (default `1000`)
- `CATEGORY_TOP_K` (default `5`)
- `CATEGORY_THRESHOLD` (default `0.40`)
- `CATEGORY_INDEX_DIR` (default `data/`)
- `DATA_DIR` (default `data/`)

## Config files

- `app/config/eligibility_config.yaml`
  - Policy defaults for blocklist strategy, eligibility gate, and all numeric scoring constants
    (`sensitivity_scale`, `commercial_min/range/scale`, per-cluster hard-gate thresholds,
    per-cluster sensitivity scales, co-activation and distress amplifier settings).
  - All values have in-code fallback defaults; the file is authoritative at runtime.
- `app/config/score_weights.yaml`
  - Component weights and penalties for the ranking pipeline.

## Editable data files

These files live under `data/` and are loaded at startup. Editing them takes effect on the next
container restart without a code release.

- `data/blocklist_builtin.txt` — built-in safety regex patterns (one per line, `#`-comment support).
  Always loaded as the baseline safety floor; `data/blocklist.txt` is additive on top.
- `data/blocklist.txt` — deploy-time blocklist additions (optional, additive with built-ins).
- `data/commercial_exemplars.json` — phrase list used to build the commercial-affinity centroid.
  Edit to shift what the scorer considers "transactional intent".
- `data/sensitivity_clusters.json` — cluster phrase lists for legacy centroid reference (currently
  superseded by the v5 insensitive model but retained for offline analysis).
- `data/attribute_keywords.json` — keyword → `[attribute_key, attribute_value]` mappings used by
  the query parser. Add entries to expand attribute extraction coverage.
- `data/warmup_queries.json` — queries embedded at startup to pre-warm ONNX graph paths.

## Safe ranges and tuning guidance

- `ELIGIBILITY_GATE`
  - safer higher values reduce risky ads but increase false-positive gating.
  - practical start: `0.25` to `0.40`.
- `FAISS_TOP_K`
  - larger values increase recall and latency.
  - default `1000` is validated by tests and current pipeline assumptions.
- `CATEGORY_THRESHOLD`
  - lower values increase category recall and noise.
  - high values risk empty categories.
- scoring weights in `score_weights.yaml`
  - keep near documented range `[-1.0, 1.0]` to prevent FAISS base score from being dominated.

## Change management

- Any change to embedding model/backend or text generation requires index rebuild.
- Any change to eligibility thresholds requires safety regression replay on sensitive test suites.
- Any score weight change should be validated against ranking tests before deploy.

## Related docs

- [Testing](./testing.md)
- [Tradeoffs](../05-design-decisions/tradeoffs.md)
