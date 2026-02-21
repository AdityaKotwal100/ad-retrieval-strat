# Ad Retrieval System

Semantic ad retrieval service with layered safety gating, query parsing, and a composable ranking pipeline. Single endpoint: `POST /api/retrieve`.

## Setup

**Prerequisites:** Python 3.11, `faiss-cpu`-compatible environment (Linux/macOS).

```bash
# Create virtualenv and install all dependencies
make install
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

During development, add `--reload` to auto-restart on file changes:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Startup takes 5–15 s (FAISS index load + embedding model warmup). Verify with:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Expected response:

```json
{
  "status": "ok",
  "campaigns_indexed": 1000,
  "embedding_model": "all-MiniLM-L6-v2",
  "version": "2.0.0"
}
```

## Make a request

```bash
curl -s -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "best running shoes for marathon training"}' \
  | python3 -m json.tool
```

With user context:

```bash
curl -s -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "affordable laptops for college students",
    "context": {"age": 21, "location": "US-CA", "device": "mobile"}
  }' | python3 -m json.tool
```

## Run tests

```bash
pytest -q tests/
```

Smoke scripts for individual components:

```bash
python3 scripts/test_embedding.py
python3 scripts/test_eligibility.py
python3 scripts/test_categories.py --demo
python3 scripts/test_retrieval.py
```

## Project structure

```
app/
  main.py                  # FastAPI app, startup, single endpoint
  config.py                # Environment variable resolution
  models.py                # Pydantic request/response schemas
  config/
    eligibility_config.yaml  # Blocklist, gate, and scoring constants
    score_weights.yaml       # Ranking component weights
  services/
    eligibility.py         # Safety gating — blocklist + model + cluster centroids
    categories.py          # Category extraction via FAISS
    retrieval.py           # Campaign FAISS index search
    ranking.py             # Scoring pipeline
    campaign_eligibility.py  # Hard campaign-level filters
    query_parser.py        # Lexical query parsing
    scoring/               # Modular ranking components
data/
  faiss.index              # Campaign embedding index
  campaigns_meta.json      # Campaign metadata rows
  insensitive_model_v5.joblib  # Sensitivity classifier
  sensitivity_clusters.json    # Cluster phrase lists (8 topic areas)
  commercial_exemplars.json    # Commercial intent phrase list
  blocklist_builtin.txt        # Built-in safety regex patterns
  blocklist.txt                # Deploy-time additions (optional)
  taxonomy.json                # Category taxonomy
tests/
docs/                      # Full documentation set (see below)
```

## Documentation

- [System Flow](./docs/00-overview/system-flow.md) — end-to-end request path
- [Request and Response Contract](./docs/00-overview/request-response-contract.md) — API schema
- [Latency Budget](./docs/00-overview/latency-budget.md) — per-phase timing targets

Pipeline phases:

- [01 Ingress and Validation](./docs/01-pipelines/01-ingress-and-validation.md)
- [02 Ad Eligibility](./docs/01-pipelines/02-ad-eligibility.md)
- [03 Category Extraction](./docs/01-pipelines/03-category-extraction.md)
- [04 Retrieval](./docs/01-pipelines/04-retrieval.md)
- [05 Ranking and Scoring](./docs/01-pipelines/05-ranking-and-scoring.md)
- [06 Post Processing and Response](./docs/01-pipelines/06-post-processing-and-response.md)
- [07 Observability and Metadata](./docs/01-pipelines/07-observability-and-metadata.md)
- [08 Error Handling and Fallbacks](./docs/01-pipelines/08-error-handling-and-fallbacks.md)

Data:

- [Synthetic Campaign Generation](./docs/02-data/synthetic-campaign-generation.md)
- [Campaign Schema](./docs/02-data/campaign-schema.md)
- [Indexing and Embeddings](./docs/02-data/indexing-and-embeddings.md)

Experiments:

- [Experiments Index](./docs/03-experiments/experiments-index.md)
- [Ad Eligibility Experiments](./docs/03-experiments/ad-eligibility-experiments.md)
- [Category Extraction Experiments](./docs/03-experiments/category-extraction-experiments.md)
- [Retrieval Experiments](./docs/03-experiments/retrieval-experiments.md)
- [Ranking Experiments](./docs/03-experiments/ranking-experiments.md)
- [Latency Experiments](./docs/03-experiments/latency-experiments.md)

Ops:

- [Local Development](./docs/04-ops/local-dev.md)
- [Configuration](./docs/04-ops/configuration.md)
- [Testing](./docs/04-ops/testing.md)
- [Deployment](./docs/04-ops/deployment.md)

Design decisions:

- [Tradeoffs](./docs/05-design-decisions/tradeoffs.md)
- [Output Design Rationale](./docs/05-design-decisions/output-design-rationale.md)
- [Safety and Sensitivity](./docs/05-design-decisions/safety-and-sensitivity.md)

Summary:

- [Brief Summary](./docs/99-summary/brief-summary.md)
- [Detailed Summary](./docs/99-summary/detailed-summary.md)
