# Local Development

## Prerequisites

- Python 3.11+ recommended.
- FAISS-compatible environment for `faiss-cpu`.
- Optional: `OPENROUTER_API_KEY` for synthetic data generation workflows.

## Setup

```bash
make install
source .venv/bin/activate
```

Alternative:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Build or refresh data and index

### Generate taxonomy

```bash
python3 scripts/generate_taxonomy.py
```

### Generate campaigns (LLM-dependent)

```bash
python3 scripts/generate_campaigns.py
python3 scripts/generate_campaigns_v3.py
```

### Build retrieval artifacts

```bash
python3 scripts/build_index.py
# or
python3 scripts/build_index_v4.py
```

## Run service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

## Run test queries and benchmarks

```bash
python3 scripts/run_test_queries.py
python3 scripts/benchmark_latency.py --base-url http://localhost:8000
```

## Smoke scripts

```bash
python3 scripts/test_embedding.py
python3 scripts/test_categories.py --demo
python3 scripts/test_retrieval.py
python3 scripts/test_eligibility.py
```

## Related docs

- [Configuration](./configuration.md)
- [Testing](./testing.md)
