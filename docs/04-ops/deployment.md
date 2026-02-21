# Deployment

## Expected deployment shape

Inference based on FastAPI app and artifact loading behavior:

- Containerized Python service running `uvicorn app.main:app`.
- Read-only mounted artifact bundle under `DATA_DIR` including FAISS and sidecar files.
- One or more stateless app replicas behind an HTTP load balancer.

## Startup and readiness

- Critical startup path loads embedding backend, eligibility scorer, category extractor, retrieval index, score config, ranker, parser, and eligibility filter.
- Startup fails fast if required artifacts are missing or inconsistent.
- Use `/health` as liveness/readiness signal only after startup completes.

## Health checks

- `GET /health` returns:
  - `status`
  - `campaigns_indexed`
  - `embedding_model`
  - `version`

Suggested policy:

- readiness should require `status=ok` and `campaigns_indexed > 0`.

## Scaling notes

- Scale horizontally on CPU and memory pressure.
- Ensure each replica has local access to index artifacts to avoid network filesystem tail latency.
- Embedding inference and reranking are CPU-bound; monitor p95 with QPS.

## Rollback plan

1. Keep previous image and artifact bundle versions available.
2. Roll back application and artifacts as a pair when embedding/index formats change.
3. Validate `/health` and run small query canary set before full traffic restore.
4. Monitor gated-rate and latency regressions for at least one full traffic cycle.

## Deployment risks

- Artifact mismatch between app version and index sidecars.
- Model path drift (`INSENSITIVE_MODEL_PATH`) across environments.
- Cold-start latency spikes if replicas churn frequently.

## Related docs

- [Configuration](./configuration.md)
- [System Flow](../00-overview/system-flow.md)
