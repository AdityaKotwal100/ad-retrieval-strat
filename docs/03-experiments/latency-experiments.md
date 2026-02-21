# Latency Experiments

## Experiment 1: Parallel request latency benchmark script

- Hypothesis
- Service can meet latency target under parallel query load from the curated test suite.

- Change made
- Benchmark harness `scripts/benchmark_latency.py` sends all test queries concurrently and reports min/avg/p50/p95/p99.

- Evaluation method
- HTTP benchmark against running service using query files in `data/test_queries`.

- Results
- Script exists and calculates summary metrics, but no committed historical benchmark output snapshot was found.

- Decision and rationale
- Kept script for repeatable local/perf checks.

## Experiment 2: Saved test run latency extraction

- Hypothesis
- Existing saved output artifacts can provide real stage-level and end-to-end latency evidence.

- Change made
- Replayed analysis of `data/test_results/*/output.json` in this docs pass.

- Evaluation method
- Parsed 56 output files and computed latency summary.

- Results
- min: 0.45ms
- p50: 9.12ms
- p95: 30.84ms
- max: 32.3ms
- mean: 10.55ms

- Decision and rationale
- Confirms large margin against 100ms p95 target for this local snapshot.
- Treated as supportive but not final production SLO evidence.

## Experiment 3: Stage timing instrumentation in runtime responses

- Hypothesis
- Per-phase timings in API metadata make bottleneck detection and regressions easier.

- Change made
- `app/main.py` emits `timing_ms` with `embed`, `parallel_stages`, `parse`, `filter_and_rerank`, `total`.

- Evaluation method
- Inspect response metadata from saved outputs and query runs.

- Results
- Stage timing keys are consistently present in ungated responses.
- Example in `experiments/relevance-scoring/response.json` shows stage decomposition for a 65.68ms request.

- Decision and rationale
- Kept instrumentation as standard metadata.

## Evidence gaps

- Not found in repo: long-run load test reports by concurrency level, CPU saturation, and p99 tail behavior.
- Not found in repo: cold-start vs warm-start latency comparison report.

## Proposed replacement experiments

1. Run `scripts/benchmark_latency.py --runs N` at multiple concurrency levels and collect p50/p95/p99.
2. Add warm/cold process restart benchmarks.
3. Record stage-level timing histograms and regressions in CI for representative hardware.
4. Track payload-size impact on total latency for debug-heavy metadata responses.
