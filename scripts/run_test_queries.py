#!/usr/bin/env python3
"""
Run all test queries against the running app and store results.

For each file in data/test_queries/*.json creates:
  data/test_results/<name>/input.json
  data/test_results/<name>/output.json

where <name> is the filename without .json (e.g. 01-running-shoes-marathon).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/api/retrieve"

QUERIES_DIR = Path(__file__).resolve().parent / "test_queries"
RESULTS_DIR = Path(__file__).resolve().parent / "test_results"


def run_all() -> None:
    query_files = sorted(QUERIES_DIR.glob("*.json"))
    if not query_files:
        print(f"No query files found in {QUERIES_DIR}", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(query_files)} query files. Sending to {ENDPOINT}\n")

    ok = 0
    errors = 0

    for qfile in query_files:
        name = qfile.stem  # filename without .json
        out_dir = RESULTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(qfile) as f:
            payload = json.load(f)

        # Write input
        (out_dir / "input.json").write_text(json.dumps(payload, indent=2))

        try:
            resp = httpx.post(ENDPOINT, json=payload, timeout=30.0)
            resp.raise_for_status()
            output = resp.json()
            status = resp.status_code
        except httpx.HTTPStatusError as e:
            output = {"error": str(e), "status_code": e.response.status_code, "body": e.response.text}
            status = e.response.status_code
            errors += 1
        except httpx.RequestError as e:
            output = {"error": str(e)}
            status = "ERR"
            errors += 1
        else:
            ok += 1

        (out_dir / "output.json").write_text(json.dumps(output, indent=2))

        n_results = output.get("campaigns") and len(output["campaigns"])
        latency = output.get("latency_ms", "?")
        print(f"  [{status}] {name:45s}  results={n_results or 0:3}  latency={latency}ms")

    print(f"\nDone. {ok} succeeded, {errors} failed.")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    run_all()
