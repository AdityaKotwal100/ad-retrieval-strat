# Test queries for POST /api/retrieve

Each JSON file is a valid request body for `POST /api/retrieve`. Use them for manual testing, load tests, or automation.

## Usage with curl

```bash
# Single query (server must be running on port 8000)
curl -s -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d @data/test_queries/01-running-shoes-marathon.json

# Pretty-print response
curl -s -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d @data/test_queries/01-running-shoes-marathon.json | python3 -m json.tool
```

## Run all queries

```bash
for f in data/test_queries/*.json; do
  [[ "$f" == *README* ]] && continue
  echo "=== $(basename "$f") ==="
  curl -s -X POST http://localhost:8000/api/retrieve \
    -H "Content-Type: application/json" \
    -d @"$f" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"eligibility={d['ad_eligibility']:.3f} latency={d['latency_ms']}ms n_campaigns={len(d['campaigns'])}\")"
  echo
done
```

## Query index

| File | Query type | Expected eligibility | Notes |
|------|------------|----------------------|--------|
| `01-running-shoes-marathon.json` | Commercial | High (0.8+) | Full context: age, gender, location, interests |
| `02-laptop-college.json` | Commercial | High | Tech + education interests |
| `03-meal-prep-weight-loss.json` | Commercial | High | Health, cooking interests |
| `04-flights-travel.json` | Commercial | High | Travel interest, no location |
| `05-crm-small-business.json` | Commercial | High | Business, software interests |
| `06-moisturizer-no-context.json` | Commercial | High | No context (tests default behaviour) |
| `07-marathon-history.json` | Informational | High | Safe to show ads (e.g. marathon gear) |
| `08-runners-blisters.json` | Informational | High | Mild medical wording, should still score high |
| `09-stressed-work.json` | Mildly sensitive | Medium (0.4–0.7) | Stress / anxiety adjacent |
| `10-file-unemployment.json` | Sensitive topic | Medium | Financial hardship, procedural |
| `11-lost-job.json` | Sensitive topic | Medium | Job loss, emotional |
| `12-grief.json` | Grief | Low / zero | Should not show ads |
| `13-self-harm-blocklist.json` | Blocklist | 0.0 | Regex blocklist hit |
| `14-medical-emergency.json` | Emergency | Low | Acute medical |
| `15-cant-afford-food.json` | Financial distress | Low | Severe hardship |
| `16-blocklist-bomb.json` | Blocklist | 0.0 | Violence blocklist hit |

## Schema

Each file must have:

- **`query`** (string, required): User's natural language query.
- **`context`** (object, optional): User context. May include:
  - `gender`: `"male"` | `"female"` | etc.
  - `age`: integer
  - `location`: e.g. `"San Francisco, CA"` (state code is extracted for targeting)
  - `interests`: array of strings, e.g. `["fitness", "technology"]`

Omit `context` or set to `null` for no user context.
