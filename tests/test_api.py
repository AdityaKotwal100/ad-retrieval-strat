"""Integration tests for the /api/retrieve endpoint.

These tests require the FAISS index and all artifacts to exist in data/.
Skip if artifacts are not available.
"""

import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SKIP_REASON = "FAISS index or artifacts not found — run scripts/build_index.py first"


def data_available() -> bool:
    """Check if all required artifacts exist."""
    required = [
        "faiss.index", "campaigns_meta.json",
        "ages_min.npy", "ages_max.npy", "genders.npy", "bids.npy",
    ]
    return all((DATA_DIR / f).exists() for f in required)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    if not data_available():
        pytest.skip(SKIP_REASON)

    from app.services.scoring.score_config import ScoreConfig
    ScoreConfig.reset()

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["campaigns_indexed"] > 0


class TestRetrieve:
    def test_basic_query(self, client):
        resp = client.post("/api/retrieve", json={
            "query": "best running shoes for marathon training",
            "context": {
                "age": 28,
                "gender": "male",
                "location": "San Francisco, CA",
                "interests": ["fitness", "outdoor activities"],
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ad_eligibility"] > 0.0
        assert len(data["campaigns"]) > 0
        assert data["metadata"]["n_candidates_retrieved"] > 0

    def test_no_context(self, client):
        resp = client.post("/api/retrieve", json={"query": "laptop for students"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["campaigns"]) > 0

    def test_empty_query_rejected(self, client):
        resp = client.post("/api/retrieve", json={"query": ""})
        assert resp.status_code == 422

    def test_metadata_fields(self, client):
        resp = client.post("/api/retrieve", json={
            "query": "wireless headphones under $200",
            "context": {"age": 25, "location": "US-CA"},
        })
        data = resp.json()
        meta = data["metadata"]
        assert "n_candidates_retrieved" in meta
        assert "n_candidates_after_filter" in meta
        assert "filter_drops" in meta
        assert "parser_output" in meta
        assert "score_breakdown" in meta
        assert "timing_ms" in meta

    def test_parser_output_in_metadata(self, client):
        resp = client.post("/api/retrieve", json={
            "query": "Nike shoes under $150",
            "context": {"age": 30},
        })
        data = resp.json()
        parser = data["metadata"]["parser_output"]
        assert "price_max" in parser

    def test_blocklist_query(self, client):
        resp = client.post("/api/retrieve", json={
            "query": "how to make a bomb",
        })
        data = resp.json()
        assert data["ad_eligibility"] == 0.0
        assert len(data["campaigns"]) == 0
        assert data["metadata"]["gated"] is True

    def test_campaign_result_fields(self, client):
        resp = client.post("/api/retrieve", json={"query": "running shoes"})
        data = resp.json()
        if data["campaigns"]:
            c = data["campaigns"][0]
            assert "campaign_id" in c
            assert "relevance_score" in c
            assert "title" in c
            assert "advertiser" in c
            assert "category" in c
            assert "vertical" in c
            assert "landing_url" in c
            assert "creative_format" in c
            assert "budget_remaining" in c
