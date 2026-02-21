"""Integration-style API tests with mocked downstream services."""

from __future__ import annotations

import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services.campaign_eligibility import FilterResult
from app.services.eligibility import BlocklistMatch
from app.services.query_parser import ParsedQuery


@pytest.fixture
def api_client(monkeypatch):
    import app.main as app_main

    class StubEmbeddingService:
        dimension = 3

        def embed(self, text: str) -> np.ndarray:
            t = text.lower()
            return np.array(
                [
                    float("suicide" in t or "kill myself" in t),
                    float(any(w in t for w in ("buy", "deal", "order", "price"))),
                    1.0,
                ],
                dtype=np.float32,
            )

    class StubEligibilityScorer:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.num_clusters = 3

        def blocklist_match(self, query: str):
            if "bomb" in query.lower():
                return BlocklistMatch(strategy="regex", rule="bomb", score=0.0)
            return None

        def score_with_metadata(self, query: str, query_embedding: np.ndarray):  # noqa: ARG002
            sensitive = "suicide" in query.lower() or "kill myself" in query.lower()
            eligibility = 0.01 if sensitive else 0.82
            return {
                "eligibility": eligibility,
                "blocklist_triggered": False,
                "hard_block_cluster": None,
                "sensitivity_penalty": 0.95 if sensitive else 0.18,
                "commercial_boost": 1.0 if sensitive else 0.97,
                "top_cluster": "self_harm_crisis" if sensitive else "none",
                "top_cluster_sim": 0.99 if sensitive else 0.12,
                "commercial_sim": 0.05 if sensitive else 0.78,
                "distress_signal": 0.8 if sensitive else 0.0,
                "amplifier": 1.0,
                "co_activation_bonus": 0.0,
            }

    class StubCategoryExtractor:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.num_categories = 3

        def extract(self, query_embedding: np.ndarray, context: dict | None = None):  # noqa: ARG002
            categories = [{"category": "running shoes", "score": 0.91}]
            if context and context.get("location") == "US-CA":
                categories.append({"category": "outdoor fitness", "score": 0.74})
            return categories

    class StubCampaignIndex:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self._meta = [
                {
                    "campaign_id": f"camp_{i:04d}",
                    "title": f"title-{i}",
                    "advertiser": "acme",
                    "category": "running shoes",
                    "vertical": "fitness",
                    "landing_url": f"https://example.test/{i}",
                    "creative_format": "native",
                    "budget_remaining": 500.0,
                }
                for i in range(1200)
            ]
            self._all_brand_names = {"nike", "adidas"}

        @property
        def size(self) -> int:
            return len(self._meta)

        @property
        def all_brand_names(self) -> set[str]:
            return self._all_brand_names

        @property
        def meta(self) -> list[dict]:
            return self._meta

        def search(self, query_embedding: np.ndarray, top_k: int = 1000):  # noqa: ARG002
            k = min(top_k, len(self._meta))
            ids = np.arange(k, dtype=np.int64)
            scores = np.linspace(0.95, 0.10, num=k, dtype=np.float32)
            return ids, scores

    class StubReranker:
        def __init__(self, index: StubCampaignIndex, config):  # noqa: ARG002
            self._index = index
            self._pipeline = [object()]

        def rerank(
            self,
            candidate_ids: np.ndarray,
            faiss_scores: np.ndarray,
            parsed_query: ParsedQuery,  # noqa: ARG002
            top_k: int = 1000,
            debug_top_n: int = 5,  # noqa: ARG002
            extracted_categories: list[dict] | None = None,  # noqa: ARG002
        ):
            k = min(top_k, len(candidate_ids))
            results = []
            for pos, row_id in enumerate(candidate_ids[:k]):
                result = dict(self._index.meta[int(row_id)])
                result["internal_score"] = round(float(faiss_scores[pos]), 6)
                result["relevance_score"] = round(float(faiss_scores[pos]), 6)
                results.append(result)

            breakdown = []
            if results:
                breakdown.append(
                    {
                        "campaign_id": results[0]["campaign_id"],
                        "base_score": float(faiss_scores[0]),
                        "final_score": float(faiss_scores[0]),
                        "active_params": 0,
                        "coverage_ratio": 1.0,
                        "query_potential_max": 1.0,
                        "deltas": [],
                    }
                )
            return results, breakdown

    class StubQueryParser:
        def __init__(self, brand_names: set[str]):  # noqa: ARG002
            pass

        def parse(self, query: str, context: dict | None = None) -> ParsedQuery:
            ctx = context or {}
            return ParsedQuery(
                raw_query=query,
                query_tokens=frozenset(re.findall(r"[a-z0-9]+", query.lower())),
                user_location=ctx.get("location"),
                user_age=ctx.get("age"),
                user_gender=ctx.get("gender"),
                user_interests=ctx.get("interests", []),
            )

    class StubCampaignEligibilityFilter:
        def __init__(self, index: StubCampaignIndex):  # noqa: ARG002
            pass

        def filter_with_refill(
            self,
            candidate_ids: np.ndarray,
            faiss_scores: np.ndarray,
            parsed_query: ParsedQuery,  # noqa: ARG002
            index: StubCampaignIndex,  # noqa: ARG002
            query_embedding: np.ndarray,  # noqa: ARG002
            target_count: int = 500,  # noqa: ARG002
            max_refill_rounds: int = 2,  # noqa: ARG002
        ) -> FilterResult:
            keep = min(len(candidate_ids), 1000)
            dropped = max(len(candidate_ids) - keep, 0)
            return FilterResult(
                surviving_ids=candidate_ids[:keep],
                surviving_scores=faiss_scores[:keep],
                drop_counts={"mock_drop": dropped} if dropped else {},
                total_dropped=dropped,
            )

    class StubScoreConfig:
        def __init__(self) -> None:
            self.data = {"schema_version": "test"}

    def _fake_load(cls, config_path):  # noqa: ARG001
        return StubScoreConfig()

    monkeypatch.setattr(app_main, "create_embedding_service", lambda: StubEmbeddingService())
    monkeypatch.setattr(app_main, "EligibilityScorer", StubEligibilityScorer)
    monkeypatch.setattr(app_main, "CategoryExtractor", StubCategoryExtractor)
    monkeypatch.setattr(app_main, "CampaignIndex", StubCampaignIndex)
    monkeypatch.setattr(app_main, "ContextReranker", StubReranker)
    monkeypatch.setattr(app_main, "QueryParser", StubQueryParser)
    monkeypatch.setattr(app_main, "CampaignEligibilityFilter", StubCampaignEligibilityFilter)
    monkeypatch.setattr(app_main, "create_blocklist_strategy", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_main.ScoreConfig, "load", classmethod(_fake_load))

    with TestClient(app_main.app) as client:
        yield client


def test_retrieve_happy_path_returns_expected_shape(api_client):
    response = api_client.post(
        "/api/retrieve",
        json={
            "query": "best running shoes under 200",
            "context": {"location": "US-CA", "age": 30, "interests": ["fitness"]},
        },
    )
    assert response.status_code == 200

    payload = response.json()
    for key in ("ad_eligibility", "extracted_categories", "campaigns", "latency_ms", "metadata"):
        assert key in payload

    assert isinstance(payload["campaigns"], list)
    assert "n_candidates_retrieved" in payload["metadata"]
    assert "n_candidates_after_filter" in payload["metadata"]
    assert "timing_ms" in payload["metadata"]
    assert "eligibility_detail" in payload["metadata"]


def test_missing_query_field_returns_4xx_with_clear_message(api_client):
    response = api_client.post("/api/retrieve", json={"context": {"age": 27}})
    assert response.status_code == 422
    assert "query" in str(response.json()).lower()


def test_malformed_json_returns_4xx_with_parse_error(api_client):
    response = api_client.post(
        "/api/retrieve",
        data='{"query": "broken"',
        headers={"content-type": "application/json"},
    )
    assert response.status_code in (400, 422)
    body = str(response.json()).lower()
    assert "json" in body or "decode" in body


def test_sensitive_query_is_gated_but_response_shape_is_valid(api_client):
    response = api_client.post(
        "/api/retrieve",
        json={"query": "i am thinking about suicide tonight"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["ad_eligibility"] <= 0.05
    assert payload["campaigns"] == []
    assert payload["metadata"]["gated"] is True
    assert "eligibility_detail" in payload["metadata"]


def test_optional_context_absent_does_not_crash(api_client):
    response = api_client.post("/api/retrieve", json={"query": "best running shoes"})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["campaigns"], list)
