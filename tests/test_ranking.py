"""Unit tests for retrieval reranking logic."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.query_parser import ParsedQuery
from app.services.ranking import ContextReranker
from app.services.scoring.score_context import ScoreDelta


class MinimalIndex:
    """Small campaign index stub used by ranking unit tests."""

    def __init__(self, n: int, include_optional_meta: bool = True):
        if include_optional_meta:
            self._meta = [
                {
                    "campaign_id": f"camp_{i:04d}",
                    "title": f"title-{i}",
                    "advertiser": "acme",
                    "category": "running shoes",
                    "vertical": "fitness",
                    "landing_url": f"https://example.test/{i}",
                    "creative_format": "native",
                    "budget_remaining": 1000.0,
                }
                for i in range(n)
            ]
        else:
            self._meta = [{"campaign_id": f"camp_{i:04d}"} for i in range(n)]

    @property
    def meta(self) -> list[dict]:
        return self._meta


def _pq(**kwargs) -> ParsedQuery:
    base = {
        "raw_query": "best running shoes",
        "query_tokens": frozenset({"best", "running", "shoes"}),
    }
    base.update(kwargs)
    return ParsedQuery(**base)


class StubComponent:
    """Deterministic component with fixed per-candidate deltas."""

    def __init__(self, name: str, deltas: list[float]):
        self._name = name
        self._deltas = deltas

    def apply_batch(self, candidate_ids, parsed_query, index, config):  # noqa: ARG002
        return [
            ScoreDelta(self._name, float(self._deltas[i]), reason=f"{self._name}:{i}")
            for i in range(len(candidate_ids))
        ]

    def apply(self, ctx, config):  # pragma: no cover - apply_batch is always used here
        raise AssertionError("apply() should not be called in this test")


def test_ordering_prefers_higher_relevance_scores(score_config):
    index = MinimalIndex(n=3)
    ranker = ContextReranker(index, score_config)
    ranker._pipeline = []

    candidate_ids = np.array([0, 1, 2], dtype=np.int64)
    faiss_scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)

    results, _ = ranker.rerank(candidate_ids, faiss_scores, _pq())
    ordered = [row["campaign_id"] for row in results]
    assert ordered == ["camp_0001", "camp_0002", "camp_0000"]


def test_ties_are_deterministic_and_follow_reverse_candidate_order(score_config):
    index = MinimalIndex(n=3)
    ranker = ContextReranker(index, score_config)
    ranker._pipeline = []

    candidate_ids = np.array([0, 1, 2], dtype=np.int64)
    faiss_scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    first, _ = ranker.rerank(candidate_ids, faiss_scores, _pq())
    second, _ = ranker.rerank(candidate_ids, faiss_scores, _pq())

    first_order = [row["campaign_id"] for row in first]
    second_order = [row["campaign_id"] for row in second]

    assert first_order == second_order
    assert first_order == ["camp_0002", "camp_0001", "camp_0000"]


def test_coverage_ratio_worked_example_matches_expected_math(score_config):
    index = MinimalIndex(n=2)
    ranker = ContextReranker(index, score_config)

    # Worked example:
    # active specs = 3 (location + demographic + interests)
    # candidate 0: spec_delta=0.30, matched=3 -> ratio=1.0
    # candidate 1: spec_delta=0.10, matched=1 -> ratio=1/3
    # always-on delta = 0.05 for both
    ranker._pipeline = [
        StubComponent("location", [0.10, 0.10]),
        StubComponent("demographic", [0.10, 0.00]),
        StubComponent("interest", [0.10, 0.00]),
        StubComponent("taxonomy_alignment", [0.05, 0.05]),
    ]

    candidate_ids = np.array([0, 1], dtype=np.int64)
    faiss_scores = np.array([0.5, 0.5], dtype=np.float32)
    pq = _pq(user_location="US-CA", user_age=30, user_interests=["running"])

    results, breakdowns = ranker.rerank(
        candidate_ids,
        faiss_scores,
        pq,
        debug_top_n=2,
    )

    assert [row["campaign_id"] for row in results][:2] == ["camp_0000", "camp_0001"]

    by_campaign = {row["campaign_id"]: row for row in breakdowns}
    assert by_campaign["camp_0000"]["coverage_ratio"] == pytest.approx(1.0)
    assert by_campaign["camp_0001"]["coverage_ratio"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert by_campaign["camp_0000"]["final_score"] == pytest.approx(0.85, abs=1e-6)
    assert by_campaign["camp_0001"]["final_score"] == pytest.approx(0.583333, abs=1e-5)


def test_empty_candidates_return_empty_outputs(score_config):
    index = MinimalIndex(n=0)
    ranker = ContextReranker(index, score_config)
    ranker._pipeline = []

    candidate_ids = np.array([], dtype=np.int64)
    faiss_scores = np.array([], dtype=np.float32)
    results, breakdowns = ranker.rerank(candidate_ids, faiss_scores, _pq())

    assert results == []
    assert breakdowns == []


def test_fewer_than_1000_candidates_returns_all(score_config):
    index = MinimalIndex(n=12)
    ranker = ContextReranker(index, score_config)
    ranker._pipeline = []

    candidate_ids = np.arange(12, dtype=np.int64)
    faiss_scores = np.linspace(1.0, 0.5, num=12, dtype=np.float32)
    results, _ = ranker.rerank(candidate_ids, faiss_scores, _pq())

    assert len(results) == 12


def test_more_than_1000_candidates_is_trimmed_to_1000(score_config):
    index = MinimalIndex(n=1500)
    ranker = ContextReranker(index, score_config)
    ranker._pipeline = []

    candidate_ids = np.arange(1500, dtype=np.int64)
    faiss_scores = np.linspace(1.0, 0.0, num=1500, dtype=np.float32)
    results, _ = ranker.rerank(candidate_ids, faiss_scores, _pq())

    assert len(results) == 1000


def test_missing_optional_metadata_fields_do_not_crash(score_config):
    index = MinimalIndex(n=2, include_optional_meta=False)
    ranker = ContextReranker(index, score_config)
    ranker._pipeline = []

    candidate_ids = np.array([0, 1], dtype=np.int64)
    faiss_scores = np.array([0.8, 0.7], dtype=np.float32)
    results, _ = ranker.rerank(candidate_ids, faiss_scores, _pq())

    assert len(results) == 2
    assert set(results[0].keys()) == {"campaign_id", "relevance_score", "internal_score"}
