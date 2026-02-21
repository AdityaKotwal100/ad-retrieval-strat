"""Unit tests for ad eligibility scoring behavior."""

from __future__ import annotations

import pytest

from app.services.eligibility import (
    EligibilityScorer,
    NoopBlocklistStrategy,
)
import app.services.eligibility as eligibility_mod


@pytest.fixture
def eligibility_scorer(monkeypatch, deterministic_embedding_service):
    """Build a scorer with tiny deterministic clusters for fast tests."""

    monkeypatch.setattr(
        eligibility_mod,
        "_SENSITIVITY_CLUSTERS",
        {
            "self_harm_crisis": ["kill myself now crisis"],
            "financial_hardship": ["cannot afford rent debt eviction"],
            "medical_emergency": ["not breathing emergency heart attack"],
        },
    )
    monkeypatch.setattr(
        eligibility_mod,
        "_COMMERCIAL_EXEMPLARS",
        [
            "best deal buy now",
            "order online at sale price",
        ],
    )
    monkeypatch.setattr(
        eligibility_mod,
        "_CLUSTER_HARD_THRESHOLDS",
        {
            "self_harm_crisis": 0.55,
            "financial_hardship": 1.10,
            "medical_emergency": 1.10,
        },
    )
    monkeypatch.setattr(
        eligibility_mod,
        "_CLUSTER_SENSITIVITY_SCALE",
        {
            "self_harm_crisis": 1.4,
            "financial_hardship": 0.6,
            "medical_emergency": 1.0,
        },
    )
    monkeypatch.setattr(eligibility_mod, "SENSITIVITY_MIN_SIM", 0.15)

    scorer = EligibilityScorer(
        deterministic_embedding_service,
        blocklist_strategy=NoopBlocklistStrategy(),
    )
    return scorer, deterministic_embedding_service


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "best deal to buy running shoes online",
        "history of marathon training",
        "this is a very long query " * 200,
        "cafe restaurantes en espana",
    ],
)
def test_score_is_bounded_for_varied_inputs(eligibility_scorer, query):
    scorer, svc = eligibility_scorer
    score = scorer.score(query, svc.embed(query))
    assert 0.0 <= score <= 1.0


def test_sensitive_query_is_suppressed_by_semantic_override(eligibility_scorer):
    scorer, svc = eligibility_scorer
    query = "i want to kill myself tonight"
    score = scorer.score(query, svc.embed(query))
    assert score == 0.0


def test_commercial_query_scores_higher_than_neutral_query(eligibility_scorer):
    scorer, svc = eligibility_scorer
    commercial = "best deal to buy running shoes online"
    neutral = "history of marathon training plans"

    commercial_score = scorer.score(commercial, svc.embed(commercial))
    neutral_score = scorer.score(neutral, svc.embed(neutral))

    assert commercial_score > neutral_score


def test_ambiguous_distress_query_is_suppressed_by_v5_model(eligibility_scorer):
    scorer, svc = eligibility_scorer
    query = "i lost my job and need help paying rent and debt"
    score = scorer.score(query, svc.embed(query))
    assert score == 0.0


def test_repeated_calls_are_deterministic(eligibility_scorer):
    scorer, svc = eligibility_scorer
    query = "best laptop deal for students"
    vec = svc.embed(query)
    scores = [scorer.score(query, vec) for _ in range(5)]
    assert scores.count(scores[0]) == len(scores)


def test_score_with_metadata_has_expected_shape(eligibility_scorer):
    scorer, svc = eligibility_scorer
    query = "best laptop deal for students"
    result = scorer.score_with_metadata(query, svc.embed(query))

    assert "eligibility" in result
    assert "sensitivity_penalty" in result
    assert "commercial_boost" in result
    assert "top_cluster" in result
    assert 0.0 <= result["eligibility"] <= 1.0
