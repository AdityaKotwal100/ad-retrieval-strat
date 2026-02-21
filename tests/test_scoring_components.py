"""Unit tests for scoring components."""

import numpy as np
import pytest

from app.services.query_parser import ParsedQuery
from app.services.scoring.components import (
    AttributeAlignmentComponent,
    BrandAlignmentComponent,
    ContradictionPenaltyComponent,
    DemographicBoostComponent,
    InterestBoostComponent,
    KeywordOverlapComponent,
    LocationBoostComponent,
    PriceAlignmentComponent,
    TaxonomyAlignmentComponent,
)
from app.services.scoring.score_config import ScoreConfig
from app.services.scoring.score_context import ScoreContext


class MockIndex:
    """Minimal mock of CampaignIndex for scoring component tests."""

    def __init__(self):
        self._ages_min = np.array([18], dtype=np.int8)
        self._ages_max = np.array([65], dtype=np.int8)
        self._genders = np.array([1], dtype=np.uint8)  # bit0=all
        self._bids = np.array([100.0], dtype=np.float32)
        self._geo_include = [frozenset(["US-CA", "US-NY"])]
        self._geo_exclude = [frozenset()]
        self._taxonomy = [{"category": "running shoes", "subcategory": "road running", "leaf": "stability"}]
        self._brands = [{"name": "Nike", "type": "manufacturer", "product_line": "Pegasus"}]
        self._keyword_sets = [{
            "brand_terms": frozenset(["nike"]),
            "model_terms": frozenset(["pegasus"]),
            "category_terms": frozenset(["running", "shoes"]),
            "feature_terms": frozenset(["cushioning", "stability"]),
            "benefit_terms": frozenset(["comfort", "injury prevention"]),
        }]
        self._negative_keywords = [frozenset(["basketball", "soccer", "cleats"])]
        self._attributes = [{"terrain": "road", "support": "stability", "waterproof": False}]
        self._interests = [frozenset(["fitness", "running", "outdoor"])]

    @property
    def ages_min(self): return self._ages_min
    @property
    def ages_max(self): return self._ages_max
    @property
    def genders(self): return self._genders
    @property
    def bids(self): return self._bids
    @property
    def geo_include(self): return self._geo_include
    @property
    def geo_exclude(self): return self._geo_exclude
    @property
    def taxonomy(self): return self._taxonomy
    @property
    def brands(self): return self._brands
    @property
    def keyword_sets(self): return self._keyword_sets
    @property
    def negative_keywords(self): return self._negative_keywords
    @property
    def attributes(self): return self._attributes
    @property
    def interests(self): return self._interests


def make_ctx(parsed_query: ParsedQuery, index=None) -> ScoreContext:
    return ScoreContext(
        parsed_query=parsed_query,
        campaign_idx=0,
        base_score=0.8,
        index=index or MockIndex(),
    )


def make_pq(**kwargs) -> ParsedQuery:
    defaults = {
        "raw_query": "test query",
        "query_tokens": frozenset(["test", "query"]),
    }
    defaults.update(kwargs)
    return ParsedQuery(**defaults)


class TestLocationBoost:
    def test_match(self, score_config):
        pq = make_pq(user_location="US-CA")
        ctx = make_ctx(pq)
        delta = LocationBoostComponent().apply(ctx, score_config)
        assert delta.delta == pytest.approx(0.05)

    def test_no_match(self, score_config):
        pq = make_pq(user_location="US-TX")
        ctx = make_ctx(pq)
        delta = LocationBoostComponent().apply(ctx, score_config)
        assert delta.delta == 0.0

    def test_no_location(self, score_config):
        pq = make_pq()
        ctx = make_ctx(pq)
        delta = LocationBoostComponent().apply(ctx, score_config)
        assert delta.delta == 0.0


class TestDemographicBoost:
    def test_age_and_gender_match(self, score_config):
        pq = make_pq(user_age=30, user_gender="male")
        ctx = make_ctx(pq)
        delta = DemographicBoostComponent().apply(ctx, score_config)
        expected = (
            score_config.get("geo_demo", "age_match")
            + score_config.get("geo_demo", "gender_match")
        )
        assert delta.delta == pytest.approx(expected)

    def test_age_out_of_range(self, score_config):
        pq = make_pq(user_age=10, user_gender="male")
        ctx = make_ctx(pq)
        delta = DemographicBoostComponent().apply(ctx, score_config)
        expected = (
            score_config.get("geo_demo", "age_mismatch")
            + score_config.get("geo_demo", "gender_match")
        )
        assert delta.delta == pytest.approx(expected)

    def test_no_context(self, score_config):
        pq = make_pq()
        ctx = make_ctx(pq)
        delta = DemographicBoostComponent().apply(ctx, score_config)
        assert delta.delta == 0.0


class TestInterestBoost:
    def test_overlap(self, score_config):
        pq = make_pq(user_interests=["fitness", "running"])
        ctx = make_ctx(pq)
        delta = InterestBoostComponent().apply(ctx, score_config)
        assert delta.delta > 0.0
        assert delta.delta <= 0.05

    def test_no_overlap(self, score_config):
        pq = make_pq(user_interests=["cooking", "baking"])
        ctx = make_ctx(pq)
        delta = InterestBoostComponent().apply(ctx, score_config)
        assert delta.delta == 0.0

    def test_no_interests(self, score_config):
        pq = make_pq()
        ctx = make_ctx(pq)
        delta = InterestBoostComponent().apply(ctx, score_config)
        assert delta.delta == 0.0


class TestTaxonomyAlignment:
    def test_category_match(self, score_config):
        pq = make_pq(raw_query="running shoes for marathon", query_tokens=frozenset(["running", "shoes", "for", "marathon"]))
        ctx = make_ctx(pq)
        delta = TaxonomyAlignmentComponent().apply(ctx, score_config)
        assert delta.delta > 0.0

    def test_no_match(self, score_config):
        pq = make_pq(raw_query="laptop for work", query_tokens=frozenset(["laptop", "for", "work"]))
        ctx = make_ctx(pq)
        delta = TaxonomyAlignmentComponent().apply(ctx, score_config)
        assert delta.delta == pytest.approx(score_config.get("taxonomy", "mismatch_penalty"))


class TestBrandAlignment:
    def test_brand_match(self, score_config):
        pq = make_pq(brand_terms=["nike"])
        ctx = make_ctx(pq)
        delta = BrandAlignmentComponent().apply(ctx, score_config)
        assert delta.delta > 0.0

    def test_no_brand(self, score_config):
        pq = make_pq(brand_terms=[])
        ctx = make_ctx(pq)
        delta = BrandAlignmentComponent().apply(ctx, score_config)
        assert delta.delta == 0.0

    def test_wrong_brand(self, score_config):
        pq = make_pq(brand_terms=["adidas"])
        ctx = make_ctx(pq)
        delta = BrandAlignmentComponent().apply(ctx, score_config)
        assert delta.delta == pytest.approx(score_config.get("brand", "mismatch_penalty"))


class TestAttributeAlignment:
    def test_matching_attribute(self, score_config):
        pq = make_pq(attribute_intents={"terrain": "road"})
        ctx = make_ctx(pq)
        delta = AttributeAlignmentComponent().apply(ctx, score_config)
        assert delta.delta > 0.0

    def test_contradicting_attribute(self, score_config):
        pq = make_pq(attribute_intents={"terrain": "trail"})
        ctx = make_ctx(pq)
        delta = AttributeAlignmentComponent().apply(ctx, score_config)
        assert delta.delta < 0.0

    def test_no_intents(self, score_config):
        pq = make_pq(attribute_intents={})
        ctx = make_ctx(pq)
        delta = AttributeAlignmentComponent().apply(ctx, score_config)
        assert delta.delta == 0.0


class TestKeywordOverlap:
    def test_overlapping_tokens(self, score_config):
        pq = make_pq(query_tokens=frozenset(["running", "shoes", "nike"]))
        ctx = make_ctx(pq)
        delta = KeywordOverlapComponent().apply(ctx, score_config)
        assert delta.delta > 0.0

    def test_no_overlap(self, score_config):
        pq = make_pq(query_tokens=frozenset(["laptop", "dell"]))
        ctx = make_ctx(pq)
        delta = KeywordOverlapComponent().apply(ctx, score_config)
        assert delta.delta == 0.0


class TestPriceAlignment:
    def test_under_budget(self, score_config):
        pq = make_pq(price_max=150.0)
        ctx = make_ctx(pq)  # campaign price = 100
        delta = PriceAlignmentComponent().apply(ctx, score_config)
        assert delta.delta > 0.0

    def test_over_budget(self, score_config):
        pq = make_pq(price_max=50.0)
        ctx = make_ctx(pq)  # campaign price = 100
        delta = PriceAlignmentComponent().apply(ctx, score_config)
        assert delta.delta < 0.0


class TestContradictionPenalty:
    def test_conflict(self, score_config):
        pq = make_pq(query_tokens=frozenset(["basketball", "shoes"]))
        ctx = make_ctx(pq)
        delta = ContradictionPenaltyComponent().apply(ctx, score_config)
        assert delta.delta < 0.0

    def test_no_conflict(self, score_config):
        pq = make_pq(query_tokens=frozenset(["running", "shoes"]))
        ctx = make_ctx(pq)
        delta = ContradictionPenaltyComponent().apply(ctx, score_config)
        assert delta.delta == 0.0
