"""Unit tests for CampaignEligibilityFilter."""

import numpy as np
import pytest

from app.services.campaign_eligibility import CampaignEligibilityFilter, FilterResult
from app.services.query_parser import ParsedQuery


class MockIndex:
    """Minimal mock of CampaignIndex for eligibility filter tests."""

    def __init__(self, n: int = 5):
        self._n = n
        # Default: all campaigns accept everyone
        self._ages_min = np.array([18] * n, dtype=np.int8)
        self._ages_max = np.array([65] * n, dtype=np.int8)
        # Bitmask: 1=all
        self._genders = np.array([1] * n, dtype=np.uint8)
        self._bids = np.array([100.0] * n, dtype=np.float32)
        self._geo_include = [frozenset(["US-CA", "US-NY"]) for _ in range(n)]
        self._geo_exclude = [frozenset() for _ in range(n)]
        self._negative_keywords = [frozenset() for _ in range(n)]
        self._taxonomy = [{"category": "running shoes"} for _ in range(n)]

    @property
    def ages_min(self):
        return self._ages_min

    @property
    def ages_max(self):
        return self._ages_max

    @property
    def genders(self):
        return self._genders

    @property
    def bids(self):
        return self._bids

    @property
    def geo_include(self):
        return self._geo_include

    @property
    def geo_exclude(self):
        return self._geo_exclude

    @property
    def negative_keywords(self):
        return self._negative_keywords

    @property
    def taxonomy(self):
        return self._taxonomy


def make_parsed_query(**kwargs) -> ParsedQuery:
    defaults = {
        "raw_query": "test query",
        "query_tokens": frozenset(["test", "query"]),
    }
    defaults.update(kwargs)
    return ParsedQuery(**defaults)


class TestGeoIncludeFilter:
    def test_passes_matching_location(self):
        idx = MockIndex(3)
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query(user_location="US-CA")
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 3

    def test_filters_non_matching_location(self):
        idx = MockIndex(3)
        idx._geo_include[1] = frozenset(["US-TX"])  # campaign 1 only targets TX
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query(user_location="US-CA")
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 2
        assert 1 not in result.surviving_ids
        assert result.drop_counts.get("geo_include", 0) == 1


class TestGeoExcludeFilter:
    def test_filters_excluded_location(self):
        idx = MockIndex(3)
        idx._geo_exclude[2] = frozenset(["US-CA"])
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query(user_location="US-CA")
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 2
        assert result.drop_counts.get("geo_exclude", 0) == 1


class TestAgeFilter:
    def test_passes_in_range(self):
        idx = MockIndex(3)
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query(user_age=30)
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 3

    def test_filters_out_of_range(self):
        idx = MockIndex(3)
        idx._ages_min[1] = 40
        idx._ages_max[1] = 60
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query(user_age=25)
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 2
        assert result.drop_counts.get("age_range", 0) == 1


class TestGenderFilter:
    def test_passes_all_gender(self):
        idx = MockIndex(3)
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query(user_gender="male")
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 3

    def test_filters_wrong_gender(self):
        idx = MockIndex(3)
        idx._genders[1] = 4  # female only (bit2)
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query(user_gender="male")
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 2
        assert result.drop_counts.get("gender", 0) == 1


class TestHardPriceFilter:
    def test_no_filter_without_price_max(self):
        idx = MockIndex(3)
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query()
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 3

    def test_filters_over_hard_limit(self):
        idx = MockIndex(3)
        idx._bids[2] = 500.0  # way over $100 * 1.5 = $150
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query(price_max=100.0)
        ids = np.array([0, 1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 2
        assert result.drop_counts.get("price_hard", 0) == 1


class TestEmptyCandidates:
    def test_empty_input(self):
        idx = MockIndex(0)
        f = CampaignEligibilityFilter(idx)
        pq = make_parsed_query()
        ids = np.array([], dtype=np.int64)
        scores = np.array([], dtype=np.float32)
        result = f.filter(ids, scores, pq)
        assert len(result.surviving_ids) == 0
