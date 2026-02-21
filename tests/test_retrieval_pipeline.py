"""Tests for retrieval and category extraction pipeline behavior."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.services.categories import CategoryExtractor
from app.services.retrieval import CampaignIndex


def _write_taxonomy(tmp_path, entries: list[dict]) -> str:
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def test_category_extraction_returns_expected_shape_and_bounds(
    tmp_path,
    deterministic_embedding_service,
):
    taxonomy_path = _write_taxonomy(
        tmp_path,
        [
            {"name": "running shoes", "description": "marathon running footwear"},
            {"name": "laptop deals", "description": "student laptop computer discounts"},
            {"name": "travel insurance", "description": "trip medical insurance coverage"},
        ],
    )

    extractor = CategoryExtractor(
        deterministic_embedding_service,
        taxonomy_path=taxonomy_path,
    )
    query_vec = deterministic_embedding_service.embed("best running shoes for marathon")

    categories = extractor.extract(query_vec, top_k=10, threshold=0.0)

    assert 1 <= len(categories) <= 10
    assert all(set(item.keys()) == {"category", "score"} for item in categories)
    scores = [item["score"] for item in categories]
    assert scores == sorted(scores, reverse=True)


def test_interest_context_can_change_category_ordering(
    tmp_path,
    deterministic_embedding_service,
):
    taxonomy_path = _write_taxonomy(
        tmp_path,
        [
            {"name": "travel deals", "description": "travel flights hotels"},
            {"name": "fitness gear", "description": "workout equipment plans"},
        ],
    )
    extractor = CategoryExtractor(
        deterministic_embedding_service,
        taxonomy_path=taxonomy_path,
    )

    # Override internals so this test directly validates context boost math.
    extractor._taxonomy_names = ["travel deals", "fitness gear"]
    extractor._taxonomy_keyword_sets = [
        {"travel", "deals"},
        {"fitness", "gear"},
    ]
    extractor._embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.99, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    query_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    baseline = extractor.extract(query_vec, context=None, top_k=2, threshold=0.0)
    with_interest = extractor.extract(
        query_vec,
        context={"interests": ["fitness"]},
        top_k=2,
        threshold=0.0,
    )

    assert baseline[0]["category"] == "travel deals"
    assert with_interest[0]["category"] == "fitness gear"


def test_search_returns_exactly_1000_when_enough_candidates_exist():
    class FakeFaissIndex:
        def __init__(self, ntotal: int):
            self.ntotal = ntotal

        def search(self, vec: np.ndarray, k: int):  # noqa: ARG002
            ids = np.arange(k, dtype=np.int64)
            scores = np.linspace(1.0, 0.0, num=k, dtype=np.float32)
            return scores.reshape(1, -1), ids.reshape(1, -1)

    index = CampaignIndex.__new__(CampaignIndex)
    index._index = FakeFaissIndex(ntotal=2500)

    ids, scores = index.search(np.ones(6, dtype=np.float32), top_k=1000)

    assert len(ids) == 1000
    assert len(scores) == 1000


def test_search_filters_out_invalid_faiss_ids():
    class FakeFaissIndex:
        ntotal = 4

        def search(self, vec: np.ndarray, k: int):  # noqa: ARG002
            ids = np.array([0, 1, -1, 2], dtype=np.int64)
            scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
            return scores.reshape(1, -1), ids.reshape(1, -1)

    index = CampaignIndex.__new__(CampaignIndex)
    index._index = FakeFaissIndex()

    ids, scores = index.search(np.ones(6, dtype=np.float32), top_k=4)

    assert ids.tolist() == [0, 1, 2]
    assert scores.tolist() == pytest.approx([0.9, 0.8, 0.6], abs=1e-6)


def test_category_extract_does_not_call_embedding_model_at_runtime(
    tmp_path,
    deterministic_embedding_service,
):
    taxonomy_path = _write_taxonomy(
        tmp_path,
        [
            {"name": "running shoes", "description": "marathon running footwear"},
            {"name": "laptop deals", "description": "student laptop computer discounts"},
        ],
    )

    before_init = deterministic_embedding_service.encode_calls
    extractor = CategoryExtractor(
        deterministic_embedding_service,
        taxonomy_path=taxonomy_path,
    )
    after_init = deterministic_embedding_service.encode_calls
    assert after_init > before_init

    query_vec = deterministic_embedding_service.embed("running shoes")
    before_extract = deterministic_embedding_service.encode_calls
    _ = extractor.extract(query_vec, context={"interests": ["running"]}, threshold=0.0)
    after_extract = deterministic_embedding_service.encode_calls

    assert after_extract == before_extract
