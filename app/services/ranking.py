"""Context-aware reranking on top of FAISS retrieval.

Responsibility:
    Apply composable score components to FAISS candidates, then return campaigns
    sorted by final relevance.

Key design:
    - Spec-gated deltas are scaled by a coverage ratio so partial matches do not
      over-rank campaigns when users specify many constraints.
    - Always-on deltas (taxonomy/category/keyword/contradiction) apply at full
      weight because they remain meaningful for both broad and specific queries.

Score interpretation:
    Relevance is normalized by a query-specific potential maximum, then clamped
    to [0, 1], so API consumers receive bounded probability-like scores.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.services.query_parser import ParsedQuery
from app.services.retrieval import CampaignIndex
from app.services.scoring.builder import ScoreBuilder
from app.services.scoring.score_config import ScoreConfig
from app.services.scoring.score_context import ScoreContext, ScoreDelta

# Components whose deltas are gated by the coverage ratio.
# Each maps to a query-level spec that must be explicitly stated for the
# component to be "active". Always-on components (taxonomy, category,
# keywords, contradiction) are NOT in this set.
_SPEC_GATED: frozenset[str] = frozenset({
    "location",
    "demographic",
    "interest",
    "brand_alignment",
    "attribute_alignment",
    "price_alignment",
})


def _active_spec_count(pq: ParsedQuery) -> int:
    """Count how many spec-gated dimensions are explicitly stated in the query."""
    count = 0
    if pq.user_location:
        count += 1
    if pq.user_age is not None or pq.user_gender:
        count += 1
    if pq.user_interests:
        count += 1
    if pq.brand_terms:
        count += 1
    if any(k != "price_intent" for k in pq.attribute_intents):
        count += 1
    if (pq.price_max is not None
            or pq.price_min is not None
            or "price_intent" in pq.attribute_intents):
        count += 1
    return count


def _query_potential_max(
    pq: ParsedQuery,
    config: ScoreConfig,
    faiss_base_max: float,
) -> float:
    """Compute the theoretical maximum score achievable for this query.

    Sums the best possible FAISS base score, the maximum positive delta from
    each always-on component, and the maximum positive delta from each
    spec-gated component that the query explicitly states. Used as the
    denominator for query-adaptive score normalisation so that
    relevance_score = actual_score / potential_max ∈ [0, 1].
    """
    # --- Always-on potential ---
    taxonomy_max = (
        config.get("taxonomy", "category_match")
        + config.get("taxonomy", "subcategory_match")
        + config.get("taxonomy", "leaf_match")
    )
    cat_align_max = config.get("category_alignment", "match_weight")
    keyword_max = (
        config.get("keywords", "brand_term_boost")
        + config.get("keywords", "model_term_boost")
        + config.get("keywords", "category_term_boost")
        + config.get("keywords", "feature_term_boost")
        + config.get("keywords", "benefit_term_boost")
    )
    always_max = taxonomy_max + cat_align_max + keyword_max

    # --- Spec-gated potential (only for dimensions active in this query) ---
    spec_max = 0.0

    if pq.user_location:
        spec_max += config.get("geo_demo", "location_match")
    if pq.user_age is not None:
        spec_max += config.get("geo_demo", "age_match")
    if pq.user_gender:
        spec_max += config.get("geo_demo", "gender_match")
    if pq.user_interests:
        spec_max += config.get("geo_demo", "interest_overlap")
    if pq.brand_terms:
        spec_max += config.get("brand", "name_match")
        spec_max += config.get("brand", "product_line_match")

    non_price_attrs = [k for k in pq.attribute_intents if k != "price_intent"]
    if non_price_attrs:
        spec_max += len(non_price_attrs) * config.get("attributes", "match_boost")

    if (
        pq.price_max is not None
        or pq.price_min is not None
        or "price_intent" in pq.attribute_intents
    ):
        spec_max += config.get("pricing", "budget_match_boost")

    return faiss_base_max + spec_max + always_max


class ContextReranker:
    """Re-ranks FAISS candidates using the scoring pipeline."""

    def __init__(self, index: CampaignIndex, config: ScoreConfig) -> None:
        self._index = index
        self._config = config

        # Build pipeline once
        self._pipeline = (
            ScoreBuilder()
            .with_location_boost()
            .with_demographic_boost()
            .with_interest_alignment()
            .with_taxonomy_alignment()
            .with_category_alignment()
            .with_brand_alignment()
            .with_attribute_alignment()
            .with_keyword_overlap()
            .with_price_alignment()
            .with_contradiction_penalties()
            .build()
        )

    def rerank(
        self,
        candidate_ids: NDArray[np.int64],
        faiss_scores: NDArray[np.float32],
        parsed_query: ParsedQuery,
        top_k: int = 1000,
        debug_top_n: int = 5,
        extracted_categories: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Apply scoring pipeline, re-sort, and build response dicts.

        Args:
            candidate_ids:         FAISS row indices, shape (k,).
            faiss_scores:          Raw cosine similarity scores, shape (k,).
            parsed_query:          Parsed query with extracted signals.
            top_k:                 Maximum results to return.
            debug_top_n:           How many top candidates to include full score breakdowns for.
            extracted_categories:  Output of CategoryExtractor — list of
                                   {"category": str, "score": float} dicts, sorted by score desc.

        Returns:
            (results, score_breakdowns) where:
            - results: list of campaign dicts with relevance_score
            - score_breakdowns: list of per-component deltas for top debug_top_n candidates

        Invariants:
            - `results[i]["relevance_score"]`.
            - Candidate metadata is copied from index meta by FAISS row id.
        """
        if len(candidate_ids) == 0:
            return [], []

        n_candidates = len(candidate_ids)
        all_deltas: list[list[ScoreDelta]] = [[] for _ in range(n_candidates)]
        categories = extracted_categories or []

        # --- Phase 1: collect all component deltas (don't accumulate into scores yet) ---
        # Components that implement apply_batch() run as a single vectorized call;
        # the rest fall back to the per-element loop.
        for component in self._pipeline:
            batch = component.apply_batch(candidate_ids, parsed_query, self._index, self._config)
            if batch is not None:
                for rank_pos, delta in enumerate(batch):
                    all_deltas[rank_pos].append(delta)
            else:
                for rank_pos in range(n_candidates):
                    idx = int(candidate_ids[rank_pos])
                    ctx = ScoreContext(
                        parsed_query=parsed_query,
                        campaign_idx=idx,
                        base_score=float(faiss_scores[rank_pos]),
                        index=self._index,
                        extracted_categories=categories,
                    )
                    all_deltas[rank_pos].append(component.apply(ctx, self._config))

        # --- Phase 2: coverage-ratio scoring ---
        # active_params: how many spec-gated dimensions the query explicitly states.
        # coverage_ratio per candidate: matched_specs / active_params.
        # Spec deltas are scaled by this ratio; always-on deltas apply at full weight.
        active_params = _active_spec_count(parsed_query)

        scores = np.empty(n_candidates, dtype=np.float64)
        coverage_ratios = np.empty(n_candidates, dtype=np.float64)

        for i in range(n_candidates):
            deltas = all_deltas[i]
            spec_delta   = sum(d.delta for d in deltas if d.component in _SPEC_GATED)
            always_delta = sum(d.delta for d in deltas if d.component not in _SPEC_GATED)

            if active_params > 0:
                matched = sum(1 for d in deltas if d.component in _SPEC_GATED and d.delta > 0)
                ratio = matched / active_params
            else:
                ratio = 1.0

            coverage_ratios[i] = ratio
            scores[i] = float(faiss_scores[i]) + spec_delta * ratio + always_delta

        # --- Phase 3: sort, normalise, and build response ---
        k = min(top_k, n_candidates)
        top_order = np.argsort(scores)[::-1][:k]

        # Query-adaptive potential max: the highest score any campaign could
        # theoretically achieve for this specific query.
        q_max = _query_potential_max(parsed_query, self._config, float(faiss_scores.max()))

        results: list[dict] = []
        meta = self._index.meta

        for i, rank_pos in enumerate(top_order):
            row_id = int(candidate_ids[rank_pos])
            result = dict(meta[row_id])
            final_score = float(scores[rank_pos])
            result["internal_score"] = round(final_score, 6)
            result["relevance_score"] = round(
                max(min(final_score / q_max, 1.0), 0.0), 6
            )
            results.append(result)

        # Debug breakdowns for top-N
        score_breakdowns: list[dict] = []
        for i, rank_pos in enumerate(top_order[:debug_top_n]):
            row_id = int(candidate_ids[rank_pos])
            breakdown = {
                "campaign_id": meta[row_id]["campaign_id"],
                "base_score": round(float(faiss_scores[rank_pos]), 6),
                "final_score": round(float(scores[rank_pos]), 6),
                "active_params": active_params,
                "coverage_ratio": round(float(coverage_ratios[rank_pos]), 4),
                "query_potential_max": round(q_max, 6),
                "deltas": [
                    {
                        "component": d.component,
                        "delta": round(d.delta, 6),
                        "reason": d.reason,
                    }
                    for d in all_deltas[rank_pos]
                    if d.delta != 0.0
                ],
            }
            score_breakdowns.append(breakdown)

        return results, score_breakdowns
