"""Post-retrieval hard filtering before ranking.

Responsibility:
    Remove candidates that are not eligible for serving under explicit targeting
    or safety constraints. This runs after FAISS retrieval and before scoring.

Inputs:
    - candidate_ids/faiss_scores from retrieval
    - ParsedQuery containing normalized user context and parsed query tokens

Outputs:
    - surviving IDs/scores plus drop counts by reason

Invariants:
    - Filtering is monotonic: once dropped by a hard constraint, a campaign is
      not re-added in the same pass.
    - Surviving scores preserve FAISS order/values until the ranking stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.services.query_parser import ParsedQuery
from app.services.retrieval import CampaignIndex

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class FilterResult:
    """Structured output of eligibility filtering.

    Attributes:
        surviving_ids: Candidate row IDs that passed all hard filters.
        surviving_scores: FAISS scores aligned 1:1 with surviving_ids.
        drop_counts: Per-filter drop counts for observability/debugging.
        total_dropped: Total number of dropped candidates.
    """

    surviving_ids: NDArray[np.int64]
    surviving_scores: NDArray[np.float32]
    drop_counts: dict[str, int] = field(default_factory=dict)
    total_dropped: int = 0


class CampaignEligibilityFilter:
    """Hard eligibility filters applied post-retrieval, pre-ranking."""

    def __init__(self, index: CampaignIndex) -> None:
        self._index = index

    def filter(
        self,
        candidate_ids: NDArray[np.int64],
        faiss_scores: NDArray[np.float32],
        parsed_query: ParsedQuery,
    ) -> FilterResult:
        """Apply hard eligibility filters to a candidate set.

        Args:
            candidate_ids: FAISS row IDs aligned with `faiss_scores`.
            faiss_scores: Retrieval scores aligned with `candidate_ids`.
            parsed_query: Parsed query and normalized user context signals.

        Returns:
            FilterResult with survivors and filter-level drop telemetry.

        Invariants:
            - Output arrays remain aligned and order-preserving for survivors.
            - No ranking boosts/penalties are applied here; this stage only drops.
        """
        if len(candidate_ids) == 0:
            return FilterResult(
                surviving_ids=candidate_ids,
                surviving_scores=faiss_scores,
            )

        # Start with all candidates passing
        mask = np.ones(len(candidate_ids), dtype=bool)
        drop_counts: dict[str, int] = {}

        # 1. Geo include filter
        if parsed_query.user_location:
            loc = parsed_query.user_location
            geo_mask = np.array(
                [
                    len(self._index.geo_include[int(i)]) == 0  # national campaign
                    or loc in self._index.geo_include[int(i)]
                    for i in candidate_ids
                ],
                dtype=bool,
            )
            drops = int(np.sum(mask & ~geo_mask))
            if drops > 0:
                drop_counts["geo_include"] = drops
                mask &= geo_mask

        # 2. Geo exclude filter
        if parsed_query.user_location:
            loc = parsed_query.user_location
            geo_exc_mask = np.array(
                [loc not in self._index.geo_exclude[int(i)] for i in candidate_ids],
                dtype=bool,
            )
            drops = int(np.sum(mask & ~geo_exc_mask))
            if drops > 0:
                drop_counts["geo_exclude"] = drops
                mask &= geo_exc_mask

        # 3. Age range filter
        # Allow ±1 year tolerance at boundaries. A 12-year-old asking about
        # games should see teen-targeted gaming ads (min_age=13), not vacation
        # packages. Strict 1-year cutoffs are not meaningful demographic signals.
        if parsed_query.user_age is not None:
            age = parsed_query.user_age
            ages_min = self._index.ages_min[candidate_ids].astype(np.int16)
            ages_max = self._index.ages_max[candidate_ids].astype(np.int16)
            age_mask = (ages_min - 1 <= age) & (age <= ages_max + 1)
            drops = int(np.sum(mask & ~age_mask))
            if drops > 0:
                drop_counts["age_range"] = drops
                mask &= age_mask

        # 4. Gender overlap filter
        if parsed_query.user_gender:
            gender_bits = {"all": 1, "male": 2, "female": 4}
            user_bit = gender_bits.get(parsed_query.user_gender.lower(), 0)
            if user_bit:
                genders = self._index.genders[candidate_ids]
                # Pass if campaign targets "all" (bit0) or user's gender bit
                gender_mask = ((genders & 1) != 0) | ((genders & user_bit) != 0)
                drops = int(np.sum(mask & ~gender_mask))
                if drops > 0:
                    drop_counts["gender"] = drops
                    mask &= gender_mask

        # 5. Category gender mismatch filter
        # Drops campaigns whose taxonomy category is gendered for the opposite gender.
        # The genders bitmask (filter 4) handles audience *targeting*; this handles
        # *product* gender semantics (e.g. "women's shoes" category → exclude for male users).
        # Uses the same first-token heuristic as CategoryExtractor.
        if parsed_query.user_gender:
            gender_str = parsed_query.user_gender.lower()
            if gender_str in {"male", "man", "m"}:
                opp_tokens: frozenset[str] = frozenset({"women", "girls", "ladies"})
            elif gender_str in {"female", "woman", "f"}:
                opp_tokens = frozenset({"men", "boys"})
            else:
                opp_tokens = frozenset()

            if opp_tokens:
                cat_gender_mask = np.ones(len(candidate_ids), dtype=bool)
                for pos, idx in enumerate(candidate_ids):
                    if not mask[pos]:
                        continue
                    tax = self._index.taxonomy[int(idx)]
                    cat = (tax.get("category") or "").lower()
                    first_tok = _WORD_RE.findall(cat)
                    if first_tok and first_tok[0] in opp_tokens:
                        cat_gender_mask[pos] = False
                drops = int(np.sum(mask & ~cat_gender_mask))
                if drops > 0:
                    drop_counts["category_gender_mismatch"] = drops
                    mask &= cat_gender_mask

        # 6. Negative keyword conflict filter
        query_tokens = parsed_query.query_tokens
        if query_tokens:
            # Check only candidates still passing
            passing_indices = np.where(mask)[0]
            neg_mask = np.ones(len(candidate_ids), dtype=bool)
            for pos in passing_indices:
                idx = int(candidate_ids[pos])
                neg_kw = self._index.negative_keywords[idx]
                if neg_kw:
                    for term in neg_kw:
                        if not isinstance(term, str):
                            continue
                        phrase_tokens = set(_WORD_RE.findall(term.lower()))
                        if not phrase_tokens:
                            continue
                        # Single-token keyword: any match in the query is enough.
                        # Multi-token keyword: require ALL phrase tokens to appear
                        # (avoids stopword false-positives, e.g. "shoes" in
                        # "dress shoes" blocking a "running shoes" query).
                        if len(phrase_tokens) == 1:
                            if phrase_tokens & query_tokens:
                                neg_mask[pos] = False
                                break
                        else:
                            if phrase_tokens.issubset(query_tokens):
                                neg_mask[pos] = False
                                break

            drops = int(np.sum(mask & ~neg_mask))
            if drops > 0:
                drop_counts["negative_keyword"] = drops
                mask &= neg_mask

        # 7. Hard price filter (only when explicit price_max in query)
        if parsed_query.price_max is not None:
            # Generous margin: 1.5x the stated max
            hard_limit = parsed_query.price_max * 1.5
            prices = self._index.bids[candidate_ids]
            # Only filter if campaign has a non-zero price
            price_mask = (prices <= hard_limit) | (prices <= 0)
            drops = int(np.sum(mask & ~price_mask))
            if drops > 0:
                drop_counts["price_hard"] = drops
                mask &= price_mask

        surviving_ids = candidate_ids[mask]
        surviving_scores = faiss_scores[mask]
        total_dropped = int(np.sum(~mask))

        return FilterResult(
            surviving_ids=surviving_ids,
            surviving_scores=surviving_scores,
            drop_counts=drop_counts,
            total_dropped=total_dropped,
        )

    def filter_with_refill(
        self,
        candidate_ids: NDArray[np.int64],
        faiss_scores: NDArray[np.float32],
        parsed_query: ParsedQuery,
        index: CampaignIndex,
        query_embedding: NDArray[np.float32],
        target_count: int = 500,
        max_refill_rounds: int = 2,
    ) -> FilterResult:
        """Filter candidates, then optionally refill from deeper FAISS neighbors.

        Refill prevents sparse outputs when strict hard filters remove a large
        share of top neighbors. It only requests deeper neighbors when both:
        1) fewer than 50% of target_count survive, and
        2) more than 50% of the original set was dropped.

        Args:
            candidate_ids: Initial FAISS row IDs.
            faiss_scores: Initial FAISS scores aligned with `candidate_ids`.
            parsed_query: Parsed query and user context.
            index: Campaign index used for additional FAISS searches.
            query_embedding: Unit-normalized query vector for refill searches.
            target_count: Desired approximate number of post-filter survivors.
            max_refill_rounds: Safety bound on additional retrieval passes.

        Returns:
            FilterResult merged across the initial pass and refill rounds.
        """
        result = self.filter(candidate_ids, faiss_scores, parsed_query)

        original_count = len(candidate_ids)
        current_top_k = original_count
        refill_round = 0

        while (
            len(result.surviving_ids) < target_count * 0.5
            and refill_round < max_refill_rounds
            and result.total_dropped > original_count * 0.5
        ):
            refill_round += 1
            current_top_k += 500
            new_ids, new_scores = index.search(query_embedding, top_k=current_top_k)

            # Only consider IDs we haven't already processed
            seen = set(candidate_ids.tolist())
            new_mask = np.array([int(i) not in seen for i in new_ids], dtype=bool)
            if not np.any(new_mask):
                break

            extra_ids = new_ids[new_mask]
            extra_scores = new_scores[new_mask]
            extra_result = self.filter(extra_ids, extra_scores, parsed_query)

            # Merge results
            result.surviving_ids = np.concatenate([result.surviving_ids, extra_result.surviving_ids])
            result.surviving_scores = np.concatenate([result.surviving_scores, extra_result.surviving_scores])
            for reason, count in extra_result.drop_counts.items():
                result.drop_counts[reason] = result.drop_counts.get(reason, 0) + count
            result.total_dropped += extra_result.total_dropped

            # Update tracking
            candidate_ids = np.concatenate([candidate_ids, extra_ids])
            original_count = len(candidate_ids)

        return result
