"""Scoring component implementations used by reranking.

Responsibility:
    Define pluggable scoring components that transform a `ScoreContext` into a
    `ScoreDelta` describing one interpretable boost/penalty contribution.

Inputs:
    - campaign/query context (`ScoreContext`)
    - scoring weights (`ScoreConfig`)

Outputs:
    - per-component `ScoreDelta` values, optionally batched

Invariant:
    Components are side-effect free and only compute score deltas.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from app.services.scoring.score_config import ScoreConfig
from app.services.scoring.score_context import ScoreContext, ScoreDelta

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from app.services.query_parser import ParsedQuery
    from app.services.retrieval import CampaignIndex

_WORD_RE = re.compile(r"[a-z0-9]+")


class ScoreComponent(ABC):
    """Base class for all scoring components."""

    @abstractmethod
    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta: ...

    def apply_batch(
        self,
        _candidate_ids: NDArray[np.int64],
        _parsed_query: ParsedQuery,
        _index: CampaignIndex,
        _config: ScoreConfig,
    ) -> list[ScoreDelta] | None:
        """Vectorized batch scoring over all candidates.

        Return a list of ScoreDelta (one per candidate) for components that
        support batch evaluation, or None to fall back to the per-element
        apply() loop in the reranker.
        """
        return None


# ---------------------------------------------------------------------------
# Migrated components (from ranking.py)
# ---------------------------------------------------------------------------


class LocationBoostComponent(ScoreComponent):
    """Boost when user location is in campaign's geo include set."""

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        user_loc = ctx.parsed_query.user_location
        if not user_loc:
            return ScoreDelta("location", 0.0)

        geo_inc = ctx.index.geo_include[ctx.campaign_idx]
        if user_loc in geo_inc:
            boost = config.get("geo_demo", "location_match")
            return ScoreDelta("location", boost, f"location_match: {user_loc}")
        return ScoreDelta("location", 0.0)

    def apply_batch(
        self,
        candidate_ids: NDArray[np.int64],
        parsed_query: ParsedQuery,
        index: CampaignIndex,
        config: ScoreConfig,
    ) -> list[ScoreDelta] | None:
        user_loc = parsed_query.user_location
        if not user_loc:
            return [ScoreDelta("location", 0.0)] * len(candidate_ids)
        boost = config.get("geo_demo", "location_match")
        result: list[ScoreDelta] = []
        for idx in candidate_ids:
            if user_loc in index.geo_include[int(idx)]:
                result.append(ScoreDelta("location", boost, f"location_match: {user_loc}"))
            else:
                result.append(ScoreDelta("location", 0.0))
        return result


class DemographicBoostComponent(ScoreComponent):
    """Boost when user age + gender match campaign targeting; penalty on mismatch."""

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        pq = ctx.parsed_query
        idx = ctx.campaign_idx
        index = ctx.index
        delta = 0.0
        reasons = []

        # Age match / mismatch
        if pq.user_age is not None:
            age_min = int(index.ages_min[idx])
            age_max = int(index.ages_max[idx])
            if age_min <= pq.user_age <= age_max:
                delta += config.get("geo_demo", "age_match")
                reasons.append(f"age {pq.user_age} in [{age_min},{age_max}]")
            else:
                delta += config.get("geo_demo", "age_mismatch")
                reasons.append(f"age {pq.user_age} outside [{age_min},{age_max}]")

        # Gender match / mismatch — bitmask: bit0=all, bit1=male, bit2=female
        if pq.user_gender:
            gender_mask = int(index.genders[idx])
            gender_bits = {"all": 1, "male": 2, "female": 4}
            user_bit = gender_bits.get(pq.user_gender.lower(), 0)
            if (gender_mask & 1) or (gender_mask & user_bit):
                delta += config.get("geo_demo", "gender_match")
                reasons.append("gender_match")
            else:
                delta += config.get("geo_demo", "gender_mismatch")
                reasons.append("gender_mismatch")

        reason = "; ".join(reasons) if reasons else ""
        return ScoreDelta("demographic", delta, reason)

    def apply_batch(
        self,
        candidate_ids: NDArray[np.int64],
        parsed_query: ParsedQuery,
        index: CampaignIndex,
        config: ScoreConfig,
    ) -> list[ScoreDelta] | None:
        pq = parsed_query
        n = len(candidate_ids)
        delta_arr = np.zeros(n, dtype=np.float64)
        reasons: list[list[str]] = [[] for _ in range(n)]

        if pq.user_age is not None:
            ages_min = index.ages_min[candidate_ids].astype(np.int16)
            ages_max = index.ages_max[candidate_ids].astype(np.int16)
            age_match = (ages_min <= pq.user_age) & (pq.user_age <= ages_max)
            delta_arr += np.where(
                age_match,
                config.get("geo_demo", "age_match"),
                config.get("geo_demo", "age_mismatch"),
            )
            for i in range(n):
                amin, amax = int(ages_min[i]), int(ages_max[i])
                if age_match[i]:
                    reasons[i].append(f"age {pq.user_age} in [{amin},{amax}]")
                else:
                    reasons[i].append(f"age {pq.user_age} outside [{amin},{amax}]")

        if pq.user_gender:
            user_bit = {"all": 1, "male": 2, "female": 4}.get(pq.user_gender.lower(), 0)
            if user_bit:
                genders = index.genders[candidate_ids]
                gender_match = ((genders & 1) != 0) | ((genders & user_bit) != 0)
                delta_arr += np.where(
                    gender_match,
                    config.get("geo_demo", "gender_match"),
                    config.get("geo_demo", "gender_mismatch"),
                )
                for i in range(n):
                    reasons[i].append("gender_match" if gender_match[i] else "gender_mismatch")

        return [
            ScoreDelta("demographic", float(delta_arr[i]), "; ".join(reasons[i]))
            for i in range(n)
        ]


class InterestBoostComponent(ScoreComponent):
    """Boost based on overlap between user interests and campaign interest tokens.

    Uses weighted overlap ratio scaled by the max interest boost.
    """

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        user_interests = ctx.parsed_query.user_interests
        if not user_interests:
            return ScoreDelta("interest", 0.0)

        # Tokenize user interests
        user_tokens = set()
        for term in user_interests:
            user_tokens.update(_WORD_RE.findall(str(term).lower()))

        if not user_tokens:
            return ScoreDelta("interest", 0.0)

        campaign_tokens = ctx.index.interests[ctx.campaign_idx]
        overlap = user_tokens & campaign_tokens
        if not overlap:
            return ScoreDelta("interest", 0.0)

        # Scale by overlap ratio
        max_boost = config.get("geo_demo", "interest_overlap")
        ratio = len(overlap) / len(user_tokens)
        delta = max_boost * min(ratio, 1.0)
        return ScoreDelta("interest", delta, f"overlap: {overlap}")


# ---------------------------------------------------------------------------
# New V1 components
# ---------------------------------------------------------------------------


class TaxonomyAlignmentComponent(ScoreComponent):
    """Boost for taxonomy alignment with query; penalty when query clearly diverges."""

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        tax = ctx.index.taxonomy[ctx.campaign_idx]
        query_lower = ctx.parsed_query.raw_query.lower()
        query_tokens = ctx.parsed_query.query_tokens
        delta = 0.0
        reasons = []
        any_match = False

        # Collect all taxonomy tokens for mismatch check
        all_taxonomy_tokens: set[str] = set()

        for level, config_key in [
            ("category", "category_match"),
            ("subcategory", "subcategory_match"),
            ("leaf", "leaf_match"),
        ]:
            val = tax.get(level, "")
            if not val:
                continue
            val_lower = val.lower()
            val_tokens = set(_WORD_RE.findall(val_lower))
            all_taxonomy_tokens |= val_tokens

            # Exact phrase match
            if val_lower in query_lower:
                if len(val_tokens) >= 2:
                    boost = config.get("taxonomy", config_key)
                else:
                    boost = config.get("taxonomy", config_key) * 0.5
                delta += boost
                reasons.append(f"{level}={val}")
                any_match = True
            else:
                # Token overlap fallback: require at least 2 matching tokens
                overlap = val_tokens & query_tokens
                if len(overlap) >= 2:
                    boost = config.get("taxonomy", config_key) * 0.5
                    delta += boost
                    reasons.append(f"{level}_partial={val}")
                    any_match = True

        # Penalty when the campaign has a specific taxonomy but the query shares
        # zero tokens with it. Only fires when the campaign has at least 2 taxonomy
        # levels (to avoid penalising very broadly-defined campaigns) and the query
        # itself has meaningful content.
        if (
            not any_match
            and len(all_taxonomy_tokens) >= 3
            and len(query_tokens) >= 2
            and not (query_tokens & all_taxonomy_tokens)
        ):
            delta += config.get("taxonomy", "mismatch_penalty")
            reasons.append(f"taxonomy_mismatch: query has no overlap with {all_taxonomy_tokens}")

        reason = "; ".join(reasons) if reasons else ""
        return ScoreDelta("taxonomy_alignment", delta, reason)


class BrandAlignmentComponent(ScoreComponent):
    """Boost when query brand terms match campaign brand; penalty when they don't."""

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        brand_terms = ctx.parsed_query.brand_terms
        if not brand_terms:
            return ScoreDelta("brand_alignment", 0.0)

        brand = ctx.index.brands[ctx.campaign_idx]
        brand_name = (brand.get("name") or "").lower()
        product_line = (brand.get("product_line") or "").lower()
        delta = 0.0
        reasons = []
        matched = False

        for term in brand_terms:
            term_lower = term.lower()
            if term_lower == brand_name or term_lower in brand_name:
                delta += config.get("brand", "name_match")
                reasons.append(f"brand_name={brand_name}")
                matched = True
            if product_line and (term_lower == product_line or term_lower in product_line):
                delta += config.get("brand", "product_line_match")
                reasons.append(f"product_line={product_line}")
                matched = True

        # User asked for a specific brand and this campaign is a different one
        if not matched:
            delta += config.get("brand", "mismatch_penalty")
            reasons.append(f"brand_mismatch: query={brand_terms}, campaign={brand_name or 'unknown'}")

        reason = "; ".join(reasons) if reasons else ""
        return ScoreDelta("brand_alignment", delta, reason)


class AttributeAlignmentComponent(ScoreComponent):
    """Boost/penalty for attribute match/contradiction with query intents."""

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        intents = ctx.parsed_query.attribute_intents
        if not intents:
            return ScoreDelta("attribute_alignment", 0.0)

        attrs = ctx.index.attributes[ctx.campaign_idx]
        if not attrs:
            return ScoreDelta("attribute_alignment", 0.0)

        delta = 0.0
        reasons = []
        match_boost = config.get("attributes", "match_boost")
        contra_penalty = config.get("attributes", "contradiction_penalty")

        for key, expected_val in intents.items():
            if key == "price_intent":
                continue  # Handled by PriceAlignmentComponent
            campaign_val = attrs.get(key)
            if campaign_val is None:
                continue

            expected_str = expected_val.lower()

            # Handle list-type attribute values (e.g. distance_focus: ["5k", "marathon"]).
            # Check if the expected value appears in the list rather than comparing
            # the raw list repr as a string.
            if isinstance(campaign_val, list):
                vals_lower = [str(v).lower() for v in campaign_val]
                if expected_str in vals_lower or any(expected_str in v for v in vals_lower):
                    delta += match_boost
                    reasons.append(f"match:{key} in {campaign_val}")
                # No penalty for list mismatches: the campaign simply may not cover
                # that specific value rather than actively contradict the query.
                continue

            # Scalar comparison
            campaign_str = str(campaign_val).lower()
            if campaign_str == expected_str or expected_str == "true" and campaign_val is True:
                delta += match_boost
                reasons.append(f"match:{key}={campaign_val}")
            elif expected_str == "true" and campaign_val is False:
                delta += contra_penalty
                reasons.append(f"contra:{key}=false")
            elif campaign_str != expected_str:
                delta += contra_penalty
                reasons.append(f"contra:{key}={campaign_val}!={expected_val}")

        reason = "; ".join(reasons) if reasons else ""
        return ScoreDelta("attribute_alignment", delta, reason)


class KeywordOverlapComponent(ScoreComponent):
    """Boost based on query token overlap with campaign keyword buckets."""

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        query_tokens = ctx.parsed_query.query_tokens
        if not query_tokens:
            return ScoreDelta("keyword_overlap", 0.0)

        kw_sets = ctx.index.keyword_sets[ctx.campaign_idx]
        delta = 0.0
        reasons = []

        bucket_config = {
            "brand_terms": "brand_term_boost",
            "model_terms": "model_term_boost",
            "category_terms": "category_term_boost",
            "feature_terms": "feature_term_boost",
            "benefit_terms": "benefit_term_boost",
        }

        for bucket, config_key in bucket_config.items():
            campaign_terms = kw_sets.get(bucket, frozenset())
            if not campaign_terms:
                continue

            # Tokenize campaign terms and find overlap with query tokens
            campaign_tokens = set()
            for term in campaign_terms:
                campaign_tokens.update(_WORD_RE.findall(term.lower()) if isinstance(term, str) else [])

            overlap = query_tokens & campaign_tokens
            if overlap:
                max_boost = config.get("keywords", config_key)
                ratio = len(overlap) / max(len(campaign_tokens), 1)
                delta += max_boost * min(ratio, 1.0)
                reasons.append(f"{bucket}: {overlap}")

        reason = "; ".join(reasons) if reasons else ""
        return ScoreDelta("keyword_overlap", delta, reason)


class PriceAlignmentComponent(ScoreComponent):
    """Boost/penalty based on price alignment with query intent."""

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        pq = ctx.parsed_query
        campaign_price = float(ctx.index.bids[ctx.campaign_idx])
        if campaign_price <= 0:
            return ScoreDelta("price_alignment", 0.0)

        delta = 0.0
        reasons = []

        # Explicit price max
        if pq.price_max is not None:
            if campaign_price <= pq.price_max:
                delta += config.get("pricing", "budget_match_boost")
                reasons.append(f"under_budget: ${campaign_price} <= ${pq.price_max}")
            else:
                # Penalty scaled by how far over budget
                overage_ratio = (campaign_price - pq.price_max) / pq.price_max
                penalty = config.get("pricing", "over_budget_penalty") * min(overage_ratio, 2.0)
                delta += penalty
                reasons.append(f"over_budget: ${campaign_price} > ${pq.price_max}")

        # Budget/premium intent (no explicit price)
        price_intent = pq.attribute_intents.get("price_intent")
        if price_intent == "budget" and pq.price_max is None:
            # Lower price campaigns get a small boost
            # Use bids array as proxy for price bucket
            delta += config.get("pricing", "budget_match_boost") * 0.5
            reasons.append("budget_intent")
        elif price_intent == "premium" and pq.price_max is None:
            delta += config.get("pricing", "budget_match_boost") * 0.5
            reasons.append("premium_intent")

        reason = "; ".join(reasons) if reasons else ""
        return ScoreDelta("price_alignment", delta, reason)


class CategoryAlignmentComponent(ScoreComponent):
    """Boost/penalty based on semantic category extracted from the query.

    Uses the output of CategoryExtractor (which runs in parallel at query time)
    rather than raw token matching. This handles synonyms and paraphrases that
    TaxonomyAlignmentComponent misses (e.g. "sneakers" → category "athletic footwear").

    Boost is weighted by the extractor's confidence score so high-confidence
    category matches contribute more than uncertain ones.
    """

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        if not ctx.extracted_categories:
            return ScoreDelta("category_alignment", 0.0)

        tax = ctx.index.taxonomy[ctx.campaign_idx]

        # Build a token set from all taxonomy levels of this campaign
        campaign_tax_tokens: set[str] = set()
        campaign_tax_strings: list[str] = []
        for level in ("category", "subcategory", "leaf"):
            val = tax.get(level, "") or ""
            if val:
                campaign_tax_strings.append(val.lower())
                campaign_tax_tokens.update(_WORD_RE.findall(val.lower()))

        if not campaign_tax_tokens:
            return ScoreDelta("category_alignment", 0.0)

        match_weight = config.get("category_alignment", "match_weight")
        mismatch_penalty = config.get("category_alignment", "mismatch_penalty")

        # Use the single best-matching category rather than summing all matches.
        # Summing would inflate scores proportionally to how many synonym categories
        # the extractor returned (e.g. "running shoes" + "trail running shoes" +
        # "men's sneakers" all matching the same campaign), which is a quirk of the
        # extractor output rather than a genuine multi-signal boost.
        best_boost = 0.0
        best_reason = ""
        any_match = False

        for cat_entry in ctx.extracted_categories:
            cat_name = cat_entry["category"].lower()
            cat_score = float(cat_entry["score"])  # extractor confidence [0, 1]
            cat_tokens = set(_WORD_RE.findall(cat_name))

            if cat_tokens & campaign_tax_tokens or any(cat_name in s for s in campaign_tax_strings):
                boost = match_weight * cat_score
                if boost > best_boost:
                    best_boost = boost
                    best_reason = f"cat_match={cat_entry['category']}(conf={cat_score:.2f})"
                any_match = True

        delta = best_boost
        reasons = [best_reason] if best_reason else []

        # Penalty: top extracted category is high-confidence but campaign taxonomy
        # has zero overlap with it. Only fires if top category is confident (>0.5)
        # to avoid penalising on uncertain extractions.
        if not any_match:
            top = ctx.extracted_categories[0]
            if float(top["score"]) > 0.5:
                delta += mismatch_penalty
                reasons.append(
                    f"cat_mismatch: query={top['category']}(conf={top['score']:.2f})"
                    f", campaign_tax={campaign_tax_strings}"
                )

        reason = "; ".join(reasons) if reasons else ""
        return ScoreDelta("category_alignment", delta, reason)


class ContradictionPenaltyComponent(ScoreComponent):
    """Penalty when query tokens match campaign negative keywords."""

    def apply(self, ctx: ScoreContext, config: ScoreConfig) -> ScoreDelta:
        query_tokens = ctx.parsed_query.query_tokens
        if not query_tokens:
            return ScoreDelta("contradiction", 0.0)

        neg_kw = ctx.index.negative_keywords[ctx.campaign_idx]
        if not neg_kw:
            return ScoreDelta("contradiction", 0.0)

        # Match negative keywords as complete phrases: every token in the phrase must
        # appear in the query. This prevents "dress shoes" from firing on a "running
        # shoes" query (no "dress" token present), while a keyword like "basketball"
        # would correctly fire on a "basketball shoes" query.
        matched_phrases = []
        for term in neg_kw:
            if not isinstance(term, str):
                continue
            phrase_tokens = set(_WORD_RE.findall(term.lower()))
            if phrase_tokens and phrase_tokens.issubset(query_tokens):
                matched_phrases.append(term)

        if not matched_phrases:
            return ScoreDelta("contradiction", 0.0)

        penalty_per = config.get("contradiction", "negative_keyword_penalty")
        total_penalty = penalty_per * len(matched_phrases)
        return ScoreDelta("contradiction", total_penalty, f"conflicts: {matched_phrases}")
