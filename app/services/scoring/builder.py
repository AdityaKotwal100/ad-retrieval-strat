"""Fluent assembly of scoring component pipelines.

Responsibility:
    Provide an explicit, readable pipeline construction order for reranking.

Input:
    Builder method calls selecting enabled components.

Output:
    Ordered list of `ScoreComponent` instances consumed by `ContextReranker`.

Invariant:
    `build()` preserves insertion order so scoring behavior remains stable across
    deployments that use the same builder sequence.
"""

from __future__ import annotations

from app.services.scoring.components import (
    AttributeAlignmentComponent,
    BrandAlignmentComponent,
    CategoryAlignmentComponent,
    ContradictionPenaltyComponent,
    DemographicBoostComponent,
    InterestBoostComponent,
    KeywordOverlapComponent,
    LocationBoostComponent,
    PriceAlignmentComponent,
    ScoreComponent,
    TaxonomyAlignmentComponent,
)


class ScoreBuilder:
    """Fluent builder for assembling a scoring pipeline."""

    def __init__(self) -> None:
        """Initialize an empty component sequence."""
        self._components: list[ScoreComponent] = []

    def with_location_boost(self) -> ScoreBuilder:
        """Append location matching component and return builder."""
        self._components.append(LocationBoostComponent())
        return self

    def with_demographic_boost(self) -> ScoreBuilder:
        """Append age/gender component and return builder."""
        self._components.append(DemographicBoostComponent())
        return self

    def with_interest_alignment(self) -> ScoreBuilder:
        """Append interest-overlap component and return builder."""
        self._components.append(InterestBoostComponent())
        return self

    def with_taxonomy_alignment(self) -> ScoreBuilder:
        """Append taxonomy alignment component and return builder."""
        self._components.append(TaxonomyAlignmentComponent())
        return self

    def with_brand_alignment(self) -> ScoreBuilder:
        """Append brand alignment component and return builder."""
        self._components.append(BrandAlignmentComponent())
        return self

    def with_attribute_alignment(self) -> ScoreBuilder:
        """Append attribute alignment component and return builder."""
        self._components.append(AttributeAlignmentComponent())
        return self

    def with_keyword_overlap(self) -> ScoreBuilder:
        """Append keyword-overlap component and return builder."""
        self._components.append(KeywordOverlapComponent())
        return self

    def with_price_alignment(self) -> ScoreBuilder:
        """Append price-alignment component and return builder."""
        self._components.append(PriceAlignmentComponent())
        return self

    def with_category_alignment(self) -> ScoreBuilder:
        """Append semantic category-alignment component and return builder."""
        self._components.append(CategoryAlignmentComponent())
        return self

    def with_contradiction_penalties(self) -> ScoreBuilder:
        """Append contradiction penalty component and return builder."""
        self._components.append(ContradictionPenaltyComponent())
        return self

    def build(self) -> list[ScoreComponent]:
        """Return an ordered snapshot of configured components."""
        return list(self._components)
