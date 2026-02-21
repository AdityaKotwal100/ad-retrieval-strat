"""Shared datatypes for scoring component inputs and outputs.

`ScoreContext` carries immutable per-candidate inputs.
`ScoreDelta` captures one component's additive score effect with a reason string
for debugging and observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.query_parser import ParsedQuery
    from app.services.retrieval import CampaignIndex


@dataclass
class ScoreContext:
    """All data a scoring component needs to compute its delta."""
    parsed_query: ParsedQuery
    campaign_idx: int                           # FAISS row index
    base_score: float                           # raw cosine similarity from FAISS
    index: CampaignIndex                        # reference to loaded artifacts
    extracted_categories: list[dict] = field(default_factory=list)  # from CategoryExtractor


@dataclass
class ScoreDelta:
    """One score adjustment from a single component."""
    component: str          # e.g., "taxonomy_alignment"
    delta: float            # positive = boost, negative = penalty
    reason: str = ""        # e.g., "category_match: running shoes"
