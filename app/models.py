"""Typed request/response schemas for the retrieval API.

Inputs:
    - RetrieveRequest from API clients

Outputs:
    - RetrieveResponse serialized by FastAPI

Invariant:
    Field constraints here define the public API contract independently of
    internal scoring/filtering implementation details.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

GENDER_VALUES = Literal["male", "female"]

class UserContext(BaseModel):
    """Optional user context passed with every query.

    Common fields are typed explicitly; any extra keys are passed through
    transparently (device info, session flags, etc.).
    """

    model_config = ConfigDict(extra="allow")

    gender: GENDER_VALUES | None = None
    age: int | None = None
    location: str | None = None
    interests: list[str] = Field(default_factory=list)
    device: str | None = None


class RetrieveRequest(BaseModel):
    """Request payload for ``POST /api/retrieve``."""

    query: str = Field(..., min_length=1, max_length=200, description="User's natural language query")
    context: UserContext | None = Field(None, description="Optional user context")


class CampaignResult(BaseModel):
    """One campaign in the response. Mirrors the response-surface fields from
    campaigns_meta.json plus the relevance_score and final_score computed by the ranker."""

    campaign_id: str
    relevance_score: float
    internal_score: float
    title: str
    advertiser: str
    category: str
    vertical: str
    landing_url: str
    creative_format: str
    budget_remaining: float


class ErrorDetail(BaseModel):
    """Single validation error item."""

    loc: list[str | int] = Field(default_factory=list, description="Location of the error in the request body")
    msg: str = Field(..., description="Human-readable error message")
    type: str = Field(..., description="Error type identifier")


class ErrorResponse(BaseModel):
    """Standardised error envelope returned on HTTP errors (4xx/5xx)."""

    status_code: int = Field(..., description="HTTP status code")
    error: str = Field(..., description="Short error category label")
    message: str = Field(..., description="Human-readable summary of the error")
    details: list[ErrorDetail] = Field(default_factory=list, description="Per-field validation errors (populated on 422)")


class RetrieveResponse(BaseModel):
    """Response payload returned by ``POST /api/retrieve``.

    Invariants:
        - `ad_eligibility` is bounded to [0.0, 1.0].
        - `campaigns` entries mirror response-surface campaign metadata plus score.
    """

    ad_eligibility: float = Field(..., ge=0.0, le=1.0)
    extracted_categories: list[str]
    campaigns: list[CampaignResult]
    latency_ms: float
    metadata: dict
