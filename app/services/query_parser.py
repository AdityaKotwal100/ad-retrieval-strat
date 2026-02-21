"""Query parsing for retrieval, filtering, and ranking signals.

Responsibility:
    Convert raw query text plus optional user context into a ParsedQuery object
    consumed by campaign filtering and scoring components.

Key outputs:
    - tokenized query terms for overlap checks
    - brand terms, attribute intents, and price bounds
    - normalized user context fields (location, age, gender, interests, device)

Performance note:
    Regexes are compiled once at import time to keep parse latency low.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedQuery:
    """Normalized query representation shared across pipeline stages.

    Invariants:
        - `query_tokens` is lowercase and tokenized with `_WORD_RE`.
        - `price_min` and `price_max` are numeric when present.
        - context fields are best-effort normalized, not guaranteed complete.
    """

    raw_query: str
    taxonomy_hints: list[str] = field(default_factory=list)
    brand_terms: list[str] = field(default_factory=list)
    model_terms: list[str] = field(default_factory=list)
    attribute_intents: dict[str, str] = field(default_factory=dict)
    price_max: float | None = None
    price_min: float | None = None
    negative_terms: list[str] = field(default_factory=list)
    query_tokens: frozenset[str] = field(default_factory=frozenset)
    # Normalized from user context
    user_location: str | None = None
    user_age: int | None = None
    user_gender: str | None = None
    user_interests: list[str] = field(default_factory=list)
    user_device: str | None = None


# Compiled regexes (module-level, compiled once)
_PRICE_UNDER = re.compile(r"(?:under|less than|below|up to|max|maximum)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_PRICE_OVER = re.compile(r"(?:over|above|more than|min|minimum|at least)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_PRICE_RANGE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)\s*[-–to]+\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
_NEGATIVE = re.compile(
    r"(?:not|no|without|avoid|exclude|excluding)\s+(?!(?:not|no|and|or|the|a|an)\b)(\w+(?:\s+\w+)?)",
    re.I
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_LOCATION_RE = re.compile(r"US-[A-Z]{2}")

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Budget/price intent keywords
_BUDGET_KEYWORDS = {"budget", "cheap", "affordable", "inexpensive", "value", "bargain", "economical"}
_PREMIUM_KEYWORDS = {"premium", "luxury", "high-end", "expensive", "top-tier", "elite", "designer"}


def _load_attribute_keywords() -> dict[str, tuple[str, str]]:
    """Load keyword → (attribute_key, attribute_value) mappings from data/attribute_keywords.json."""
    path = _DATA_DIR / "attribute_keywords.json"
    try:
        with open(path) as f:
            raw: dict = json.load(f)
        # JSON values are [key, val] lists; convert to tuples. Skip the _comment key.
        return {k: (v[0], v[1]) for k, v in raw.items() if k != "_comment" and isinstance(v, list)}
    except FileNotFoundError:
        return {}


# Attribute dictionaries — keyword → (attribute_key, attribute_value)
_ATTRIBUTE_KEYWORDS: dict[str, tuple[str, str]] = _load_attribute_keywords()


def _parse_user_location(location: str | None) -> str | None:
    """Normalize a location string to US-STATE form (for example, ``US-CA``).

    The filter stage only supports state-level geo matching. Constraining output
    to a simple `US-XX` code keeps downstream lookups deterministic.
    """
    if not location:
        return None
    location = location.strip()
    if location.upper().startswith("US-") and len(location) == 5:
        return location.upper()
    parts = re.split(r"[\s,]+", location)
    for part in reversed(parts):
        if len(part) == 2 and part.isalpha():
            return f"US-{part.upper()}"
    return None


class QueryParser:
    """Fast query parser using compiled regexes and dictionary lookups."""

    def __init__(self, brand_names: set[str] | None = None) -> None:
        self._brand_names: set[str] = brand_names or set()
        # Build regex for multi-word brand matching
        if self._brand_names:
            escaped = [re.escape(b) for b in sorted(self._brand_names, key=len, reverse=True)]
            self._brand_re = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.I)
        else:
            self._brand_re = None

    def parse(self, query: str, context: dict | None = None) -> ParsedQuery:
        """Parse query text and optional context into a structured ParsedQuery.

        Args:
            query: Raw user query string.
            context: Optional context dict (age, gender, location, interests, device).

        Returns:
            ParsedQuery with extracted lexical signals and normalized context.

        Notes:
            - Taxonomy hints intentionally keep the full lowercased query as a
              high-recall signal for downstream taxonomy/category components.
            - Missing context fields are left as None/default empty values.
        """
        ctx = context or {}
        query_lower = query.lower().strip()
        tokens = set(_WORD_RE.findall(query_lower))

        result = ParsedQuery(
            raw_query=query,
            query_tokens=frozenset(tokens),
        )

        # --- Brand detection ---
        if self._brand_re:
            brand_matches = self._brand_re.findall(query)
            result.brand_terms = list({m.lower() for m in brand_matches})

        # --- Price extraction ---
        range_match = _PRICE_RANGE.search(query)
        if range_match:
            result.price_min = float(range_match.group(1))
            result.price_max = float(range_match.group(2))
        else:
            under_match = _PRICE_UNDER.search(query)
            if under_match:
                result.price_max = float(under_match.group(1))
            over_match = _PRICE_OVER.search(query)
            if over_match:
                result.price_min = float(over_match.group(1))

        # Budget/premium intent (no explicit dollar amount)
        if result.price_max is None and tokens & _BUDGET_KEYWORDS:
            result.attribute_intents["price_intent"] = "budget"
        elif tokens & _PREMIUM_KEYWORDS:
            result.attribute_intents["price_intent"] = "premium"

        # --- Attribute extraction ---
        for keyword, (attr_key, attr_val) in _ATTRIBUTE_KEYWORDS.items():
            if keyword in query_lower:
                result.attribute_intents[attr_key] = attr_val

        # --- Negative term detection ---
        for neg_match in _NEGATIVE.finditer(query):
            neg_term = neg_match.group(1).lower().strip()
            if neg_term:
                result.negative_terms.append(neg_term)

        # --- Taxonomy hints (extracted from remaining significant tokens) ---
        # Use the full query minus brand terms and price as taxonomy hints
        result.taxonomy_hints = [query_lower]

        # --- User context ---
        result.user_location = _parse_user_location(ctx.get("location"))
        age = ctx.get("age")
        if age is not None:
            result.user_age = int(age)
        result.user_gender = ctx.get("gender")
        result.user_interests = ctx.get("interests", [])
        result.user_device = ctx.get("device")

        return result
