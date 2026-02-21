"""
Category Extraction Service
============================
Supports two backends:

1. **FAISS query-index** (preferred): A separate FAISS index populated with
   representative user queries per category.  At query time we search this index
   for top-K nearest query vectors and aggregate by category (max cosine
   similarity).  Because both the index and the query are in "user query"
   vocabulary, the cosine scores are far more reliable than description-based
   matching (typically 0.65-0.75 vs 0.35-0.45 for keyword bags).

   Required files (built by model-test/category/build_index.py):
       <category_index_dir>/category_index.faiss
       <category_index_dir>/category_meta.json

2. **Description-matrix** (legacy fallback): Embeds taxonomy descriptions into a
   matrix and dot-products against the query embedding.  Used automatically when
   no category_index_dir is supplied or the FAISS files are missing.

All vectors are unit-normalised (guaranteed by EmbeddingService), so inner
product == cosine similarity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from app.config import CATEGORY_THRESHOLD, CATEGORY_TOP_K
from app.services.embedding import EmbeddingService

_WORD_RE = re.compile(r"[a-z0-9]+")

# How many FAISS vectors to retrieve per query before category aggregation.
# Must be ≥ (queries_per_cat) to ensure every category has at least one shot.
# 50 covers 15-query-per-cat indexes with room for distant queries.
_FAISS_SEARCH_K: int = 50

# --- Coherence gate (description-matrix mode only) ---
# Two-path OR logic:
# 1. Inter-category: secondary embedding must be within _COHERENCE_MIN of top-1.
#    Catches description-close siblings while blocking cross-domain bleed.
# 2. High-ratio bypass: keep any category that scores ≥ 80% of top-1 query score.
_COHERENCE_MIN: float = 0.45
_COHERENCE_RATIO: float = 0.80

# In FAISS mode the high-ratio bypass still applies; inter-category coherence is
# not needed because query-to-query matching already filters irrelevant categories.
_FAISS_COHERENCE_RATIO: float = 0.80

# Fallback floor: when no category clears CATEGORY_THRESHOLD, return the
# single best match only if it clears this floor.
_FALLBACK_MIN_SCORE: float = 0.40


class CategoryExtractor:
    """Extracts relevant product/service categories from a query embedding."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        taxonomy_path: str | Path,
        overrides_path: Optional[str | Path] = None,
        category_index_dir: Optional[str | Path] = None,
    ) -> None:
        """
        Args:
            embedding_service: Used to embed taxonomy descriptions (legacy mode).
            taxonomy_path: Path to taxonomy.json (legacy mode + keyword sets).
            overrides_path: Optional path to taxonomy_overrides.json.
            category_index_dir: Directory containing category_index.faiss and
                category_meta.json.  When supplied and the files exist, the
                FAISS query-index backend is used instead of description embeddings.
        """
        # ── Load taxonomy (needed for keyword sets in both modes) ────────────
        taxonomy_path = Path(taxonomy_path)
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)

        overrides: dict[str, list[str]] = {}
        if overrides_path is not None:
            overrides_path = Path(overrides_path)
            if overrides_path.exists():
                with open(overrides_path) as f:
                    raw = json.load(f)
                overrides = {k: v for k, v in raw.items() if not k.startswith("_")}

        # Build keyword sets for context boosting (both modes use these)
        # taxonomy names serve as the canonical category namespace for the legacy mode.
        self._taxonomy_names: list[str] = [entry["name"] for entry in taxonomy]
        self._taxonomy_descriptions: list[str] = [entry["description"] for entry in taxonomy]
        self._taxonomy_keyword_sets: list[set[str]] = []
        for entry in taxonomy:
            base_text = entry["name"].lower() + " " + entry["description"].lower()
            keywords = set(_WORD_RE.findall(base_text))
            extra = overrides.get(entry["name"], [])
            for term in extra:
                keywords.update(_WORD_RE.findall(term.lower()))
            self._taxonomy_keyword_sets.append(keywords)

        # ── Choose backend ───────────────────────────────────────────────────
        self._use_faiss_index = False

        if category_index_dir is not None:
            cat_dir = Path(category_index_dir)
            faiss_path = cat_dir / "category_index.faiss"
            meta_path = cat_dir / "category_meta.json"

            if faiss_path.exists() and meta_path.exists():
                try:
                    import faiss as _faiss

                    self._faiss_index = _faiss.read_index(str(faiss_path))
                    with open(meta_path) as f:
                        self._faiss_meta: list[dict] = json.load(f)

                    # Build unique ordered category list + per-category keyword sets
                    seen: set[str] = set()
                    self._faiss_cats: list[str] = []
                    for row in self._faiss_meta:
                        cat = row["category"]
                        if cat not in seen:
                            seen.add(cat)
                            self._faiss_cats.append(cat)

                    self._faiss_keyword_sets: dict[str, set[str]] = {}
                    for cat in self._faiss_cats:
                        kw = set(_WORD_RE.findall(cat.lower()))
                        self._faiss_keyword_sets[cat] = kw

                    self._use_faiss_index = True
                    print(
                        f"  CategoryExtractor: FAISS query-index mode "
                        f"({self._faiss_index.ntotal} vectors, "
                        f"{len(self._faiss_cats)} categories)"
                    )
                except Exception as exc:
                    print(f"  CategoryExtractor: FAISS index load failed ({exc}), "
                          "falling back to description-matrix mode.")

        if not self._use_faiss_index:
            # Legacy: pre-compute taxonomy description embedding matrix
            self._embeddings: NDArray[np.float32] = embedding_service.embed_batch(
                self._taxonomy_descriptions
            )

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def num_categories(self) -> int:
        """Return the active category cardinality for the selected backend."""
        if self._use_faiss_index:
            return len(self._faiss_cats)
        return len(self._taxonomy_names)

    def extract(
        self,
        query_embedding: NDArray[np.float32],
        context: dict | None = None,
        top_k: int = CATEGORY_TOP_K,
        threshold: float = CATEGORY_THRESHOLD,
    ) -> list[dict]:
        """Extract top-K categories above threshold for a query.

        Args:
            query_embedding: Unit-normalized query vector, shape (dimension,).
            context: Optional user context dict with "interests" and/or "gender".
            top_k: Maximum number of categories to return.
            threshold: Minimum similarity score to include.

        Returns:
            List of {"category": str, "score": float} sorted by score descending.
        """
        if self._use_faiss_index:
            return self._extract_faiss(query_embedding, context, top_k, threshold)
        return self._extract_matrix(query_embedding, context, top_k, threshold)

    # ── FAISS query-index backend ─────────────────────────────────────────────

    def _extract_faiss(
        self,
        query_embedding: NDArray[np.float32],
        context: dict | None,
        top_k: int,
        threshold: float,
    ) -> list[dict]:
        q = query_embedding.astype(np.float32).reshape(1, -1)
        k = min(_FAISS_SEARCH_K, self._faiss_index.ntotal)
        distances, indices = self._faiss_index.search(q, k)

        # Aggregate: max cosine similarity per category
        cat_scores: dict[str, float] = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            cat = self._faiss_meta[idx]["category"]
            if dist > cat_scores.get(cat, float("-inf")):
                cat_scores[cat] = float(dist)

        if not cat_scores:
            return []

        # Context boost: +0.05 if user interests share keywords with category name
        if context and "interests" in context:
            interest_words: set[str] = set()
            for interest in context["interests"]:
                interest_words.update(_WORD_RE.findall(interest.lower()))
            if interest_words:
                for cat in cat_scores:
                    if interest_words & self._faiss_keyword_sets.get(cat, set()):
                        cat_scores[cat] = cat_scores[cat] + 0.05

        # Gender suppression
        if context and "gender" in context:
            opposite_tokens = self._opposite_gender_tokens(str(context.get("gender", "")))
            if opposite_tokens:
                for cat in list(cat_scores):
                    tokens = _WORD_RE.findall(cat.lower())
                    if tokens and tokens[0] in opposite_tokens:
                        cat_scores[cat] = -1.0

        # Sort descending
        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)

        # Filter by threshold
        above = [(c, s) for c, s in sorted_cats if s >= threshold]
        if not above:
            # Fallback: return best if above floor
            best_cat, best_score = sorted_cats[0]
            if best_score < _FALLBACK_MIN_SCORE:
                return []
            return [{"category": best_cat, "score": round(best_score, 4)}]

        # High-ratio coherence gate: keep only categories within 80% of top-1
        top_score = above[0][1]
        coherent = [
            (c, s) for c, s in above
            if s >= _FAISS_COHERENCE_RATIO * top_score
        ]

        return [
            {"category": c, "score": round(s, 4)}
            for c, s in coherent[:top_k]
        ]

    # ── Description-matrix backend (legacy) ───────────────────────────────────

    def _extract_matrix(
        self,
        query_embedding: NDArray[np.float32],
        context: dict | None,
        top_k: int,
        threshold: float,
    ) -> list[dict]:
        base_scores = self._embeddings @ query_embedding
        scores = base_scores.copy()

        if context and "interests" in context:
            interest_words: set[str] = set()
            for interest in context["interests"]:
                interest_words.update(_WORD_RE.findall(interest.lower()))
            boost = np.array(
                [
                    0.05 if interest_words & kw_set else 0.0
                    for kw_set in self._taxonomy_keyword_sets
                ],
                dtype=np.float32,
            )
            scores = scores + boost

        if context and "gender" in context:
            opposite_tokens = self._opposite_gender_tokens(str(context.get("gender", "")))
            if opposite_tokens:
                for i, name in enumerate(self._taxonomy_names):
                    tokens = _WORD_RE.findall(name.lower())
                    if tokens and tokens[0] in opposite_tokens:
                        scores[i] = -1.0

        mask = scores >= threshold
        if not mask.any():
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score < _FALLBACK_MIN_SCORE:
                return []
            return [{"category": self._taxonomy_names[best_idx], "score": round(best_score, 4)}]

        eligible_indices = np.where(mask)[0]
        eligible_scores = scores[eligible_indices]
        top_order = np.argsort(eligible_scores)[::-1][:top_k]
        top_indices = eligible_indices[top_order]

        # Coherence gate (description-matrix mode)
        if len(top_indices) > 1:
            top_vec = self._embeddings[int(top_indices[0])]
            top_base_score = float(base_scores[int(top_indices[0])])
            top_indices = [
                idx for idx in top_indices
                if idx == top_indices[0]
                or float(self._embeddings[int(idx)] @ top_vec) >= _COHERENCE_MIN
                or float(base_scores[int(idx)]) >= _COHERENCE_RATIO * top_base_score
            ]

        return [
            {"category": self._taxonomy_names[idx], "score": round(float(scores[idx]), 4)}
            for idx in top_indices
        ]

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _opposite_gender_tokens(gender_str: str) -> frozenset[str]:
        g = gender_str.strip().lower()
        if g in {"male", "man", "m"}:
            return frozenset({"women", "girls", "ladies"})
        if g in {"female", "woman", "f"}:
            return frozenset({"men", "boys"})
        return frozenset()
