"""FAISS-backed campaign retrieval index.

Responsibility:
    Load retrieval artifacts at startup and serve low-latency top-K nearest
    campaign candidates for each query embedding.

Inputs:
    - startup artifact directory (FAISS index + aligned metadata arrays)
    - query embedding vectors at request time

Outputs:
    - candidate FAISS row IDs and raw inner-product scores

Invariants:
    - All loaded artifact arrays must have exactly one entry per FAISS row.
    - Returned IDs index into every companion artifact and `campaigns_meta.json`.

Artifact layout (produced by scripts/build_index.py):

    data/
    ├── faiss.index          # IndexFlatIP, N × 384
    ├── campaigns_meta.json  # list[dict] indexed by FAISS row — response surface
    ├── ages_min.npy         # int8
    ├── ages_max.npy         # int8
    ├── genders.npy          # uint8 bitmask (bit0=all, bit1=male, bit2=female)
    ├── bids.npy             # float32
    ├── budgets.npy          # float32
    ├── geo_include.json     # list[list[str]]
    ├── geo_exclude.json     # list[list[str]]
    ├── campaign_taxonomy.json  # list[dict]
    ├── brands.json          # list[dict]
    ├── keyword_sets.json    # list[dict[str, list[str]]]
    ├── negative_keywords.json # list[list[str]]
    ├── attributes.json      # list[dict]
    ├── interests.json       # list[list[str]]
    └── index_meta.json      # {schema_version, n_campaigns, ...}

"""

from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from app.config import DATA_DIR, FAISS_TOP_K

_WORD_RE = re.compile(r"[a-z0-9]+")


class CampaignIndex:
    """Loads FAISS index and all supporting artifacts; serves top-K search."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)

        # FAISS index
        index_path = base / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run `python scripts/build_index.py` first."
            )
        self._index: faiss.IndexFlatIP = faiss.read_index(str(index_path))
        n = self._index.ntotal

        # Response surface
        with open(base / "campaigns_meta.json") as f:
            self._meta: list[dict] = json.load(f)

        # Numpy filter/boost arrays
        self._ages_min: NDArray[np.int8] = np.load(base / "ages_min.npy")
        self._ages_max: NDArray[np.int8] = np.load(base / "ages_max.npy")
        self._genders: NDArray[np.uint8] = np.load(base / "genders.npy")
        self._bids: NDArray[np.float32] = np.load(base / "bids.npy")

        budgets_path = base / "budgets.npy"
        if budgets_path.exists():
            self._budgets: NDArray[np.float32] = np.load(budgets_path)
        else:
            self._budgets = np.zeros(n, dtype=np.float32)

        # Geo targeting
        geo_inc_path = base / "geo_include.json"
        geo_exc_path = base / "geo_exclude.json"
        if geo_inc_path.exists():
            with open(geo_inc_path) as f:
                raw_inc: list[list[str]] = json.load(f)
            self._geo_include: list[frozenset[str]] = [frozenset(locs) for locs in raw_inc]
        else:
            # Fallback to old location_sets.json
            loc_path = base / "location_sets.json"
            if loc_path.exists():
                with open(loc_path) as f:
                    raw_locs: list[list[str]] = json.load(f)
                self._geo_include = [frozenset(locs) for locs in raw_locs]
            else:
                self._geo_include = [frozenset() for _ in range(n)]

        if geo_exc_path.exists():
            with open(geo_exc_path) as f:
                raw_exc: list[list[str]] = json.load(f)
            self._geo_exclude: list[frozenset[str]] = [frozenset(locs) for locs in raw_exc]
        else:
            self._geo_exclude = [frozenset() for _ in range(n)]

        # Taxonomy
        tax_path = base / "campaign_taxonomy.json"
        if tax_path.exists():
            with open(tax_path) as f:
                self._taxonomy: list[dict] = json.load(f)
        else:
            self._taxonomy = [{"category": "", "subcategory": "", "leaf": ""} for _ in range(n)]

        # Sync check: campaign_taxonomy categories must exist in taxonomy.json
        query_tax_path = base / "taxonomy.json"
        if query_tax_path.exists():
            with open(query_tax_path) as f:
                query_tax_names: set[str] = {e["name"] for e in json.load(f)}
            camp_cats: set[str] = {
                e["category"] for e in self._taxonomy if e.get("category")
            }
            orphaned = camp_cats - query_tax_names
            if orphaned:
                print(
                    f"  [WARN] taxonomy sync: {len(orphaned)} campaign categories not in "
                    f"taxonomy.json — run scripts/normalize_campaign_taxonomy.py to fix:\n"
                    + "".join(f"    • {c}\n" for c in sorted(orphaned))
                )

        # Brands
        brands_path = base / "brands.json"
        if brands_path.exists():
            with open(brands_path) as f:
                self._brands: list[dict] = json.load(f)
        else:
            self._brands = [{"name": "", "type": "", "product_line": ""} for _ in range(n)]

        # Keyword sets — pre-tokenize into frozensets for fast lookup
        kw_path = base / "keyword_sets.json"
        if kw_path.exists():
            with open(kw_path) as f:
                raw_kw: list[dict] = json.load(f)
            self._keyword_sets: list[dict[str, frozenset[str]]] = [
                {
                    bucket: frozenset(terms)
                    for bucket, terms in kw.items()
                }
                for kw in raw_kw
            ]
        else:
            empty_kw: dict[str, frozenset[str]] = {
                b: frozenset() for b in
                ("brand_terms", "model_terms", "category_terms", "feature_terms", "benefit_terms")
            }
            self._keyword_sets = [empty_kw.copy() for _ in range(n)]

        # Negative keywords — pre-tokenize into frozensets
        neg_path = base / "negative_keywords.json"
        if neg_path.exists():
            with open(neg_path) as f:
                raw_neg: list[list[str]] = json.load(f)
            self._negative_keywords: list[frozenset[str]] = [
                frozenset(terms) for terms in raw_neg
            ]
        else:
            self._negative_keywords = [frozenset() for _ in range(n)]

        # Attributes
        attr_path = base / "attributes.json"
        if attr_path.exists():
            with open(attr_path) as f:
                self._attributes: list[dict] = json.load(f)
        else:
            self._attributes = [{} for _ in range(n)]

        # Interests
        int_path = base / "interests.json"
        if int_path.exists():
            with open(int_path) as f:
                raw_int: list[list[str]] = json.load(f)
            self._interests: list[frozenset[str]] = [
                frozenset(_WORD_RE.findall(" ".join(terms).lower()))
                for terms in raw_int
            ]
        else:
            self._interests = [frozenset() for _ in range(n)]

        # Build brand name lookup for query parser
        self._all_brand_names: set[str] = set()
        for b in self._brands:
            name = b.get("name", "")
            if name:
                self._all_brand_names.add(name.lower())

        # Validate artifact lengths
        expected = n
        for name, arr in [
            ("meta", self._meta), ("ages_min", self._ages_min),
            ("ages_max", self._ages_max), ("genders", self._genders),
            ("bids", self._bids), ("budgets", self._budgets),
            ("geo_include", self._geo_include), ("geo_exclude", self._geo_exclude),
            ("taxonomy", self._taxonomy), ("brands", self._brands),
            ("keyword_sets", self._keyword_sets),
            ("negative_keywords", self._negative_keywords),
            ("attributes", self._attributes), ("interests", self._interests),
        ]:
            actual = len(arr)
            if actual != expected:
                raise ValueError(
                    f"Artifact '{name}' has {actual} entries but FAISS index has {expected}. "
                    "Re-run scripts/build_index.py."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = FAISS_TOP_K,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Run FAISS inner-product search and return valid candidate rows.

        Args:
            query_embedding: Query vector with shape (dimension,).
            top_k: Maximum number of neighbors to request from FAISS.

        Returns:
            Tuple `(ids, scores)` where:
            - `ids` are FAISS row indices as int64
            - `scores` are raw FAISS inner-product scores as float32

        Notes:
            FAISS may return -1 IDs when fewer than `top_k` rows are available.
            Those rows are removed here so downstream stages can index safely.
        """
        k = min(top_k, self._index.ntotal)
        if k == 0:
            empty_ids = np.empty(0, dtype=np.int64)
            empty_scores = np.empty(0, dtype=np.float32)
            return empty_ids, empty_scores

        vec = query_embedding.reshape(1, -1)
        scores_2d, ids_2d = self._index.search(vec, k)

        ids = ids_2d[0]
        scores = scores_2d[0]

        valid = ids >= 0
        return ids[valid].astype(np.int64), scores[valid]

    # ------------------------------------------------------------------
    # Filter/boost arrays
    # ------------------------------------------------------------------

    @property
    def ages_min(self) -> NDArray[np.int8]:
        """Minimum target age per campaign row."""
        return self._ages_min

    @property
    def ages_max(self) -> NDArray[np.int8]:
        """Maximum target age per campaign row."""
        return self._ages_max

    @property
    def genders(self) -> NDArray[np.uint8]:
        """Gender targeting bitmask per campaign row."""
        return self._genders

    @property
    def bids(self) -> NDArray[np.float32]:
        """Bid/price proxy per campaign row."""
        return self._bids

    @property
    def budgets(self) -> NDArray[np.float32]:
        """Budget remaining per campaign row."""
        return self._budgets

    @property
    def geo_include(self) -> list[frozenset[str]]:
        """Allowed geo codes per campaign row."""
        return self._geo_include

    @property
    def geo_exclude(self) -> list[frozenset[str]]:
        """Excluded geo codes per campaign row."""
        return self._geo_exclude

    @property
    def location_sets(self) -> list[frozenset[str]]:
        """Backward-compatible alias for geo_include."""
        return self._geo_include

    @property
    def taxonomy(self) -> list[dict]:
        """Campaign taxonomy metadata aligned to FAISS rows."""
        return self._taxonomy

    @property
    def brands(self) -> list[dict]:
        """Campaign brand metadata aligned to FAISS rows."""
        return self._brands

    @property
    def keyword_sets(self) -> list[dict[str, frozenset[str]]]:
        """Pre-tokenized keyword buckets aligned to FAISS rows."""
        return self._keyword_sets

    @property
    def negative_keywords(self) -> list[frozenset[str]]:
        """Negative keyword phrases aligned to FAISS rows."""
        return self._negative_keywords

    @property
    def attributes(self) -> list[dict]:
        """Campaign attribute maps aligned to FAISS rows."""
        return self._attributes

    @property
    def interests(self) -> list[frozenset[str]]:
        """Tokenized campaign interest terms aligned to FAISS rows."""
        return self._interests

    @property
    def all_brand_names(self) -> set[str]:
        """Normalized set of all known campaign brand names."""
        return self._all_brand_names

    @property
    def meta(self) -> list[dict]:
        """Response-surface campaign objects aligned to FAISS rows."""
        return self._meta

    @property
    def size(self) -> int:
        """Number of campaigns in the index."""
        return self._index.ntotal
