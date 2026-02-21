"""
Ad Retrieval API (V2 — Ranking V1 Pipeline)
=============================================
Single endpoint: POST /api/retrieve

Pipeline:
    1. Blocklist check              ~microseconds
    2. Embed query                  ~10-15ms  (once, shared vector)
    3. Eligibility + Categories + FAISS search  (parallel)  ~3-5ms each
    4. Query parsing                ~<2ms
    5. Campaign eligibility filters ~1-2ms
    6. Scoring pipeline (replaces old reranker) ~2-5ms
    ─────────────────────────────────────────────────
    Total target            < 100ms p95

Startup loads all services into app.state so they are warm for every request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import (
    BLOCKLIST_STRATEGY,
    CATEGORY_INDEX_DIR,
    DATA_DIR,
    ELIGIBILITY_GATE,
    EMBEDDING_MODEL,
    FAISS_TOP_K,
    INSENSITIVE_MODEL_PATH,
    LR_MODEL_PATH,
    TOXIC_THRESHOLD,
)
from app.models import CampaignResult, ErrorResponse, RetrieveRequest, RetrieveResponse
from app.services import create_embedding_service
from app.services.campaign_eligibility import CampaignEligibilityFilter
from app.services.categories import CategoryExtractor
from app.services.eligibility import (
    EligibilityScorer,
    create_blocklist_strategy,
)
from app.services.query_parser import QueryParser
from app.services.ranking import ContextReranker
from app.services.retrieval import CampaignIndex
from app.services.scoring.score_config import ScoreConfig

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all services once at startup. Store in app.state."""
    t0 = time.perf_counter()
    data_dir = Path(DATA_DIR)
    loop = asyncio.get_event_loop()

    # Phase 1: Load embedding service (required by scorer and extractor)
    logger.info("Loading embedding service...")
    svc = create_embedding_service()
    logger.info(f"  {type(svc).__name__} initialized")

    # Warm up the model on the executor threads, not the event loop thread.
    # run_in_executor uses a ThreadPoolExecutor that creates worker threads lazily;
    # the first real query would otherwise pay thread-spawn + PyTorch per-thread
    # init costs (~90ms). Running warmup through the executor pre-warms those threads.
    _warmup_path = data_dir / "warmup_queries.json"
    try:
        warmup_queries: list[str] = json.loads(_warmup_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        warmup_queries = ["warmup short", "best running shoes for marathon training"]
    await asyncio.gather(*[loop.run_in_executor(None, svc.embed, q) for q in warmup_queries * 2])
    logger.info(f"  Model ready: {EMBEDDING_MODEL} ({svc.dimension}d)")

    # Phase 2: Load independent services in parallel
    blocklist_path = data_dir / "blocklist.txt"
    blocklist_file = blocklist_path if blocklist_path.exists() else None
    config_path = Path(__file__).parent / "config" / "score_weights.yaml"

    def load_scorer():
        logger.info("Loading eligibility scorer...")
        scorer = EligibilityScorer(
            svc,
            blocklist_path=blocklist_file,
            blocklist_strategy=create_blocklist_strategy(
                BLOCKLIST_STRATEGY,
                blocklist_path=blocklist_file,
                toxic_threshold=TOXIC_THRESHOLD,
                lr_model_path=LR_MODEL_PATH,
            ),
            insensitive_model_path=INSENSITIVE_MODEL_PATH,
        )
        logger.info(f"  {scorer.num_clusters} sensitivity clusters")
        return scorer

    def load_extractor():
        logger.info("Loading category extractor...")
        extractor = CategoryExtractor(
            svc,
            taxonomy_path=data_dir / "taxonomy.json",
            overrides_path=data_dir / "taxonomy_overrides.json",
            category_index_dir=CATEGORY_INDEX_DIR,
        )
        logger.info(f"  {extractor.num_categories} categories")
        return extractor

    def load_index():
        logger.info("Loading campaign index...")
        index = CampaignIndex(data_dir)
        logger.info(f"  {index.size} campaigns indexed")
        return index

    def load_score_config():
        logger.info("Loading score config...")
        score_config = ScoreConfig.load(config_path)
        logger.info(f"  Score config loaded (schema_version={score_config.data.get('schema_version', '?')})")
        return score_config

    # Run Phase 2 in parallel
    scorer, extractor, index, score_config = await asyncio.gather(
        loop.run_in_executor(None, load_scorer),
        loop.run_in_executor(None, load_extractor),
        loop.run_in_executor(None, load_index),
        loop.run_in_executor(None, load_score_config),
    )

    # Phase 3: Load dependent services in parallel (need index/score_config)
    def load_ranker():
        logger.info("Initializing scoring pipeline...")
        ranker = ContextReranker(index, score_config)
        logger.info(f"  {len(ranker._pipeline)} scoring components")
        return ranker

    def load_query_parser():
        logger.info("Initializing query parser...")
        query_parser = QueryParser(brand_names=index.all_brand_names)
        logger.info(f"  {len(index.all_brand_names)} brand names loaded")
        return query_parser

    def load_eligibility_filter():
        logger.info("Initializing eligibility filter...")
        return CampaignEligibilityFilter(index)

    # Run Phase 3 in parallel
    ranker, query_parser, eligibility_filter = await asyncio.gather(
        loop.run_in_executor(None, load_ranker),
        loop.run_in_executor(None, load_query_parser),
        loop.run_in_executor(None, load_eligibility_filter),
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Startup complete in {elapsed:.0f}ms")

    # Store everything in app.state
    app.state.svc = svc
    app.state.scorer = scorer
    app.state.extractor = extractor
    app.state.index = index
    app.state.ranker = ranker
    app.state.query_parser = query_parser
    app.state.eligibility_filter = eligibility_filter

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ad Retrieval API",
    description="Semantic ad retrieval with eligibility scoring, query parsing, and composable ranking.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

_HTTP_ERROR_LABELS: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "unprocessable_entity",
    429: "too_many_requests",
    500: "internal_server_error",
    503: "service_unavailable",
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic / FastAPI request-validation failures (HTTP 422)."""
    details = [
        {
            "loc": list(err.get("loc", [])),
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    body = ErrorResponse(
        status_code=422,
        error=_HTTP_ERROR_LABELS[422],
        message="Request validation failed. Check the `details` field for per-field errors.",
        details=details,  # type: ignore[arg-type]
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Normalise all HTTPException responses into the ErrorResponse envelope."""
    status = exc.status_code
    body = ErrorResponse(
        status_code=status,
        error=_HTTP_ERROR_LABELS.get(status, "http_error"),
        message=str(exc.detail),
    )
    headers = getattr(exc, "headers", None)
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness check. Returns index size so you can verify data was loaded."""
    return {
        "status": "ok",
        "campaigns_indexed": app.state.index.size,
        "embedding_model": EMBEDDING_MODEL,
        "version": "2.0.0",
    }


@app.post("/api/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Main retrieval endpoint.

    Returns the top most relevant campaigns for the query, along with
    an ad eligibility score, extracted categories, and debug metadata.
    """
    t_total = time.perf_counter()

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    context_dict: dict = (
        request.context.model_dump(exclude_none=True) if request.context else {}
    )

    svc = app.state.svc
    scorer: EligibilityScorer = app.state.scorer
    extractor: CategoryExtractor = app.state.extractor
    index: CampaignIndex = app.state.index
    ranker: ContextReranker = app.state.ranker
    query_parser: QueryParser = app.state.query_parser
    eligibility_filter: CampaignEligibilityFilter = app.state.eligibility_filter

    loop = asyncio.get_event_loop()

    # ------------------------------------------------------------------
    # Phase 1: Blocklist check (hard blocks only — score == 0.0)
    # Soft caps (0 < score < 1) fall through to Phase 3 where score_with_metadata
    # applies the cap after semantic scoring.
    # ------------------------------------------------------------------
    match = scorer.blocklist_match(query)
    if match is not None and match.score == 0.0:
        parsed_query = query_parser.parse(query, context=context_dict)
        parser_summary = {
            "brand_terms": parsed_query.brand_terms,
            "price_min": parsed_query.price_min,
            "price_max": parsed_query.price_max,
            "attribute_intents": parsed_query.attribute_intents,
            "negative_terms": parsed_query.negative_terms,
            "user_location": parsed_query.user_location,
        }
        total_ms = (time.perf_counter() - t_total) * 1000
        blocked_meta = {
            "eligibility": 0.0,
            "blocklist_triggered": True,
            "blocklist_strategy": match.strategy,
            "blocklist_rule": match.rule,
            "sensitivity_penalty": 1.0,
            "commercial_boost": 0.0,
            "top_cluster": "blocklist",
            "top_cluster_sim": 1.0,
            "commercial_sim": 0.0,
        }
        return RetrieveResponse(
            ad_eligibility=0.0,
            extracted_categories=[],
            campaigns=[],
            latency_ms=round(total_ms, 2),
            metadata={
                "embedding_model": EMBEDDING_MODEL,
                "n_campaigns_indexed": index.size,
                "n_candidates_retrieved": 0,
                "n_candidates_after_filter": 0,
                "filter_drops": {},
                "n_results": 0,
                "gated": True,
                "timing_ms": {"total": round(total_ms, 2)},
                "eligibility_detail": {
                    k: v for k, v in blocked_meta.items() if k != "eligibility"
                },
                "n_categories": 0,
                "categories": [],
                "parser_output": parser_summary,
                "score_breakdown": [],
            },
        )


    # ------------------------------------------------------------------
    # Phase 2: Embed query
    # ------------------------------------------------------------------
    t_embed = time.perf_counter()
    query_vec = await loop.run_in_executor(None, svc.embed, query)
    embed_ms = (time.perf_counter() - t_embed) * 1000

    # ------------------------------------------------------------------
    # Phase 3: Eligibility, categories, FAISS search — parallel
    # ------------------------------------------------------------------
    t_parallel = time.perf_counter()

    elig_task = loop.run_in_executor(
        None, scorer.score_with_metadata, query, query_vec,
    )
    cat_task = loop.run_in_executor(
        None, partial(extractor.extract, query_vec, context=context_dict),
    )
    search_task = loop.run_in_executor(
        None, partial(index.search, query_vec, FAISS_TOP_K),
    )

    elig_meta, raw_categories, (cand_ids, faiss_scores) = await asyncio.gather(
        elig_task, cat_task, search_task
    )

    parallel_ms = (time.perf_counter() - t_parallel) * 1000

    eligibility_score: float = elig_meta["eligibility"]
    gated: bool = eligibility_score < ELIGIBILITY_GATE
    n_candidates_retrieved = len(cand_ids)

    # Suppress categories for gated queries — returning categories like
    # "baby food", "life insurance", or "robot vacuums" for crisis queries
    # is insensitive and serves no downstream purpose when no ads are shown.
    if gated:
        raw_categories = []

    # ------------------------------------------------------------------
    # Phase 4: Query parsing
    # ------------------------------------------------------------------
    t_parse = time.perf_counter()
    parsed_query = query_parser.parse(query, context=context_dict)
    parse_ms = (time.perf_counter() - t_parse) * 1000

    # ------------------------------------------------------------------
    # Phase 5: Campaign eligibility filtering
    # ------------------------------------------------------------------
    t_filter = time.perf_counter()
    if gated:
        ranked = []
        score_breakdowns = []
        filter_drops: dict[str, int] = {}
        n_after_filter = 0
    else:
        filter_result = eligibility_filter.filter_with_refill(
            cand_ids, faiss_scores, parsed_query,
            index=index, query_embedding=query_vec,
        )
        filter_drops = filter_result.drop_counts
        n_after_filter = len(filter_result.surviving_ids)
        n_candidates_retrieved = n_after_filter + filter_result.total_dropped
        filtered_ids = filter_result.surviving_ids
        filtered_scores = filter_result.surviving_scores

        # ------------------------------------------------------------------
        # Phase 6: Scoring pipeline
        # ------------------------------------------------------------------
        ranked, score_breakdowns = ranker.rerank(
            filtered_ids, filtered_scores, parsed_query,
            extracted_categories=raw_categories,
        )

    filter_ms = (time.perf_counter() - t_filter) * 1000

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    campaigns = [CampaignResult(**c) for c in ranked]

    # Parser output summary (safe for metadata)
    parser_summary = {
        "brand_terms": parsed_query.brand_terms,
        "price_min": parsed_query.price_min,
        "price_max": parsed_query.price_max,
        "attribute_intents": parsed_query.attribute_intents,
        "negative_terms": parsed_query.negative_terms,
        "user_location": parsed_query.user_location,
    }

    total_ms = (time.perf_counter() - t_total) * 1000

    return RetrieveResponse(
        ad_eligibility=eligibility_score,
        extracted_categories=[c["category"] for c in raw_categories],
        campaigns=campaigns,
        latency_ms=round(total_ms, 2),
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "n_campaigns_indexed": index.size,
            "n_candidates_retrieved": n_candidates_retrieved,
            "n_candidates_after_filter": n_after_filter,
            "filter_drops": filter_drops,
            "n_results": len(campaigns),
            "gated": gated,
            "timing_ms": {
                "embed": round(embed_ms, 2),
                "parallel_stages": round(parallel_ms, 2),
                "parse": round(parse_ms, 2),
                "filter_and_rerank": round(filter_ms, 2),
                "total": round(total_ms, 2),
            },
            "eligibility_detail": {
                k: v for k, v in elig_meta.items() if k != "eligibility"
            },
            "n_categories": len(raw_categories),
            "categories": [
                {"category": c["category"], "score": c["score"]}
                for c in raw_categories
            ],
            "parser_output": parser_summary,
            "score_breakdown": score_breakdowns,
        },
    )
