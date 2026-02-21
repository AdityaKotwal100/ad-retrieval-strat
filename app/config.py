"""Application runtime configuration constants.

Responsibility:
    Resolve runtime settings from YAML-backed defaults plus environment variable
    overrides, then expose them as module-level constants.

Inputs:
    - `app/config/eligibility_config.yaml` (when available)
    - environment variables (authoritative overrides)

Outputs:
    - typed constants consumed by startup wiring and request pipeline code

Invariant:
    Config resolution is deterministic at import time for a given process
    environment.
"""

import os
from pathlib import Path

# Restrict ONNX Runtime's OpenMP thread pool before any import loads the library.
# Without this, ONNX Runtime spawns threads that conflict with Python's GIL on
# macOS/ARM and cause a segfault during the first multi-sentence embed_batch() call.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load eligibility config from YAML (with env var fallback)
try:
    import importlib.util

    _eligibility_config_module_path = Path(__file__).parent / "config" / "eligibility_config.py"
    if _eligibility_config_module_path.exists():
        spec = importlib.util.spec_from_file_location(
            "eligibility_config", _eligibility_config_module_path
        )
        eligibility_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eligibility_config_module)  # type: ignore
        EligibilityConfig = eligibility_config_module.EligibilityConfig

        _eligibility_config_path = Path(__file__).parent / "config" / "eligibility_config.yaml"
        _eligibility_config = EligibilityConfig.load(_eligibility_config_path)
    else:
        _eligibility_config = None
except (ImportError, FileNotFoundError, ValueError, AttributeError):
    # Fallback to env vars if YAML not available
    _eligibility_config = None

# Embedding
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# "sentence_transformer" | "onnx"
EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "onnx")

# Blocklist strategy for eligibility scoring.
# Loaded from eligibility_config.yaml, override via BLOCKLIST_STRATEGY env var.
# "regex" uses patterns from data/blocklist.txt (if present) or built-ins.
# "toxic_bert" uses ML model (unitary/toxic-bert) to detect harmful content.
# "none" disables blocklist matching entirely (useful for benchmarking).
BLOCKLIST_STRATEGY: str = os.getenv(
    "BLOCKLIST_STRATEGY",
    _eligibility_config.get_blocklist_strategy() if _eligibility_config else "all",
)

# Path to the pre-trained LR harmful-content model (joblib).
# Used by the "lr" blocklist strategy.
_DEFAULT_LR_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "model-test", "harmful-content", "harmful_lr.joblib"
)
LR_MODEL_PATH: str | None = os.getenv("LR_MODEL_PATH", _DEFAULT_LR_MODEL)

# Path to insensitive-query suppression model (generate-ads-or-not toxicity v5).
_DEFAULT_INSENSITIVE_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "data", "insensitive_model_v5.joblib"
)
INSENSITIVE_MODEL_PATH: str = os.getenv("INSENSITIVE_MODEL_PATH", _DEFAULT_INSENSITIVE_MODEL)

# Toxic probability threshold for toxic_bert strategy (0.0-1.0).
# Loaded from eligibility_config.yaml, override via TOXIC_THRESHOLD env var.
# Queries with toxic probability >= this value are blocked.
TOXIC_THRESHOLD: float = float(
    os.getenv(
        "TOXIC_THRESHOLD",
        str(_eligibility_config.get_toxic_threshold()) if _eligibility_config else "0.4",
    )
)

# Safety gate — queries with eligibility score below this threshold return no
# campaigns. Loaded from eligibility_config.yaml, override via ELIGIBILITY_GATE env var.
# Blocklist hits score 0.0; the gate at 0.1 adds a small buffer for near-blocklist queries.
# Raise to e.g. 0.5 for stricter filtering.
ELIGIBILITY_GATE: float = float(
    os.getenv(
        "ELIGIBILITY_GATE",
        str(_eligibility_config.get_eligibility_gate()) if _eligibility_config else "0.3",
    )
)


# Retrieval
FAISS_TOP_K: int = int(os.getenv("FAISS_TOP_K", "1000"))

# Category extraction
CATEGORY_TOP_K: int = int(os.getenv("CATEGORY_TOP_K", "5"))
CATEGORY_THRESHOLD: float = float(os.getenv("CATEGORY_THRESHOLD", "0.40"))

# Path to the FAISS query-index for category extraction.
# When set (and files exist), CategoryExtractor uses query-to-query matching
# instead of description embeddings — far more accurate at retrieval time.
_DEFAULT_CAT_INDEX = os.path.join(
    os.path.dirname(__file__), "..", "data"
)
CATEGORY_INDEX_DIR: str | None = os.getenv("CATEGORY_INDEX_DIR", _DEFAULT_CAT_INDEX)

# Data paths
DATA_DIR: str = os.getenv(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")
)
