"""Service factory entry points.

This module centralizes lightweight construction helpers used during app startup.
"""

from app.config import EMBEDDING_BACKEND, EMBEDDING_MODEL
from app.services.embedding import (
    EmbeddingService,
    ONNXBackend,
    SentenceTransformerBackend,
)

_BACKENDS: dict[str, type[EmbeddingService]] = {
    "sentence_transformer": SentenceTransformerBackend,
    "onnx": ONNXBackend,
}


def create_embedding_service() -> EmbeddingService:
    """Create the configured embedding backend instance.

    Returns:
        EmbeddingService selected by `EMBEDDING_BACKEND`.

    Raises:
        ValueError: if `EMBEDDING_BACKEND` is not a supported key.
    """
    backend_cls = _BACKENDS.get(EMBEDDING_BACKEND)
    if backend_cls is None:
        raise ValueError(
            f"Unknown embedding backend: {EMBEDDING_BACKEND!r}. "
            f"Choose from: {list(_BACKENDS.keys())}"
        )
    return backend_cls(model_name=EMBEDDING_MODEL)
