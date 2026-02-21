"""
Embedding Service
=================
Strategy pattern for embedding backends. Every vector returned is L2-normalized.

Usage:
    service = SentenceTransformerBackend(model_name="all-MiniLM-L6-v2")
    vec = service.embed("running shoes for marathon")        # shape: (384,)
    vecs = service.embed_batch(["shoes", "laptops", "food"]) # shape: (3, 384)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingService(ABC):
    """Interface that all embedding backends must implement."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""
        ...

    @abstractmethod
    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode texts into raw vectors. Shape: (n, dimension).

        Subclasses implement this. Normalization is handled by the base class.
        """
        ...

    def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text. Returns a unit-length vector of shape (dimension,)."""
        vec = self._encode([text])[0]
        return self._normalize(vec)

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed multiple texts. Returns unit-length vectors of shape (n, dimension)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vecs = self._encode(texts)
        return self._normalize(vecs)

    @staticmethod
    def _normalize(vecs: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2-normalize vectors. Works for both 1D and 2D arrays."""
        if vecs.ndim == 1:
            norm = np.linalg.norm(vecs)
            return vecs / norm if norm > 0 else vecs
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        return vecs / norms


class SentenceTransformerBackend(EmbeddingService):
    """Local CPU embedding using sentence-transformers.

    Model loads once on init (~80MB memory for MiniLM-L6-v2).
    Single-sentence encode: ~5-15ms on CPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


class ONNXBackend(EmbeddingService):
    """ONNX Runtime backend for faster CPU inference.

    Drop-in replacement — same models, ~2-3x faster.
    Requires: pip install optimum[onnxruntime]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        import os
        import platform
        from sentence_transformers import SentenceTransformer

        # Restrict ONNX Runtime to a single intra-op thread.
        # Without this, ONNX Runtime spawns OpenMP threads for batch inference
        # that conflict with Python's GIL on macOS/ARM, causing a segfault in
        # embed_batch() even though single-string embed() works fine.
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        # Pick the model file best suited to the current CPU.
        # model_qint8_arm64.onnx is quantized for Apple Silicon and is faster
        # and more stable than the generic model.onnx on ARM Macs.
        is_arm = platform.machine() in ("arm64", "aarch64")
        file_name = "onnx/model_qint8_arm64.onnx" if is_arm else "onnx/model.onnx"

        self._model = SentenceTransformer(
            model_name,
            backend="onnx",
            model_kwargs={
                "provider": "CPUExecutionProvider",
                "file_name": file_name,
            },
        )
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
