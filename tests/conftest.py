"""Shared fixtures for tests."""

from __future__ import annotations

import re
import sys
import types
import importlib.machinery
from pathlib import Path

import numpy as np
import pytest

from app.services.embedding import EmbeddingService
from app.services.query_parser import ParsedQuery, QueryParser
from app.services.scoring.score_config import ScoreConfig

_WORD_RE = re.compile(r"[a-z0-9]+")


def _install_optional_dependency_stubs() -> None:
    """Provide tiny stubs when heavy optional ML deps are unavailable.

    The production code imports torch/transformers at module import time.
    These tests avoid loading real models, so lightweight stubs are enough.
    """
    stubbed_torch = False
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        stubbed_torch = True
        torch_stub = types.ModuleType("torch")

        class _MPSBackend:
            @staticmethod
            def is_available() -> bool:
                return False

        class _Backends:
            mps = _MPSBackend()

        def _device(name: str):
            return name

        def _inference_mode():
            def _decorator(fn):
                return fn

            return _decorator

        torch_stub.backends = _Backends()
        torch_stub.device = _device
        torch_stub.inference_mode = _inference_mode
        torch_stub.compile = lambda model, mode=None: model
        torch_stub.float16 = "float16"
        torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

        torch_nn_stub = types.ModuleType("torch.nn")
        torch_nn_func_stub = types.ModuleType("torch.nn.functional")
        torch_nn_func_stub.softmax = lambda x, dim=-1: x
        torch_nn_stub.__spec__ = importlib.machinery.ModuleSpec("torch.nn", loader=None)
        torch_nn_func_stub.__spec__ = importlib.machinery.ModuleSpec(
            "torch.nn.functional", loader=None
        )

        torch_nn_stub.functional = torch_nn_func_stub
        torch_stub.nn = torch_nn_stub

        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = torch_nn_stub
        sys.modules["torch.nn.functional"] = torch_nn_func_stub

    if stubbed_torch:
        transformers_stub = types.ModuleType("transformers")

        class _AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
                return cls()

            def __call__(self, *args, **kwargs):  # noqa: ARG002
                return {}

        class _AutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
                return cls()

            def to(self, device):  # noqa: ARG002
                return self

            def eval(self):
                return self

            def __call__(self, *args, **kwargs):  # noqa: ARG002
                class _Output:
                    logits = np.zeros((1, 2), dtype=np.float32)

                return _Output()

        transformers_stub.AutoTokenizer = _AutoTokenizer
        transformers_stub.AutoModelForSequenceClassification = _AutoModel
        transformers_stub.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
        sys.modules["transformers"] = transformers_stub
        return

    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        transformers_stub = types.ModuleType("transformers")

        class _AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
                return cls()

            def __call__(self, *args, **kwargs):  # noqa: ARG002
                return {}

        class _AutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
                return cls()

            def to(self, device):  # noqa: ARG002
                return self

            def eval(self):
                return self

            def __call__(self, *args, **kwargs):  # noqa: ARG002
                class _Output:
                    logits = np.zeros((1, 2), dtype=np.float32)

                return _Output()

        transformers_stub.AutoTokenizer = _AutoTokenizer
        transformers_stub.AutoModelForSequenceClassification = _AutoModel
        transformers_stub.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
        sys.modules["transformers"] = transformers_stub


_install_optional_dependency_stubs()


@pytest.fixture
def brand_names() -> set[str]:
    return {"nike", "adidas", "apple", "samsung", "tesla", "gucci", "coursera"}


@pytest.fixture
def parser(brand_names: set[str]) -> QueryParser:
    return QueryParser(brand_names=brand_names)


@pytest.fixture
def score_config() -> ScoreConfig:
    ScoreConfig.reset()
    config_path = Path(__file__).parent.parent / "app" / "config" / "score_weights.yaml"
    return ScoreConfig.load(config_path)


class DeterministicEmbeddingService(EmbeddingService):
    """Lightweight deterministic embedding backend for tests.

    The vector space is intentionally small and keyword-driven so tests can
    assert score relationships without downloading models.
    """

    def __init__(self) -> None:
        self._dimension = 6
        self.encode_calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: list[str]) -> np.ndarray:
        self.encode_calls += len(texts)
        vecs = np.zeros((len(texts), self._dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = set(_WORD_RE.findall((text or "").lower()))
            vecs[i] = np.array(
                [
                    # self-harm / crisis
                    float(
                        len(
                            tokens
                            & {
                                "suicide",
                                "suicidal",
                                "kill",
                                "myself",
                                "self",
                                "harm",
                                "hurt",
                                "crisis",
                            }
                        )
                    ),
                    # medical emergency
                    float(len(tokens & {"emergency", "breathing", "unconscious", "stroke", "heart"})),
                    # financial hardship
                    float(len(tokens & {"rent", "eviction", "debt", "bills", "afford", "job"})),
                    # commercial intent
                    float(
                        len(
                            tokens
                            & {"buy", "shop", "deal", "price", "order", "sale", "best", "discount"}
                        )
                    ),
                    # topic words
                    float(len(tokens & {"running", "shoes", "laptop", "history", "guide", "tips"})),
                    # bias keeps vectors non-zero even for empty strings
                    1.0,
                ],
                dtype=np.float32,
            )

        return vecs


@pytest.fixture
def deterministic_embedding_service() -> DeterministicEmbeddingService:
    return DeterministicEmbeddingService()


@pytest.fixture
def parsed_query_factory():
    def _make(**kwargs) -> ParsedQuery:
        base = {
            "raw_query": "test query",
            "query_tokens": frozenset({"test", "query"}),
        }
        base.update(kwargs)
        return ParsedQuery(**base)

    return _make
