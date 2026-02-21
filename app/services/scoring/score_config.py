"""Score-weight configuration loader for ranking components.

Responsibility:
    Load score weights from YAML and expose typed accessors used by scoring
    components during reranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ScoreConfig:
    """Singleton access to scoring weights."""

    _instance: ScoreConfig | None = None
    _data: dict[str, Any]

    def __init__(self, config_path: str | Path) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Score config not found: {path}")
        with open(path) as f:
            self._data = yaml.safe_load(f)
        self._validate()

    def _validate(self) -> None:
        """Ensure required scoring sections exist before runtime use."""
        required_sections = [
            "taxonomy", "brand", "attributes", "keywords",
            "pricing", "geo_demo", "contradiction",
        ]
        for section in required_sections:
            if section not in self._data:
                raise ValueError(f"Missing required config section: '{section}'")

    def get(self, section: str, key: str, default: float = 0.0) -> float:
        """Return a weight as float.

        Args:
            section: Top-level YAML section name.
            key: Weight key inside the section.
            default: Fallback value when section/key is missing.
        """
        return float(self._data.get(section, {}).get(key, default))

    @classmethod
    def load(cls, config_path: str | Path) -> ScoreConfig:
        """Load config once and return the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton state (used by tests)."""
        cls._instance = None

    @property
    def data(self) -> dict[str, Any]:
        """Return raw parsed YAML mapping."""
        return self._data
