"""Eligibility configuration loader.

Responsibility:
    Load `eligibility_config.yaml`, validate required sections, and provide typed
    accessors used by runtime config wiring in `app/config.py`.

Invariants:
    - Required sections (`blocklist`, `eligibility`) must
      exist at load time.
    - Accessors return runtime-friendly scalar types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class EligibilityConfig:
    """Singleton access to eligibility configuration values."""

    _instance: EligibilityConfig | None = None
    _data: dict[str, Any]

    def __init__(self, config_path: str | Path) -> None:
        """Load YAML config from disk and validate required structure."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Eligibility config not found: {path}")
        with open(path) as f:
            self._data = yaml.safe_load(f) or {}
        self._validate()

    def _validate(self) -> None:
        """Validate required sections exist."""
        if "blocklist" not in self._data:
            raise ValueError("Missing required config section: 'blocklist'")
        if "eligibility" not in self._data:
            raise ValueError("Missing required config section: 'eligibility'")

    def get_blocklist_strategy(self) -> str:
        """Return configured blocklist strategy name."""
        return str(self._data.get("blocklist", {}).get("strategy", "regex"))

    def get_toxic_threshold(self) -> float:
        """Return toxic_bert threshold as float."""
        return float(self._data.get("blocklist", {}).get("toxic_threshold", 0.5))

    def get_eligibility_gate(self) -> float:
        """Return API eligibility gate threshold as float."""
        return float(self._data.get("eligibility", {}).get("gate", 0.1))

    def get_scoring_constants(self) -> dict:
        """Return the scoring constants dict (from the `scoring` section).

        Keys mirror the YAML names: sensitivity_scale, commercial_min,
        cluster_hard_thresholds, etc. Returns an empty dict if the section
        is absent so callers can always use .get(key, default).
        """
        return dict(self._data.get("scoring", {}))

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> EligibilityConfig:
        """Load config once and return the singleton instance."""
        if cls._instance is None:
            if config_path is None:
                # Default path relative to this file
                config_path = Path(__file__).parent / "eligibility_config.yaml"
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
