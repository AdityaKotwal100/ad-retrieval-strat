"""Public exports for the scoring subsystem.

This package exposes:
    - pipeline construction (`ScoreBuilder`)
    - component interfaces (`ScoreComponent`)
    - configuration and context contracts (`ScoreConfig`, `ScoreContext`, `ScoreDelta`)
"""

from app.services.scoring.builder import ScoreBuilder
from app.services.scoring.components import ScoreComponent
from app.services.scoring.score_config import ScoreConfig
from app.services.scoring.score_context import ScoreContext, ScoreDelta

__all__ = ["ScoreBuilder", "ScoreComponent", "ScoreConfig", "ScoreContext", "ScoreDelta"]
