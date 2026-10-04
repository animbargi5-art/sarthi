"""
SĀRTHI Physics & Reproducibility Validation Framework.

Provides systematic validation of physical simulation and integrated repeatability:
- 12 canonical physics dimensions (V3-8)
- 3-level reproducibility validation (V3-9):
    * Level A: Physics-Only Replay
    * Level B: Decision-Pipeline Replay
    * Level C: Full Live V3 Repeatability
"""

from backend.app.validation.framework import SarthiPhysicsValidator
from backend.app.validation.models import (
    PhysicsValidationResult,
    PhysicsValidationSuiteReport,
    ValidationDimension,
    sanitize_secrets,
)
from backend.app.validation.reproducibility import SarthiReproducibilityValidator
from backend.app.validation.reproducibility_models import (
    FailureClassification,
    LevelAPhysicsReplayResult,
    LevelBDecisionReplayResult,
    LevelCLiveRunResult,
    ReproducibilityMetrics,
    V39ReproducibilityReport,
)

__all__ = [
    "FailureClassification",
    "LevelAPhysicsReplayResult",
    "LevelBDecisionReplayResult",
    "LevelCLiveRunResult",
    "PhysicsValidationResult",
    "PhysicsValidationSuiteReport",
    "ReproducibilityMetrics",
    "SarthiPhysicsValidator",
    "SarthiReproducibilityValidator",
    "V39ReproducibilityReport",
    "ValidationDimension",
    "sanitize_secrets",
]
