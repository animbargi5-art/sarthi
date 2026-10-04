"""
SĀRTHI Model Integration Package.
Provider-independent semantic task understanding boundary for foundation models,
and V3 bounded fast decision interfaces.
"""

from backend.app.model.models import TaskUnderstanding
from backend.app.model.provider import ModelProvider
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.model.config import NebiusConfig
from backend.app.model.nebius_provider import (
    NebiusNemotronProvider,
    NebiusProviderError,
    NebiusConfigurationError,
    NebiusAPIError,
    NebiusNetworkError,
    NebiusTimeoutError,
    NebiusParsingError,
    NebiusValidationError,
)
from backend.app.model.v3_models import (
    HumanInstruction,
    PhysicalSituation,
    DecisionContext,
    DecisionQuestionType,
    DecisionQuestion,
    JevDecision,
    DecisionLatencyRecord,
    PhysicsValidationResult,
    RecoveryEvent,
    TelemetryEvent,
)
from backend.app.model.jev_provider import JevDecisionProvider
from backend.app.model.mock_jev_provider import MockJevDecisionProvider
from backend.app.model.jev_config import JevConfig
from backend.app.model.real_jev_provider import (
    RealJevDecisionProvider,
    JevProviderError,
    JevConfigurationError,
    JevConnectionError,
    JevTimeoutError,
    JevResponseParsingError,
    JevValidationError,
)
from backend.app.model.laya_config import LayaConfig
from backend.app.model.laya_provider import LayaDecisionProvider

__all__ = [
    "TaskUnderstanding",
    "ModelProvider",
    "MockModelProvider",
    "TaskUnderstandingService",
    "NebiusConfig",
    "NebiusNemotronProvider",
    "NebiusProviderError",
    "NebiusConfigurationError",
    "NebiusAPIError",
    "NebiusNetworkError",
    "NebiusTimeoutError",
    "NebiusParsingError",
    "NebiusValidationError",
    # V3 Interfaces & Schemas
    "HumanInstruction",
    "PhysicalSituation",
    "DecisionContext",
    "DecisionQuestionType",
    "DecisionQuestion",
    "JevDecision",
    "DecisionLatencyRecord",
    "PhysicsValidationResult",
    "RecoveryEvent",
    "TelemetryEvent",
    # V3 Jev Providers & Config
    "JevDecisionProvider",
    "MockJevDecisionProvider",
    "RealJevDecisionProvider",
    "JevConfig",
    "JevProviderError",
    "JevConfigurationError",
    "JevConnectionError",
    "JevTimeoutError",
    "JevResponseParsingError",
    "JevValidationError",
    "LayaConfig",
    "LayaDecisionProvider",
]
