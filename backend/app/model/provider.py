"""
SĀRTHI Model Integration — Abstract Model Provider Interface.
Defines the contract for foundation model backends (Nebius Token Factory, local mocks).
Strictly decoupled from physical control and actuation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from backend.app.model.models import TaskUnderstanding


class ModelProvider(ABC):
    """
    Abstract interface for AI model providers.
    Enforces that model providers only synthesize structured task understandings
    and never directly access robot motors, simulation adapters, or the decision engine.
    """

    @abstractmethod
    def understand_task(
        self,
        instruction: str,
        world_context: Optional[Dict[str, Any]] = None,
    ) -> TaskUnderstanding:
        """
        Interprets natural language human instruction within an optional world context
        and returns a structured, type-safe TaskUnderstanding instance.
        """
        pass
