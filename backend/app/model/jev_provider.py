"""
SĀRTHI V3 — Abstract Jev Decision Provider Interface.
Defines the contract for fast bounded decision layers.
Decoupled entirely from physical control, motors, joint limits, or trajectory execution.
"""

from abc import ABC, abstractmethod
from backend.app.model.v3_models import DecisionQuestion, JevDecision


class JevDecisionProvider(ABC):
    """
    Abstract interface for fast bounded decision providers.
    Enforces that providers evaluate bounded decision questions over compact context
    and NEVER directly access robot motors, simulation controllers, or authoritative decisions.
    """

    @abstractmethod
    def ask(self, question: DecisionQuestion) -> JevDecision:
        """
        Synchronous evaluation of a bounded decision question against compact context.
        Returns a structured, non-authoritative candidate recommendation.
        """
        pass

    def decide(self, question: DecisionQuestion) -> JevDecision:
        """Alias for ask() for interface convenience."""
        return self.ask(question)

    @abstractmethod
    async def ask_async(self, question: DecisionQuestion) -> JevDecision:
        """
        Asynchronous evaluation of a bounded decision question against compact context.
        Returns a structured, non-authoritative candidate recommendation.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Returns True if the provider is healthy, authenticated, and reachable.
        """
        pass
