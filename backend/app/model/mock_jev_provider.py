"""
SĀRTHI V3 — Deterministic Mock Jev Decision Provider.
Used for offline unit testing and integration tests without network calls.
"""

from typing import Dict, List, Optional
import time
from backend.app.model.v3_models import DecisionQuestion, JevDecision
from backend.app.model.jev_provider import JevDecisionProvider


class MockJevDecisionProvider(JevDecisionProvider):
    """
    Deterministic mock implementation of JevDecisionProvider.
    Selects options based on configured preferences or falls back to the first available option.
    Zero external network calls or secrets required.
    """

    def __init__(
        self,
        default_choices: Optional[Dict[str, str]] = None,
        confidence: float = 0.95,
        simulated_latency_ms: float = 5.0,
    ) -> None:
        self._default_choices = default_choices or {
            "RECOVERY_SELECTION": "REPOSITION",
            "ACTION_SELECTION": "APPROACH",
            "RECOVERY_REQUIRED": "RECOVER",
            "TARGET_SELECTION": "red_object",
        }
        self._confidence = confidence
        self._simulated_latency_ms = simulated_latency_ms
        self._query_history: List[DecisionQuestion] = []

    @property
    def query_history(self) -> List[DecisionQuestion]:
        """Inspection property returning questions queried to this mock."""
        return list(self._query_history)

    def ask(self, question: DecisionQuestion) -> JevDecision:
        """Synchronously returns a deterministic bounded candidate decision."""
        self._query_history.append(question)
        q_type_str = (
            question.question_type.value
            if hasattr(question.question_type, "value")
            else str(question.question_type)
        )

        # Check configured choice for this question type
        preferred = self._default_choices.get(q_type_str)
        if preferred and preferred in question.available_options:
            selected = preferred
        else:
            selected = question.available_options[0]

        # Calculate normalized probability distribution
        num_options = len(question.available_options)
        probs: Dict[str, float] = {}
        for opt in question.available_options:
            if opt == selected:
                probs[opt] = round(self._confidence, 4)
            else:
                remaining = (1.0 - self._confidence) / max(1, num_options - 1)
                probs[opt] = round(remaining, 4)

        return JevDecision(
            selected_option=selected,
            option_probabilities=probs,
            confidence=self._confidence,
            rationale=f"Mock decision selected '{selected}' for question type '{q_type_str}'",
            provider="mock_jev",
            model_version="mock-v1.0",
            latency_ms=self._simulated_latency_ms,
            timestamp_ns=time.time_ns(),
            request_id=f"mock_req_{len(self._query_history)}",
        )

    async def ask_async(self, question: DecisionQuestion) -> JevDecision:
        """Asynchronously returns a deterministic bounded candidate decision."""
        return self.ask(question)

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True
