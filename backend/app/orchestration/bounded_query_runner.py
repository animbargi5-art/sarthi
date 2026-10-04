"""
SĀRTHI V3-4 — Bounded Jev Decision Query Orchestrator.
Orchestrates a controlled, bounded fast decision query against TypeSafe AI / Jev.
Strictly enforces context minimization, schema bounds, input/output validation,
sanitized telemetry recording, and non-authoritative boundary guarantees.

Non-Negotiable Architecture Invariants:
1. AI models propose or select bounded semantic candidates; deterministic software validates
   physical feasibility; only validated actions reach the robot controller.
2. Jev must never bypass deterministic safety validation.
3. The output of this query runner strictly terminates at JevDecision.
   It must NEVER:
   - execute robot actions
   - call MuJoCo runtime
   - modify WorldState
   - modify robot joints
   - command motors
   - create ActionExecution
   - bypass SarthiDecisionEngine
4. Credentials must be loaded ONLY from environment variables, never hardcoded.
5. In case of provider failure, report the exact failure category; never fabricate
   responses or silently substitute MockJevDecisionProvider as a live success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from backend.app.decision_engine.models import (
    ActionType,
    ActiveConstraint,
    LastActionOutcome,
    LastActionStatus,
    ObjectState,
    Point3D,
    RobotState,
    TargetZone,
    TaskObjective,
    WorldObject,
    WorldState,
)
from backend.app.model.jev_config import JevConfig
from backend.app.model.jev_provider import JevDecisionProvider
from backend.app.model.real_jev_provider import (
    JevConfigurationError,
    JevConnectionError,
    JevProviderError,
    JevResponseParsingError,
    JevTimeoutError,
    JevValidationError,
    RealJevDecisionProvider,
)
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionQuestion,
    DecisionQuestionType,
    JevDecision,
    PhysicalSituation,
)
from backend.app.orchestration.context_builder import DecisionContextBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry and Execution Result Structures
# ---------------------------------------------------------------------------

def _sanitize_telemetry_str(val: Optional[str]) -> Optional[str]:
    """Sanitizes sensitive patterns from telemetry strings."""
    if not val:
        return val
    sanitized = val
    for pattern in ["sk-", "nbf_", "Bearer ", "API_KEY", "SECRET", "TOKEN", "PASSWORD"]:
        if pattern.lower() in sanitized.lower():
            sanitized = re.sub(rf"{re.escape(pattern)}[A-Za-z0-9_\-\.]*", "[REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized


@dataclass(frozen=True)
class BoundedQueryTelemetry:
    """Safe, sanitized telemetry record for a bounded Jev query."""
    question_id: str
    question_type: str
    available_options: List[str]
    selected_option: Optional[str]
    confidence: Optional[float]
    probabilities: Dict[str, float]
    latency_ms: Optional[float]
    success: bool
    provider_status: str
    provider_name: str
    model_version: Optional[str]
    answer_confidence: Optional[float] = None
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes telemetry record to a secret-free dictionary."""
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "available_options": list(self.available_options),
            "selected_option": _sanitize_telemetry_str(self.selected_option),
            "confidence": self.confidence,
            "answer_confidence": self.answer_confidence,
            "probabilities": dict(self.probabilities),
            "latency_ms": self.latency_ms,
            "success": self.success,
            "provider_status": self.provider_status,
            "provider_name": _sanitize_telemetry_str(self.provider_name) or "jev",
            "model_version": _sanitize_telemetry_str(self.model_version),
            "error_category": _sanitize_telemetry_str(self.error_category),
            "error_message": _sanitize_telemetry_str(self.error_message),
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass(frozen=True)
class BoundedQueryResult:
    """Encapsulates the complete outcome of a bounded Jev query."""
    question: DecisionQuestion
    decision: Optional[JevDecision]
    telemetry: BoundedQueryTelemetry
    validation_passed: bool
    success: bool = True
    error_category: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Bounded Jev Query Runner
# ---------------------------------------------------------------------------

class BoundedJevQueryRunner:
    """
    Executes controlled bounded decision queries against Jev for SĀRTHI V3-4.
    Guarantees that context is built deterministically through DecisionContextBuilder,
    credentials come only from environment, responses undergo strict validation,
    and no physical action or state modification is ever performed.
    """

    DEFAULT_OBJECTIVE: str = "Move the red object to the blue target."
    DEFAULT_QUESTION_TEXT: str = "Which recovery action should be considered?"
    DEFAULT_OPTIONS: List[str] = ("REPOSITION", "STOP")

    def __init__(
        self,
        context_builder: Optional[DecisionContextBuilder] = None,
        coordinate_precision: int = 3,
    ) -> None:
        self._builder = context_builder or DecisionContextBuilder(coordinate_precision=coordinate_precision)

    @property
    def context_builder(self) -> DecisionContextBuilder:
        return self._builder

    def build_recovery_scenario(
        self,
        *,
        question_id: str = "q_v3_4_recovery_01",
        context_id: str = "ctx_v3_4_recovery_01",
        situation_id: str = "sit_v3_4_recovery_01",
        objective: str = DEFAULT_OBJECTIVE,
        available_options: Optional[List[str]] = None,
    ) -> tuple[PhysicalSituation, DecisionQuestion]:
        """
        Builds the deterministic V3-4 recovery situation and bounded question:
        - Robot is holding the red object.
        - Direct path to the destination is blocked.
        - The blocked-path constraint is active.
        - Previous MOVE action was rejected.
        - Question: "Which recovery action should be considered?"
        - Available options: discrete set ["REPOSITION", "STOP"].

        Constructs DecisionContext strictly through DecisionContextBuilder.
        """
        options = list(available_options) if available_options is not None else list(self.DEFAULT_OPTIONS)

        # 1. Robot holding red object (post-grasp, transit position)
        robot_pos = Point3D(x=0.550, y=0.000, z=0.520)
        robot = RobotState(
            position=robot_pos,
            gripper_open=False,
            holding_object_id="red_object_01",
            payload_mass_kg=0.100,
            is_moving=False,
            max_payload_kg=3.0,
            max_reach_m=0.85,
        )

        # 2. Target object held in gripper
        target_obj = WorldObject(
            id="red_object_01",
            name="red_object",
            position=robot_pos,
            bounding_radius_m=0.020,
            mass_kg=0.100,
            state=ObjectState.GRASPED,
            is_target=True,
            is_obstacle=False,
        )

        # 3. Destination target zone
        destination = TargetZone(
            id="blue_target",
            position=Point3D(x=0.400, y=-0.190, z=0.120),
            tolerance_radius_m=0.060,
        )

        # 4. Active blocked-path constraint
        blocked_constraint = ActiveConstraint(
            constraint_id="c_blocked_path_01",
            description="BlockedPath: Obstacle 'blocking_barrier_01' intersects trajectory (clearance 0.066m < required 0.082m)",
            required_clearance_m=0.082,
        )

        # 5. Rejected previous MOVE action outcome
        failed_outcome = LastActionOutcome(
            action_type=ActionType.MOVE,
            status=LastActionStatus.FAILURE,
            error_message="Constraint validation rejected 'act_move_to_target_zone': BlockedPath: Obstacle intersects trajectory",
        )

        # 6. Canonical WorldState
        world_state = WorldState(
            version=4,
            timestamp_ns=time.time_ns(),
            robot=robot,
            objects=[target_obj],
            target=destination,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=[blocked_constraint],
            last_action_outcome=failed_outcome,
        )

        # 7. Canonical PhysicalSituation wrapper
        situation = PhysicalSituation(
            situation_id=situation_id,
            world_state=world_state,
            active_constraints=[blocked_constraint],
            last_action_outcome=failed_outcome,
            task_context={"task_phase": "DISTURBANCE_RECOVERY"},
            timestamp_ns=world_state.timestamp_ns,
        )

        # 8. Deterministic DecisionContext synthesis via DecisionContextBuilder
        disturbance_info = {
            "type": "PATH_BLOCKED",
            "obstacle_id": "blocking_barrier_01",
            "measured_clearance_m": 0.066,
            "required_clearance_m": 0.082,
        }

        context = self._builder.build(
            context_id=context_id,
            situation=situation,
            task_objective=objective,
            candidate_actions=options,
            disturbance_info=disturbance_info,
        )

        # 9. Bounded DecisionQuestion
        question = DecisionQuestion(
            question_id=question_id,
            question_type=DecisionQuestionType.RECOVERY_SELECTION,
            question_text=self.DEFAULT_QUESTION_TEXT,
            available_options=options,
            context=context,
        )

        return situation, question

    def validate_decision(
        self,
        decision: JevDecision,
        question: DecisionQuestion,
    ) -> None:
        """
        Enforces strict bounded validation on JevDecision before acceptance:
        - selected_option must be in available_options
        - confidence must be within [0.0, 1.0] if provided
        - all probabilities must be within [0.0, 1.0]
        - no unknown or free-form actions accepted
        Raises JevValidationError on any violation.
        """
        if not decision.selected_option:
            raise JevValidationError("JevDecision has empty selected_option.")

        if decision.selected_option not in question.available_options:
            raise JevValidationError(
                f"Selected option '{decision.selected_option}' is not one of bounded "
                f"available options: {question.available_options}"
            )

        if decision.confidence is not None:
            if not (0.0 <= decision.confidence <= 1.0):
                raise JevValidationError(
                    f"Confidence score {decision.confidence} is out of bounds [0.0, 1.0]."
                )

        for opt, prob in decision.option_probabilities.items():
            if not (0.0 <= prob <= 1.0):
                raise JevValidationError(
                    f"Option probability {prob} for '{opt}' is out of bounds [0.0, 1.0]."
                )

    def execute_bounded_query(
        self,
        *,
        provider: Optional[JevDecisionProvider] = None,
        question: Optional[DecisionQuestion] = None,
    ) -> BoundedQueryResult:
        """
        Executes ONE controlled bounded decision query.

        Strict safety boundaries:
        - Never commands robot motors or joints.
        - Never calls MuJoCo runtime.
        - Never modifies WorldState.
        - Execution strictly terminates at JevDecision.
        - If provider fails, captures exact provider failure category and sanitized info;
          never fabricates a response or silently substitutes MockJevDecisionProvider.
        """
        # 1. Resolve Question
        if question is None:
            _, question = self.build_recovery_scenario()

        # 2. Resolve Provider (Defaults to Laya if DECISION_PROVIDER=laya or LAYA_BASE_URL set, else RealJevDecisionProvider)
        resolved_provider: JevDecisionProvider
        if provider is not None:
            resolved_provider = provider
        elif os.environ.get("DECISION_PROVIDER", "").lower() == "laya" or os.environ.get("LAYA_BASE_URL"):
            from backend.app.model.laya_config import LayaConfig
            from backend.app.model.laya_provider import LayaDecisionProvider
            resolved_provider = LayaDecisionProvider(LayaConfig.from_env())
        else:
            config = JevConfig.from_env()
            resolved_provider = RealJevDecisionProvider(config)

        # 3. Perform the query with strict failure containment
        start_ns = time.time_ns()
        try:
            decision = resolved_provider.ask(question)
            latency_ms = decision.latency_ms or round((time.time_ns() - start_ns) / 1_000_000.0, 2)

            # 4. Strict bounded option & schema validation
            self.validate_decision(decision, question)

            # 5. Build successful telemetry
            telemetry = BoundedQueryTelemetry(
                question_id=question.question_id,
                question_type=question.question_type.value,
                available_options=question.available_options,
                selected_option=decision.selected_option,
                confidence=decision.confidence,
                answer_confidence=decision.answer_confidence,
                probabilities=decision.option_probabilities,
                latency_ms=latency_ms,
                success=True,
                provider_status="AVAILABLE",
                provider_name=decision.provider or "jev",
                model_version=decision.model_version,
            )

            return BoundedQueryResult(
                question=question,
                decision=decision,
                telemetry=telemetry,
                validation_passed=True,
            )

        except JevProviderError as jpe:
            latency_ms = round((time.time_ns() - start_ns) / 1_000_000.0, 2)
            category = jpe.__class__.__name__
            sanitized_msg = str(jpe)

            telemetry = BoundedQueryTelemetry(
                question_id=question.question_id,
                question_type=question.question_type.value,
                available_options=question.available_options,
                selected_option=None,
                confidence=None,
                probabilities={},
                latency_ms=latency_ms,
                success=False,
                provider_status="FAILED",
                provider_name=getattr(resolved_provider, "config", None) and resolved_provider.config.model or "jev",
                model_version=getattr(resolved_provider, "config", None) and resolved_provider.config.model or None,
                error_category=category,
                error_message=sanitized_msg,
            )

            return BoundedQueryResult(
                question=question,
                decision=None,
                telemetry=telemetry,
                validation_passed=False,
                success=False,
                error_category=category,
                error_message=sanitized_msg,
            )

        except Exception as ex:
            latency_ms = round((time.time_ns() - start_ns) / 1_000_000.0, 2)
            category = ex.__class__.__name__
            sanitized_msg = str(ex)

            telemetry = BoundedQueryTelemetry(
                question_id=question.question_id,
                question_type=question.question_type.value,
                available_options=question.available_options,
                selected_option=None,
                confidence=None,
                probabilities={},
                latency_ms=latency_ms,
                success=False,
                provider_status="ERROR",
                provider_name="unknown",
                model_version=None,
                error_category=category,
                error_message=sanitized_msg,
            )

            return BoundedQueryResult(
                question=question,
                decision=None,
                telemetry=telemetry,
                validation_passed=False,
                success=False,
                error_category=category,
                error_message=sanitized_msg,
            )
