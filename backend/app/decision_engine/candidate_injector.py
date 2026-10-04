"""
SĀRTHI V3-5 — Safe Candidate Action Injection.
Translates bounded fast decision recommendations (JevDecision from Laya/Jev)
into canonical SĀRTHI CandidateAction models for evaluation by SarthiDecisionEngine.

Non-Negotiable Architecture Invariant:
AI models (Laya/Jev) propose or select bounded semantic candidates;
deterministic software (SarthiDecisionEngine) validates physical feasibility;
only validated actions reach lower-level execution.

Jev/Laya MUST NEVER:
- execute actions directly
- create ActionExecution instances
- bypass SarthiDecisionEngine
- command robot motors or joints
- call MuJoCo runtime
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    CandidateEvaluation,
    Decision,
    Point3D,
    WorldState,
)

if TYPE_CHECKING:
    from backend.app.model.real_jev_provider import JevValidationError
    from backend.app.model.v3_models import DecisionQuestion, JevDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateInjectionTelemetry:
    """Audit record capturing provenance and evaluation of an injected candidate."""
    provider: str
    selected_option: str
    candidate_action: str
    candidate_source: str
    validation_result: str  # "accepted" or "rejected"
    rejection_reasons: List[str]
    confidence: Optional[float] = None
    answer_confidence: Optional[float] = None
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes telemetry record to a secret-free dictionary."""
        return {
            "provider": self.provider,
            "selected_option": self.selected_option,
            "candidate_action": self.candidate_action,
            "candidate_source": self.candidate_source,
            "validation_result": self.validation_result,
            "rejection_reasons": list(self.rejection_reasons),
            "confidence": self.confidence,
            "answer_confidence": self.answer_confidence,
            "timestamp_ns": self.timestamp_ns,
        }


class CandidateActionInjector:
    """
    Dedicated adapter converting validated JevDecision proposals
    into canonical CandidateAction objects for deterministic evaluation.
    Enforces strict bounded question option validation and preserves provenance.
    """

    SUPPORTED_ACTIONS: Set[ActionType] = {
        ActionType.APPROACH,
        ActionType.REPOSITION,
        ActionType.GRASP,
        ActionType.MOVE,
        ActionType.RELEASE,
        ActionType.STOP,
    }

    def inject_candidate(
        self,
        decision: JevDecision,
        question: Optional[DecisionQuestion] = None,
        world_state: Optional[WorldState] = None,
        action_id: Optional[str] = None,
    ) -> CandidateAction:
        """
        Converts a validated JevDecision into a canonical SĀRTHI CandidateAction.

        Args:
            decision: Validated JevDecision from Laya or Jev.
            question: Optional DecisionQuestion containing bounded available options.
            world_state: Optional WorldState for contextual kinematic parameter synthesis.
            action_id: Optional custom action identifier.

        Returns:
            CandidateAction: Ready for submission to SarthiDecisionEngine.

        Raises:
            JevValidationError: If decision is invalid, selected option is outside
                               question available options, or action type is unsupported.
        """
        from backend.app.model.real_jev_provider import JevValidationError
        from backend.app.model.v3_models import JevDecision

        if not isinstance(decision, JevDecision):
            raise JevValidationError(f"Expected JevDecision instance, got {type(decision).__name__}")

        if not decision.selected_option:
            raise JevValidationError("JevDecision has empty selected_option.")

        # 1. Enforce Bounded Question Options
        if question is not None:
            if decision.selected_option not in question.available_options:
                raise JevValidationError(
                    f"Selected option '{decision.selected_option}' is not in question's "
                    f"available options: {question.available_options}"
                )

        # 2. Map to ActionType
        selected_raw = decision.selected_option.strip().upper()
        try:
            action_type = ActionType(selected_raw)
        except ValueError as e:
            raise JevValidationError(
                f"Unknown action '{decision.selected_option}'. Must be one of: "
                f"{[a.value for a in self.SUPPORTED_ACTIONS]}"
            ) from e

        if action_type not in self.SUPPORTED_ACTIONS:
            raise JevValidationError(
                f"ActionType '{action_type}' is not supported for candidate injection."
            )

        # 3. Provenance Tracking Metadata
        provider_name = decision.provider or "jev"
        provenance_source = f"bounded_{provider_name}_decision"
        resolved_action_id = (
            action_id
            or f"cand_{provider_name}_{action_type.value.lower()}_{int(decision.timestamp_ns % 100000)}"
        )

        parameters: Dict[str, Any] = {
            "candidate_source": provenance_source,
            "provider": provider_name,
            "model_version": str(decision.model_version or ""),
            "confidence": float(decision.confidence) if decision.confidence is not None else 0.0,
            "answer_confidence": float(decision.answer_confidence) if decision.answer_confidence is not None else 0.0,
            "raw_selected_option": str(decision.selected_option),
            "decision_timestamp_ns": int(decision.timestamp_ns),
        }

        # 4. Contextual Kinematic Parameter Synthesis
        target_pos: Optional[Point3D] = None
        target_obj_id: Optional[str] = None
        speed_scale: float = 0.3
        expected_force_n: float = 0.0

        if action_type == ActionType.STOP:
            speed_scale = 0.0
            expected_force_n = 0.0
            parameters["mode"] = "hold_current_joint_positions"

        elif action_type == ActionType.REPOSITION:
            speed_scale = 0.3
            expected_force_n = 0.0
            parameters["purpose"] = "vertical_clearance_lift"
            if world_state is not None:
                robot = world_state.robot
                env = world_state.environment
                # Lift vertically while respecting robot maximum reach and ceiling
                xy_dist_sq = robot.position.x ** 2 + robot.position.y ** 2
                max_reach_sq = max(0.0, (robot.max_reach_m - 0.03) ** 2)
                if max_reach_sq > xy_dist_sq:
                    max_z_reach = math.sqrt(max_reach_sq - xy_dist_sq)
                    lift_delta = 0.15 if robot.position.z < 0.40 else 0.08
                    lift_z = min(env.max_z - 0.05, max_z_reach, robot.position.z + lift_delta)
                else:
                    lift_z = min(env.max_z - 0.05, robot.position.z)
                target_pos = Point3D(x=robot.position.x, y=robot.position.y, z=round(lift_z, 3))
            else:
                target_pos = Point3D(x=0.55, y=0.0, z=0.60)

        elif action_type == ActionType.MOVE:
            speed_scale = 0.4
            expected_force_n = 0.0
            parameters["trajectory_profile"] = "trapezoidal"
            if world_state is not None:
                target_pos = world_state.target.position
                target_obj_id = world_state.robot.holding_object_id
            else:
                target_pos = Point3D(x=0.40, y=-0.19, z=0.12)

        elif action_type == ActionType.APPROACH:
            speed_scale = 0.5
            expected_force_n = 0.0
            parameters["approach_axis"] = "Z"
            parameters["standoff_m"] = 0.05
            if world_state is not None:
                target_obj = next((obj for obj in world_state.objects if obj.is_target), None)
                if target_obj:
                    target_obj_id = target_obj.id
                    target_pos = target_obj.position

        elif action_type == ActionType.GRASP:
            speed_scale = 0.2
            expected_force_n = 10.0
            parameters["grip_force_n"] = 10.0
            if world_state is not None:
                target_obj = next((obj for obj in world_state.objects if obj.is_target), None)
                if target_obj:
                    target_obj_id = target_obj.id
                    target_pos = target_obj.position

        elif action_type == ActionType.RELEASE:
            speed_scale = 0.1
            expected_force_n = 0.0
            parameters["open_width_m"] = 0.08
            if world_state is not None:
                target_obj_id = world_state.robot.holding_object_id
                target_pos = world_state.robot.position

        return CandidateAction(
            action_id=resolved_action_id,
            action_type=action_type,
            target_object_id=target_obj_id,
            target_position=target_pos,
            speed_scale=speed_scale,
            expected_force_n=expected_force_n,
            parameters=parameters,
        )

    def evaluate_candidate_with_engine(
        self,
        engine: SarthiDecisionEngine,
        decision: JevDecision,
        world_state: WorldState,
        question: Optional[DecisionQuestion] = None,
        include_baseline_candidates: bool = True,
        decision_id: str = "dec_injected",
    ) -> tuple[Decision, CandidateInjectionTelemetry]:
        """
        Safely injects the JevDecision into SarthiDecisionEngine, executes deterministic
        constraint validation and scoring, and returns the Decision and audit telemetry.

        CRITICAL INVARIANTS:
        - The Decision Engine remains the sole physical authority.
        - High Jev/Laya confidence does NOT override physical constraint validation.
        - If the injected candidate violates physical constraints, the engine REJECTS it.
        """
        # 1. Inject candidate action
        injected_action = self.inject_candidate(
            decision, question=question, world_state=world_state
        )

        # 2. Form candidate pool
        candidates_to_evaluate: List[CandidateAction] = [injected_action]

        if include_baseline_candidates:
            # Add fail-safe STOP if not already the injected candidate
            if injected_action.action_type != ActionType.STOP:
                candidates_to_evaluate.append(
                    CandidateAction(
                        action_id="act_stop_failsafe",
                        action_type=ActionType.STOP,
                        speed_scale=0.0,
                        expected_force_n=0.0,
                        parameters={"mode": "hold_current_joint_positions", "source": "failsafe"},
                    )
                )

        # 3. Deterministic Decision Engine Evaluation
        engine_decision = engine.decide(
            world_state=world_state,
            candidate_actions=candidates_to_evaluate,
            decision_id=decision_id,
        )

        # 4. Extract Injected Action Evaluation Status
        injected_eval = next(
            (ev for ev in engine_decision.candidate_evaluations if ev.action.action_id == injected_action.action_id),
            None,
        )

        is_accepted = bool(injected_eval and injected_eval.is_valid and engine_decision.selected_action.action_id == injected_action.action_id)
        rejection_reasons = injected_eval.rejection_reasons if injected_eval else ["Candidate not evaluated"]

        # 5. Build Safe Telemetry
        telemetry = CandidateInjectionTelemetry(
            provider=decision.provider or "jev",
            selected_option=decision.selected_option,
            candidate_action=injected_action.action_type.value,
            candidate_source=injected_action.parameters.get("candidate_source", "unknown"),
            validation_result="accepted" if is_accepted else "rejected",
            rejection_reasons=rejection_reasons,
            confidence=decision.confidence,
            answer_confidence=decision.answer_confidence,
        )

        return engine_decision, telemetry
