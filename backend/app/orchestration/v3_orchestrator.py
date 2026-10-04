"""
SĀRTHI V3-6 — Human Command + Physical Situation End-to-End Orchestrator.
Connects the full SĀRTHI V3 cognitive-to-physical pipeline:
  Human Instruction + Physical Situation (MuJoCo)
  -> Task Understanding (Nemotron / ModelProvider)
  -> DecisionContextBuilder (Context synthesis & minimization)
  -> DecisionQuestion (Strictly bounded candidate options)
  -> Local Laya (Fast bounded decision proposal via JevDecisionProvider)
  -> CandidateActionInjector (Non-authoritative candidate translation & provenance)
  -> SarthiDecisionEngine (Deterministic constraint validation & action selection)
  -> SarthiMuJoCoRuntime (Closed-loop physical execution & verification)
  -> Verification & Dynamic Obstacle Recovery
  -> Secret-free Telemetry Audit Record

Strict Architectural Invariants:
1. Nemotron remains responsible for task understanding only; zero motor/execution access.
2. Laya is the fast bounded decision layer; zero motor/execution access.
3. JevDecisionProvider remains the provider abstraction boundary.
4. CandidateActionInjector converts only bounded decisions into canonical CandidateActions.
5. SarthiDecisionEngine remains the sole physical action authority.
6. ConstraintValidator remains authoritative for physical feasibility.
7. Laya confidence/probabilities NEVER override deterministic physical validation.
8. No direct AI -> MuJoCo / robot execution path exists.
9. Secrets and API credentials are NEVER serialized in telemetry, logs, or exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from backend.app.decision_engine.candidate_injector import (
    CandidateActionInjector,
    CandidateInjectionTelemetry,
)
from backend.app.decision_engine.engine import SarthiDecisionEngine
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Decision,
    WorldState,
)
from backend.app.model.laya_provider import LayaDecisionProvider
from backend.app.model.mock_provider import MockModelProvider
from backend.app.model.models import TaskUnderstanding
from backend.app.model.nebius_provider import (
    NebiusNemotronProvider,
    NebiusProviderError,
    _sanitize_error_message,
)
from backend.app.model.provider import ModelProvider
from backend.app.model.real_jev_provider import (
    JevProviderError,
    JevValidationError,
)
from backend.app.model.task_understanding import TaskUnderstandingService
from backend.app.model.v3_models import (
    DecisionContext,
    DecisionQuestion,
    DecisionQuestionType,
    JevDecision,
)
from backend.app.orchestration.context_builder import DecisionContextBuilder
from backend.app.orchestration.loop import TaskRunStatus
from backend.app.orchestration.task_interpreter import TaskInterpreter
from simulation.adapters.base import SimulationAdapter
from simulation.core.events import ActionExecutionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry Sanitization Helper
# ---------------------------------------------------------------------------

def _sanitize_data(data: Any) -> Any:
    """Recursively scrubs API keys, bearer tokens, and secrets from data structures."""
    if isinstance(data, str):
        # Redact known credential patterns
        sanitized = data
        for pattern in ["sk-", "nbf_", "Bearer ", "API_KEY", "SECRET", "TOKEN", "PASSWORD"]:
            if pattern.lower() in sanitized.lower():
                sanitized = re.sub(
                    rf"{re.escape(pattern)}[A-Za-z0-9_\-\.]*",
                    "[REDACTED]",
                    sanitized,
                    flags=re.IGNORECASE,
                )
        return sanitized
    elif isinstance(data, dict):
        scrubbed = {}
        for k, v in data.items():
            if any(s in k.lower() for s in ["key", "secret", "token", "password", "auth"]):
                scrubbed[k] = "[REDACTED]"
            else:
                scrubbed[k] = _sanitize_data(v)
        return scrubbed
    elif isinstance(data, (list, tuple)):
        return [_sanitize_data(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# Structured Telemetry Records
# ---------------------------------------------------------------------------

@dataclass
class V3StepTelemetry:
    """Audit record capturing a single V3 closed-loop execution step."""
    step_number: int
    question_id: str
    question_text: str
    available_options: List[str]
    laya_provider: str
    laya_model: str
    selected_option: Optional[str]
    laya_probabilities: Dict[str, float]
    laya_confidence: Optional[float]
    answer_confidence: Optional[float]
    laya_latency_ms: Optional[float]
    candidate_action: Optional[str]
    deterministic_validation_result: str  # "accepted", "rejected", "fallback"
    rejection_reasons: List[str]
    final_decision_action: str
    final_decision_score: float
    execution_success: bool
    verification_success: bool
    world_state_version_before: Union[int, str]
    world_state_version_after: Union[int, str]
    timestamp_ns: int = field(default_factory=time.time_ns)
    latency: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_data({
            "step_number": self.step_number,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "available_options": list(self.available_options),
            "laya_provider": self.laya_provider,
            "laya_model": self.laya_model,
            "selected_option": self.selected_option,
            "laya_probabilities": dict(self.laya_probabilities),
            "laya_confidence": self.laya_confidence,
            "answer_confidence": self.answer_confidence,
            "laya_latency_ms": self.laya_latency_ms,
            "candidate_action": self.candidate_action,
            "deterministic_validation_result": self.deterministic_validation_result,
            "rejection_reasons": list(self.rejection_reasons),
            "final_decision_action": self.final_decision_action,
            "final_decision_score": self.final_decision_score,
            "execution_success": self.execution_success,
            "verification_success": self.verification_success,
            "world_state_version_before": self.world_state_version_before,
            "world_state_version_after": self.world_state_version_after,
            "timestamp_ns": self.timestamp_ns,
            "latency": dict(self.latency),
        })


@dataclass
class V3ExecutionTelemetry:
    """Comprehensive end-to-end telemetry record for SĀRTHI V3 execution."""
    run_id: str
    human_instruction: str
    task_understanding: Dict[str, Any]
    context_id: str
    decision_question: Dict[str, Any]
    laya_provider: str
    laya_model: str
    selected_option: Optional[str]
    laya_probabilities: Dict[str, float]
    laya_confidence: Optional[float]
    answer_confidence: Optional[float]
    candidate_action: Optional[str]
    deterministic_validation_result: str
    rejection_reasons: List[str]
    final_decision: Dict[str, Any]
    execution_result: Dict[str, Any]
    recovery_events: List[Dict[str, Any]]
    final_task_status: str
    final_world_state_version: Union[int, str]
    completed: bool
    latency_markers: Dict[str, float]
    timestamp_iso: str
    latency: Dict[str, float] = field(default_factory=dict)
    steps: List[V3StepTelemetry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_data({
            "run_id": self.run_id,
            "human_instruction": self.human_instruction,
            "task_understanding": self.task_understanding,
            "context_id": self.context_id,
            "decision_question": self.decision_question,
            "laya_provider": self.laya_provider,
            "laya_model": self.laya_model,
            "selected_option": self.selected_option,
            "laya_probabilities": dict(self.laya_probabilities),
            "laya_confidence": self.laya_confidence,
            "answer_confidence": self.answer_confidence,
            "candidate_action": self.candidate_action,
            "deterministic_validation_result": self.deterministic_validation_result,
            "rejection_reasons": list(self.rejection_reasons),
            "final_decision": self.final_decision,
            "execution_result": self.execution_result,
            "recovery_events": list(self.recovery_events),
            "final_task_status": self.final_task_status,
            "final_world_state_version": self.final_world_state_version,
            "completed": self.completed,
            "latency_markers": dict(self.latency_markers),
            "latency": dict(self.latency),
            "timestamp_iso": self.timestamp_iso,
            "steps": [s.to_dict() for s in self.steps],
        })

    def save_to_file(self, filepath: str) -> None:
        """Persists secret-free telemetry to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# V3 Orchestration Service
# ---------------------------------------------------------------------------

class V3OrchestrationService:
    """
    SĀRTHI V3-6 Closed-Loop Orchestration Coordinator.
    Wired strictly according to the non-authoritative architectural invariants.
    """

    def __init__(
        self,
        adapter: Optional[SimulationAdapter] = None,
        engine: Optional[SarthiDecisionEngine] = None,
        context_builder: Optional[DecisionContextBuilder] = None,
        candidate_injector: Optional[CandidateActionInjector] = None,
        model_provider: Optional[ModelProvider] = None,
        laya_provider: Optional[Any] = None,
        max_steps: int = 10,
        sim_adapter: Optional[SimulationAdapter] = None,
        decision_engine: Optional[SarthiDecisionEngine] = None,
    ) -> None:
        actual_adapter = adapter or sim_adapter
        if actual_adapter is None:
            raise ValueError("A simulation adapter must be provided via 'adapter' or 'sim_adapter'.")
        self.adapter = actual_adapter
        self.engine = engine or decision_engine or SarthiDecisionEngine()
        self.context_builder = context_builder or DecisionContextBuilder(coordinate_precision=3)
        self.candidate_injector = candidate_injector or CandidateActionInjector()

        # Model provider setup with graceful fallback / env loading
        if model_provider is not None:
            self.model_provider = model_provider
        else:
            # Check for configs/.env
            if not os.environ.get("NEBIUS_API_KEY"):
                env_path = os.path.join(os.getcwd(), "configs", ".env")
                if os.path.exists(env_path):
                    try:
                        from dotenv import load_dotenv
                        load_dotenv(env_path)
                    except ImportError:
                        pass

            if os.environ.get("NEBIUS_API_KEY"):
                try:
                    self.model_provider = NebiusNemotronProvider()
                except Exception as exc:
                    logger.warning("Could not initialize NebiusNemotronProvider: %s", exc)
                    self.model_provider = MockModelProvider()
            else:
                self.model_provider = MockModelProvider()

        # Laya provider setup
        if laya_provider is not None:
            self.laya_provider = laya_provider
        else:
            self.laya_provider = LayaDecisionProvider()

        self.task_service = TaskUnderstandingService(self.model_provider)
        self.max_steps = max_steps

    # -----------------------------------------------------------------------
    # Task Completion Predicate
    # -----------------------------------------------------------------------

    def is_task_completed(self, world_state: WorldState) -> bool:
        """
        Determines whether the pick-and-place task is physically completed:
        1. Target object is not held.
        2. Gripper is open.
        3. Target object is inside target zone tolerance.
        4. Last executed action was RELEASE.
        """
        robot = world_state.robot
        target_zone = world_state.target

        if robot.holding_object_id is not None or not robot.gripper_open:
            return False

        target_obj = next((obj for obj in world_state.objects if obj.is_target), None)
        if target_obj is None:
            return False

        horizontal_dist = (
            (target_obj.position.x - target_zone.position.x) ** 2 +
            (target_obj.position.y - target_zone.position.y) ** 2
        ) ** 0.5

        if horizontal_dist > target_zone.tolerance_radius_m:
            return False

        last_outcome = world_state.last_action_outcome
        if not last_outcome:
            return False

        action_type_val = (
            last_outcome.action_type.value
            if isinstance(last_outcome.action_type, ActionType)
            else str(last_outcome.action_type)
        )
        status_val = (
            last_outcome.status.value
            if hasattr(last_outcome.status, "value")
            else str(last_outcome.status)
        )
        return action_type_val == "RELEASE" and status_val == "SUCCESS"

    # -----------------------------------------------------------------------
    # Bounded Question Synthesis
    # -----------------------------------------------------------------------

    def build_bounded_question(
        self,
        world_state: WorldState,
        step_number: int,
        task_objective: str = "Move the red object to the blue target.",
    ) -> DecisionQuestion:
        """
        Synthesizes a strictly bounded DecisionQuestion based on the current physical situation.
        Available options are restricted to discrete, valid domain choices.
        """
        robot = world_state.robot
        has_obstacles = any(obj.is_obstacle for obj in world_state.objects)
        has_blocked_constraint = any(
            "blocked" in c.description.lower() or "obstacle" in c.description.lower()
            for c in world_state.active_constraints
        )

        question_id = f"q_v3_step_{step_number}_{int(time.time() * 1000) % 100000}"

        is_elevated = robot.position.z >= 0.28
        target_zone = world_state.target
        dist_to_zone = (
            (robot.position.x - target_zone.position.x) ** 2 +
            (robot.position.y - target_zone.position.y) ** 2
        ) ** 0.5
        at_destination = dist_to_zone <= (target_zone.tolerance_radius_m + 0.05)

        # 1. Object grasped and arrived at destination: RELEASE
        if robot.holding_object_id is not None and at_destination:
            question_type = DecisionQuestionType.RECOVERY_SELECTION
            available_options = ["RELEASE", "STOP"]
            question_text = "Which action should be executed to release the object?"
            disturbance_info = None

        # 2. Lift / clearance transit: Robot holding object at low altitude
        elif robot.holding_object_id is not None and not is_elevated:
            question_type = DecisionQuestionType.RECOVERY_SELECTION
            available_options = ["REPOSITION", "STOP"]
            question_text = "Which action should be executed to elevate the object for transit?"
            disturbance_info = {
                "type": "PATH_BLOCKED",
                "obstacle_detected": True,
            } if (has_obstacles or has_blocked_constraint) else None

        # 3. Nominal pre-approach: Robot at home, gripper open, not holding
        elif robot.holding_object_id is None and robot.gripper_open:
            # Check distance to target object
            target_obj = next((o for o in world_state.objects if o.is_target), None)
            if target_obj:
                dist = (
                    (robot.position.x - target_obj.position.x) ** 2 +
                    (robot.position.y - target_obj.position.y) ** 2 +
                    (robot.position.z - target_obj.position.z) ** 2
                ) ** 0.5
            else:
                dist = 1.0

            if dist > 0.15:
                question_type = DecisionQuestionType.RECOVERY_SELECTION
                available_options = ["APPROACH", "STOP"]
                question_text = "Which action should be executed to initiate the task?"
            else:
                question_type = DecisionQuestionType.RECOVERY_SELECTION
                available_options = ["GRASP", "STOP"]
                question_text = "Which action should be executed to grasp the object?"
            disturbance_info = None

        # 4. Object grasped and in transit: transport to target zone
        elif robot.holding_object_id is not None:
            question_type = DecisionQuestionType.RECOVERY_SELECTION
            available_options = ["MOVE", "STOP"]
            question_text = "Which action should be executed to transport the object?"
            disturbance_info = None

        else:
            question_type = DecisionQuestionType.RECOVERY_SELECTION
            available_options = ["STOP", "REPOSITION"]
            question_text = "Which safety action should be executed?"
            disturbance_info = None

        # Build DecisionContext using DecisionContextBuilder
        context = self.context_builder.build_context(
            world_state=world_state,
            task_objective=task_objective,
            candidate_actions=available_options,
            context_id=f"ctx_v3_{question_id}",
            disturbance_info=disturbance_info,
        )

        return DecisionQuestion(
            question_id=question_id,
            question_type=question_type,
            question_text=question_text,
            available_options=available_options,
            context=context,
        )

    # -----------------------------------------------------------------------
    # Target Resolution
    # -----------------------------------------------------------------------

    def _resolve_targets(
        self,
        understanding: TaskUnderstanding,
        world_state: WorldState,
    ) -> Optional[str]:
        """
        Validates that targets specified in TaskUnderstanding physically exist in WorldState.
        Returns error string if missing, None if verified.
        """
        target_obj_label = understanding.target_object
        target_loc_label = understanding.target_location

        if target_obj_label and target_obj_label not in ("unspecified_object",):
            has_target = any(obj.is_target for obj in world_state.objects)
            if not has_target:
                return (
                    f"TARGET_OBJECT_NOT_FOUND: '{target_obj_label}' "
                    f"not found in current WorldState. No physical action executed."
                )

        if target_loc_label and target_loc_label not in ("unspecified_location",):
            zone = world_state.target
            zone_id_norm = zone.id.lower().replace("_", "").replace(" ", "")
            label_norm = target_loc_label.lower().replace("_", "").replace(" ", "")
            if label_norm not in zone_id_norm and zone_id_norm not in label_norm:
                found = any(label_norm in obj.id.lower() for obj in world_state.objects)
                if not found:
                    return (
                        f"TARGET_LOCATION_NOT_FOUND: '{target_loc_label}' "
                        f"not found in current WorldState. No physical action executed."
                    )

        return None

    # -----------------------------------------------------------------------
    # Atomic Bounded Decision & Execution Step
    # -----------------------------------------------------------------------

    def execute_v3_step(
        self,
        world_state: WorldState,
        step_number: int,
        task_objective: str = "Move the red object to the blue target.",
        custom_question: Optional[DecisionQuestion] = None,
    ) -> Tuple[TaskRunStatus, V3StepTelemetry, WorldState]:
        """
        Executes one atomic V3 closed-loop control step:
        1. Synthesize bounded DecisionQuestion and DecisionContext.
        2. Query Laya for fast bounded proposal (with deterministic fallback).
        3. Inject candidate via CandidateActionInjector.
        4. Evaluate candidate with SarthiDecisionEngine (authoritative constraints).
        5. Execute ONLY validated Decision through SimulationAdapter.
        6. Verify outcome and update WorldState.
        """
        version_before = world_state.version
        t_step_start = time.perf_counter()

        # 1. Question Formulation (tau_context)
        t_ctx0 = time.perf_counter()
        question = custom_question or self.build_bounded_question(
            world_state=world_state,
            step_number=step_number,
            task_objective=task_objective,
        )
        tau_context_ms = (time.perf_counter() - t_ctx0) * 1000.0

        # 2. Query Laya Provider (tau_jev)
        laya_decision: Optional[JevDecision] = None
        laya_error: Optional[str] = None
        laya_provider_name = getattr(self.laya_provider, "provider_name", "laya")
        laya_model = getattr(self.laya_provider, "model", "laya-rl-agent")

        t_jev0 = time.perf_counter()
        try:
            if hasattr(self.laya_provider, "decide"):
                laya_decision = self.laya_provider.decide(question)
            else:
                laya_decision = self.laya_provider.ask(question)
        except Exception as exc:
            laya_error = _sanitize_error_message(str(exc))
            logger.warning(
                "Laya decision query failed at step %d: %s. Using deterministic fallback.",
                step_number,
                laya_error,
            )
        tau_jev_ms = (time.perf_counter() - t_jev0) * 1000.0
        laya_latency_ms = tau_jev_ms

        # 3. Candidate Injection & Decision Engine Evaluation (tau_validation)
        t_val0 = time.perf_counter()
        candidate_action: Optional[CandidateAction] = None
        engine_decision: Optional[Decision] = None
        validation_result: str = "fallback"
        rejection_reasons: List[str] = []

        if laya_decision is not None:
            try:
                candidate_action = self.candidate_injector.inject_candidate(
                    decision=laya_decision,
                    question=question,
                    world_state=world_state,
                    action_id=f"act_v3_step_{step_number}_{laya_decision.selected_option.lower()}",
                )
                engine_decision, injection_telemetry = self.candidate_injector.evaluate_candidate_with_engine(
                    engine=self.engine,
                    decision=laya_decision,
                    world_state=world_state,
                    question=question,
                    decision_id=f"dec_v3_step_{step_number}",
                )
                validation_result = injection_telemetry.validation_result
                rejection_reasons = injection_telemetry.rejection_reasons

            except JevValidationError as val_err:
                logger.warning("Candidate injection rejected by validator: %s", val_err)
                validation_result = "rejected"
                rejection_reasons = [str(val_err)]
                # Deterministic engine fallback
                engine_decision = self.engine.decide(
                    world_state=world_state,
                    decision_id=f"dec_v3_fallback_step_{step_number}",
                )
        else:
            # Deterministic engine fallback when Laya failed/unavailable
            engine_decision = self.engine.decide(
                world_state=world_state,
                decision_id=f"dec_v3_fallback_step_{step_number}",
            )
            validation_result = "fallback"
            if laya_error:
                rejection_reasons = [f"LayaProviderUnavailable: {laya_error}"]

        tau_validation_ms = (time.perf_counter() - t_val0) * 1000.0

        # Extract selected action to execute
        selected_action = engine_decision.selected_action

        # 4. Physical Execution in Simulation Runtime (tau_ik and tau_sim)
        t_exec0 = time.perf_counter()
        exec_result: ActionExecutionResult = self.adapter.execute_action(selected_action)
        raw_exec_ms = (time.perf_counter() - t_exec0) * 1000.0

        details = getattr(exec_result, "details", {}) or {}
        tau_ik_ms = float(details.get("tau_ik_ms", 0.0))
        tau_sim_ms = float(details.get("tau_sim_ms", max(0.0, raw_exec_ms - tau_ik_ms)))
        internal_verify_ms = float(details.get("tau_verify_ms", 0.0))

        # 5. Verification (tau_verify)
        expected_outcome: Dict[str, Any] = {}
        if selected_action.action_type == ActionType.GRASP:
            expected_outcome["gripper_open"] = False
            if selected_action.target_object_id:
                expected_outcome["carrying_object_id"] = selected_action.target_object_id
        elif selected_action.action_type == ActionType.RELEASE:
            expected_outcome["gripper_open"] = True
            expected_outcome["carrying_object_id"] = None

        t_ver0 = time.perf_counter()
        verify_success = self.adapter.verify_action_result(
            selected_action, expected_outcome if expected_outcome else None
        )
        external_verify_ms = (time.perf_counter() - t_ver0) * 1000.0
        tau_verify_ms = internal_verify_ms + external_verify_ms

        # 6. Read updated WorldState
        new_world_state = self.adapter.get_world_state()

        # Step total duration measured independently
        tau_step_total_ms = (time.perf_counter() - t_step_start) * 1000.0

        step_latency = {
            "tau_nemotron_ms": 0.0,
            "tau_context_ms": round(max(0.0, tau_context_ms), 3),
            "tau_jev_ms": round(max(0.0, tau_jev_ms), 3),
            "tau_validation_ms": round(max(0.0, tau_validation_ms), 3),
            "tau_ik_ms": round(max(0.0, tau_ik_ms), 3),
            "tau_sim_ms": round(max(0.0, tau_sim_ms), 3),
            "tau_verify_ms": round(max(0.0, tau_verify_ms), 3),
            "tau_total_ms": round(max(0.0, tau_step_total_ms), 3),
        }

        # Build Step Telemetry
        step_telemetry = V3StepTelemetry(
            step_number=step_number,
            question_id=question.question_id,
            question_text=question.question_text,
            available_options=list(question.available_options),
            laya_provider=laya_provider_name,
            laya_model=laya_model,
            selected_option=laya_decision.selected_option if laya_decision else None,
            laya_probabilities=laya_decision.option_probabilities if laya_decision else {},
            laya_confidence=laya_decision.confidence if laya_decision else None,
            answer_confidence=laya_decision.answer_confidence if laya_decision else None,
            laya_latency_ms=round(max(0.0, laya_latency_ms), 3) if laya_decision else None,
            candidate_action=candidate_action.action_type.value if candidate_action else None,
            deterministic_validation_result=validation_result,
            rejection_reasons=rejection_reasons,
            final_decision_action=selected_action.action_type.value,
            final_decision_score=engine_decision.decision_factors.get("overall_score", 0.0),
            execution_success=exec_result.success,
            verification_success=verify_success,
            world_state_version_before=version_before,
            world_state_version_after=new_world_state.version,
            latency=step_latency,
        )

        # Determine terminal step status
        if selected_action.action_type == ActionType.STOP:
            status = TaskRunStatus.STOPPED_SAFELY
        elif self.is_task_completed(new_world_state):
            status = TaskRunStatus.COMPLETED
        else:
            status = TaskRunStatus.COMPLETED if False else TaskRunStatus.MAX_STEPS_REACHED

        return status, step_telemetry, new_world_state

    # -----------------------------------------------------------------------
    # End-to-End Execution Pipeline
    # -----------------------------------------------------------------------

    def run_instruction(
        self,
        instruction: str,
        step_callback: Optional[Callable[[int, WorldState], None]] = None,
        run_id: Optional[str] = None,
        output_telemetry_path: Optional[str] = None,
    ) -> V3ExecutionTelemetry:
        """
        Executes the complete V3-6 end-to-end integration pipeline from human instruction.
        Flow:
          Human Instruction
          -> TaskUnderstandingService (via ModelProvider)
          -> Target resolution against initial physical WorldState
          -> Closed-loop execution with Laya proposals & deterministic Decision Engine
          -> Physical placement verification
          -> Structured Secret-Free Telemetry Output
        """
        active_run_id = run_id or f"v3_run_{int(time.time() * 1000)}"
        start_run = time.perf_counter()
        latency_markers: Dict[str, float] = {}

        # 1. Cognitive Task Understanding (tau_nemotron)
        start_cognition = time.perf_counter()
        try:
            understanding = self.task_service.understand(instruction)
        except Exception as exc:
            sanitized = _sanitize_error_message(str(exc))
            logger.error("Task understanding failed: %s", sanitized)
            tau_nemo_err_ms = (time.perf_counter() - start_cognition) * 1000.0
            total_run_err_ms = (time.perf_counter() - start_run) * 1000.0
            ws_initial = self.adapter.get_world_state()
            telemetry = V3ExecutionTelemetry(
                run_id=active_run_id,
                human_instruction=instruction,
                task_understanding={"error": sanitized},
                context_id="none",
                decision_question={"error": "cognition_failed"},
                laya_provider=getattr(self.laya_provider, "provider_name", "laya"),
                laya_model=getattr(self.laya_provider, "model", "laya-rl-agent"),
                selected_option=None,
                laya_probabilities={},
                laya_confidence=None,
                answer_confidence=None,
                candidate_action=None,
                deterministic_validation_result="rejected",
                rejection_reasons=[f"CognitionFailure: {sanitized}"],
                final_decision={"action": "NONE"},
                execution_result={"success": False, "failure_reason": sanitized},
                recovery_events=[],
                final_task_status="FAILED",
                final_world_state_version=ws_initial.version,
                completed=False,
                latency_markers={
                    "cognition_latency_ms": round(tau_nemo_err_ms, 3),
                    "total_run_duration_ms": round(total_run_err_ms, 3),
                    "steps_executed": 0.0,
                },
                timestamp_iso=datetime.now(timezone.utc).isoformat(),
                latency={
                    "tau_nemotron_ms": round(tau_nemo_err_ms, 3),
                    "tau_context_ms": 0.0,
                    "tau_jev_ms": 0.0,
                    "tau_validation_ms": 0.0,
                    "tau_ik_ms": 0.0,
                    "tau_sim_ms": 0.0,
                    "tau_verify_ms": 0.0,
                    "tau_total_ms": round(total_run_err_ms, 3),
                },
                steps=[],
            )
            if output_telemetry_path:
                telemetry.save_to_file(output_telemetry_path)
            return telemetry

        tau_nemotron_ms = (time.perf_counter() - start_cognition) * 1000.0
        latency_markers["cognition_latency_ms"] = round(tau_nemotron_ms, 3)

        # 2. Target Resolution against actual initial WorldState
        ws_initial = self.adapter.get_world_state()
        resolution_error = self._resolve_targets(understanding, ws_initial)
        if resolution_error:
            logger.error("Target resolution failed: %s", resolution_error)
            total_run_res_ms = (time.perf_counter() - start_run) * 1000.0
            telemetry = V3ExecutionTelemetry(
                run_id=active_run_id,
                human_instruction=instruction,
                task_understanding=understanding.model_dump(),
                context_id="none",
                decision_question={"error": "target_resolution_failed"},
                laya_provider=getattr(self.laya_provider, "provider_name", "laya"),
                laya_model=getattr(self.laya_provider, "model", "laya-rl-agent"),
                selected_option=None,
                laya_probabilities={},
                laya_confidence=None,
                answer_confidence=None,
                candidate_action=None,
                deterministic_validation_result="rejected",
                rejection_reasons=[resolution_error],
                final_decision={"action": "NONE"},
                execution_result={"success": False, "failure_reason": resolution_error},
                recovery_events=[],
                final_task_status="FAILED",
                final_world_state_version=ws_initial.version,
                completed=False,
                latency_markers={
                    "cognition_latency_ms": round(tau_nemotron_ms, 3),
                    "total_run_duration_ms": round(total_run_res_ms, 3),
                    "steps_executed": 0.0,
                },
                timestamp_iso=datetime.now(timezone.utc).isoformat(),
                latency={
                    "tau_nemotron_ms": round(tau_nemotron_ms, 3),
                    "tau_context_ms": 0.0,
                    "tau_jev_ms": 0.0,
                    "tau_validation_ms": 0.0,
                    "tau_ik_ms": 0.0,
                    "tau_sim_ms": 0.0,
                    "tau_verify_ms": 0.0,
                    "tau_total_ms": round(total_run_res_ms, 3),
                },
                steps=[],
            )
            if output_telemetry_path:
                telemetry.save_to_file(output_telemetry_path)
            return telemetry

        # 3. Closed-Loop Execution Loop
        step_telemetries: List[V3StepTelemetry] = []
        recovery_events: List[Dict[str, Any]] = []
        current_world_state = ws_initial
        terminal_status = TaskRunStatus.MAX_STEPS_REACHED
        completed = False

        step_counter = 0
        while step_counter < self.max_steps:
            step_counter += 1

            # Callback hook (for dynamic disturbance injection, e.g. post-grasp)
            if step_callback:
                step_callback(step_counter, current_world_state)
                current_world_state = self.adapter.get_world_state()

            # Execute atomic step
            status, step_tel, current_world_state = self.execute_v3_step(
                world_state=current_world_state,
                step_number=step_counter,
                task_objective=understanding.objective,
            )
            step_telemetries.append(step_tel)

            # Record recovery events
            if step_tel.final_decision_action == "REPOSITION":
                recovery_events.append({
                    "step": step_counter,
                    "event": "DISTURBANCE_RECOVERY_REPOSITION",
                    "validation_result": step_tel.deterministic_validation_result,
                    "score": step_tel.final_decision_score,
                })

            if status == TaskRunStatus.COMPLETED or self.is_task_completed(current_world_state):
                terminal_status = TaskRunStatus.COMPLETED
                completed = True
                break

            if status == TaskRunStatus.STOPPED_SAFELY:
                terminal_status = TaskRunStatus.STOPPED_SAFELY
                completed = False
                break

        # Compute full-run latency breakdowns
        total_tau_context_ms = sum(s.latency.get("tau_context_ms", 0.0) for s in step_telemetries)
        total_tau_jev_ms = sum(s.latency.get("tau_jev_ms", 0.0) for s in step_telemetries)
        total_tau_validation_ms = sum(s.latency.get("tau_validation_ms", 0.0) for s in step_telemetries)
        total_tau_ik_ms = sum(s.latency.get("tau_ik_ms", 0.0) for s in step_telemetries)
        total_tau_sim_ms = sum(s.latency.get("tau_sim_ms", 0.0) for s in step_telemetries)
        total_tau_verify_ms = sum(s.latency.get("tau_verify_ms", 0.0) for s in step_telemetries)

        total_run_ms = (time.perf_counter() - start_run) * 1000.0
        latency_markers["total_run_duration_ms"] = round(total_run_ms, 3)
        latency_markers["steps_executed"] = float(step_counter)

        latency_breakdown = {
            "tau_nemotron_ms": round(max(0.0, tau_nemotron_ms), 3),
            "tau_context_ms": round(max(0.0, total_tau_context_ms), 3),
            "tau_jev_ms": round(max(0.0, total_tau_jev_ms), 3),
            "tau_validation_ms": round(max(0.0, total_tau_validation_ms), 3),
            "tau_ik_ms": round(max(0.0, total_tau_ik_ms), 3),
            "tau_sim_ms": round(max(0.0, total_tau_sim_ms), 3),
            "tau_verify_ms": round(max(0.0, total_tau_verify_ms), 3),
            "tau_total_ms": round(max(0.0, total_run_ms), 3),
        }

        # Extract representative latest step for top-level telemetry summary
        representative_step = step_telemetries[-1] if step_telemetries else None

        telemetry = V3ExecutionTelemetry(
            run_id=active_run_id,
            human_instruction=instruction,
            task_understanding=understanding.model_dump(),
            context_id=representative_step.question_id if representative_step else "none",
            decision_question={
                "question_id": representative_step.question_id if representative_step else "none",
                "question_text": representative_step.question_text if representative_step else "",
                "available_options": representative_step.available_options if representative_step else [],
            },
            laya_provider=getattr(self.laya_provider, "provider_name", "laya"),
            laya_model=getattr(self.laya_provider, "model", "laya-rl-agent"),
            selected_option=representative_step.selected_option if representative_step else None,
            laya_probabilities=representative_step.laya_probabilities if representative_step else {},
            laya_confidence=representative_step.laya_confidence if representative_step else None,
            answer_confidence=representative_step.answer_confidence if representative_step else None,
            candidate_action=representative_step.candidate_action if representative_step else None,
            deterministic_validation_result=representative_step.deterministic_validation_result if representative_step else "unknown",
            rejection_reasons=representative_step.rejection_reasons if representative_step else [],
            final_decision={
                "action": representative_step.final_decision_action if representative_step else "NONE",
                "score": representative_step.final_decision_score if representative_step else 0.0,
            },
            execution_result={
                "success": representative_step.execution_success if representative_step else False,
                "verified": representative_step.verification_success if representative_step else False,
            },
            recovery_events=recovery_events,
            final_task_status=terminal_status.value,
            final_world_state_version=current_world_state.version,
            completed=completed,
            latency_markers=latency_markers,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            latency=latency_breakdown,
            steps=step_telemetries,
        )

        if output_telemetry_path:
            telemetry.save_to_file(output_telemetry_path)

        return telemetry
