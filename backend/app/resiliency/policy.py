"""
SĀRTHI V3-10 — Resiliency Fallback Policy.
Formalizes the explicit 3-tier fallback hierarchy:

Tier 1: Valid AI proposal
        ↓
        Deterministic constraint validation (SarthiDecisionEngine)
        ↓
        Execute only validated action

Tier 2: If AI proposal unavailable or rejected:
        Deterministic candidate if context permits
        ↓
        Deterministic constraint validation (SarthiDecisionEngine)
        ↓
        Execute only validated action

Tier 3: If no candidate is physically valid:
        Deterministic SAFE STOP (hold joint positions, zero velocity)

Strict Architectural Invariant:
Fallback logic does NOT become a second AI engine. It is strictly rule-bounded,
deterministic, and enforced by SarthiDecisionEngine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

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
from backend.app.model.real_jev_provider import JevValidationError
from backend.app.model.v3_models import DecisionQuestion, JevDecision
from backend.app.resiliency.models import (
    DetectionStage,
    FailureType,
    FallbackStrategy,
    ResiliencyRecord,
)

logger = logging.getLogger(__name__)


class SarthiFallbackPolicy:
    """
    Formalized fallback coordinator executing the 3-tier resiliency hierarchy.
    Ensures that every selected action, whether originating from AI or fallback,
    is validated by SarthiDecisionEngine before physical simulation.
    """

    def __init__(
        self,
        engine: Optional[SarthiDecisionEngine] = None,
        candidate_injector: Optional[CandidateActionInjector] = None,
    ) -> None:
        self.engine = engine or SarthiDecisionEngine()
        self.candidate_injector = candidate_injector or CandidateActionInjector()

    def evaluate_with_fallback(
        self,
        world_state: WorldState,
        question: DecisionQuestion,
        ai_decision: Optional[JevDecision] = None,
        ai_error: Optional[str] = None,
        injected_failure_type: Optional[FailureType] = None,
        step_id: str = "step_0",
    ) -> Tuple[Decision, ResiliencyRecord]:
        """
        Executes the 3-tier fallback evaluation for a single decision cycle.

        Args:
            world_state: Active physical WorldState.
            question: Bounded DecisionQuestion presented for this step.
            ai_decision: Optional JevDecision returned by the AI provider.
            ai_error: Optional error string if AI query failed.
            injected_failure_type: Optional failure tag for audit/benchmark scenarios.
            step_id: Identifier for telemetry records.

        Returns:
            Tuple of (authoritative Decision, ResiliencyRecord).
        """
        rejection_reasons: List[str] = []
        source_component = "LayaDecisionProvider" if ai_decision else "UnknownProvider"
        detected_stage = DetectionStage.DECISION_PROPOSAL
        f_type = injected_failure_type or FailureType.NONE

        # -------------------------------------------------------------------
        # TIER 1: Valid AI proposal -> Candidate Action -> Engine Validation
        # -------------------------------------------------------------------
        if ai_decision is not None and not ai_error:
            source_component = ai_decision.provider or "laya"
            try:
                candidate_action = self.candidate_injector.inject_candidate(
                    decision=ai_decision,
                    question=question,
                    world_state=world_state,
                    action_id=f"act_ai_{step_id}_{ai_decision.selected_option.lower()}",
                )

                # Authoritative deterministic validation with fail-safe STOP in candidate pool
                decision, inj_telem = self.candidate_injector.evaluate_candidate_with_engine(
                    engine=self.engine,
                    decision=ai_decision,
                    world_state=world_state,
                    question=question,
                    include_baseline_candidates=True,
                    decision_id=f"dec_tier1_{step_id}",
                )

                if inj_telem.validation_result == "accepted":
                    # Tier 1 Success: AI proposal was safe, valid, and selected
                    record = ResiliencyRecord(
                        failure_id=f"rec_tier1_{step_id}",
                        failure_type=f_type,
                        source_component=source_component,
                        detected_at_stage=DetectionStage.DECISION_PROPOSAL,
                        recovery_strategy=FallbackStrategy.PROPOSAL_ACCEPTED,
                        fallback_action=None,
                        validation_result="accepted",
                        final_status="COMPLETED",
                        rejection_reason=[],
                        safe_stop=decision.selected_action.action_type == ActionType.STOP,
                        provenance=inj_telem.to_dict(),
                    )
                    return decision, record

                # Proposal was physically invalid -> Rejected by engine!
                detected_stage = DetectionStage.DETERMINISTIC_VALIDATION
                if f_type == FailureType.NONE:
                    f_type = FailureType.DETERMINISTIC_VALIDATION_REJECTION
                rejection_reasons = list(inj_telem.rejection_reasons)
                logger.info(
                    "Tier 1 rejection for option '%s': %s",
                    ai_decision.selected_option,
                    rejection_reasons,
                )

            except JevValidationError as val_err:
                # Injection rejected (e.g. out-of-bounds or malformed decision)
                detected_stage = DetectionStage.CANDIDATE_INJECTION
                if f_type == FailureType.NONE:
                    f_type = FailureType.LAYA_OUT_OF_BOUNDS
                rejection_reasons = [str(val_err)]
                logger.warning("Candidate injection error: %s", val_err)

        else:
            # AI proposal unavailable (timeout, connection error, parsing error)
            detected_stage = DetectionStage.DECISION_PROPOSAL
            if ai_error:
                rejection_reasons.append(f"AIProposalUnavailable: {ai_error}")

        # -------------------------------------------------------------------
        # TIER 2: Deterministic Recovery Candidate
        # -------------------------------------------------------------------
        logger.info("Engaging Tier 2 deterministic candidate generation for %s", step_id)
        deterministic_candidates = self._generate_deterministic_candidates(
            world_state=world_state,
            question=question,
            step_id=step_id,
        )

        tier2_decision = self.engine.decide(
            world_state=world_state,
            candidate_actions=deterministic_candidates,
            decision_id=f"dec_tier2_{step_id}",
        )

        selected_act = tier2_decision.selected_action
        if selected_act.action_type != ActionType.STOP:
            # Tier 2 Success: A safe deterministic recovery/nominal candidate was validated
            record = ResiliencyRecord(
                failure_id=f"rec_tier2_{step_id}",
                failure_type=f_type,
                source_component=source_component,
                detected_at_stage=detected_stage,
                recovery_strategy=FallbackStrategy.DETERMINISTIC_RECOVERY,
                fallback_action=selected_act.action_type.value,
                validation_result="fallback",
                final_status="COMPLETED",
                rejection_reason=rejection_reasons,
                safe_stop=False,
                provenance={
                    "strategy": "deterministic_recovery",
                    "action_id": selected_act.action_id,
                    "action_type": selected_act.action_type.value,
                },
            )
            return tier2_decision, record

        # -------------------------------------------------------------------
        # TIER 3: Deterministic SAFE STOP
        # -------------------------------------------------------------------
        logger.warning("Engaging Tier 3 SAFE STOP for %s: No candidate was physically valid", step_id)
        stop_action = CandidateAction(
            action_id=f"act_safe_stop_{step_id}",
            action_type=ActionType.STOP,
            speed_scale=0.0,
            expected_force_n=0.0,
            parameters={
                "mode": "hold_current_joint_positions",
                "reason": "Tier 3 Safe Stop engaged; no safe motion candidate",
            },
        )
        stop_decision = Decision(
            decision_id=f"dec_tier3_stop_{step_id}",
            world_state_version=world_state.version,
            selected_action=stop_action,
            candidate_evaluations=[],
            rejection_reasons={"all": rejection_reasons or ["Safety constraint violation"]},
            decision_factors={"overall_score": 0.0, "safe_stop": 1.0},
            timestamp_ns=world_state.timestamp_ns,
        )

        record = ResiliencyRecord(
            failure_id=f"stop_tier3_{step_id}",
            failure_type=f_type,
            source_component=source_component,
            detected_at_stage=detected_stage,
            recovery_strategy=FallbackStrategy.SAFE_STOP,
            fallback_action="STOP",
            validation_result="rejected",
            final_status="STOPPED_SAFELY",
            rejection_reason=rejection_reasons or ["All candidate actions violated safety constraints"],
            safe_stop=True,
            provenance={"strategy": "safe_stop", "reason": "No safe candidates"},
        )
        return stop_decision, record

    def _generate_deterministic_candidates(
        self,
        world_state: WorldState,
        question: DecisionQuestion,
        step_id: str,
    ) -> List[CandidateAction]:
        """
        Synthesizes standard deterministic candidate actions from WorldState and question.
        Includes candidate actions matching available options, plus fail-safe STOP.
        """
        candidates: List[CandidateAction] = []
        robot = world_state.robot
        target_zone = world_state.target
        target_obj = next((o for o in world_state.objects if o.is_target), None)

        # Always include STOP
        candidates.append(
            CandidateAction(
                action_id=f"act_tier2_stop_{step_id}",
                action_type=ActionType.STOP,
                speed_scale=0.0,
                expected_force_n=0.0,
                parameters={"mode": "hold_current_joint_positions"},
            )
        )

        # Check options in question
        for opt in question.available_options:
            opt_upper = opt.upper()
            if opt_upper == "REPOSITION":
                # Elevated clearance lift
                lift_z = min(world_state.environment.max_z - 0.05, max(0.32, robot.position.z + 0.15))
                candidates.append(
                    CandidateAction(
                        action_id=f"act_tier2_reposition_{step_id}",
                        action_type=ActionType.REPOSITION,
                        target_position={
                            "x": robot.position.x,
                            "y": robot.position.y,
                            "z": lift_z,
                        },
                        speed_scale=0.3,
                        expected_force_n=0.0,
                        parameters={"purpose": "clearance_lift"},
                    )
                )
            elif opt_upper == "MOVE":
                candidates.append(
                    CandidateAction(
                        action_id=f"act_tier2_move_{step_id}",
                        action_type=ActionType.MOVE,
                        target_position=target_zone.position,
                        target_object_id=robot.holding_object_id,
                        speed_scale=0.4,
                        expected_force_n=0.0,
                        parameters={"trajectory_profile": "trapezoidal"},
                    )
                )
            elif opt_upper == "APPROACH" and target_obj is not None:
                candidates.append(
                    CandidateAction(
                        action_id=f"act_tier2_approach_{step_id}",
                        action_type=ActionType.APPROACH,
                        target_object_id=target_obj.id,
                        target_position=target_obj.position,
                        speed_scale=0.5,
                        expected_force_n=0.0,
                        parameters={"approach_axis": "Z", "standoff_m": 0.05},
                    )
                )
            elif opt_upper == "GRASP" and target_obj is not None:
                candidates.append(
                    CandidateAction(
                        action_id=f"act_tier2_grasp_{step_id}",
                        action_type=ActionType.GRASP,
                        target_object_id=target_obj.id,
                        target_position=target_obj.position,
                        speed_scale=0.2,
                        expected_force_n=10.0,
                        parameters={"grip_force_n": 10.0},
                    )
                )
            elif opt_upper == "RELEASE":
                candidates.append(
                    CandidateAction(
                        action_id=f"act_tier2_release_{step_id}",
                        action_type=ActionType.RELEASE,
                        speed_scale=0.2,
                        expected_force_n=0.0,
                        parameters={"action": "open_gripper"},
                    )
                )

        return candidates
