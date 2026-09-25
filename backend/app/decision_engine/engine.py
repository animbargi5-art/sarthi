"""
SĀRTHI Decision Engine — Core Engine Coordinator.
Orchestrates deterministic constraint validation, responsibility scoring,
relevance evaluation, consequence estimation, and action selection.
CRITICAL INVARIANT: This engine never directly accesses or commands robot motors.
"""

from typing import List, Optional
from backend.app.decision_engine.constraints import ConstraintValidator
from backend.app.decision_engine.consequences import ConsequenceEvaluator
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    CandidateEvaluation,
    Decision,
    Point3D,
    WorldState,
)
from backend.app.decision_engine.relevance import RelevanceEvaluator
from backend.app.decision_engine.responsibility import ResponsibilityEvaluator
from backend.app.decision_engine.selector import ActionSelector


class SarthiDecisionEngine:
    """
    Production-grade, fully deterministic physical AI decision engine.
    Computes structured, auditable decisions from WorldState snapshots without stochastic LLMs.
    """

    def __init__(
        self,
        weight_relevance: float = 0.40,
        weight_responsibility: float = 0.35,
        weight_consequence: float = 0.25,
    ):
        self.selector = ActionSelector(
            weight_relevance=weight_relevance,
            weight_responsibility=weight_responsibility,
            weight_consequence=weight_consequence,
        )

    def generate_candidate_actions(self, world_state: WorldState) -> List[CandidateAction]:
        """
        Synthesizes a canonical set of candidate actions tailored to the active WorldState.
        Supports all required action primitives: APPROACH, REPOSITION, GRASP, MOVE, RELEASE, STOP.
        """
        candidates: List[CandidateAction] = []
        robot = world_state.robot
        target_zone = world_state.target

        # Identify target object
        target_obj = None
        for obj in world_state.objects:
            if obj.is_target:
                target_obj = obj
                break

        # 1. STOP candidate (Always present fail-safe)
        candidates.append(CandidateAction(
            action_id="act_stop_hold",
            action_type=ActionType.STOP,
            speed_scale=0.0,
            expected_force_n=0.0,
            parameters={"mode": "hold_current_joint_positions"}
        ))

        # 2. APPROACH candidate
        if target_obj is not None:
            candidates.append(CandidateAction(
                action_id="act_approach_target_obj",
                action_type=ActionType.APPROACH,
                target_object_id=target_obj.id,
                target_position=target_obj.position,
                speed_scale=0.5,
                expected_force_n=0.0,
                parameters={"approach_axis": "Z", "standoff_m": 0.05}
            ))

        # 3. GRASP candidate
        if target_obj is not None:
            candidates.append(CandidateAction(
                action_id="act_grasp_target_obj",
                action_type=ActionType.GRASP,
                target_object_id=target_obj.id,
                target_position=target_obj.position,
                speed_scale=0.2,
                expected_force_n=10.0,
                parameters={"grip_force_n": 10.0}
            ))

        # 4. MOVE candidate (Move toward destination zone)
        candidates.append(CandidateAction(
            action_id="act_move_to_target_zone",
            action_type=ActionType.MOVE,
            target_object_id=robot.holding_object_id,
            target_position=target_zone.position,
            speed_scale=0.4,
            expected_force_n=0.0,
            parameters={"trajectory_profile": "trapezoidal"}
        ))

        # 5. REPOSITION candidate (Vertical lift or clearance waypoint)
        lift_position = Point3D(
            x=robot.position.x,
            y=robot.position.y,
            z=min(world_state.environment.max_z - 0.1, robot.position.z + 0.15)
        )
        candidates.append(CandidateAction(
            action_id="act_reposition_clearance",
            action_type=ActionType.REPOSITION,
            target_position=lift_position,
            speed_scale=0.3,
            expected_force_n=0.0,
            parameters={"purpose": "vertical_clearance_lift"}
        ))

        # 6. RELEASE candidate
        candidates.append(CandidateAction(
            action_id="act_release_gripper",
            action_type=ActionType.RELEASE,
            target_object_id=robot.holding_object_id,
            target_position=robot.position,
            speed_scale=0.1,
            expected_force_n=0.0,
            parameters={"open_width_m": 0.08}
        ))

        return candidates

    def decide(
        self,
        world_state: WorldState,
        candidate_actions: Optional[List[CandidateAction]] = None,
        decision_id: str = "dec_auto",
    ) -> Decision:
        """
        Executes deterministic evaluation and action selection for a WorldState.
        Returns:
            Structured Decision object containing the selected action, candidate
            evaluations, rejection reasons, and numerical decision factors.
        """
        # If no explicit candidates passed, generate canonical set
        actions_to_evaluate = candidate_actions if candidate_actions is not None else self.generate_candidate_actions(world_state)

        evaluations: List[CandidateEvaluation] = []

        for action in actions_to_evaluate:
            # 1. Deterministic Constraint Validation
            is_valid, rejection_reasons = ConstraintValidator.validate(action, world_state)

            # 2. Responsibility Evaluation
            resp_score, resp_factors = ResponsibilityEvaluator.evaluate(action, world_state)

            # 3. Relevance Evaluation
            rel_score, rel_factors = RelevanceEvaluator.evaluate(action, world_state)

            # 4. Consequence Evaluation
            con_score, con_factors = ConsequenceEvaluator.evaluate(action, world_state)

            # 5. Candidate Synthesis via ActionSelector
            ev = self.selector.evaluate_candidate(
                action=action,
                is_valid=is_valid,
                rejection_reasons=rejection_reasons,
                responsibility_score=resp_score,
                responsibility_factors=resp_factors,
                relevance_score=rel_score,
                relevance_factors=rel_factors,
                consequence_score=con_score,
                consequence_factors=con_factors,
            )
            evaluations.append(ev)

        # Final deterministic selection
        return self.selector.select(
            evaluations=evaluations,
            world_state=world_state,
            decision_id=decision_id,
        )
