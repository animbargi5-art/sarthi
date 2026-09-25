"""
SĀRTHI Decision Engine — Responsibility Evaluation Module.
Evaluates candidate actions against safety preservation, hazard minimization,
payload stewardship, and past failure mitigation.
"""

from typing import List, Tuple
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    EvaluationFactor,
    LastActionStatus,
    WorldState,
)


class ResponsibilityEvaluator:
    """
    Deterministic evaluator measuring the responsibility score of candidate actions.
    Responsibility reflects hazard prevention, payload stewardship, and prudent failure recovery.
    """

    @staticmethod
    def evaluate(action: CandidateAction, world_state: WorldState) -> Tuple[float, List[EvaluationFactor]]:
        """
        Evaluate candidate action responsibility.
        Returns:
            Tuple of (responsibility_score [0.0, 1.0], list of EvaluationFactors)
        """
        factors: List[EvaluationFactor] = []

        # 1. Base Action Prudence Factor
        # STOP is inherently the most prudent fail-safe action
        if action.action_type == ActionType.STOP:
            base_score = 0.95 if world_state.last_action_outcome.status == LastActionStatus.FAILURE else 0.85
        elif action.action_type == ActionType.REPOSITION:
            base_score = 0.80
        elif action.action_type in (ActionType.APPROACH, ActionType.MOVE):
            base_score = 0.75
        elif action.action_type == ActionType.GRASP:
            base_score = 0.70
        elif action.action_type == ActionType.RELEASE:
            base_score = 0.75
        else:
            base_score = 0.50

        factors.append(EvaluationFactor(
            name="base_prudence",
            score=base_score,
            weight=0.20,
            description="Baseline prudence inherent to action primitive"
        ))

        # 2. Failure Recovery & Disturbance Mitigation Factor
        # If the prior action experienced failure/force anomaly, prioritize stabilization
        last_outcome = world_state.last_action_outcome
        if last_outcome.status == LastActionStatus.FAILURE:
            if action.action_type == ActionType.STOP:
                mitigation_score = 1.0
            elif action.action_type == ActionType.REPOSITION:
                mitigation_score = 0.85
            else:
                # Continuing aggressive move or grasp after failure without repositioning is irresponsible
                mitigation_score = 0.15
        elif last_outcome.status == LastActionStatus.INTERRUPTED:
            mitigation_score = 0.90 if action.action_type in (ActionType.STOP, ActionType.REPOSITION) else 0.40
        else:
            mitigation_score = 0.90

        factors.append(EvaluationFactor(
            name="failure_mitigation",
            score=mitigation_score,
            weight=0.30,
            description="Appropriateness of response given previous execution outcome"
        ))

        # 3. Payload Stewardship & Capacity Factor
        payload_stewardship_score = 1.0
        robot = world_state.robot
        target_obj = None
        if action.target_object_id:
            for obj in world_state.objects:
                if obj.id == action.target_object_id:
                    target_obj = obj
                    break

        # Check payload capacity
        if action.action_type == ActionType.GRASP and target_obj:
            if target_obj.mass_kg > robot.max_payload_kg:
                payload_stewardship_score = 0.0  # Exceeds rated arm payload
            else:
                ratio = target_obj.mass_kg / robot.max_payload_kg
                payload_stewardship_score = max(0.2, 1.0 - (0.5 * ratio))
        elif robot.holding_object_id is not None:
            # Carrying an object requires speed caution
            if action.speed_scale > 0.8:
                payload_stewardship_score = 0.50
            else:
                payload_stewardship_score = 0.90
        elif action.action_type == ActionType.RELEASE:
            # Releasing when not near target is reckless
            dist_to_target = robot.position.distance_to(world_state.target.position)
            if dist_to_target > world_state.target.tolerance_radius_m * 3:
                payload_stewardship_score = 0.20  # Releasing in inappropriate location
            else:
                payload_stewardship_score = 0.95

        factors.append(EvaluationFactor(
            name="payload_stewardship",
            score=payload_stewardship_score,
            weight=0.25,
            description="Preservation and stewardship of physical payload"
        ))

        # 4. Environmental Dynamics & Friction Prudence Factor
        env = world_state.environment
        speed = action.speed_scale
        if env.slip_risk_level > 0.4:
            # High slip environment penalizes high speed
            friction_score = max(0.1, 1.0 - (speed * env.slip_risk_level))
        else:
            friction_score = 0.90

        factors.append(EvaluationFactor(
            name="friction_prudence",
            score=friction_score,
            weight=0.25,
            description="Sensitivity to surface friction and slip vulnerability"
        ))

        # Compute composite weighted responsibility score
        total_weight = sum(f.weight for f in factors)
        composite_score = sum(f.score * f.weight for f in factors) / total_weight
        final_score = round(max(0.0, min(1.0, composite_score)), 4)

        return final_score, factors
