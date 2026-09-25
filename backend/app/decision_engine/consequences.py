"""
SĀRTHI Decision Engine — Consequence Evaluation Module.
Predicts physical consequences, dynamic stability, slip vulnerability,
and proximity margins under changing physical conditions.
"""

from typing import List, Tuple
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    EvaluationFactor,
    Point3D,
    WorldState,
)


class ConsequenceEvaluator:
    """
    Deterministic evaluator predicting physical outcome stability and risk.
    Scores candidate actions based on dynamic margin, slip susceptibility, and kinetic settling.
    """

    @staticmethod
    def evaluate(action: CandidateAction, world_state: WorldState) -> Tuple[float, List[EvaluationFactor]]:
        """
        Evaluate physical consequences and stability of the candidate action.
        Returns:
            Tuple of (consequence_score [0.0, 1.0], list of EvaluationFactors)
        """
        factors: List[EvaluationFactor] = []
        robot = world_state.robot
        env = world_state.environment

        # 1. Kinetic Stability & Inertial Settling Factor
        # STOP guarantees total kinetic dissipation; high speed induces inertial disturbance
        if action.action_type == ActionType.STOP:
            stability_score = 0.98
        elif action.action_type in (ActionType.GRASP, ActionType.RELEASE):
            stability_score = 0.88
        else:
            # Movement actions: higher speed reduces kinetic stability margin
            stability_score = max(0.2, 1.0 - (0.5 * action.speed_scale))

        factors.append(EvaluationFactor(
            name="kinetic_stability",
            score=round(stability_score, 4),
            weight=0.35,
            description="Anticipated inertial stability during and following execution"
        ))

        # 2. Slip & Contact Margin Factor
        slip_margin_score = 0.90
        if robot.holding_object_id is not None:
            # Carrying payload under variable friction
            slip_vulnerability = env.slip_risk_level * action.speed_scale
            slip_margin_score = max(0.1, 1.0 - slip_vulnerability)
        elif action.action_type == ActionType.GRASP:
            # Clamping force adequacy vs expected force
            if action.expected_force_n > 20.0:
                slip_margin_score = 0.70  # Excessive clamping risk of crushing
            elif action.expected_force_n < 2.0:
                slip_margin_score = 0.40  # Inadequate grip force
            else:
                slip_margin_score = 0.95

        factors.append(EvaluationFactor(
            name="slip_margin",
            score=round(slip_margin_score, 4),
            weight=0.35,
            description="Margin against surface slippage and payload drop"
        ))

        # 3. Obstacle Proximity Buffer Factor
        proximity_buffer_score = 0.85
        if action.target_position is not None:
            min_dist_to_obstacle = 999.0
            for obj in world_state.objects:
                if obj.is_obstacle:
                    dist = action.target_position.distance_to(obj.position)
                    if dist < min_dist_to_obstacle:
                        min_dist_to_obstacle = dist

            if min_dist_to_obstacle < 0.10:
                proximity_buffer_score = 0.30
            elif min_dist_to_obstacle < 0.20:
                proximity_buffer_score = 0.65
            else:
                proximity_buffer_score = 0.95

        factors.append(EvaluationFactor(
            name="obstacle_proximity_buffer",
            score=round(proximity_buffer_score, 4),
            weight=0.30,
            description="Clearance buffer maintained from surrounding physical obstacles"
        ))

        # Compute composite weighted consequence score
        total_weight = sum(f.weight for f in factors)
        composite_score = sum(f.score * f.weight for f in factors) / total_weight
        final_score = round(max(0.0, min(1.0, composite_score)), 4)

        return final_score, factors
