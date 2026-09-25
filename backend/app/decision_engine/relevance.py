"""
SĀRTHI Decision Engine — Relevance Evaluation Module.
Evaluates candidate actions against task objective alignment, operational state progression,
and physical workflow continuity.
"""

from typing import List, Tuple
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    EvaluationFactor,
    ObjectState,
    TaskObjective,
    WorldState,
)


class RelevanceEvaluator:
    """
    Deterministic evaluator measuring the task relevance score of candidate actions.
    Relevance measures how effectively an action advances the active task objective.
    """

    @staticmethod
    def evaluate(action: CandidateAction, world_state: WorldState) -> Tuple[float, List[EvaluationFactor]]:
        """
        Evaluate candidate action relevance to current task objective and world state.
        Returns:
            Tuple of (relevance_score [0.0, 1.0], list of EvaluationFactors)
        """
        factors: List[EvaluationFactor] = []
        robot = world_state.robot
        target_zone = world_state.target
        objective = world_state.task_objective

        # Find designated target object
        target_obj = None
        for obj in world_state.objects:
            if obj.is_target:
                target_obj = obj
                break

        # 1. State Phase Alignment Factor
        phase_score = 0.5
        if objective == TaskObjective.PICK_AND_PLACE:
            if robot.holding_object_id is None:
                # Phase: Object Acquisition
                if target_obj is not None:
                    dist_to_obj = robot.position.distance_to(target_obj.position)
                    if dist_to_obj > 0.08:
                        # Far from object: APPROACH is optimal
                        if action.action_type == ActionType.APPROACH:
                            phase_score = 0.95
                        elif action.action_type == ActionType.REPOSITION:
                            phase_score = 0.60
                        elif action.action_type == ActionType.GRASP:
                            phase_score = 0.10  # Premature grasp
                        elif action.action_type == ActionType.RELEASE:
                            phase_score = 0.00  # Nothing to release
                        elif action.action_type == ActionType.STOP:
                            phase_score = 0.30
                        else:
                            phase_score = 0.40
                    else:
                        # Close to object: GRASP is optimal if gripper open
                        if action.action_type == ActionType.GRASP:
                            phase_score = 0.95 if robot.gripper_open else 0.40
                        elif action.action_type == ActionType.APPROACH:
                            phase_score = 0.40  # Already close
                        elif action.action_type == ActionType.REPOSITION:
                            phase_score = 0.50
                        elif action.action_type == ActionType.STOP:
                            phase_score = 0.30
                        else:
                            phase_score = 0.20
            else:
                # Phase: Object Transport & Delivery
                dist_to_target = robot.position.distance_to(target_zone.position)
                if dist_to_target > target_zone.tolerance_radius_m:
                    # In transit to target zone
                    if action.action_type == ActionType.MOVE:
                        phase_score = 0.95
                    elif action.action_type == ActionType.REPOSITION:
                        phase_score = 0.70
                    elif action.action_type == ActionType.RELEASE:
                        phase_score = 0.05  # Releasing in mid-air
                    elif action.action_type == ActionType.GRASP:
                        phase_score = 0.00  # Already holding
                    elif action.action_type == ActionType.STOP:
                        phase_score = 0.35
                    else:
                        phase_score = 0.40
                else:
                    # At target zone: RELEASE is optimal
                    if action.action_type == ActionType.RELEASE:
                        phase_score = 0.98
                    elif action.action_type == ActionType.MOVE:
                        phase_score = 0.30
                    elif action.action_type == ActionType.STOP:
                        phase_score = 0.40
                    else:
                        phase_score = 0.10

        elif objective == TaskObjective.HOLD_POSITION:
            phase_score = 0.95 if action.action_type == ActionType.STOP else 0.10
        elif objective == TaskObjective.CLEAR_OBSTACLE:
            if action.action_type in (ActionType.REPOSITION, ActionType.MOVE):
                phase_score = 0.90
            elif action.action_type == ActionType.STOP:
                phase_score = 0.50
            else:
                phase_score = 0.30

        factors.append(EvaluationFactor(
            name="phase_alignment",
            score=phase_score,
            weight=0.45,
            description="Relevance of action to the current operational task phase"
        ))

        # 2. Target Proximity Progress Factor
        progress_score = 0.5
        if action.target_position is not None:
            current_dist = robot.position.distance_to(target_zone.position)
            projected_dist = action.target_position.distance_to(target_zone.position)
            if robot.holding_object_id is not None:
                # Holding object: moving closer to target zone is good
                if projected_dist < current_dist:
                    progress_score = 0.90
                else:
                    progress_score = 0.40
            elif target_obj is not None:
                # Not holding: moving closer to target object is good
                cur_to_obj = robot.position.distance_to(target_obj.position)
                proj_to_obj = action.target_position.distance_to(target_obj.position)
                if proj_to_obj < cur_to_obj:
                    progress_score = 0.92
                else:
                    progress_score = 0.45
        else:
            if action.action_type == ActionType.GRASP:
                progress_score = 0.85 if target_obj and target_obj.state == ObjectState.FREE else 0.20
            elif action.action_type == ActionType.RELEASE:
                progress_score = 0.85 if robot.holding_object_id is not None else 0.05
            elif action.action_type == ActionType.STOP:
                progress_score = 0.40

        factors.append(EvaluationFactor(
            name="goal_progress",
            score=progress_score,
            weight=0.35,
            description="Metric indicating forward progress toward objective endpoint"
        ))

        # 3. Parameter Suitability Factor
        param_score = 0.85
        if action.action_type in (ActionType.APPROACH, ActionType.MOVE):
            if action.target_position is None:
                param_score = 0.10  # Missing target position for movement action
        elif action.action_type == ActionType.GRASP:
            if action.target_object_id is None and target_obj is None:
                param_score = 0.20
        factors.append(EvaluationFactor(
            name="parameter_suitability",
            score=param_score,
            weight=0.20,
            description="Completeness and correctness of action parameters"
        ))

        # Compute composite weighted relevance score
        total_weight = sum(f.weight for f in factors)
        composite_score = sum(f.score * f.weight for f in factors) / total_weight
        final_score = round(max(0.0, min(1.0, composite_score)), 4)

        return final_score, factors
