"""
SĀRTHI Decision Engine — Action Selector Module.
Deterministic ranking and selection of candidate actions based on explicit numerical factors.
"""

from typing import Dict, List, Optional
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    CandidateEvaluation,
    Decision,
    EvaluationFactor,
    LastActionStatus,
    WorldState,
)


class ActionSelector:
    """
    Deterministic selector that evaluates candidate actions using explicit numerical weights.
    Guarantees reproducible, transparent action selection without stochastic sampling.
    """

    def __init__(
        self,
        weight_relevance: float = 0.40,
        weight_responsibility: float = 0.35,
        weight_consequence: float = 0.25,
    ):
        total = weight_relevance + weight_responsibility + weight_consequence
        self.w_relevance = weight_relevance / total
        self.w_responsibility = weight_responsibility / total
        self.w_consequence = weight_consequence / total

    def evaluate_candidate(
        self,
        action: CandidateAction,
        is_valid: bool,
        rejection_reasons: List[str],
        responsibility_score: float,
        responsibility_factors: List[EvaluationFactor],
        relevance_score: float,
        relevance_factors: List[EvaluationFactor],
        consequence_score: float,
        consequence_factors: List[EvaluationFactor],
    ) -> CandidateEvaluation:
        """
        Synthesizes modular factor scores into an overall deterministic evaluation.
        """
        all_factors = responsibility_factors + relevance_factors + consequence_factors

        if not is_valid:
            overall_score = 0.0
        else:
            weighted_sum = (
                self.w_relevance * relevance_score +
                self.w_responsibility * responsibility_score +
                self.w_consequence * consequence_score
            )
            overall_score = round(max(0.0, min(1.0, weighted_sum)), 4)

        return CandidateEvaluation(
            action=action,
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            responsibility_score=responsibility_score,
            relevance_score=relevance_score,
            consequence_score=consequence_score,
            overall_score=overall_score,
            factors=all_factors,
        )

    def select(
        self,
        evaluations: List[CandidateEvaluation],
        world_state: WorldState,
        decision_id: str = "dec_001",
    ) -> Decision:
        """
        Selects the optimal candidate action based on explicit numerical criteria.
        Guarantees deterministic output for identical inputs.
        """
        if not evaluations:
            # Deterministic failsafe if empty candidate list passed
            failsafe_action = CandidateAction(
                action_id="failsafe_stop",
                action_type=ActionType.STOP,
                parameters={"reason": "Empty candidate evaluation list"}
            )
            return Decision(
                decision_id=decision_id,
                world_state_version=world_state.version,
                selected_action=failsafe_action,
                candidate_evaluations=[],
                rejection_reasons={},
                decision_factors={"overall_score": 1.0, "responsibility_score": 1.0},
                timestamp_ns=world_state.timestamp_ns,
            )

        rejection_map: Dict[str, List[str]] = {}
        for ev in evaluations:
            if not ev.is_valid or ev.rejection_reasons:
                rejection_map[ev.action.action_id] = list(ev.rejection_reasons)

        valid_evaluations = [ev for ev in evaluations if ev.is_valid]

        if not valid_evaluations:
            # If all candidates are invalid, fallback to deterministic STOP
            failsafe_action = CandidateAction(
                action_id="failsafe_stop",
                action_type=ActionType.STOP,
                parameters={"reason": "All candidate actions violated safety constraints"}
            )
            return Decision(
                decision_id=decision_id,
                world_state_version=world_state.version,
                selected_action=failsafe_action,
                candidate_evaluations=evaluations,
                rejection_reasons=rejection_map,
                decision_factors={
                    "overall_score": 0.0,
                    "relevance_score": 0.0,
                    "responsibility_score": 1.0,
                    "consequence_score": 1.0,
                    "fallback_engaged": 1.0,
                },
                timestamp_ns=world_state.timestamp_ns,
            )

        # Deterministic sorting key:
        # 1. Overall score (descending)
        # 2. Responsibility score (descending)
        # 3. Preference for STOP if prior action failed
        # 4. Action ID (alphabetical ascending for absolute tie-break consistency)
        is_failure_state = (world_state.last_action_outcome.status == LastActionStatus.FAILURE)

        def sort_key(ev: CandidateEvaluation):
            stop_bias = 1 if (is_failure_state and ev.action.action_type == ActionType.STOP) else 0
            # We negate numerical scores for descending order
            return (
                -ev.overall_score,
                -stop_bias,
                -ev.responsibility_score,
                -ev.relevance_score,
                ev.action.action_id,
            )

        sorted_evaluations = sorted(valid_evaluations, key=sort_key)
        best_ev = sorted_evaluations[0]

        decision_factors = {
            "overall_score": best_ev.overall_score,
            "relevance_score": best_ev.relevance_score,
            "responsibility_score": best_ev.responsibility_score,
            "consequence_score": best_ev.consequence_score,
            "weight_relevance": self.w_relevance,
            "weight_responsibility": self.w_responsibility,
            "weight_consequence": self.w_consequence,
        }

        return Decision(
            decision_id=decision_id,
            world_state_version=world_state.version,
            selected_action=best_ev.action,
            candidate_evaluations=evaluations,
            rejection_reasons=rejection_map,
            decision_factors=decision_factors,
            timestamp_ns=world_state.timestamp_ns,
        )
