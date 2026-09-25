"""
SĀRTHI Decision Engine — Constraint Validation Module.
Enforces deterministic physical boundaries, kinematic reachability,
obstacle collision avoidance, and operational preconditions.
"""

import math
from typing import List, Tuple
from backend.app.decision_engine.models import (
    ActionType,
    CandidateAction,
    Point3D,
    WorldObject,
    WorldState,
)


class ConstraintValidator:
    """
    Deterministic validator enforcing hard physical, geometric, and kinematic constraints.
    Rejects any candidate action that violates workspace limits, intersects obstacles,
    or breaches active safety thresholds.
    """

    @staticmethod
    def _distance_point_to_segment(p: Point3D, a: Point3D, b: Point3D) -> float:
        """
        Calculate shortest Euclidean distance from 3D point p to line segment ab.
        """
        ab_x = b.x - a.x
        ab_y = b.y - a.y
        ab_z = b.z - a.z
        ab_len_sq = ab_x ** 2 + ab_y ** 2 + ab_z ** 2

        if ab_len_sq < 1e-9:
            # Segment is essentially a single point
            return p.distance_to(a)

        # Vector ap
        ap_x = p.x - a.x
        ap_y = p.y - a.y
        ap_z = p.z - a.z

        # Project point p onto segment ab normalized parameter t
        t = (ap_x * ab_x + ap_y * ab_y + ap_z * ab_z) / ab_len_sq
        t = max(0.0, min(1.0, t))

        # Closest point on segment
        closest_x = a.x + t * ab_x
        closest_y = a.y + t * ab_y
        closest_z = a.z + t * ab_z

        return math.sqrt(
            (p.x - closest_x) ** 2 +
            (p.y - closest_y) ** 2 +
            (p.z - closest_z) ** 2
        )

    @classmethod
    def validate(cls, action: CandidateAction, world_state: WorldState) -> Tuple[bool, List[str]]:
        """
        Validate candidate action against all active boundaries, obstacles, and constraints.
        Returns:
            Tuple of (is_valid: bool, rejection_reasons: List[str])
        """
        rejection_reasons: List[str] = []
        robot = world_state.robot
        env = world_state.environment

        # STOP action is always kinematically and operationally valid (fail-safe primitive)
        if action.action_type == ActionType.STOP:
            return True, []

        # 1. Parameter & Target Geometry Validation
        if action.action_type in (ActionType.APPROACH, ActionType.MOVE, ActionType.REPOSITION):
            if action.target_position is None:
                rejection_reasons.append("ParameterViolation: Action requires a valid target_position")
            else:
                tp = action.target_position

                # Workspace Enclosure Check
                if not (env.min_x <= tp.x <= env.max_x and
                        env.min_y <= tp.y <= env.max_y and
                        env.min_z <= tp.z <= env.max_z):
                    rejection_reasons.append(
                        f"WorkspaceViolation: Target position ({tp.x:.2f}, {tp.y:.2f}, {tp.z:.2f}) "
                        f"exceeds workspace bounding volume"
                    )

                # Kinematic Arm Reachability Check (Distance from base origin [0,0,0])
                origin = Point3D(x=0.0, y=0.0, z=0.0)
                if origin.distance_to(tp) > robot.max_reach_m:
                    rejection_reasons.append(
                        f"ReachabilityViolation: Target distance {origin.distance_to(tp):.3f}m "
                        f"exceeds robot maximum reach {robot.max_reach_m:.3f}m"
                    )

                # Obstacle Trajectory Collision Check (Blocked Path)
                clearance = 0.04
                for obj in world_state.objects:
                    if obj.is_obstacle:
                        min_dist = cls._distance_point_to_segment(obj.position, robot.position, tp)
                        safe_radius = obj.bounding_radius_m + clearance
                        if min_dist < safe_radius:
                            rejection_reasons.append(
                                f"BlockedPath: Obstacle '{obj.id}' intersects trajectory "
                                f"(min distance {min_dist:.3f}m < required {safe_radius:.3f}m)"
                            )

        # 2. Action-Specific Precondition Checks
        if action.action_type == ActionType.GRASP:
            if robot.holding_object_id is not None:
                rejection_reasons.append(
                    f"PreconditionViolation: Gripper already holding object '{robot.holding_object_id}'"
                )

            # Target Object Check
            target_obj: WorldObject = None
            if action.target_object_id:
                for obj in world_state.objects:
                    if obj.id == action.target_object_id:
                        target_obj = obj
                        break

            if target_obj is None:
                rejection_reasons.append("PreconditionViolation: Target object for grasp not found")
            else:
                # Proximity Check (Must be within 0.12m to grasp)
                dist_to_obj = robot.position.distance_to(target_obj.position)
                if dist_to_obj > 0.12:
                    rejection_reasons.append(
                        f"ProximityViolation: Distance to object {dist_to_obj:.3f}m exceeds grasp threshold (0.12m)"
                    )

                # Payload Capacity Check
                if target_obj.mass_kg > robot.max_payload_kg:
                    rejection_reasons.append(
                        f"PayloadViolation: Object mass {target_obj.mass_kg:.2f}kg exceeds "
                        f"robot payload limit {robot.max_payload_kg:.2f}kg"
                    )

        elif action.action_type == ActionType.RELEASE:
            if robot.holding_object_id is None:
                rejection_reasons.append("PreconditionViolation: Gripper is not currently holding any object to release")

        # 3. Active Safety Constraints Verification
        for constraint in world_state.active_constraints:
            # Force Limit Constraint
            if constraint.max_force_newtons is not None:
                if action.expected_force_n > constraint.max_force_newtons:
                    rejection_reasons.append(
                        f"ForceLimitViolation: Expected force {action.expected_force_n:.1f}N "
                        f"exceeds active constraint threshold {constraint.max_force_newtons:.1f}N"
                    )

            # Speed Constraint
            if constraint.max_speed_mps is not None:
                # Assuming nominal max velocity is 1.0 m/s
                actual_speed = action.speed_scale * 1.0
                if actual_speed > constraint.max_speed_mps:
                    rejection_reasons.append(
                        f"SpeedLimitViolation: Commanded speed {actual_speed:.2f}m/s "
                        f"exceeds constraint {constraint.max_speed_mps:.2f}m/s"
                    )

            # Keep-out Zone Constraint
            if constraint.keep_out_center and constraint.keep_out_radius and action.target_position:
                dist_to_center = action.target_position.distance_to(constraint.keep_out_center)
                if dist_to_center < constraint.keep_out_radius:
                    rejection_reasons.append(
                        f"KeepOutZoneViolation: Target inside exclusion zone '{constraint.constraint_id}' "
                        f"({dist_to_center:.3f}m < {constraint.keep_out_radius:.3f}m)"
                    )

        is_valid = len(rejection_reasons) == 0
        return is_valid, rejection_reasons
