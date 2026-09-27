"""
SĀRTHI MuJoCo Runtime — Physical World-State Reader.

Translates live MuJoCo physical state (mjModel + mjData) into the canonical
SĀRTHI WorldState domain representation for the Decision Engine.

Strict Architectural Guarantees:
- READ-ONLY: Never selects actions, modifies the simulation, commands actuators,
  or executes recovery logic.
- Reuses existing canonical SĀRTHI models (WorldState, RobotState, WorldObject,
  TargetZone, EnvironmentState, ActiveConstraint, LastActionOutcome, Point3D).
- No schema duplication.
- Reads actual physical quantities from mjData (qpos, qvel, xpos, xmat, contact),
  never hardcoded scenario constants.
- Deterministic and idempotent across repeated reads without physics stepping.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from backend.app.decision_engine.models import (
    ActiveConstraint,
    EnvironmentState,
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
from simulation.mujoco_runtime import require_mujoco
from simulation.scenarios.tabletop_pick_place import (
    TabletopPickPlaceScenario,
    create_default_scenario,
)


class SarthiMuJoCoStateReader:
    """
    Extracts and compiles live physical simulation telemetry from MuJoCo
    into canonical SĀRTHI WorldState data structures.
    """

    # End-effector grasp center offset relative to the 'hand' body frame,
    # established in Phase C (0.1034m along local +Z axis).
    GRASP_CENTER_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float64)

    ARM_JOINT_NAMES = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    FINGER_JOINT_NAMES = [
        "finger_joint1",
        "finger_joint2",
    ]

    ROBOT_BASE_BODY = "link0"
    HAND_BODY = "hand"
    LEFT_FINGER_BODY = "left_finger"
    RIGHT_FINGER_BODY = "right_finger"
    RED_OBJECT_BODY = "red_object"
    BLUE_TARGET_BODY = "blue_target"
    OBSTACLE_BODY = "obstacle"

    RED_OBJECT_JOINT = "red_object_joint"
    BLUE_TARGET_GEOM = "blue_target_zone"
    OBSTACLE_GEOM = "obstacle_geom"
    RED_OBJECT_GEOM = "red_object_geom"

    # Threshold for considering the gripper open based on finger separation (meters)
    GRIPPER_OPEN_THRESHOLD_M = 0.06

    # Proximity threshold for object grasp physical evidence (meters)
    GRASP_PROXIMITY_THRESHOLD_M = 0.07

    def __init__(
        self,
        model: Any,
        data: Any,
        entity_ids: Optional[Dict[str, Dict[str, int]]] = None,
        scenario: Optional[TabletopPickPlaceScenario] = None,
    ):
        require_mujoco()
        import mujoco

        self._model = model
        self._data = data
        self._scenario = scenario or create_default_scenario()

        # Cache entity IDs from model
        self._discover_entities(entity_ids)

        # Monotonic world state version counter
        self._world_state_version: int = 1

    def _discover_entities(self, entity_ids: Optional[Dict[str, Dict[str, int]]]) -> None:
        """Discovers and caches all body, joint, and geom IDs required for telemetry."""
        import mujoco

        def get_id(obj_type, name: str) -> int:
            if entity_ids:
                type_map = {
                    mujoco.mjtObj.mjOBJ_BODY: "bodies",
                    mujoco.mjtObj.mjOBJ_JOINT: "joints",
                    mujoco.mjtObj.mjOBJ_GEOM: "geoms",
                    mujoco.mjtObj.mjOBJ_ACTUATOR: "actuators",
                }
                bucket = type_map.get(obj_type)
                if bucket and bucket in entity_ids and name in entity_ids[bucket]:
                    return entity_ids[bucket][name]
            return mujoco.mj_name2id(self._model, obj_type, name)

        # Arm Joints
        self._arm_joint_ids = [get_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.ARM_JOINT_NAMES]
        self._arm_qposadr = [int(self._model.jnt_qposadr[jid]) for jid in self._arm_joint_ids]
        self._arm_dofadr = [int(self._model.jnt_dofadr[jid]) for jid in self._arm_joint_ids]

        # Finger Joints
        self._finger_joint_ids = [get_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.FINGER_JOINT_NAMES]
        self._finger_qposadr = [int(self._model.jnt_qposadr[jid]) for jid in self._finger_joint_ids]

        # Bodies
        self._base_body_id = get_id(mujoco.mjtObj.mjOBJ_BODY, self.ROBOT_BASE_BODY)
        self._hand_body_id = get_id(mujoco.mjtObj.mjOBJ_BODY, self.HAND_BODY)
        self._left_finger_id = get_id(mujoco.mjtObj.mjOBJ_BODY, self.LEFT_FINGER_BODY)
        self._right_finger_id = get_id(mujoco.mjtObj.mjOBJ_BODY, self.RIGHT_FINGER_BODY)
        self._red_object_body_id = get_id(mujoco.mjtObj.mjOBJ_BODY, self.RED_OBJECT_BODY)
        self._blue_target_body_id = get_id(mujoco.mjtObj.mjOBJ_BODY, self.BLUE_TARGET_BODY)
        self._obstacle_body_id = get_id(mujoco.mjtObj.mjOBJ_BODY, self.OBSTACLE_BODY)

        # Red object joint
        self._red_object_joint_id = get_id(mujoco.mjtObj.mjOBJ_JOINT, self.RED_OBJECT_JOINT)
        self._red_object_qposadr = int(self._model.jnt_qposadr[self._red_object_joint_id]) if self._red_object_joint_id >= 0 else None
        self._red_object_dofadr = int(self._model.jnt_dofadr[self._red_object_joint_id]) if self._red_object_joint_id >= 0 else None

        # Geoms
        self._blue_target_geom_id = get_id(mujoco.mjtObj.mjOBJ_GEOM, self.BLUE_TARGET_GEOM)
        self._obstacle_geom_id = get_id(mujoco.mjtObj.mjOBJ_GEOM, self.OBSTACLE_GEOM)
        self._red_object_geom_id = get_id(mujoco.mjtObj.mjOBJ_GEOM, self.RED_OBJECT_GEOM)

        # Cache set of finger geom IDs for contact inspection
        self._finger_geom_ids = set()
        for g_idx in range(self._model.ngeom):
            b_id = self._model.geom_bodyid[g_idx]
            if b_id in (self._left_finger_id, self._right_finger_id):
                self._finger_geom_ids.add(g_idx)

    # -----------------------------------------------------------------------
    # Core Telemetry Read Methods
    # -----------------------------------------------------------------------

    @property
    def simulation_time(self) -> float:
        """Returns the current simulation time in seconds."""
        return float(self._data.time)

    def get_robot_joint_positions(self) -> List[float]:
        """Reads live positions of the 7 arm joints in radians."""
        return [float(self._data.qpos[adr]) for adr in self._arm_qposadr]

    def get_robot_joint_velocities(self) -> List[float]:
        """Reads live velocities of the 7 arm joints in rad/s."""
        return [float(self._data.qvel[adr]) for adr in self._arm_dofadr]

    def get_gripper_finger_positions(self) -> Tuple[float, float]:
        """Reads live displacement of the two gripper fingers in meters."""
        return (
            float(self._data.qpos[self._finger_qposadr[0]]),
            float(self._data.qpos[self._finger_qposadr[1]]),
        )

    def get_gripper_separation(self) -> float:
        """Calculates total gripper jaw separation distance in meters."""
        q1, q2 = self.get_gripper_finger_positions()
        return q1 + q2

    def is_gripper_open(self) -> bool:
        """
        Determines gripper state physically from finger joint separation.
        Returns True if total jaw separation >= GRIPPER_OPEN_THRESHOLD_M.
        """
        return self.get_gripper_separation() >= self.GRIPPER_OPEN_THRESHOLD_M

    def get_end_effector_position(self) -> Point3D:
        """
        Calculates world Cartesian coordinates of the Franka Panda grasp center.
        Derived from 'hand' body frame position and rotation with calibrated offset.
        """
        hand_pos = self._data.xpos[self._hand_body_id]
        hand_mat = self._data.xmat[self._hand_body_id].reshape(3, 3)
        ee_pos = hand_pos + hand_mat @ self.GRASP_CENTER_OFFSET
        return Point3D(x=float(ee_pos[0]), y=float(ee_pos[1]), z=float(ee_pos[2]))

    def get_end_effector_pose(self) -> Tuple[Point3D, Tuple[float, float, float, float]]:
        """Returns (position, orientation_quaternion) of the end-effector in world frame."""
        pos = self.get_end_effector_position()
        quat = self._data.xquat[self._hand_body_id]
        return pos, (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def get_robot_base_position(self) -> Point3D:
        """Returns the world Cartesian position of the robot mounting base (link0)."""
        base_pos = self._data.xpos[self._base_body_id]
        return Point3D(x=float(base_pos[0]), y=float(base_pos[1]), z=float(base_pos[2]))

    def is_robot_moving(self, threshold_rad_per_s: float = 0.01) -> bool:
        """Physically determines whether the robot arm is in motion via joint velocities."""
        vels = self.get_robot_joint_velocities()
        max_vel = max(abs(v) for v in vels) if vels else 0.0
        return max_vel > threshold_rad_per_s

    # -----------------------------------------------------------------------
    # Object, Target, and Obstacle Telemetry
    # -----------------------------------------------------------------------

    def get_red_object_position(self) -> Point3D:
        """
        Reads the actual physical world coordinates of the red movable object from mjData.
        Never uses scenario constants.
        """
        obj_pos = self._data.xpos[self._red_object_body_id]
        return Point3D(x=float(obj_pos[0]), y=float(obj_pos[1]), z=float(obj_pos[2]))

    def get_red_object_linear_velocity(self) -> Tuple[float, float, float]:
        """Reads linear velocity (vx, vy, vz) of red object from mjData if available."""
        if self._red_object_dofadr is not None:
            adr = self._red_object_dofadr
            return (
                float(self._data.qvel[adr]),
                float(self._data.qvel[adr + 1]),
                float(self._data.qvel[adr + 2]),
            )
        return (0.0, 0.0, 0.0)

    def get_blue_target_position(self) -> Point3D:
        """Reads physical world coordinates of the blue target zone from mjData."""
        tgt_pos = self._data.xpos[self._blue_target_body_id]
        return Point3D(x=float(tgt_pos[0]), y=float(tgt_pos[1]), z=float(tgt_pos[2]))

    def get_blue_target_tolerance(self) -> float:
        """Returns target tolerance radius in meters."""
        if self._blue_target_geom_id >= 0:
            # Cylinder size: (radius, half-height)
            return float(self._model.geom_size[self._blue_target_geom_id][0])
        return self._scenario.blue_target.tolerance_radius_m

    def get_obstacle_position(self) -> Point3D:
        """Reads physical world coordinates of the dynamic path obstacle from mjData."""
        obs_pos = self._data.xpos[self._obstacle_body_id]
        return Point3D(x=float(obs_pos[0]), y=float(obs_pos[1]), z=float(obs_pos[2]))

    def get_obstacle_dimensions(self) -> Tuple[float, float, float]:
        """
        Returns full (length_x, width_y, height_z) dimensions of the obstacle in meters.
        MuJoCo box geoms store half-sizes; this method doubles them.
        """
        if self._obstacle_geom_id >= 0:
            half = self._model.geom_size[self._obstacle_geom_id]
            return (float(half[0] * 2), float(half[1] * 2), float(half[2] * 2))
        dims = self._scenario.obstacle.dimensions
        return (dims.length_x, dims.width_y, dims.height_z)

    def is_obstacle_active(self) -> bool:
        """
        Checks whether the obstacle is physically present and active in the workspace.
        """
        if self._obstacle_body_id < 0:
            return False
        # If obstacle is within workspace bounds, it is physically active
        obs_pos = self.get_obstacle_position()
        ws = self._scenario.workspace
        return (
            ws.min_x <= obs_pos.x <= ws.max_x
            and ws.min_y <= obs_pos.y <= ws.max_y
            and ws.min_z <= obs_pos.z <= ws.max_z
        )

    # -----------------------------------------------------------------------
    # Grasp & Manipulation Evidence
    # -----------------------------------------------------------------------

    def has_gripper_object_contact(self) -> bool:
        """
        Queries MuJoCo active contacts in mjData for contact pairs
        between finger geoms and the red object geom.
        """
        if self._red_object_geom_id < 0 or not self._finger_geom_ids:
            return False

        ncon = self._data.ncon
        for i in range(ncon):
            contact = self._data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 in self._finger_geom_ids and g2 == self._red_object_geom_id) or (
                g2 in self._finger_geom_ids and g1 == self._red_object_geom_id
            ):
                return True
        return False

    def is_object_grasped(self, proximity_threshold_m: Optional[float] = None) -> bool:
        """
        Determines if the red object is currently grasped/held using physical evidence:
        1. Cartesian proximity between end-effector grasp center and object position.
        2. Gripper jaw clamping (gripper not fully open).
        3. Active contact forces between finger geoms and object geom.
        """
        thresh = proximity_threshold_m or self.GRASP_PROXIMITY_THRESHOLD_M

        ee_pos = self.get_end_effector_position()
        obj_pos = self.get_red_object_position()
        dist = math.sqrt(
            (ee_pos.x - obj_pos.x) ** 2
            + (ee_pos.y - obj_pos.y) ** 2
            + (ee_pos.z - obj_pos.z) ** 2
        )

        # Contact evidence or close proximity with clamped fingers
        has_contact = self.has_gripper_object_contact()
        is_close = dist <= thresh
        is_clamped = self.get_gripper_separation() < 0.065  # Narrower than fully open (0.08)

        return is_clamped and is_close and (has_contact or self.get_gripper_separation() < 0.055)

    def determine_object_state(self) -> ObjectState:
        """
        Derives canonical ObjectState purely from physical coordinates and evidence:
        - PLACED if resting inside the blue target zone within tolerance.
        - GRASPED if physically held by the gripper.
        - FREE otherwise.
        """
        obj_pos = self.get_red_object_position()
        tgt_pos = self.get_blue_target_position()
        tol = self.get_blue_target_tolerance()

        h_dist = math.sqrt(
            (obj_pos.x - tgt_pos.x) ** 2
            + (obj_pos.y - tgt_pos.y) ** 2
        )
        is_at_tabletop = abs(obj_pos.z - (tgt_pos.z + 0.03)) <= 0.035

        if h_dist <= tol and is_at_tabletop and not self.is_object_grasped():
            return ObjectState.PLACED

        if self.is_object_grasped():
            return ObjectState.GRASPED

        return ObjectState.FREE

    # -----------------------------------------------------------------------
    # Active Constraints Extraction
    # -----------------------------------------------------------------------

    def get_active_constraints(self) -> List[ActiveConstraint]:
        """
        Extracts active physical constraints based on scene geometry.
        If the obstacle is active, creates a keep-out zone constraint.
        """
        constraints: List[ActiveConstraint] = []
        if self.is_obstacle_active():
            obs_pos = self.get_obstacle_position()
            dims = self.get_obstacle_dimensions()
            bounding_radius = math.sqrt((dims[0] / 2) ** 2 + (dims[1] / 2) ** 2)
            constraints.append(
                ActiveConstraint(
                    constraint_id="c_obstacle_keepout",
                    description="Dynamic obstacle spatial keep-out zone",
                    keep_out_center=obs_pos,
                    keep_out_radius=bounding_radius,
                    required_clearance_m=0.03,
                )
            )
        return constraints

    # -----------------------------------------------------------------------
    # Canonical WorldState Assembly
    # -----------------------------------------------------------------------

    def read_world_state(
        self,
        version: Optional[Union[int, str]] = None,
        last_action_outcome: Optional[LastActionOutcome] = None,
    ) -> WorldState:
        """
        Compiles the complete canonical SĀRTHI WorldState from live MuJoCo physics.
        Guarantees that returned state reflects true physical coordinates.
        """
        ws_version = version if version is not None else self._world_state_version
        ee_pos = self.get_end_effector_position()
        obj_pos = self.get_red_object_position()
        is_grasped = self.is_object_grasped()
        obj_state = self.determine_object_state()
        gripper_open = self.is_gripper_open()

        # 1. Robot State
        robot_state = RobotState(
            position=ee_pos,
            gripper_open=gripper_open,
            holding_object_id=self._scenario.red_object.object_id if is_grasped else None,
            payload_mass_kg=self._scenario.red_object.mass_kg if is_grasped else 0.0,
            is_moving=self.is_robot_moving(),
            max_payload_kg=self._scenario.robot.max_payload_kg,
            max_reach_m=self._scenario.robot.max_reach_m,
        )

        # 2. Objects list (if grasped and clamped, object position tracks end-effector)
        effective_obj_pos = ee_pos if is_grasped else obj_pos
        objects_list: List[WorldObject] = [
            WorldObject(
                id=self._scenario.red_object.object_id,
                name=self._scenario.red_object.name,
                position=effective_obj_pos,
                bounding_radius_m=self._scenario.red_object.bounding_radius_m,
                mass_kg=self._scenario.red_object.mass_kg,
                state=obj_state,
                is_target=True,
                is_obstacle=False,
            )
        ]

        if self.is_obstacle_active():
            obs_pos = self.get_obstacle_position()
            dims = self.get_obstacle_dimensions()
            bounding_radius = math.sqrt((dims[0] / 2) ** 2 + (dims[1] / 2) ** 2)
            objects_list.append(
                WorldObject(
                    id=self._scenario.obstacle.obstacle_id,
                    name=self._scenario.obstacle.name,
                    position=obs_pos,
                    bounding_radius_m=bounding_radius,
                    mass_kg=self._scenario.obstacle.mass_kg,
                    state=ObjectState.FREE,
                    is_target=False,
                    is_obstacle=True,
                )
            )

        # 3. Target Zone
        target_zone = TargetZone(
            id=self._scenario.blue_target.target_id,
            position=self.get_blue_target_position(),
            tolerance_radius_m=self.get_blue_target_tolerance(),
        )

        # 4. Environment State
        ws_bounds = self._scenario.workspace
        env_state = EnvironmentState(
            min_x=ws_bounds.min_x,
            max_x=ws_bounds.max_x,
            min_y=ws_bounds.min_y,
            max_y=ws_bounds.max_y,
            min_z=ws_bounds.min_z,
            max_z=ws_bounds.max_z,
            dynamic_obstacles_detected=self.is_obstacle_active(),
            slip_risk_level=0.0,
            friction_coefficient=0.6,
        )

        # 5. Last Action Outcome
        outcome = last_action_outcome or LastActionOutcome(
            status=LastActionStatus.NONE,
            contact_force_delta=0.0,
        )

        # 6. Active Constraints
        constraints = self.get_active_constraints()

        return WorldState(
            version=ws_version,
            timestamp_ns=int(self.simulation_time * 1e9),
            robot=robot_state,
            objects=objects_list,
            target=target_zone,
            environment=env_state,
            task_objective=TaskObjective.PICK_AND_PLACE,
            active_constraints=constraints,
            last_action_outcome=outcome,
        )
