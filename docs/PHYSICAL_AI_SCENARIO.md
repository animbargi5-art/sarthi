# Physical AI Benchmark Scenario Specification: Tabletop Pick-and-Place

## 1. Problem Statement

Standard robotic task execution pipelines rely heavily on open-loop scripted trajectories or brittle state machines. When unexpected physical perturbations occur—such as dynamic obstacles obstructing a pre-planned path—traditional systems either collide, trigger emergency stops requiring human intervention, or blindly re-attempt failed actions.

This benchmark establishes a reproducible, deterministic Physical AI scenario to validate autonomous cognitive perception, physical constraint evaluation, and adaptive recovery in robotic manipulation.

---

## 2. Target Application & Operational Context

- **Application Domain:** Autonomous industrial kitting, flexible warehouse sorting, and laboratory tabletop manipulation.
- **System Objective:** Execute human natural language instructions within dynamic, partially uncertain workspaces without manual waypoint re-programming.
- **Target Robot:** Common 6-DOF and 7-DOF tabletop articulated manipulators (e.g., Franka Emika Panda, Universal Robots UR10e) in simulation (NVIDIA Isaac Sim / Local Core) and subsequent hardware deployment.

---

## 3. Scenario Configuration & Physical Parameters

The benchmark scenario is deterministically locked in `simulation/scenarios/tabletop_pick_place.py` using explicit metric units (meters and kilograms).

| Component | Identifier | Parameter / Property | Metric Value |
| :--- | :--- | :--- | :--- |
| **Scenario** | `tabletop_pick_and_place_mvp` | Canonical Human Instruction | `"Move the red object to the blue target."` |
| **Robot** | `tabletop_manipulator` | Base Mounting Coordinates | $(x=0.00, y=0.00, z=0.20)\,\text{m}$ |
| | | Kinematic Reach Envelope | $0.85\,\text{m}$ maximum reach |
| | | Rated Payload Capacity | $3.00\,\text{kg}$ |
| | | Degrees of Freedom | 7-DOF articulated arm |
| **Manipulated Entity** | `red_object_01` | Initial Centroid Pose | $(x=0.25, y=0.15, z=0.20)\,\text{m}$ |
| | | Bounding Dimensions $(L \times W \times H)$ | $0.05 \times 0.05 \times 0.06\,\text{m}$ |
| | | Collision Bounding Radius | $0.05\,\text{m}$ |
| | | Payload Mass | $0.50\,\text{kg}$ (within rated payload) |
| | | Manipulation Properties | Movable (`True`), Graspable (`True`) |
| **Destination Area** | `blue_target_zone` | Destination Centroid Pose | $(x=0.40, y=-0.20, z=0.20)\,\text{m}$ |
| | | Placement Tolerance Radius | $0.06\,\text{m}$ horizontal tolerance |
| | | Fixation Properties | Movable (`False`), Static Surface Zone |
| **Workspace Bounds** | `workspace` | Spatial Limits $(X, Y, Z)$ | $X \in [-0.8, 0.8], Y \in [-0.8, 0.8], Z \in [0.0, 1.2]\,\text{m}$ |

---

## 4. Nominal Task Execution Flow

In the absence of disturbances, the robot executes the nominal 4-phase manipulation sequence:

```
Human Instruction: "Move the red object to the blue target."
                         │
                         ▼
             [1. APPROACH] -> End-effector reaches pre-grasp pose at (0.25, 0.15, 0.20)
                         │
                         ▼
             [2. GRASP]    -> Gripper closes, confirms contact, attaches payload (0.50 kg)
                         │
                         ▼
             [3. MOVE]     -> Linear/Cartesian transport to target at (0.40, -0.20, 0.20)
                         │
                         ▼
             [4. RELEASE]  -> Gripper opens, detaches payload at blue target zone
```

---

## 5. Dynamic Disturbance: `PATH_BLOCKED`

To evaluate adaptive decision-making, an environmental disturbance is dynamically introduced:

- **Disturbance Primitive:** `PATH_BLOCKED` (`DisturbanceType.PATH_BLOCKED`)
- **Obstacle Identifier:** `blocking_barrier_01`
- **Obstacle Position:** $(x=0.325, y=-0.025, z=0.12)\,\text{m}$
- **Obstacle Dimensions:** $0.06 \times 0.06 \times 0.08\,\text{m}$ ($5.0\,\text{kg}$ static mass)
- **Injection Timing:** `DisturbanceTiming.AFTER_GRASP` (introduced immediately after grasping completes, before transport begins).

### Geometric Trajectory Obstruction Proof:
- Nominal path line segment endpoints in 2D $(X, Y)$:
  $$A = (0.25, 0.15), \quad B = (0.40, -0.20)$$
- Path midpoint:
  $$M = \left(\frac{0.25 + 0.40}{2}, \frac{0.15 - 0.20}{2}\right) = (0.325, -0.025)$$
- Obstacle center $O = (0.325, -0.025)$ coincides exactly with the midpoint $M$.
- Perpendicular distance from obstacle center to nominal transport segment: $d = 0.000\,\text{m}$.
- Any direct linear `MOVE` action from $A$ to $B$ intersects the obstacle collision envelope.

---

## 6. Autonomous Recovery Pipeline

The scenario specification strictly separates physical reality from cognitive deliberation:

1. **State Observation:** Following disturbance injection, the simulator exports the updated `WorldState`, registering `blocking_barrier_01` with `is_obstacle=True` and `dynamic_obstacles_detected=True`.
2. **Infeasibility Detection:** The SĀRTHI Decision Engine evaluates candidate actions. The nominal direct `MOVE` candidate is rejected with deterministic reason `BlockedPath`.
3. **Alternative Evaluation:** The Decision Engine evaluates candidate alternatives against hard constraints (kinematic reach, workspace limits, obstacle clearance) and soft objectives (responsibility, relevance, consequence scores).
4. **Adaptive Detour:** The Decision Engine autonomously selects `REPOSITION` to navigate around the obstacle.
5. **Goal Completion:** Once clear of the obstruction, subsequent cycles evaluate and execute `MOVE` followed by `RELEASE`.

```
WorldState (Obstacle Active)
           │
           ▼
Decision Engine Evaluation
           ├── Direct MOVE -> REJECTED (BlockedPath)
           └── REPOSITION  -> FEASIBLE (Clears Obstacle)
           │
           ▼
[REPOSITION] -> Detour Waypoint
           │
           ▼
[MOVE]       -> Destination Approach
           │
           ▼
[RELEASE]    -> Task Achieved
```

> [!NOTE]
> The scenario specification **never hardcodes `REPOSITION`**. The adapter and scenario only report the physical state; action selection is governed entirely by the SĀRTHI Decision Engine.

---

## 7. Success Criteria

A task run is verified as successful if and only if all physical criteria are met simultaneously:

1. **Placement Accuracy:** Horizontal Euclidean distance between the red object centroid and the blue target center satisfies:
   $$\sqrt{(x_{\text{obj}} - x_{\text{target}})^2 + (y_{\text{obj}} - y_{\text{target}})^2} \le 0.06\,\text{m}$$
2. **Detachment Verification:** Robot `holding_object_id` is `None` and gripper jaws are open (`gripper_open == True`).
3. **Constraint Integrity:** Zero kinematic, workspace, or payload limit violations occurred during execution.
4. **Final Action Status:** The last executed action was `RELEASE` with verified outcome status `SUCCESS`.

---

## 8. Structured Failure Classifications

Structured error categories are defined in `ScenarioFailureReason`:

| Failure Code | Condition Description |
| :--- | :--- |
| `OBJECT_UNREACHABLE` | Red object position exceeds robot kinematic reach limit ($> 0.85\,\text{m}$). |
| `TARGET_UNREACHABLE` | Blue target zone coordinates exceed robot kinematic reach limit. |
| `BLOCKED_PATH_NO_ALTERNATIVE` | Planned path obstructed and all candidate detours violate safety constraints. |
| `GRASP_FAILURE` | Gripper closes without establishing force contact with target object. |
| `PLACEMENT_FAILURE` | Object released outside destination tolerance radius ($> 0.06\,\text{m}$). |
| `WORKSPACE_VIOLATION` | Manipulator or payload breaches configured workspace Cartesian limits. |
| `PAYLOAD_VIOLATION` | Object mass exceeds rated robot payload capacity ($> 3.00\,\text{kg}$). |
| `MAXIMUM_STEPS_EXCEEDED` | Task does not reach termination within allotted step budget (safety cutoff). |

---

## 9. Architectural Significance for Physical AI

This tabletop scenario benchmarks the central thesis of Physical AI:
- **Separation of Cognitive and Physical Authority:** High-level models interpret goals ("Move the red object to the blue target"), but physical safety and trajectory viability are continuously verified against active physical state.
- **Closed-Loop Adaptability:** Deterministic re-evaluation in response to dynamic disturbances without manual intervention or task aborts.
- **Platform Agnostic Parity:** The exact scenario specification executes identically across in-memory simulation (`LocalSimulationAdapter`), high-fidelity physics (`IsaacSimAdapter`), and physical robot hardware.
