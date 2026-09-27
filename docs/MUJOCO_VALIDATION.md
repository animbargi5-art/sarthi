# SĀRTHI MuJoCo Demonstration, Validation & Evidence Guide

## Purpose

This document specifies the validation architecture, execution workflows, telemetry schemas, and reproducibility guidelines for the SĀRTHI physical AI decision framework integrated with the MuJoCo physics engine. It serves as the primary technical reference for verifying autonomous embodied task execution, dynamic obstacle disturbance injection, constraint-driven recovery replanning, and physical placement verification.

---

## System Flow

The end-to-end execution pipeline connects cognitive intent to deterministic physical manipulation:

```
[Human Operator / Natural Language Instruction]
                        │
                        ▼
           [TaskUnderstandingService]
                        │  (ModelProvider boundary)
                        ▼
          [NVIDIA Nemotron via Nebius Token Factory]
                        │  (Extracts high-level task semantics & target entities)
                        ▼
               [TaskUnderstanding]
                        │
                        ▼
              [SarthiTaskRunner]
                        │  (Coordinates closed-loop observe-decide-execute-verify)
                        ▼
               [SimulationAdapter] ◄───► [SarthiMuJoCoRuntime]
                        │                         │
                        ▼                         ▼
             [Canonical WorldState]         [MuJoCo mjModel / mjData]
                        │                         │
                        ▼                         ▼
            [SarthiDecisionEngine]        [SarthiMuJoCoArticulation]
                        │  (Deterministic candidate evaluation & constraints)
                        ▼
                 [Decision]
                        │
                        ▼
          [SarthiMuJoCoRuntime.execute_action()]
                        │  (DLS IK + Smooth trajectory interpolation)
                        ▼
            [Franka Emika Panda Robot] (7-DOF Arm + Actuated Gripper)
                        │
                        ▼
          [Physical State & Contact Verification]
                        │
                        ▼
             [Next Closed-Loop Cycle / Completion]
```

### Architectural Separation
1. **Cognitive Layer**: Parses unstructured natural language into structured semantic goals (`target_object`, `destination_zone`, high-level symbolic actions). Operates via the `ModelProvider` boundary without direct motor access.
2. **Decision Layer**: Deterministic decision core evaluating candidate actions against physical constraints, safety bounds, consequence projections, and task objectives.
3. **Execution Layer**: Numerical inverse kinematics, smooth trajectory interpolation, and PD control of the 7-DOF Franka Panda arm and parallel gripper.
4. **Observation & Verification Layer**: Extracts canonical `WorldState` from active `mjData` coordinates, velocities, and contact manifolds.

---

## Environment

### Hardware and OS Requirements
- **Operating System**: Windows 10/11 (64-bit) or Linux (Ubuntu 20.04/22.04 LTS).
- **Architecture**: x86_64 or ARM64.
- **Python**: Version 3.10 or higher.
- **MuJoCo**: `mujoco >= 3.1.0`.

### Dependencies
- `mujoco >= 3.1.0`
- `numpy >= 1.24.0`
- `pydantic >= 2.0`
- `openai >= 1.0.0`
- `python-dotenv >= 1.0.0`
- `pyyaml >= 6.0`

### External Model Attribution
- **Manipulator**: Franka Emika Panda 7-DOF arm with parallel jaw gripper.
- **Model Source**: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (`franka_emika_panda`).
- **Upstream License**: Apache 2.0. The upstream model files remain untouched; scenario definitions are modularly layered via MJCF inclusion.

---

## Running Headless Validation

The primary validation runner executes the entire pipeline autonomously without requiring a GUI window.

### Command Line
```bash
python scripts/mujoco/run_sarthi_mujoco_e2e.py --headless --output records/mujoco_e2e_telemetry.json
```

### Deterministic Test Mode (Offline / CI)
To run the validation suite deterministically without network calls to the Nebius endpoint:
```bash
python scripts/mujoco/run_sarthi_mujoco_e2e.py --mock --output records/mujoco_e2e_telemetry.json
```

### Optional Arguments
- `--instruction "<text>"`: Custom natural-language instruction (default: `"Move the red object to the blue target."`).
- `--output <path>`: Destination path for structured JSON telemetry.
- `--seed <int>`: Random seed for execution reproducibility.
- `--max-steps <int>`: Upper bound on closed-loop execution cycles (default: `10`).
- `--steps-per-action <int>`: Simulation ticks per action execution (default: `400`).

---

## Running Viewer Demonstration

For live interactive visualization of the Franka Panda arm, tabletop environment, and obstacle recovery:

```bash
python scripts/mujoco/run_sarthi_mujoco_e2e.py --viewer
```

*Note: Requires an active graphical display environment (OpenGL/GLFW). On headless Linux servers or virtual machines without X11/Wayland, use `--headless`.*

---

## Expected Execution Trace

When executed, the runner outputs a structured, human-readable execution trace generated from actual runtime events:

```
--- HUMAN-READABLE EVENT TRACE ---
  [01] TASK_RECEIVED: Move the red object to the blue target.
  [02] NEMOTRON_TASK_UNDERSTANDING: target='red_object', zone='blue_target', actions=['APPROACH', 'GRASP', 'MOVE', 'RELEASE']
  [03] WORLD_STATE_OBSERVED: v1, Robot=(0.55, -0.00, 0.52)
  [04] DECISION_SELECTED: Selected APPROACH (act_approach_target_obj)
  [05] APPROACH_EXECUTED: EE reached standoff near object (v2)
  [06] GRASP_VERIFIED: Gripper clamped target object 'red_object_01'
  [07] PATH_BLOCKED_INJECTED: Obstacle placed at (0.325, -0.025, 0.120)
  [08] WORLD_STATE_CHANGED: Version advanced v3 -> v4 with keepout constraint
  [09] MOVE_REJECTED_BLOCKED_PATH: Constraint validation rejected 'act_move_to_target_zone': BlockedPath: Obstacle intersects trajectory
  [10] RECOVERY_DECISION_REPOSITION: Selected REPOSITION (act_reposition_clearance) with clearance z=0.351m
  [11] REPOSITION_VERIFIED: Elevated object to clearance altitude z=0.333m
  [12] MOVE_VERIFIED: Navigated over obstacle to target zone (0.40, -0.19)
  [13] RELEASE_VERIFIED: Gripper opened; object released in target zone
  [14] FINAL_PLACEMENT_VERIFIED: Object at (0.413, -0.148, 0.200), error=0.0535m <= tol=0.0600m
  [15] TASK_COMPLETED: Physical pick-and-place with obstacle avoidance succeeded.
```

---

## Disturbance & Recovery

### Disturbance Mechanism (`PATH_BLOCKED`)
1. **Timing**: Injected immediately after the object is physically grasped (Step 2).
2. **Physical Manifestation**: An obstacle body (`obstacle_geom`, mass: 5.0 kg) is dynamically translated into the direct linear transit corridor at coordinates `(0.325, -0.025, 0.120)`.
3. **State Reflection**: MuJoCo `mjData` reflects the new obstacle location, advancing canonical `WorldState` version from `v3` to `v4`.
4. **Constraint Generation**: A spatial keep-out zone constraint (`c_obstacle_keepout`) is registered with required clearance radius `0.082 m`.

### Causal Recovery Replanning
- **Nominal Candidate Rejection**: Direct ground-level transit (`act_move_to_target_zone`) passes within `0.066 m` of the obstacle, violating the `0.082 m` clearance threshold. The Decision Engine's `ConstraintValidator` marks the action invalid (`is_valid = False`).
- **Autonomous Recovery Selection**: The Decision Engine evaluates remaining valid candidates. Detour action `act_reposition_clearance` (`ActionType.REPOSITION`) achieves the highest composite score by projecting elevation over the obstacle height.
- **Physical Recovery Execution**: The Franka Panda arm elevates the grasped object to `z = 0.333 m`, successfully maintaining grip force.
- **Task Continuation**: From the elevated recovery waypoint, a collision-free transit trajectory to `(0.40, -0.20, 0.171)` is executed, followed by controlled release.

---

## Physical Verification

Physical success is verified against ground-truth simulation coordinates measured directly from `mjData`:

| Criterion | Measured Value | Threshold / Tolerance | Status |
| :--- | :--- | :--- | :--- |
| **Object Horizontal Offset** | `0.0535 m` | $\le 0.0600\text{ m}$ (target radius) | **PASS** |
| **Tabletop Elevation** | `0.0012 m` | $\le 0.0350\text{ m}$ (table plane offset) | **PASS** |
| **Gripper Separation** | `0.0811 m` | $> 0.0650\text{ m}$ (open threshold) | **PASS** |
| **Contact Forces** | None | 0 active contacts with fingers | **PASS** |
| **Object State** | `ObjectState.PLACED` | Canonical target condition | **PASS** |

The task is marked successful if and only if all physical criteria are met. Action execution sequence alone does not determine task success.

---

## Telemetry

The structured JSON telemetry report (`--output <path>`) contains complete execution metadata:

```json
{
  "timestamp": "2026-09-27T11:32:36.382929+00:00",
  "project": "SĀRTHI",
  "simulator": "MuJoCo",
  "simulator_version": "3.14.0",
  "robot": "Franka Emika Panda",
  "model": "nvidia/Nemotron-3-Ultra-550b-a55b",
  "model_provider": "NebiusNemotronProvider",
  "inference_endpoint": "https://api.tokenfactory.nebius.com/v1",
  "instruction": "Move the red object to the blue target.",
  "task_understanding": {
    "task_id": "task_move_red_to_blue_001",
    "objective": "Move the red object to the blue target location",
    "target_object": "red_object",
    "target_location": "blue_target",
    "required_actions": ["APPROACH", "GRASP", "MOVE", "RELEASE"],
    "confidence": 0.95,
    "reasoning_summary": "Standard pick-and-place sequence inferred."
  },
  "actions": ["APPROACH", "GRASP", "REPOSITION", "MOVE", "RELEASE"],
  "disturbance": {
    "type": "PATH_BLOCKED",
    "injected": true,
    "injected_at_step": 2,
    "obstacle_position": {"x": 0.325, "y": -0.025, "z": 0.12},
    "pre_disturbance_ws_version": 3,
    "post_disturbance_ws_version": 4,
    "recovery_action": "REPOSITION"
  },
  "recovery_count": 1,
  "final_object_position": {"x": 0.4129, "y": -0.1481, "z": 0.1998},
  "target_position": {"x": 0.4, "y": -0.2, "z": 0.171},
  "placement_error": 0.0535,
  "tolerance": 0.06,
  "gripper_state": "OPEN",
  "gripper_released": true,
  "final_object_state": "PLACED",
  "overall_result": "COMPLETED",
  "physical_verification_passed": true,
  "event_trace": [ ... ]
}
```

*Security Invariant: The telemetry export contains zero secret keys, tokens, or authorization headers.*

---

## Reproducibility

### Determinism Guarantees
MuJoCo employs an analytical constrained dynamics formulation (Euler/RK4 integrator) that is inherently deterministic. Given identical initial coordinates and control inputs, numerical trajectory deviation is zero across repeated executions.

### System Configuration
- **Simulation Timestep**: `dt = 0.002 s` (500 Hz physics rate).
- **Action Execution Budget**: `400 ticks` (0.80 s simulation time) per action.
- **IK Algorithm**: Damped Least Squares (DLS) with downward vertical orientation constraint:
  - Damping parameter $\lambda = 0.05$
  - Maximum iterations: `300`
  - Convergence tolerance: `0.004 m`
  - Orientation error threshold: `0.15 rad`
- **Trajectory Interpolation**: 85% linear joint velocity interpolation + 15% settling phase.
- **Model Inference**: NVIDIA Nemotron model hosted on Nebius Token Factory.

### Environment Configuration (Sanitized)
```env
NEBIUS_API_KEY=<configured-token-factory-key>
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
NEBIUS_MODEL=nvidia/Nemotron-3-Ultra-550b-a55b
```

---

## Troubleshooting

1. **`UnboundLocalError` or Missing Display**:
   - Running in headless environments without an active X server will cause `--viewer` to fail. Use `--headless` instead.
2. **Nebius API Timeout or Network Errors**:
   - Check internet connectivity and verify `NEBIUS_BASE_URL` is reachable.
   - For offline test verification, append `--mock` to use the deterministic mock provider.
3. **IK Non-Convergence**:
   - Ensure the commanded target coordinates reside within the Panda workspace reach (`reach <= 0.855 m`).

---

## Known Limitations

1. **Planar Tabletop Geometry**: The baseline demonstration scene utilizes a flat horizontal table plane. Irregular uneven terrain is not modeled in the current MVP benchmark.
2. **Rigid Cylinder Target**: The manipulation target is modeled as a rigid cylinder with rubberized surface friction. Deformable, fragile, or compliant objects require dedicated soft-body simulation.
3. **Kinematic Obstacle Injection**: The obstacle is statically translated into the workspace upon disturbance trigger rather than dynamically rolling or dropping from an aerial trajectory.
