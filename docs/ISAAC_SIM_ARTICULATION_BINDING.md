# SĀRTHI — Isaac Sim Robot Articulation Binding (Phase 7A)

This document specifies the architectural boundary, articulation controller, asset configuration, telemetry conversion, and physical action dispatch connecting SĀRTHI to an articulated robotic manipulator (Franka Emika Panda) in NVIDIA Isaac Sim.

---

## 1. Architectural Boundary

SĀRTHI enforces strict responsibility separation across cognitive reasoning, physical decision-making, simulation adaptation, and low-level robot joint control:

```
Human Instruction
       │
       ▼
TaskUnderstandingService (Nemotron / Model Provider)
       │  (Extracts high-level task understanding; CANNOT actuate)
       ▼
SĀRTHI TaskRunner & Decision Engine
       │  (Evaluates physical rules, monitors safety, selects actions)
       ▼
SimulationAdapter / IsaacSimAdapter
       │  (Formats CandidateAction into IsaacSimAction; translates state)
       ▼
SarthiIsaacRuntime
       │
       ▼
SarthiArticulationController (Phase 7A)
       │
       ├── Joint-Space & Cartesian Pose Commands
       ├── Gripper Actuation (OPEN, CLOSE, HOLD)
       └── Immediate Safety STOP (zero velocity, hold position)
       │
       ▼
NVIDIA Isaac Sim (OpenUSD Articulation / PhysX)
```

### Strict Non-Negotiable Guarantees:
1. **Decision Authority:** The external Decision Engine remains the sole authority for physical action selection. No candidate evaluation or autonomous recovery heuristics are implemented inside the Isaac Sim runtime or articulation controller.
2. **LLM Sandboxing:** The Nemotron / LLM provider has zero actuation capability and cannot command joints, grippers, or poses directly.
3. **Canonical Schema:** Telemetry is compiled directly into the canonical `backend.app.decision_engine.models.WorldState`. No competing state schema is introduced.
4. **Lazy Imports:** Isaac Sim packages (`isaacsim`, `omni.isaac.core`, `pxr`) are imported lazily, allowing the entire SĀRTHI test suite to run offline in lightweight environments.

---

## 2. Franka Emika Panda USD Asset Configuration

The robot USD asset path is configured via `configs/isaac_sim_validation.yaml`:

```yaml
scenario_id: "tabletop_pick_and_place_mvp"
headless: true
output_directory: "logs/isaac_validation"
max_steps: 10
enable_trace: true

# Configurable robot USD asset path.
# Default intended robot: Standard NVIDIA Franka Emika Panda asset.
# Must point to a valid USD on the target GPU host (Omniverse Nucleus or local filesystem).
# Note: Do not assume this asset physically exists on the local development machine.
# Real Isaac Sim execution will fail clearly if this asset path is missing or invalid.
robot_asset_path: "omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Robots/Franka/franka.usd"
default_robot_asset_path: "omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Robots/Franka/franka.usd"
```

### Asset Validation & Fail-Fast Behavior:
- In `SarthiRobotPrim.create()`:
  - If `robot_asset_path` is `None` or whitespace, execution raises `IsaacSimRuntimeError("Missing robot asset path for 'tabletop_manipulator'...")`.
  - If `robot_asset_path` is a local filesystem path that does not exist, execution raises `IsaacSimRuntimeError("Configured robot asset path not found on disk: ...")`.
  - Silent substitution of robot models is strictly disallowed.

---

## 3. Live State Flow into SĀRTHI WorldState

The live stage state is collected through `SarthiArticulationController` and `SarthiIsaacRuntime.read_live_world_state()`:

| Component | Source in Isaac Sim | Mapping in SĀRTHI WorldState |
|---|---|---|
| **Joint Positions** | `robot.get_joint_positions()` | Exposed via `SarthiArticulationController.get_joint_positions()` and runtime telemetry |
| **Joint Velocities** | `robot.get_joint_velocities()` | Exposed via `SarthiArticulationController.get_joint_velocities()` and runtime telemetry |
| **End-Effector Pose** | `robot.end_effector.get_world_pose()` | `WorldState.robot.position` (Point3D) |
| **Gripper State** | `controller.gripper_state` | `WorldState.robot.gripper_open` (bool) |
| **Holding Object** | `controller.holding_object_id` | `WorldState.robot.holding_object_id`, `payload_mass_kg` |
| **Movable Object** | Stage prim `/World/Objects/red_movable_object` | `WorldState.objects[0].position`, `state` (`GRASPED` or `FREE`) |
| **Destination Zone** | Stage prim `/World/Targets/blue_target_zone` | `WorldState.target.position`, `tolerance_radius_m` |
| **Dynamic Obstacle** | Stage prim `/World/Obstacles/dynamic_obstacle` | `WorldState.environment.dynamic_obstacles_detected` |

The resulting structure is an immutable, canonical `WorldState` validated by Pydantic.

---

## 4. CandidateAction Dispatch to Articulation Controller

The Decision Engine produces actions of type `CandidateAction`. These are translated and dispatched through `SarthiArticulationController.dispatch_action()`:

1. **`APPROACH`**:
   - Commands gripper jaw opening: `controller.open_gripper()`
   - Commands end-effector Cartesian translation to pre-grasp waypoint: `controller.command_cartesian_position(target_position, speed_scale)`
2. **`REPOSITION`**:
   - Holds gripper separation: `controller.hold_gripper()`
   - Commands Cartesian translation to detour waypoint around obstacle: `controller.command_cartesian_position(target_position, speed_scale)`
3. **`GRASP`**:
   - Closes gripper jaw fingers with force target: `controller.close_gripper()`
   - Links target object ID to gripper payload tracking: `_is_holding_object = True`
4. **`MOVE`**:
   - Keeps gripper closed holding payload
   - Commands Cartesian translation along trajectory: `controller.command_cartesian_position(target_position, speed_scale)`
5. **`RELEASE`**:
   - Opens gripper jaw fingers: `controller.open_gripper()`
   - Detaches payload tracking: `_is_holding_object = False`, `holding_object_id = None`
   - Persists released coordinates: `_released_object_positions[object_id] = ee_position`
6. **`STOP`**:
   - Halts all joint motions immediately: `controller.stop()`

---

## 5. Closed-Loop Physical State Verification (Phase 7B)

Phase 7B closes the runtime verification loop between commanded actions, physical stage effects, and canonical WorldState:

### A. Live Object Pose Synchronization
- **While Grasped:** When `controller.is_holding_object` is True, object Cartesian position dynamically follows the live end-effector position (`ee_pos`). `WorldObject.state` is set to `ObjectState.GRASPED`.
- **After Release:** After a verified `RELEASE`, the object does NOT snap back to `scenario.initial_pose`. Instead:
  1. Live USD stage prim transforms (`/World/Objects/{object_id}`) are queried if Isaac Sim stage access is available.
  2. If the USD stage query is unavailable, the last physically released coordinates stored in `_released_object_positions` are used as the deterministic fallback.
  3. When within `blue_target.tolerance_radius_m` of the destination zone, `WorldObject.state` transitions to `ObjectState.PLACED`.

### B. Action Verification Matrix (`verify_action_result`)
Unconditional success reporting is strictly replaced by explicit geometric and state invariants:

| Action Primitive | Physical Invariants Verified | Failure Condition & Diagnostics |
|---|---|---|
| **`APPROACH`** | End-effector reached target position within configured tolerance ($\le 0.05\,\text{m}$) | End-effector Cartesian distance exceeds tolerance |
| **`REPOSITION`** | End-effector reached detour waypoint within tolerance ($\le 0.05\,\text{m}$) | End-effector Cartesian distance exceeds tolerance |
| **`MOVE`** | End-effector reached destination; collision margin to dynamic obstacle observed | Path obstructed by obstacle ($< 0.10\,\text{m}$ margin) or destination not reached |
| **`GRASP`** | Gripper jaws closed; payload attached; target object within grasp radius ($\le 0.12\,\text{m}$) | Gripper open, payload missing, or target out of reach |
| **`RELEASE`** | Gripper jaws open; payload detached; object remains at expected release coordinates | Gripper closed, payload still attached, or object displaced |
| **`STOP`** | Robot articulation is in halted STOP state (`is_stopped == True`) | Articulation velocities non-zero or not stopped |

Failed verifications populate `SarthiIsaacRuntime.last_verification_failure` with actionable diagnostic messages.

### C. Structured Grasp Failure Handling
Grasp execution in `SarthiArticulationController.dispatch_action()` validates preconditions deterministically:
1. **Safety STOP check:** Fails with error code `ROBOT_STOPPED` if commanded while halted.
2. **Missing payload check:** Fails with error code `MISSING_PAYLOAD` if `target_object_id` is empty or unspecified.
3. **Proximity check:** Fails with error code `OBJECT_OUT_OF_REACH` if distance from end-effector to target exceeds `grasp_threshold_m` (default $0.12\,\text{m}$).
4. **State preservation:** Failed grasps return `ActionExecutionResult(success=False, failure_reason=...)` and **never corrupt** `_world_state_version`, payload ownership, or robot state.

### D. Disturbance Parameter Normalization
The runtime boundary (`SarthiIsaacRuntime.inject_disturbance` and `IsaacSimAdapter.inject_disturbance`) polymorphically accepts both:
- `DisturbanceEvent` (conceptual SĀRTHI event from Decision Engine / test harness)
- `IsaacSimDisturbance` (USD stage specification model)
Both are normalized internally into canonical USD obstacle coordinates and update the active collision geometry.

---

## 6. Safety Behavior

1. **Emergency STOP:**
   - `SarthiArticulationController.stop()` immediately:
     - Sets joint velocity targets to `0.0 rad/s` for all 7 Franka joints.
     - Commands static hold targets to current joint angles.
     - Locks the gripper in its current state (preventing accidental dropping of payloads).
     - Enters `_is_stopped = True` state.
   - Any motion commands received while in STOP state immediately raise `IsaacSimRuntimeError` until `controller.resume()` is explicitly triggered.
2. **Fail-Fast Environment Guards:**
   - Attempting to call articulation control without an active Isaac Sim environment or bound articulation raises `IsaacSimRuntimeError`.
   - Never falls back to silent mock execution in live deployment.

---

## 7. What Still Requires Real Isaac Sim Execution (Current Limitations)

The offline testbed completely validates state invariants, geometric tolerances, and action dispatch without requiring NVIDIA GPU hardware. However, the following physical phenomena strictly require live NVIDIA Isaac Sim PhysX execution:

1. **Real PhysX Multi-Tick Trajectory Convergence:**
   - Physical convergence depends on joint motor PD gains (stiffness $K_p$, damping $K_d$) and PhysX substepping against contact forces.
2. **Contact Friction & Payload Slip Dynamics:**
   - Physical finger friction and normal force calculations against the cylinder mesh require PhysX contact solvers.
3. **Launch Command on GPU Host:**
   ```bash
   python.bat scripts/isaac_sim/run_sarthi_validation.py --config configs/isaac_sim_validation.yaml --headless
   ```
