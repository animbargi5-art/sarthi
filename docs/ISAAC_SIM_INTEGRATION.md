# NVIDIA Isaac Sim Integration Architecture

## 1. Overview & Physical AI Rationale

SĀRTHI is designed to bridge cognitive intent with physical robotics. While the deterministic `LocalSimulationAdapter` provides fast, offline, and bit-for-bit reproducible unit testing, **NVIDIA Isaac Sim 6.x** represents SĀRTHI's designated **Physical AI runtime environment**.

### Why Isaac Sim?
- **High-Fidelity Physics:** Powered by NVIDIA PhysX 5, delivering GPU-accelerated rigid-body dynamics, compliant contact modeling, and realistic friction.
- **Universal Scene Description (USD):** Composable, scalable stage composition for photorealistic lighting, materials, and complex industrial environments.
- **Sensor Simulation:** Ray-traced depth sensing, RGB-D cameras, tactile sensors, and contact force reporting.
- **Sim-to-Real Transfer:** Direct parity with NVIDIA Jetson and real-world Franka Emika / UR manipulators.

> [!IMPORTANT]
> **Status:** Live Isaac Sim runtime execution is **PENDING**. The local machine operates in a lightweight CPU environment. All Phase 6A components define the strict integration boundary and data translations without requiring Isaac Sim installation.

---

## 2. SimulationAdapter Architecture

Both local deterministic simulation and future Isaac Sim environments implement the unified `SimulationAdapter` abstract interface:

```
                          SĀRTHI TaskRunner
                                 │
                                 ▼
                         SimulationAdapter  (Contract)
                        /                 \
                       /                   \
                      ▼                     ▼
            LocalSimulationAdapter     IsaacSimAdapter
            (Local Core World)         (Future Omniverse Stage)
                                            │
                                            ▼
                                     NVIDIA Isaac Sim 6.x
```

### Core Interface Contract:
- `get_world_state() -> WorldState`: Polls current simulator telemetry and returns an immutable, validated SĀRTHI `WorldState`.
- `execute_action(action: CandidateAction) -> ActionExecutionResult`: Formats and commands a low-level physical action primitive.
- `inject_disturbance(disturbance: DisturbanceEvent) -> bool`: Injects dynamic perturbations (e.g., dynamic path obstructions) into the active scene.
- `verify_action_result(action, expected_outcome) -> bool`: Verifies post-execution physical criteria.

---

## 3. LocalSimulationAdapter vs. IsaacSimAdapter

| Capability / Property | `LocalSimulationAdapter` | `IsaacSimAdapter` |
| :--- | :--- | :--- |
| **Physics Backend** | Deterministic analytical kinematics | NVIDIA PhysX 5 / GPU dynamics |
| **Rendering / Stage** | Pure data structures (in-memory) | OpenUSD Stage (`/World/...`) |
| **Hardware Requirement** | CPU only (< 50MB RAM) | NVIDIA RTX GPU with >= 8GB VRAM |
| **Execution Speed** | Sub-millisecond step times | Real-time or stepped physics ticks (60Hz) |
| **Primary Role** | Fast CI/CD, logic regression tests | Physical verification, sensor rendering, sim-to-real |
| **Current Status** | Active & Production-Ready | Boundary Implemented, Live Runtime **PENDING** |

---

## 4. WorldState Translation Boundary

The `IsaacSimAdapter` enforces clean translation from raw Isaac Sim stage prims into the canonical SĀRTHI `WorldState`:

```
Isaac Sim Stage Prim Telemetry
(Transforms, Joint States, USD Attributes)
                    │
                    ▼
   IsaacSimAdapter.translate_world_state()
                    │
                    ▼
            SĀRTHI WorldState
 (RobotState, WorldObject, TargetZone, EnvironmentState)
```

- **No Schema Duplication:** The adapter reuses the canonical schemas in `backend/app/decision_engine/models.py`.
- **Entity Grounding:** USD prims under `/World/Objects` and `/World/Obstacles` are translated into `WorldObject` instances with explicit `is_target` and `is_obstacle` designations.
- **Auditability:** Every state translation records timestamp nanoseconds and monotonically advances the `version` counter.

---

## 5. Action Translation Mapping

The SĀRTHI Decision Engine retains **exclusive authority** over action selection. The adapter never decides what action to execute; it only maps the chosen `CandidateAction` to the corresponding Isaac Sim controller primitive:

| SĀRTHI `ActionType` | Isaac Sim Primitive (`primitive_name`) | Gripper Command (`gripper_target`) | Execution Description |
| :--- | :--- | :--- | :--- |
| `APPROACH` | `cartesian_approach` | `OPEN` | Move end-effector to pre-grasp offset pose |
| `REPOSITION` | `cartesian_reposition` | `HOLD` | Execute spatial detour or clearance waypoint |
| `GRASP` | `gripper_close` | `CLOSE` | Close finger articulation with force feedback |
| `MOVE` | `cartesian_move` | `HOLD` | Transport grasped payload to destination |
| `RELEASE` | `gripper_open` | `OPEN` | Open gripper fingers to detach payload |
| `STOP` | `emergency_hold` | `HOLD` | Zero velocities and engage joint brakes |

---

## 6. Disturbance Handling (PATH_BLOCKED)

The conceptual disturbance `PATH_BLOCKED` simulates dynamic real-world interference (e.g. human entry, fallen container):

1. **Injection:** `DisturbanceEvent.create_path_blocked(...)` is submitted to `IsaacSimAdapter.inject_disturbance(...)`.
2. **USD Translation:** The adapter translates the event into an `IsaacSimDisturbance` defining a dynamic collision obstacle at `/World/Obstacles/<obstacle_id>` with physics rigid-body attributes enabled.
3. **WorldState Propagation:** The next `get_world_state()` call exports the obstacle in `WorldState.objects` with `is_obstacle=True` and sets `environment.dynamic_obstacles_detected=True`.
4. **Autonomous Reassessment:** The SĀRTHI Decision Engine reads the modified `WorldState`, recognizes that a direct `MOVE` trajectory collides with the obstacle, rejects direct motion, and evaluates alternate candidates (such as `REPOSITION`).
5. **No Hard-coding:** The adapter does **not** hardcode a `REPOSITION` action; recovery remains the autonomous domain of the Decision Engine.

---

## 7. Lazy Dependency Design & Error Handling

To support development and CI/CD on machines without NVIDIA Omniverse / Isaac Sim:
- **Zero Top-Level Imports:** `import isaacsim` or `import omni.isaac.core` are never imported at module load time.
- **Availability Guard:** The helper `is_isaac_sim_available()` safely checks module availability without raising unhandled errors.
- **Actionable Exception:** When live simulation methods are invoked without an active backend or runtime, the adapter raises:
  ```python
  IsaacSimUnavailableError: Isaac Sim is not installed in this environment. Use the Isaac Sim runtime environment to execute IsaacSimAdapter.
  ```

---

## 8. Future Deployment & GPU/Cloud Runtime Plan

When Isaac Sim runtime testing is initiated:
1. **Target Environment:** Cloud instance or workstation equipped with NVIDIA RTX GPU (RTX 4080/4090 or A10G/L4) running Ubuntu 22.04 or Windows 11 with NVIDIA driver >= 535.
2. **Runtime Container:** Official `nvcr.io/nvidia/isaac-sim:latest` container or Isaac Sim workstation installer.
3. **Execution Script:**
   ```bash
   # Run SĀRTHI against live Isaac Sim runtime headless
   ./isaac-sim.sh --headless --exec "python scripts/run_sarthi_isaac_sim.py"
   ```
4. **Verification Milestone:** Smoke test loading the SĀRTHI Franka arm USD asset, verifying Isaac Sim articulation controllers, and running the standard pick-and-place task.
