# SĀRTHI — NVIDIA Isaac Sim Real Runtime Validation Guide

## 1. Purpose & Scope

This document specifies the canonical real-runtime validation protocol for SĀRTHI inside **NVIDIA Isaac Sim 6.x**. It bridges the offline, deterministic development testbed with live, GPU-accelerated PhysX simulation.

> [!IMPORTANT]
> **Execution Status: PENDING**
> Real runtime execution has **NOT** been performed yet. SĀRTHI is currently running in a CPU development environment. The validation scripts, closed-loop controllers, trace loggers, and test suites are fully prepared and verified for execution on an NVIDIA RTX GPU system.

---

## 2. Locked Benchmark Scenario

The validation uses the canonical benchmark scenario locked in `simulation/scenarios/tabletop_pick_place.py`:

- **Scenario Identifier:** `tabletop_pick_and_place_mvp`
- **Natural Language Instruction:** `"Move the red object to the blue target."`
- **Robot:** `tabletop_manipulator` (7-DOF manipulator, mount pose: $(0.0, 0.0, 0.20)\,\text{m}$)
- **Red Object:** Dynamic cylinder at $(0.25, 0.15, 0.20)\,\text{m}$ ($0.50\,\text{kg}$)
- **Blue Target:** Placement zone at $(0.40, -0.20, 0.20)\,\text{m}$ ($0.06\,\text{m}$ tolerance)
- **Tabletop:** Supporting surface at $Z=0.20\,\text{m}$ ($1.40\,\text{m} \times 1.20\,\text{m} \times 0.05\,\text{m}$)
- **Dynamic Obstacle:** `blocking_barrier_01` at $(0.325, -0.025, 0.12)\,\text{m}$ ($5.0\,\text{kg}$)

---

## 3. Strict Authority Boundaries

```
                   Human Natural Language Instruction
                                   │
                                   ▼
                       TaskUnderstandingService
                                   │
                                   ▼
                        SĀRTHI Decision Engine
                   (SOLE Physical Action Authority)
                                   │
                            CandidateAction
                                   │
                                   ▼
                           IsaacSimAdapter
                                   │
                            IsaacSimAction
                                   │
                                   ▼
                          SarthiIsaacRuntime
                                   │
                                   ▼
                        NVIDIA Isaac Sim 6.x
```

### Non-Negotiable Architectural Rules:
1. **Decision Engine Exclusivity:** The Decision Engine selects every physical action. The Isaac runtime possesses zero cognitive logic and never selects actions autonomously.
2. **Adapter Bridge:** Actions flow strictly via `IsaacSimAdapter`, which translates high-level `CandidateAction` to low-level controller primitives (`IsaacSimAction`).
3. **No Hardcoded Recovery:** Dynamic disturbances modify the physical USD stage; recovery trajectories (`REPOSITION`) are synthesized autonomously by the Decision Engine.

---

## 4. Execution Phases & Logging Format

During execution, `scripts/isaac_sim/run_sarthi_validation.py` emits structured console telemetry in the canonical format:

| Sequence | Phase Name | Log Output Example | Description |
| :---: | :--- | :--- | :--- |
| 1 | `INITIALIZE` | `[ISAAC] phase=INITIALIZE headless=True status=ok` | SimulationApp starts |
| 2 | `SCENE_BUILT` | `[ISAAC] phase=SCENE_BUILT status=ok` | OpenUSD stage populated |
| 3 | `TASK_LOADED` | `[ISAAC] phase=TASK_LOADED scenario=tabletop_pick_and_place_mvp status=ok` | Scenario constraints bound |
| 4 | `APPROACH` | `[ISAAC] phase=APPROACH action=APPROACH status=ok` | Arm reaches pre-grasp pose |
| 5 | `GRASP` | `[ISAAC] phase=GRASP action=GRASP status=ok` | Gripper grasps red object |
| 6 | `DISTURBANCE_INJECTED` | `[ISAAC] phase=DISTURBANCE_INJECTED type=PATH_BLOCKED obstacle_id=blocking_barrier_01 status=ok` | Obstacle spawned in path |
| 7 | `RECOVERY_DECISION` | `[ISAAC] phase=RECOVERY_DECISION action=REPOSITION status=ok` | Engine discovers blocked path |
| 8 | `REPOSITION` | `[ISAAC] phase=REPOSITION action=REPOSITION status=ok` | Detour trajectory executed |
| 9 | `MOVE` | `[ISAAC] phase=MOVE action=MOVE status=ok` | Payload transported to target |
| 10 | `RELEASE` | `[ISAAC] phase=RELEASE action=RELEASE status=ok` | Gripper releases payload |
| 11 | `FINAL_VERIFICATION` | `[ISAAC] phase=FINAL_VERIFICATION status=ok` | Physical invariants verified |
| 12 | `SUCCESS` | `[ISAAC] phase=SUCCESS status=ok` | Closed-loop task complete |

---

## 5. Execution Trace Schema

Each cycle step is logged to an auditable JSON trace file (`validation_trace.json`):

```json
{
  "step_number": 3,
  "world_state_version_before": 3,
  "selected_action": {
    "action_id": "act_reposition_01",
    "action_type": "REPOSITION",
    "target_position": {"x": 0.20, "y": -0.10, "z": 0.28}
  },
  "isaac_action": {
    "action_id": "act_reposition_01",
    "action_type": "REPOSITION",
    "primitive_name": "cartesian_reposition",
    "gripper_target": "HOLD"
  },
  "execution_result": {
    "success": true,
    "action_type": "REPOSITION",
    "simulation_time": 0.15
  },
  "world_state_version_after": 4,
  "verification_result": true,
  "disturbance_state": {
    "is_active": true,
    "type": "PATH_BLOCKED"
  },
  "final_status": "STEP_OK"
}
```

---

## 6. Success & Failure Criteria

### Success Criteria:
- **Final Placement:** Red object centroid within $0.06\,\text{m}$ horizontal tolerance of blue target.
- **Gripper State:** Jaws open (`gripper_open == True`), holding object ID is `None`.
- **Constraint Compliance:** Zero collisions with dynamic obstacle; zero workspace or payload breaches.
- **Action Verification:** Last executed action was `RELEASE` with verified outcome `SUCCESS`.

### Failure Criteria:
- Robot fails to detach or grasp object.
- Gripper collides with obstacle.
- Total steps exceed configured `max_steps` (default 10).
- Kinematic reach or workspace limits breached.

---

## 7. How to Run Inside Isaac Sim

When deploying to an NVIDIA RTX GPU workstation or cloud container:

### 1. Headless Execution (Automated Validation / CI):
```bash
# Linux / Docker
./isaac-sim.sh --headless --exec "python scripts/isaac_sim/run_sarthi_validation.py --config configs/isaac_sim_validation.yaml --headless"

# Windows Workstation
C:\path\to\isaac-sim\python.bat scripts\isaac_sim\run_sarthi_validation.py --config configs/isaac_sim_validation.yaml --headless
```

### 2. Interactive GUI Inspection:
```bash
python scripts/isaac_sim/run_sarthi_validation.py --robot-asset /Isaac/Robots/Franka/franka.usd
```

### 3. Stage & Trace Export:
```bash
python scripts/isaac_sim/run_sarthi_validation.py --headless --save-stage stages/validation_complete.usd
```
Execution traces are written to `logs/isaac_validation/validation_trace.json`.

### 4. Safety Guard Check Outside Isaac Sim:
If executed on a non-Isaac Sim environment, the script safely halts:
```bash
python scripts/isaac_sim/run_sarthi_validation.py
# Output:
# ERROR: Isaac Sim runtime is required. Run this script using the Isaac Sim Python environment.
# Exit code: 1
```
