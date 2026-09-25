# NVIDIA Isaac Sim 6.x Runtime & Scene Builder Specification

## 1. Runtime Architecture

The SĀRTHI Isaac Sim runtime package (`simulation/isaac_runtime/`) provides the native scene assembly, prim definition, and controller dispatch boundary for NVIDIA Isaac Sim 6.x.

```
                    Human Instruction
                            │
                            ▼
                TaskUnderstandingService
                            │
                            ▼
                 SĀRTHI Decision Engine
              (Sole Physical Action Authority)
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
               (OpenUSD Stage / PhysX 5)
```

> [!IMPORTANT]
> **Execution Status: PENDING**
> SĀRTHI is currently running in a lightweight development environment without NVIDIA Isaac Sim installed. The runtime package and assembly scripts are architecturally complete, typed, and locked to the benchmark specification, ready for execution when connected to an NVIDIA RTX GPU environment.

---

## 2. Scene Composition

The scene builder (`SarthiSceneBuilder`) constructs the stage using OpenUSD APIs:

| Prim Path | Entity Class | USD Type | Purpose |
| :--- | :--- | :--- | :--- |
| `/World/Table` | `TablePrim` | `FixedCuboid` | Tabletop surface supporting robot, objects, and workspace ($1.4\,\text{m} \times 1.2\,\text{m} \times 0.05\,\text{m}$). |
| `/World/Robots/tabletop_manipulator` | `SarthiRobotPrim` | `UsdReference` / `Robot` | Configurable articulated robot manipulator mounted at $(0.0, 0.0, 0.20)\,\text{m}$. |
| `/World/Objects/red_object_01` | `RedObjectPrim` | `DynamicCylinder` | Graspable red target cylinder at $(0.25, 0.15, 0.20)\,\text{m}$ ($0.50\,\text{kg}$). |
| `/World/Targets/blue_target_zone` | `BlueTargetPrim` | `FixedCylinder` | Blue static placement disc at $(0.40, -0.20, 0.20)\,\text{m}$ ($0.06\,\text{m}$ tolerance). |
| `/World/Obstacles/blocking_barrier_01` | `PathBlockedDisturbance` | `DynamicCuboid` | Dynamically spawned obstacle at $(0.325, -0.025, 0.12)\,\text{m}$ ($5.0\,\text{kg}$). |
| `/World/KeyLight` | Lighting | `DistantLight` | High-angle key directional light ($2500.0\,\text{lux}$). |
| `/World/DomeLight` | Lighting | `DomeLight` | Diffuse ambient studio illumination ($800.0\,\text{lux}$). |
| `/World/Camera` | Viewport Camera | `Camera` | Calibrated perspective demonstration camera. |

All physical parameters, coordinates, masses, and tolerances are sourced directly from the locked scenario specification in `simulation/scenarios/tabletop_pick_place.py`.

---

## 3. Robot Configuration & Asset Management

The robot configuration is managed via `RobotSceneConfig`:

- **Identifier:** `tabletop_manipulator` (default)
- **Mount Pose:** $(x=0.00, y=0.00, z=0.20)\,\text{m}$
- **Kinematic Constraints:** $0.85\,\text{m}$ max reach, $3.00\,\text{kg}$ max payload, 7-DOF.
- **Asset Path:** Configurable via `robot_asset_path` (e.g., Omniverse Nucleus asset `/Isaac/Robots/Franka/franka.usd`).

> [!WARNING]
> **No Silent Substitution:** If a configured robot asset is missing, `SarthiRobotPrim` raises a clear `IsaacSimRuntimeError`. The system never silently substitutes an alternate robot model.

---

## 4. Dynamic Disturbance (`PATH_BLOCKED`)

The dynamic disturbance is implemented in `PathBlockedDisturbance`:
- **State Before Injection:** Inactive and absent from the active collision physics stage.
- **State After Injection:** Spawned as a dynamic collision cuboid at the nominal path midpoint $(0.325, -0.025, 0.12)\,\text{m}$.
- **Decoupled Architecture:** The disturbance class only modifies physical reality. It **never commands `REPOSITION`**. The SĀRTHI Decision Engine evaluates the new `WorldState` and independently determines the adaptive recovery trajectory.

---

## 5. Perspective Demonstration Camera

`SceneCameraConfig` establishes a deterministic camera framing:
- **Eye Location:** $(x=1.15, y=0.45, z=0.85)\,\text{m}$ (diagonal overhead perspective)
- **Look-At Target:** $(x=0.30, y=-0.05, z=0.20)\,\text{m}$ (workspace centroid)
- **Optics:** $24.0\,\text{mm}$ focal length, $55.0^{\circ}$ horizontal field of view
- **Framing Verification:** Captures the robot base, gripper, red object, blue target zone, tabletop, and the exact spatial zone where the dynamic obstacle appears.

---

## 6. Execution Modes & Commands

### Standalone Scene Builder Script
The entry point script `scripts/isaac_sim/build_sarthi_scene.py` supports both headless and visual execution modes:

#### 1. Headless Construction & USD Export (Cloud / CI):
```bash
python scripts/isaac_sim/build_sarthi_scene.py --headless --save-stage stages/sarthi_tabletop.usd
```

#### 2. Interactive GUI Inspection:
```bash
python scripts/isaac_sim/build_sarthi_scene.py --robot-asset /Isaac/Robots/Franka/franka.usd --duration 10.0
```

#### 3. Disturbance Verification Run:
```bash
python scripts/isaac_sim/build_sarthi_scene.py --headless --inject-disturbance --save-stage stages/sarthi_blocked.usd
```

### Safety Enforcement Outside Isaac Sim
When executed outside the NVIDIA Isaac Sim environment, the script terminates immediately with exit code 1:
```
ERROR: Isaac Sim runtime is required. Run this script using the Isaac Sim Python environment.
```
It does not fall back to `LocalSimulationAdapter` and does not fake stage generation.

---

## 7. Cloud & GPU Deployment Plan

When moving to an NVIDIA Omniverse / Isaac Sim workstation or cloud instance:
1. Ensure NVIDIA GPU driver $\ge 535$ and Isaac Sim 6.x installed (or pull `nvcr.io/nvidia/isaac-sim:latest`).
2. Run using the Isaac Sim Python interpreter:
   ```bash
   # Windows Workstation
   C:\path\to\isaac-sim\python.bat scripts\isaac_sim\build_sarthi_scene.py --headless

   # Linux / Container
   ./isaac-sim.sh --headless --exec "python scripts/isaac_sim/build_sarthi_scene.py"
   ```
3. Attach the SĀRTHI `IsaacSimAdapter(sim_backend=SarthiIsaacRuntime())` to `SarthiTaskRunner` for end-to-end closed-loop execution.
