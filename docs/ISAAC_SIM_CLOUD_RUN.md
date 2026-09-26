# SĀRTHI — NVIDIA Isaac Sim 6.1.0 Cloud & Container Execution Guide

This document specifies the exact, reproducible execution protocol for running SĀRTHI closed-loop physical validation inside the official **NVIDIA Isaac Sim 6.1.0** container on a cloud GPU instance (e.g. Nebius L40S Ubuntu VM).

---

## 1. Architectural Workflow

SĀRTHI enforces strict decoupling between offline local development, containerized physical simulation, and cloud compute lifecycle:

```
Windows Development Machine (Local Git / Tests)
        │
        ▼ (git push)
GitHub Repository (animbargi5-art/sarthi)
        │
        ▼ (git clone / pull)
Nebius Ubuntu GPU VM (NVIDIA L40S 48GB VRAM)
        │
        ▼ (Docker + NVIDIA Container Toolkit)
Isaac Sim 6.1.0 Container (nvcr.io/nvidia/isaac-sim:6.1.0)
        │
        ▼ (Volume mount: -v /home/ubuntu/sarthi:/workspace/sarthi)
Container Workspace (/workspace/sarthi)
        │
        ├── 1. Host & Driver Check (nvidia-smi)
        ├── 2. Docker GPU Passthrough Verification
        ├── 3. Isaac Sim Compatibility Check
        ├── 4. SĀRTHI Scene Construction & Export
        └── 5. SĀRTHI Closed-Loop Physical Validation
        │
        ▼ (Trace & Stage artifacts written to /workspace/sarthi/artifacts/)
Validation Artifacts Saved on Host Disk
        │
        ▼ (Stop VM via Nebius CLI / Console)
Cloud VM Stopped (Zero Idle Cost)
```

---

## 2. Critical Cloud & Security Principles

1. **Headless Cloud Execution:** All cloud runs execute strictly in headless mode (`--headless`). No X11/display server is required.
2. **Ephemeral VM Lifecycle:** The Nebius L40S GPU VM must **only be started when execution is needed** and **stopped immediately after the run concludes** to minimize cloud compute cost.
3. **No Nebius API Key Required for Isaac Sim:** The Isaac Sim container and simulation runtime do **not** require any Nebius API keys or credentials.
4. **Credential Protection:** The SĀRTHI Nebius Nemotron API key (`NEBIUS_API_KEY`) is used strictly for TaskUnderstanding NLP evaluation in separate steps and must **NEVER be committed to Git or baked into Docker images**.
5. **Canonical Physical Source of Truth:** All physical dimensions, masses, and initial coordinates remain strictly governed by `TabletopPickPlaceScenario`.

---

## 3. Container Image & Prerequisites

- **Official Container Image:**
  ```text
  nvcr.io/nvidia/isaac-sim:6.1.0
  ```
- **Target GPU Architecture:** NVIDIA Ada Lovelace L40S (48 GB VRAM) or equivalent RTX GPU.
- **NVIDIA Driver:** >= 535.129.03 (CUDA 12.2+).
- **Prerequisites Installed on Host VM:**
  - Docker CE
  - NVIDIA Container Toolkit (`nvidia-container-toolkit`)

---

## 4. Phase 1: Host Verification & Diagnostic Checks

Log into the Nebius Ubuntu VM via SSH and execute the diagnostic baseline:

### A. Host GPU Verification
```bash
nvidia-smi
```
*Expected: Confirms NVIDIA L40S GPU detected with Driver >= 535 and 48GB VRAM available.*

### B. Docker GPU Passthrough Test
```bash
docker run --rm --gpus all ubuntu nvidia-smi
```
*Expected: Displays GPU details inside an ephemeral container, confirming NVIDIA Container Toolkit is operational.*

---

## 5. Phase 2: Pulling the Isaac Sim Container

Pull the verified Isaac Sim 6.1.0 image from NVIDIA NGC (requires free NGC login or acceptance of NVIDIA Isaac Sim EULA):

```bash
docker pull nvcr.io/nvidia/isaac-sim:6.1.0
```

---

## 6. Phase 3: Hardware Compatibility Check

Before mounting project code, verify that the host graphics drivers and Vulkan ray tracing subsystem meet Isaac Sim requirements:

```bash
docker run --name isaac-sim-compat --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" --rm --network=host \
  nvcr.io/nvidia/isaac-sim:6.1.0 \
  ./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

*Expected output: Diagnostic check completes with zero critical driver or Vulkan errors, exiting cleanly after 10 ticks.*

---

## 7. Phase 4: Container Workspace & Repository Mounting

Clone or pull the verified SĀRTHI repository on the host machine:

```bash
cd /home/ubuntu
git clone https://github.com/animbargi5-art/sarthi.git
cd sarthi
git checkout main
```

### Volume Mount Structure
Mount the host repository directory `/home/ubuntu/sarthi` to `/workspace/sarthi` inside the container:

```bash
docker run --name sarthi-validation -it --gpus all \
  -e "ACCEPT_EULA=Y" \
  --rm \
  --network=host \
  -v /home/ubuntu/sarthi:/workspace/sarthi \
  -w /workspace/sarthi \
  nvcr.io/nvidia/isaac-sim:6.1.0 \
  bash
```

Inside the container, all essential project modules are directly accessible:
- `/workspace/sarthi/configs/` — Validation runtime configuration (`isaac_sim_validation.yaml`).
- `/workspace/sarthi/scripts/` — Standalone scene builder and validation runners.
- `/workspace/sarthi/simulation/` — Articulation controller, runtime lifecycle, and scenarios.
- `/workspace/sarthi/backend/` — Canonical Decision Engine models and arbitration logic.
- `/workspace/sarthi/tests/` — Full offline test suite.
- `/workspace/sarthi/docs/` — Architectural specifications.
- `/workspace/sarthi/artifacts/` — Host-persisted output destination.

---

## 8. Phase 5: Executing SĀRTHI Inside the Container

Once inside the container shell (`/workspace/sarthi`), execute the validation sequence:

### Step 1: Pre-create Artifacts Directory
```bash
mkdir -p /workspace/sarthi/artifacts
```

### Step 2: Standalone Scene Construction & Stage Export
Construct the locked tabletop pick-and-place scene and save the OpenUSD stage:

```bash
./python.sh scripts/isaac_sim/build_sarthi_scene.py \
  --headless \
  --save-stage /workspace/sarthi/artifacts/sarthi_tabletop.usd \
  --duration 2.0
```

*Expected output:*
- Initializes Omniverse `SimulationApp(headless=True)`.
- Instantiates tabletop, Franka arm, red movable cylinder, and blue target zone.
- Saves the assembled OpenUSD stage to `/workspace/sarthi/artifacts/sarthi_tabletop.usd`.

### Step 3: Closed-Loop Physical Validation
Run the complete closed-loop benchmark under Decision Engine authority:

```bash
./python.sh scripts/isaac_sim/run_sarthi_validation.py \
  --config configs/isaac_sim_validation.yaml \
  --headless \
  --output /workspace/sarthi/artifacts/validation_trace.json
```

*Expected output:*
```text
[ISAAC] phase=INITIALIZE headless=True status=ok
[ISAAC] phase=SCENE_BUILT status=ok
[ISAAC] phase=TASK_LOADED scenario=tabletop_pick_and_place_mvp status=ok
[ISAAC] phase=APPROACH action=APPROACH status=ok
[ISAAC] phase=GRASP action=GRASP status=ok
[ISAAC] phase=DISTURBANCE_INJECTED type=PATH_BLOCKED obstacle_id=blocking_barrier_01 status=ok
[ISAAC] phase=RECOVERY_DECISION action=REPOSITION status=ok
[ISAAC] phase=REPOSITION action=REPOSITION status=ok
[ISAAC] phase=MOVE action=MOVE status=ok
[ISAAC] phase=RELEASE action=RELEASE status=ok
[ISAAC] phase=FINAL_VERIFICATION status=ok
[ISAAC] phase=SUCCESS status=ok
[ISAAC] Execution trace saved to: /workspace/sarthi/artifacts/validation_trace.json
[ISAAC] VALIDATION RESULT: PASSED (Task completed successfully)
```

---

## 9. Phase 6: Collecting Artifacts & Stopping Cloud VM

After exiting the container, all validation artifacts remain persisted on the host VM under `/home/ubuntu/sarthi/artifacts/`:

1. `sarthi_tabletop.usd` — Assembled OpenUSD physics stage.
2. `validation_trace.json` — Step-by-step auditable execution log containing world state versions, candidate actions, controller primitives, and verification results.

### Retrieve Artifacts to Local Machine (from Windows PowerShell)
```powershell
scp -r ubuntu@<VM_IP>:/home/ubuntu/sarthi/artifacts ./cloud_validation_artifacts
```

### Stop the Cloud VM Immediately
Via Nebius CLI or Cloud Console:
```bash
nebius compute instance stop --id <INSTANCE_ID>
```
*Verify in Nebius Console that instance status displays `STOPPED`.*
