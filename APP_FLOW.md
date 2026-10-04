# SĀRTHI — Application Execution Flow

**Project Name:** SĀRTHI  
**Tagline:** An adaptive Physical AI architecture for context-aware, responsibility-driven robotic decision-making under changing physical conditions.  
**Target Event:** Nebius × NVIDIA Global AI Hackathon 2026  
**Document Version:** Version: 3.0  
**Status:** Draft — V3 Architecture Update  

---

## 1. Safety Architecture & Core Invariant

All operational execution paths across SĀRTHI V3 adhere strictly to the fundamental safety invariant:

> [!IMPORTANT]
> **Core System Invariant:**  
> **"AI models propose or select bounded semantic candidates; deterministic software validates physical feasibility; only validated actions reach the robot controller."**  
> **"Jev must never bypass deterministic safety validation."**

No model output—from NVIDIA Nemotron or the planned Jev fast decision provider—is permitted to bypass deterministic validation or dispatch direct motor control commands.

---

## 2. End-to-End Operational Lifecycle

The operational lifecycle connects natural-language human intent to verified physical manipulation:

```text
Human natural-language command
              ↓
Task Understanding — Nemotron via Nebius Token Factory
              ↓
Structured TaskUnderstanding (semantic goals & entities)
              ↓
Physical WorldState observation (MuJoCo State Reader)
              ↓
Decision Context Builder (compact situation synthesis)
              ↓
Jev bounded decision (fast candidate ranking)
              ↓
Candidate action proposal
              ↓
SĀRTHI Deterministic Decision Engine (constraint validation)
              ↓
Approved physical action
              ↓
DLS IK & Trajectory Controller
              ↓
MuJoCo physics stepping
              ↓
State Reader extraction
              ↓
Physical outcome verification
              ↓
Task Complete / Reassess / Dynamic Recovery
              ↺
```

---

## 3. High-Level Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Human as Human Operator
    participant TaskRunner as SarthiTaskRunner
    participant Nemotron as Nebius Token Factory (Nemotron)
    participant StateReader as SarthiMuJoCoStateReader
    participant CtxBuilder as Decision Context Builder
    participant Jev as Jev Fast Decision Layer (Planned)
    participant Engine as SĀRTHI Decision Engine
    participant Articulation as SarthiMuJoCoArticulation
    participant MuJoCo as MuJoCo Physics Sim

    Note over Human,MuJoCo: 1. Human Command & Task Understanding
    Human->>TaskRunner: "Move the red object to the blue target."
    TaskRunner->>Nemotron: Parse instruction (Target, Zone, Semantic Actions)
    Nemotron-->>TaskRunner: Return structured TaskUnderstanding

    Note over Human,MuJoCo: 2. Closed-Loop Observe-Decide-Execute Cycle
    loop For each task phase until completion or terminal stop
        TaskRunner->>StateReader: Extract active WorldState from mjData
        StateReader-->>TaskRunner: Return canonical WorldState (Robot, Object, Obstacle)
        
        TaskRunner->>CtxBuilder: Build DecisionContext(WorldState, TaskUnderstanding)
        CtxBuilder-->>TaskRunner: Return compact DecisionContext
        
        opt Fast Candidate Decision via Jev
            TaskRunner->>Jev: Evaluate DecisionQuestion(Context, Candidates)
            alt Jev Responds Successfully
                Jev-->>TaskRunner: Return JevDecision(Ranked Candidates, Scores)
            else Jev Timeout / Provider Unavailable
                TaskRunner->>TaskRunner: Fallback to deterministic heuristic ranking
            end
        end

        Note over TaskRunner,Engine: 3. Deterministic Safety & Constraint Validation
        TaskRunner->>Engine: Validate candidate action against active constraints
        alt Constraint Violation Detected (e.g. PATH_BLOCKED)
            Engine-->>TaskRunner: REJECT candidate action; select REPOSITION recovery
        else Action Is Feasible & Clear
            Engine-->>TaskRunner: APPROVE candidate action (CandidateEvaluation APPROVED)
        end

        Note over TaskRunner,MuJoCo: 4. Numerical Articulation & Physical Execution
        TaskRunner->>Articulation: Dispatch approved action (Cartesian target)
        Articulation->>MuJoCo: DLS IK stepping + joint PD torque actuation
        MuJoCo-->>Articulation: Physics steps advance; contacts resolve
        
        Note over TaskRunner,MuJoCo: 5. Physical Outcome Verification
        TaskRunner->>StateReader: Inspect post-action physical state & contacts
        StateReader-->>TaskRunner: Verify action success (GRASP, REPOSITION, MOVE, RELEASE)
    end
```

---

## 4. Execution Scenarios & Branching Paths

### 4.1 Nominal Path: Unobstructed Pick-and-Place
1. **Instruction:** `"Move the red object to the blue target."`
2. **Task Understanding:** Nemotron identifies `red_object` and `blue_target`; proposes nominal sequence: `APPROACH`, `GRASP`, `MOVE`, `RELEASE`.
3. **Execution Phase 1 (APPROACH):**
   - Context built $\to$ Jev ranks `APPROACH` as top candidate.
   - Decision Engine verifies Cartesian reachability and workspace envelope $\to$ Approved.
   - DLS IK steps Panda arm to pre-grasp standoff pose ($z = 0.25\text{ m}$).
   - State Reader confirms end-effector standoff achieved.
4. **Execution Phase 2 (GRASP):**
   - Candidate `GRASP` approved $\to$ Parallel gripper closes.
   - State Reader detects persistent bilateral normal forces on `red_object` $\to$ `GRASP_VERIFIED`.
5. **Execution Phase 3 (MOVE):**
   - Clearance check verifies path is clear ($d_{\text{clearance}} \ge 0.082\text{ m}$) $\to$ Approved.
   - Arm navigates payload to target zone coordinates.
6. **Execution Phase 4 (RELEASE):**
   - Gripper opens $\to$ Object settles under gravity on target zone.
   - State Reader measures final placement error $\le 0.0600\text{ m}$ $\to$ `TASK_COMPLETED`.

---

### 4.2 Disturbance Path: Dynamic Obstacle Injection (`PATH_BLOCKED`) & Recovery
This path reflects the **empirically validated benchmark** of the SĀRTHI system:
1. **Initial Stages:** Robot completes `APPROACH` and successfully verifies `GRASP`.
2. **Disturbance Trigger:** A dynamic physical obstacle ($0.08 \times 0.08 \times 0.16\text{ m}$) is injected into the direct transit corridor.
3. **WorldState Ingestion:** State Reader advances `WorldState` version ($v3 \to v4$) and tags active constraint `PATH_BLOCKED`.
4. **Candidate Action Evaluation:** Nominal `MOVE` action evaluated:
   - Minimum trajectory distance to obstacle: **`0.066 m`**
   - Required clearance buffer: **`0.082 m`**
   - **Deterministic Verdict:** SĀRTHI Decision Engine **REJECTS** candidate `MOVE` due to path intersection.
5. **Recovery Deliberation:**
   - Decision question: *"Which recovery strategy is most appropriate for PATH_BLOCKED?"*
   - Options evaluated: `["REPOSITION", "REPLAN", "REGRASP", "STOP"]`.
   - Decision Engine selects **`REPOSITION`** with clearance altitude $z \ge 0.351\text{ m}$.
6. **Physical Recovery Execution:**
   - Franka arm elevates grasped object vertically to clearance altitude ($z \approx 0.333\text{–}0.351\text{ m}$).
   - `REPOSITION_VERIFIED`: Confirmed elevation exceeds obstacle height.
7. **Resume & Delivery:**
   - Subsequent `MOVE` over the obstacle is now verified and cleared.
   - Robot moves horizontally above obstacle, descends into target zone, and executes `RELEASE`.
8. **Final Verification:**
   - Final placement error: **`0.0535 m`** (allowed tolerance: **`0.0600 m`**).
   - Recovery count: **`1`**.
   - Status: **`COMPLETED`**.

---

### 4.3 Grasp Failure / Slip Scenario
1. Robot executes `GRASP` clamp motion.
2. State Reader inspects contact manifold: bilateral normal forces are missing or below threshold (object slipped or was misaligned).
3. Post-condition check fails: `GRASP_FAILED`.
4. Decision Engine does **not** proceed to `MOVE`.
5. Bounded query evaluated: `["REGRASP", "STANDOFF_RETRACT", "STOP"]`.
6. System executes standoff retraction, opens gripper, re-centers end-effector over detected object coordinates, and re-attempts grasp.
7. If repeated attempts exceed maximum retry threshold, robot halts in safe position hold.

---

### 4.4 Invalid Target / Missing Entity Scenario
1. Human command: *"Move the red object to the nonexistent zone."*
2. Nemotron parses intent: `target='red_object'`, `zone='nonexistent_zone'`.
3. Task Runner inspects active `WorldState` target zones: `'nonexistent_zone'` does not exist.
4. **Immediate Safe Termination:**
   - Failure Reason: `TARGET_LOCATION_NOT_FOUND: 'nonexistent_zone' not found in current WorldState.`
   - Robot remains in initial home position; zero motor actuation occurs.
   - Task terminates safely with status `INCOMPLETE`.

---

### 4.5 Jev Provider Timeout / Fallback Scenario
1. Task Runner builds `DecisionContext` and queries Jev for candidate action ranking.
2. Jev provider exceeds timeout threshold or experiences a network error.
3. System records a `DecisionLatencyRecord` with `provider_status: TIMEOUT_FALLBACK`.
4. **Deterministic Fallback:** The Task Runner invokes the Decision Engine's native deterministic candidate evaluation without interruption.
5. Execution continues safely; Jev failure causes zero downtime or unhandled exceptions.

---

### 4.6 Nemotron Timeout / Cloud Network Failure Scenario
1. Natural language instruction submitted during cloud network partition.
2. `NebiusNemotronProvider` exceeds HTTP client timeout ($30.0\text{ s}$).
3. Exception is caught, sanitized of any credentials, and reported as `COGNITIVE_SERVICE_UNAVAILABLE`.
4. Robot controller remains in default home position with brakes engaged. No unvalidated actions are generated.

---

### 4.7 Malformed Model Output / Schema Rejection Scenario
1. Generative or fast decision model produces invalid JSON or fails Pydantic schema validation.
2. Pydantic parser raises `ValidationError`.
3. The invalid payload is quarantined in telemetry.
4. For task understanding: task aborts safely. For Jev: system falls back to deterministic candidate ranking.

---

### 4.8 Deterministic Validation Rejection Scenario
1. A candidate action recommends moving to coordinates outside the robot's physical reach or workspace boundaries.
2. The SĀRTHI Decision Engine evaluates Cartesian bounds ($x \notin [0.1, 0.8]$, etc.).
3. Action status flagged `REJECTED_WORKSPACE_BREACH`.
4. Robot controller does not receive waypoints; system transitions to safe controlled hold.

---

### 4.9 Controlled STOP / Emergency Safe Termination
1. Operator triggers manual abort or critical unrecoverable physical anomaly occurs.
2. System transitions immediately to `SAFE_CONTROLLED_STOP`.
3. Trajectory generator applies smooth deceleration profile to zero joint velocities.
4. Franka arm holds current joint configuration via high-damping PD control.
