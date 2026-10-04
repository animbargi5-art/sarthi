# SĀRTHI — Product Requirements Document (PRD)

**Project Name:** SĀRTHI  
**Tagline:** An adaptive Physical AI architecture for context-aware, responsibility-driven robotic decision-making under changing physical conditions.  
**Target Event:** Nebius × NVIDIA Global AI Hackathon 2026  
**Document Version:** Version: 3.0  
**Status:** Draft — V3 Architecture Update  

---

## 1. Safety Architecture & Core Invariant

The following foundational safety principles govern all functional and technical requirements across SĀRTHI V3:

> [!IMPORTANT]
> **Core System Invariant:**  
> **"AI models propose or select bounded semantic candidates; deterministic software validates physical feasibility; only validated actions reach the robot controller."**  
> **"Jev must never bypass deterministic safety validation."**

No artificial intelligence model—whether foundation large language model or fast structured decision model—possesses direct motor actuation authority or raw trajectory publishing rights.

---

## 2. Product Overview & Problem Statement

Conventional robotic automation relies on rigid state machines and open-loop motion planning. When physical reality diverges from planned assumptions—such as an obstacle appearing in a transit corridor, an object shifting within a gripper, or clearance boundaries tightening mid-motion—traditional systems trigger hard emergency stops (E-Stops) or catastrophic collisions.

Conversely, ungrounded generative AI approaches that attempt direct end-to-end motor control introduce intolerable latency, hallucinated kinematic trajectories, and non-deterministic safety risks.

### The SĀRTHI V3 Vision
SĀRTHI V3 evolves from the validated MuJoCo pick-and-place disturbance recovery baseline into a high-speed **human-command-to-robot-decision architecture**. The system couples:
1. **Natural-Language Task Understanding:** Driven by **NVIDIA Nemotron** (hosted via **Nebius Token Factory**), converting human instructions into structured semantic intents.
2. **Fast Bounded Decision Layer (Planned):** Driven by **Jev**, providing rapid evaluation and selection among constrained, discrete candidate choices.
3. **Deterministic Physical Action Authority:** Governed strictly by the **SĀRTHI Decision Engine**, enforcing physical kinematic limits, collision clearance, reachability, and grasp stability.
4. **Physics-Based Simulation & Grounding:** Executing validated trajectories on a 7-DOF Franka Emika Panda arm in **MuJoCo**, extracting live contact dynamics and verifying execution outcomes.
5. **Physics Validation Framework:** Quantifying simulation fidelity, repeatability, and numerical stability across defined physical metrics.

---

## 3. Validated Baseline & V3 Evolution

### 3.1 Validated Baseline (Current State)
The V3 specification builds upon the empirically validated SĀRTHI Phase A–H implementation:
- **Task:** `"Move the red object to the blue target."`
- **Robotic Platform:** Franka Emika Panda (7-DOF arm, parallel-jaw gripper, Damped-Least-Squares IK) in MuJoCo physics.
- **Cognitive Model:** `nvidia/Nemotron-3-Ultra-550b-a55b` via Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1`).
- **Dynamic Disturbance:** Ingestion and recovery from a dynamic blocking obstacle (`PATH_BLOCKED`) post-grasp.
- **Recovery Outcome:** Direct `MOVE` rejected (clearance `0.066 m` < required `0.082 m`), `REPOSITION` selected, object lifted over obstacle ($z \approx 0.333\text{–}0.351\text{ m}$), verified `MOVE` and `RELEASE`.
- **Validation Metrics:** Placement error `0.0535 m` $\le$ tolerance `0.0600 m`, recovery count `1`, task status `COMPLETED`.
- **Automated Tests:** 186 unit/integration tests passing ($100\%$).

### 3.2 V3 System Evolution
V3 accelerates the deliberation cycle by introducing a dedicated fast structured decision layer (**Jev**), formalizing the combined **Human Instruction + Physical Situation** input model, establishing a comprehensive **Physics Validation Framework**, and systematically instrumenting **Decision Latency**.

---

## 4. Dual Input Model: Command + Physical Situation

SĀRTHI V3 explicitly ingests two distinct input streams to formulate robotic decisions:

### A. Human Natural-Language Instruction
High-level operational intent communicated by a human operator:
- Example: *"Move the red object to the blue target."*
- Processed by NVIDIA Nemotron to extract semantic entities (`target_object`, `destination_zone`, expected action sequence).

### B. Physical Situation
The real-time empirical state of the physical environment, robot, and task context:
- **RobotState:** Joint angles, joint velocities, end-effector 6D pose, gripper width, actuation efforts.
- **WorldObject State:** 3D coordinates, bounding box, grasp state (`FREE`, `IN_TRANSIT`, `PLACED`), contact forces.
- **TargetZone State:** Center coordinates, spatial tolerance radius, occupancy status.
- **EnvironmentState & Obstacles:** Obstacle locations, geometry, dynamic activation flags (`PATH_BLOCKED`).
- **ActiveConstraints:** Spatial keepout zones, velocity ceilings, minimum clearance buffers.
- **LastActionOutcome:** Status, execution duration, tracking error, and contact verification flags of the preceding action.
- **Telemetry History:** Rolling window of force/torque and contact manifold metrics.

The system synthesizes these inputs into a compact, bounded **DecisionContext** tailored to specific decision questions, avoiding raw, uncurated simulation dumps.

---

## 5. Architectural Roles & Model Boundaries

| Subsystem | Core Technology | Primary Responsibility | Hard Invariants & Constraints |
| :--- | :--- | :--- | :--- |
| **Cognitive Task Understanding** | NVIDIA Nemotron (via Nebius Token Factory) | Natural-language instruction comprehension; entity extraction; structured `TaskUnderstanding` generation. | **Zero Motor Control:** Outputs semantic intent only. Does not publish trajectories or modify `WorldState`. |
| **Fast Bounded Decision Layer** *(Planned)* | Jev (Fast Structured Decision Provider) | Rapid evaluation and ranking of discrete candidate actions; answering bounded decision questions. | **Candidate Proposal Only:** Does not authorize actuation. Output is treated as an unverified candidate decision. |
| **Deterministic Action Authority** | SĀRTHI Decision Engine | Evaluates candidate actions against physical constraints; selects recovery primitives; final gatekeeper. | **Sole Physical Authority:** Validates reachability, clearance, and safety. Can reject any AI recommendation. |
| **Robot Execution & Physics** | MuJoCo (`mujoco >= 3.1.0`) / Franka Panda | DLS numerical IK; joint PD control; rigid-body contact dynamics; live physical state extraction. | **Untrusted AI Inputs:** Only executes actions approved by the deterministic Decision Engine. |

---

## 6. Functional Requirements (FR)

### 6.1 Task Understanding & Cognitive Interpretation
- **FR-1.1 (Human Instruction Ingestion):** Ingest natural-language instructions from operators or task dispatchers.
- **FR-1.2 (Nemotron Semantic Parsing):** Dispatch instructions to NVIDIA Nemotron hosted on Nebius Token Factory. Produce Pydantic-validated `TaskUnderstanding` containing target entities, destination zones, and semantic action sequences.
- **FR-1.3 (Semantic Fallback):** If Nemotron times out or returns malformed output, report descriptive task-understanding errors and halt execution safely without moving the robot.

### 6.2 Situation Ingestion & Context Synthesis
- **FR-2.1 (Live State Extraction):** Read physical state directly from active simulation coordinates and contact manifolds using `SarthiMuJoCoStateReader` without secondary estimation models.
- **FR-2.2 (Decision Context Builder):** Assemble a compact `DecisionContext` merging semantic task goals with active `WorldState`, obstacle clearance metrics, and previous action outcomes.

### 6.3 Fast Bounded Decision Layer (Jev — Planned)
- **FR-3.1 (Bounded Question Interface):** Formulate targeted decision queries (`DecisionQuestion`) with discrete, enumerated alternatives:
  - *Recovery Selection:* e.g., `["REPOSITION", "REPLAN", "REGRASP", "STOP"]`
  - *Candidate Action Selection:* e.g., `["APPROACH", "GRASP", "MOVE", "RELEASE", "REPOSITION", "STOP"]`
  - *Recovery Necessity:* e.g., `["RECOVER", "CONTINUE", "STOP"]`
  - *Target Disambiguation:* Select among multiple valid physical candidates.
- **FR-3.2 (Fast Candidate Scoring):** Jev evaluates the bounded question and outputs a structured `JevDecision` with candidate ranking and probability distribution.
- **FR-3.3 (Jev Resiliency & Fallback):** If Jev is unreachable, times out, or produces invalid schema, the system must seamlessly fall back to deterministic Decision Engine heuristic ranking. Jev must never be a single point of failure.

### 6.4 Deterministic Safety & Constraint Validation
- **FR-4.1 (Physical Constraint Checking):** Validate every candidate action against:
  - Workspace boundaries ($x, y, z$).
  - Kinematic reachability and joint limits.
  - Obstacle keepout volumes and clearance buffers ($d_{\text{clearance}} \ge \delta_{\text{required}}$).
  - Gripper state and grasp prerequisites.
- **FR-4.2 (Autonomous Recovery Selection):** If a nominal action violates constraints (e.g., path blocked by dynamic obstacle), reject the action deterministically and trigger approved recovery maneuvers (`REPOSITION`).

### 6.5 Physical Execution & Outcome Verification
- **FR-5.1 (DLS Inverse Kinematics):** Convert approved Cartesian targets into joint trajectories using Damped-Least-Squares IK with joint-limit clamping.
- **FR-5.2 (Physical Verification):** Following execution, inspect actual MuJoCo physics state (object position, contact manifolds, gripper closure). Confirm success against defined tolerances before advancing task stage.

### 6.6 Physics Validation Framework (New V3 Requirement)
- **FR-6.1 (12-Point Physics Verification):** Execute automated physical verification benchmarks covering:
  1. *Timestep Sensitivity:* Stability across controlled simulation timestep configurations.
  2. *Joint-Limit Correctness:* Trajectories strictly respect configured joint limits.
  3. *Joint Velocity Limits:* Commanded velocities respect configured bounds.
  4. *EE Trajectory Repeatability:* Deviation across repeated deterministic runs.
  5. *Contact Stability:* Grasp and contact manifold stability under repeated trials.
  6. *Placement Repeatability:* Target zone settling variance across repeated runs.
  7. *Collision/Clearance Validation:* Rejection of unsafe paths when clearance is insufficient.
  8. *Gravity & Settling Behavior:* Physical settling behavior of released objects.
  9. *IK Convergence:* Residual error tracking of the DLS IK solver.
  10. *Numerical Stability:* Zero NaN/Inf occurrences across all state tensors.
  11. *Deterministic Replay:* Identical initial seed/state produces identical trajectories within defined tolerances.
  12. *Physics Regression:* Automated comparison against baseline telemetry records.

---

## 7. Non-Functional Requirements (NFR)

### 7.1 Decision Latency & Performance Measurement
V3 defines explicit latency metrics to be benchmarked across the decision lifecycle (no unverified latency claims are assumed):
- **NFR-1.1 (Task Understanding Latency):** Duration from human command submission to validated `TaskUnderstanding` from Nebius Token Factory.
- **NFR-1.2 (Jev Decision Latency):** Duration from `DecisionQuestion` dispatch to structured `JevDecision` receipt.
- **NFR-1.3 (Deterministic Validation Latency):** Duration for the Decision Engine to validate candidate actions against active constraints.
- **NFR-1.4 (Total Decision-Cycle Latency):** Combined duration: Observation $\to$ Context $\to$ Fast Decision $\to$ Validation $\to$ Dispatch.
- **NFR-1.5 (Physical Execution Duration):** Total simulated/wall-clock time required for trajectory execution in MuJoCo.
- **NFR-1.6 (Verification Latency):** Time required to query `mjData` and verify physical state predicates.
- **NFR-1.7 (Recovery Decision Latency):** End-to-end deliberation time from disturbance detection to recovery action dispatch.

### 7.2 Safety, Robustness & Fail-Safe Invariants
- **NFR-2.1 (Zero Direct Actuation):** Zero code paths shall exist whereby model outputs bypass the deterministic Decision Engine and directly drive actuators.
- **NFR-2.2 (Deterministic Fallback on Model Failure):** If either Nemotron or Jev encounters a network error, timeout, or malformed response, the robot must immediately hold position or initiate controlled safe stopping.
- **NFR-2.3 (Credential Security):** API keys (`NEBIUS_API_KEY`, Jev credentials) must be managed exclusively through environment variables and never logged or serialized.

### 7.3 Simulation vs. Hardware Boundary
- **NFR-3.1 (Simulation Grounding):** All V3 capabilities are validated within high-fidelity MuJoCo physics.
- **NFR-3.2 (Hardware Boundary):** Physical hardware deployment on Franka Panda remains future work; no claims of real-world physical hardware validation shall be made until physical tests are executed.

---

## 8. Success Metrics & Key Performance Indicators (KPIs)

1. **Closed-Loop Task Completion:** $\ge 95\%$ successful pick-place-recover runs in MuJoCo under dynamic obstacle disturbance.
2. **Deterministic Constraint Enforcement:** $100\%$ rejection rate of kinematically infeasible or collision-violating candidate actions (zero unconstrained collisions).
3. **Automated Test Coverage:** Maintain $100\%$ passing tests across existing 186 unit/integration tests and all new V3 validation suites.
4. **Reproducibility & Repeatability:** Zero non-deterministic state divergence under identical seed and initial conditions in MuJoCo.
5. **Fast Decision Latency Differential:** Quantified empirical comparison of Jev bounded decision latency versus full LLM generative re-prompting.
