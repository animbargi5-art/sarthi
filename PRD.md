# SĀRTHI — Product Requirements Document (PRD)

**Project Name:** SĀRTHI  
**Tagline:** An adaptive Physical AI architecture for context-aware, responsibility-driven robotic decision-making under changing physical conditions.  
**Target Event:** Nebius × NVIDIA Global AI Hackathon 2026  
**Document Version:** 1.0 (Phase 0 Foundation)

---

## 1. Problem Statement

Conventional robotic automation relies on deterministic trajectory generators and rigid state machines. In controlled settings, these systems excel; however, when deployed in dynamic physical environments, they exhibit extreme fragility:
1. **Unpredictable Physical Disturbances:** Transient contact shifts, payload slips, fluctuating friction, unexpected obstacles, and external force impulses instantly invalidate precomputed motion plans.
2. **Brittle Exception Handlers:** Classical error handlers typically execute an emergency stop (E-Stop), requiring manual human intervention, causing industrial downtime, and failing to achieve task autonomy.
3. **The Grounding Gap in AI:** Generic LLM-based planners lack physical grounding, output infeasible kinematic trajectories, and introduce severe latency penalties unsuitable for closed-loop robotic control.

Robotic agents require an architecture that can sense physical disturbances, preserve safety boundaries deterministically, deliberate over root causes using high-reasoning foundation models, and execute adaptive recovery maneuvers in real time.

---

## 2. Product Vision & Goals

**SĀRTHI** couples the advanced reasoning capabilities of **NVIDIA Nemotron** (accelerated via **Nebius Token Factory**) with the high-fidelity physics environment of **NVIDIA Isaac Sim**.

### Primary Objectives:
- **Autonomous Disturbance Recovery:** Detect, classify, and recover from physical disturbances without human intervention.
- **Sub-500ms Deliberation Loop:** Utilize Nebius Token Factory's inference throughput to achieve near real-time re-planning.
- **Physical AI Grounding:** Validate all high-level cognitive recovery strategies inside NVIDIA Isaac Sim prior to and during execution.
- **Strict Deterministic Safety:** Enforce non-negotiable kinematic and workspace limits, ensuring zero hallucinated motion execution.

---

## 3. Target Personas & Core Use Cases

### 3.1 Target Personas
- **Robotics Systems Engineer:** Needs reliable, modular recovery primitives that integrate with standard robot middleware.
- **Industrial Automation Operator:** Monitors fleet status, requiring real-time observability of disturbance events and agent reasoning logs.
- **Physical AI Researcher:** Evaluates closed-loop reasoning, foundation model grounding, and disturbance benchmarks in physics simulation.

### 3.2 Key Use Cases
1. **Payload Slip during High-Speed Pick-and-Place:** An object begins slipping due to surface oil or inertial acceleration. SĀRTHI detects the normal force drop, transitions to hold, deliberates an adaptive grip adjustment, and re-secures the payload before continuing.
2. **Unexpected Spatial Obstacle / Collision:** An unexpected crate or barrier blocks the nominal trajectory. SĀRTHI senses the reaction torque, aborts the path safely, queries Nemotron for an alternative detour, and maneuvers around the obstacle.
3. **Payload Mass / Inertia Discrepancy:** The grasped object is 200% heavier than nominal. The system detects joint effort saturation, computes center-of-mass compensation, and adjusts acceleration profiles.

---

## 4. Functional Requirements

| ID | Feature | Description | Priority |
| :--- | :--- | :--- | :--- |
| **FR-1** | **Telemetry Ingestion** | Ingest joint positions, velocities, efforts, end-effector poses, and 6-axis F/T data at 100 Hz from NVIDIA Isaac Sim. | P0 |
| **FR-2** | **Disturbance Detection** | Automatically trigger an anomaly event when observed telemetry exceeds dynamic error or force thresholds ($> 15\text{ N}$ delta or $> 0.05\text{ m}$ tracking error). | P0 |
| **FR-3** | **Cognitive Deliberation** | Query NVIDIA Nemotron hosted on Nebius Token Factory with a structured snapshot of the disturbance and available recovery primitives. | P0 |
| **FR-4** | **Structured Strategy Output** | Enforce JSON schema validation on Nemotron responses to extract recovery tactics, waypoints, compliance parameters, and diagnostic rationale. | P0 |
| **FR-5** | **Deterministic Safety Filter** | Reject any generated plan exceeding joint torque limits, velocity limits, or workspace boundaries ($x, y, z$). | P0 |
| **FR-6** | **Closed-Loop Execution** | Dispatch verified recovery primitives to the robot inside NVIDIA Isaac Sim and monitor convergence until nominal state is restored. | P0 |
| **FR-7** | **Operator Observability** | Stream live telemetry, disturbance triggers, and Nemotron reasoning chains over WebSocket for real-time visualization. | P1 |

---

## 5. Non-Functional Requirements

### 5.1 Performance & Latency
- **NFR-1 (Inference Latency):** Cloud inference roundtrip via Nebius Token Factory must complete within $\le 400\text{ ms}$.
- **NFR-2 (End-to-End Recovery Time):** Total time from disturbance detection to active recovery motion dispatch must be $\le 500\text{ ms}$.
- **NFR-3 (Simulation Fidelity):** NVIDIA Isaac Sim physics simulation must run at $\ge 60\text{ Hz}$ physics step with accurate contact solving.

### 5.2 Safety & Reliability
- **NFR-4 (Deterministic Fail-Safe):** If inference fails, times out ($> 500\text{ ms}$), or outputs an invalid trajectory, the system must trigger an immediate local position hold or controlled safe descent.
- **NFR-5 (Zero Collision Ingestion):** Unvalidated trajectories must never be dispatched to the physical/simulated controllers.

### 5.3 Modularity & Code Quality
- **NFR-6 (Clean Architecture):** Strict separation of concerns between backend orchestration, inference client, simulation bridge, and operator UI. Zero fake implementations or mock shortcuts.

---

## 6. Success Metrics & Hackathon KPIs

1. **Autonomous Recovery Success Rate:** $\ge 90\%$ successful task completion across randomized disturbance scenarios (slips, obstacles, mass changes).
2. **Mean Recovery Deliberation Time:** $\le 350\text{ ms}$ average cognitive response time.
3. **Safety Violations:** Exactly $0$ unconstrained workspace limit violations across all test runs.
4. **Cognitive Coherence:** Accurate diagnosis of disturbance root-cause by NVIDIA Nemotron in $\ge 95\%$ of test trials.
