# SĀRTHI — Feedback & Evaluation Framework

This document tracks technical evaluations, judge feedback, architectural reviews, and iterative improvements for **SĀRTHI** during the **Nebius × NVIDIA Global AI Hackathon 2026**.

---

## Hackathon Evaluation Rubric Mapping

| Criterion | Target Metric / Standard | SĀRTHI Focus Area | Current Status |
| :--- | :--- | :--- | :--- |
| **Physical AI Depth** | Closed-loop interaction between physics simulation and cognitive model | NVIDIA Isaac Sim contact dynamics coupled with NVIDIA Nemotron cognitive reasoning | Defined (Phase 0) |
| **Inference Efficiency** | Predictable token throughput and low recovery decision latency | Powered by Nebius Token Factory with strict latency budgeting (< 500 ms roundtrip) | Architecture Specified |
| **Disturbance Robustness** | Measurable recovery rate under dynamic mechanical disturbances | Autonomous re-planning upon slip, collision, or unexpected mass variance | Test Scenarios Outlined |
| **Safety & Determinism** | 100% boundary compliance; zero unconstrained motor actuation | Deterministic kinematic & workspace validation filter before Isaac Sim command dispatch | Contract Specified |
| **Engineering Rigor** | Clean modularity, reproducible builds, full observability | Type-safe schemas, comprehensive documentation, and structured logging | Phase 0 Baseline Complete |

---

## Review Log

| Date | Reviewer / Role | Focus Area | Observations & Action Items | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-09-25 | Core Architecture Team | Phase 0: Repository Foundation | Verified directory layout, technical specifications, and strict exclusion of mock implementations. | Approved |
| Pending | Technical Mentor / Judge | Nebius Token Factory Integration | Validate API handshake and structured output schema conformance for Nemotron. | Upcoming |
| Pending | Simulation Engineer | Isaac Sim Environment | Benchmark PhysX 5 contact stability and disturbance injector responsiveness. | Upcoming |
| Pending | Red Team / Safety Review | Kinematic Guardrails | Verify emergency brake and workspace limit clamping under divergent LLM outputs. | Upcoming |

---

## Feedback Collection Template

For reviewers submitting feedback, please record issues and recommendations in the format below:

```markdown
### Review Entry: [YYYY-MM-DD] - [Reviewer Name / Org]
- **Category:** [Architecture | Nebius Inference | Isaac Sim | Safety Guard | UI/Telemetry]
- **Severity:** [Low | Medium | High | Critical]
- **Observed Behavior:**
  <Description of current behavior or architectural gap>
- **Expected / Target Behavior:**
  <Target standard or performance metric>
- **Recommended Action:**
  <Specific remediation steps or refactor plan>
- **Resolution Tracking:**
  - Owner: [Unassigned]
  - Target Milestone: [Phase X]
  - Resolution Note: [Pending]
```
