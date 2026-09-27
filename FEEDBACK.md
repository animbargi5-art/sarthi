# SĀRTHI — Feedback & Evaluation

This document records technical evaluation notes, implementation feedback, and final validation evidence for SĀRTHI.

## Project Validation Status

SĀRTHI has been validated as a closed-loop Physical AI prototype using:

- NVIDIA Nemotron-3-Ultra for task understanding
- Nebius Token Factory for live model inference
- MuJoCo for physics-based robot simulation
- A deterministic Decision Engine for physical action authorization
- Closed-loop world-state observation, action execution, verification, and recovery

## Final Validation Evidence

The validated task is:

> Move the red object to the blue target.

The live validation demonstrated:

1. Natural-language task received.
2. Nemotron interpreted the task.
3. Robot approached the target object.
4. Physical grasp was verified.
5. A path-blocking disturbance was introduced.
6. The original MOVE action was rejected by the deterministic constraint validator.
7. The Decision Engine selected REPOSITION as the recovery action.
8. The robot repositioned above the obstacle.
9. The object was moved to the target zone.
10. Release was verified.
11. Final placement was verified within the configured tolerance.
12. The task completed successfully.

### Key Recovery Evidence

- Blocked trajectory minimum distance: `0.066 m`
- Required clearance: `0.082 m`
- Original MOVE action: rejected
- Recovery action: `REPOSITION`
- Recovery clearance target: approximately `0.351 m`
- Final placement error: `0.0535 m`
- Allowed placement tolerance: `0.0600 m`
- Recovery count: `1`

## Test Validation

Final automated test suite:

- Tests: `186`
- Passed: `186`
- Failed: `0`
- Errors: `0`

## Nebius Token Factory Feedback

Nebius Token Factory provided a practical OpenAI-compatible inference interface for integrating Nemotron into the SĀRTHI decision pipeline.

Observed strengths:

- Straightforward API integration
- Structured JSON response handling
- Clear model selection
- Suitable for rapid prototyping and evaluation
- Enabled live Nemotron inference without requiring a local large-model deployment

Engineering consideration:

- Production deployments should continue to enforce strict timeouts, structured output validation, secret protection, and deterministic downstream action validation.

## NVIDIA Model Feedback

Nemotron was used for natural-language task understanding.

The model is not granted direct physical control authority. Its structured interpretation is passed into the deterministic SĀRTHI execution pipeline, where physical actions remain subject to world-state and constraint validation.

## MuJoCo Feedback

MuJoCo provided a practical local simulation environment for validating the closed-loop robotics workflow.

Advantages observed:

- Lightweight local execution
- Fast iteration
- Deterministic test execution
- Physics-based joint and object state
- Suitable for repeatable development and validation on the available hardware

Current simulation limitations:

- The scenario is a tabletop manipulation task.
- The disturbance is currently represented by activating a blocking obstacle.
- The environment does not yet model the full range of dynamic real-world disturbances.
- Physical hardware validation has not been performed.

## Safety and Determinism

The Decision Engine remains the physical action authority.

Model output does not directly command robot motors.

Before an action is executed, the system evaluates the current world state and active constraints. If an action violates a constraint, it is rejected and the system can reassess the situation and select a recovery action.

## Final Review Notes

The current implementation is considered submission-ready for the implemented MuJoCo validation scope.

Future extensions may include:

- Additional disturbance types
- More complex manipulation tasks
- Physical robot deployment
- Additional perception modalities
- Broader robustness evaluation
- Hardware-specific inference optimization
