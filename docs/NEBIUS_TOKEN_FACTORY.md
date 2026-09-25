# Nebius Token Factory — NVIDIA Nemotron Provider Guide

## 1. Overview

SĀRTHI integrates with **Nebius Token Factory** as its cloud foundation model inference provider. Nebius Token Factory hosts high-performance open foundation models, with **NVIDIA Nemotron** (e.g. `nvidia/nemotron-4-340b-instruct` or related variants) serving as the primary cognitive engine for natural language instruction parsing and semantic task understanding.

Nebius Token Factory exposes a standard **OpenAI-compatible REST API**, enabling SĀRTHI to interface via the official `openai` Python SDK by specifying a custom `base_url`.

> [!IMPORTANT]
> **Status:** Real provider verification is **PENDING** until Nebius billing and API credentials are provided. All local unit and integration tests use strictly mocked endpoints with **ZERO** live network calls.

---

## 2. Architectural Role and Safety Invariants

In the SĀRTHI architecture, the foundation model operates exclusively as a **cognitive interpreter**:

```
Human Natural Language Instruction
             ↓
  TaskUnderstandingService
             ↓
   NebiusNemotronProvider  (OpenAI-compatible endpoint)
             ↓
    TaskUnderstanding      (Type-safe, Pydantic validated)
             ↓
      TaskInterpreter
             ↓
        TaskRunner         (Target resolution against WorldState)
             ↓
     SĀRTHI Decision Engine (SOLE Physical-Action Authority)
             ↓
     SimulationAdapter     (Simulation / Hardware Actuation)
```

### Strict Non-Execution Guarantees:
- **No Physical Execution:** Nemotron never directly commands robot joints, motors, or grippers.
- **No World State Modification:** Nemotron cannot alter `WorldState` or create phantom objects.
- **No Direct Simulation Access:** Nemotron does not interface with `SimulationAdapter` or physics engines.
- **Semantic Intent Only:** Fields such as `required_actions` (e.g., `["APPROACH", "GRASP", "MOVE", "RELEASE"]`) are semantic expectations only. Physical trajectories, collision checks, inverse kinematics, and recovery loops remain strictly governed by the SĀRTHI Decision Engine.

---

## 3. Configuration Variables

SĀRTHI reads provider configuration from environment variables or explicit constructor parameters:

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `NEBIUS_API_KEY` | Yes (for live calls) | *None* | Authentication token for Nebius Token Factory. |
| `NEBIUS_BASE_URL` | No | `https://api.tokenfactory.nebius.com/v1` | OpenAI-compatible endpoint URL. |
| `NEBIUS_MODEL` | Yes (for live calls) | *None* | Exact Nemotron model identifier exposed in Token Factory. |
| `NEBIUS_TIMEOUT` | No | `30.0` | Client network request timeout in seconds. |

Template configuration is documented in `.env.example`:

```bash
NEBIUS_API_KEY=
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
NEBIUS_MODEL=
```

---

## 4. Setup & Obtaining Credentials

When Nebius account billing setup is resumed:

1. **Obtain API Key:**
   - Log into the [Nebius Token Factory Console](https://tokenfactory.nebius.com).
   - Generate an API Key in the IAM / API Keys dashboard.
2. **Determine Available Models:**
   - Query the Nebius Token Factory models endpoint:
     ```bash
     curl -H "Authorization: Bearer $NEBIUS_API_KEY" https://api.tokenfactory.nebius.com/v1/models
     ```
   - Identify the exact Nemotron model ID (e.g., `nvidia/nemotron-4-340b-instruct` or other available NVIDIA Nemotron variant).
3. **Configure Environment:**
   - Export environment variables in your local shell or deployment container:
     ```bash
     export NEBIUS_API_KEY="your-actual-api-key"
     export NEBIUS_BASE_URL="https://api.tokenfactory.nebius.com/v1"
     export NEBIUS_MODEL="<confirmed-nemotron-model-id>"
     ```
   - **DO NOT** commit real keys to source control or `.env` files.

---

## 5. Security & Credential Protection

SĀRTHI implements strict defenses against credential leaks:

- **No Hard-Coded Keys:** Source code and tests contain zero hard-coded API credentials.
- **Sanitized Exceptions:** `NebiusNemotronProvider` scrubs authentication headers, Bearer tokens, and key strings from all exception messages, logs, and `__repr__` output.
- **Git Protection:** `.gitignore` explicitly excludes `.env`, `.env.*`, and secret credential files while preserving `.env.example`.
- **Traceback Isolation:** Underlying client exceptions are re-raised using `from None` to avoid leaking client session context in tracebacks.

---

## 6. Running the Test Suite (Offline / Mocked)

The SĀRTHI test suite is designed to run completely offline without an active Nebius account or API key:

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This verifies:
1. Provider configuration handling (defaults and explicit overrides).
2. Missing key / model rejection (`NebiusConfigurationError`).
3. Endpoint URL configuration.
4. Correct model ID dispatch.
5. Structured JSON response parsing.
6. Markdown-fenced (` ```json ... ``` `) response unwrapping.
7. Malformed JSON rejection (`NebiusParsingError`).
8. Pydantic schema validation enforcement (`NebiusValidationError`).
9. Network, timeout, and API status error translation.
10. Credential redaction in exception messages and representations.
11. Enforcement that the provider possesses zero physical actuation capability.
12. Mocked end-to-end integration through `TaskUnderstandingService` -> `TaskRunner.run_instruction()` -> `Decision Engine` -> `Simulation`.

---

## 7. Next Steps: Live Verification

Live provider verification will be conducted in Phase 5B once the Nebius Token Factory account setup is finalized:
- Authenticated smoke test against `https://api.tokenfactory.nebius.com/v1/models`.
- Model availability and token context size validation.
- Live instruction test: `"Move the red object to the blue target."`
- Structured response latency benchmarking.
