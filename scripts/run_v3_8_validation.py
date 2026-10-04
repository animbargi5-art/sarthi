"""
SĀRTHI V3-8 Physics Validation Suite Runner.

Executes all 12 physics validation dimensions:
1. timestep sensitivity
2. joint limits
3. joint velocities
4. end-effector repeatability
5. contact stability
6. placement repeatability
7. collision / clearance
8. gravity / settling
9. IK convergence
10. numerical stability
11. deterministic replay
12. physics regression

Serializes the complete results into `records/v3_8_physics_validation.json`.
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.validation.framework import SarthiPhysicsValidator


def main():
    print("=" * 70)
    print("SARTHI V3-8: Physics Validation Framework Execution")
    print("=" * 70)

    validator = SarthiPhysicsValidator()
    print("\nRunning all 12 physics validation suites...")
    report = validator.run_all_validations()

    print("\n--- Suite Summary ---")
    print(f"Total Validations:  {report.total_validations}")
    print(f"Passed Validations: {report.passed_validations}")
    print(f"Failed Validations: {report.failed_validations}")
    print(f"All Passed:         {report.all_passed}")

    print("\n--- Dimension Results ---")
    for dim_name, res in report.results.items():
        status = "PASS" if res.pass_fail else "FAIL"
        print(f"  [{status}] {dim_name}")
        if not res.pass_fail:
            for reason in res.failure_reasons:
                print(f"       -> {reason}")

    # Save to records/v3_8_physics_validation.json
    records_dir = REPO_ROOT / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    out_file = records_dir / "v3_8_physics_validation.json"

    data = report.to_dict()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved benchmark record to: records/v3_8_physics_validation.json ({out_file.stat().st_size} bytes)")
    if not report.all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
