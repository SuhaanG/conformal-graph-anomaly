"""
run_all_sanity_checks.py

Runs every test_*.py sanity/integration script built for the new
additions (cola, weighted_conformal, the strategy comparisons) in
sequence. One command to verify everything before spending real compute.

Usage:
  python3 scripts/run_all_sanity_checks.py
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(__file__)

CHECKS = [
    "test_selection_bias_sanity.py",
    "test_calibration_strategy_integration.py",
]

MODULE_SELFTESTS = [
    os.path.join(SCRIPT_DIR, "..", "src", "weighted_conformal.py"),
]


def run(cmd_path):
    print(f"\n{'='*70}\nRUNNING: {cmd_path}\n{'='*70}")
    result = subprocess.run([sys.executable, cmd_path], capture_output=False)
    return result.returncode == 0


def main():
    results = {}

    for check in CHECKS:
        path = os.path.join(SCRIPT_DIR, check)
        results[check] = run(path)

    for path in MODULE_SELFTESTS:
        results[os.path.basename(path)] = run(path)

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    all_ok = True
    for name, ok in results.items():
        status = "PASSED" if ok else "FAILED"
        print(f"  {status:<8} {name}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\nAt least one check FAILED. Do not trust results built on")
        print("these modules until this is resolved.")
        sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()