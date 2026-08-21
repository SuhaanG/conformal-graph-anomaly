"""
audit_results_published.py

Flags numeric claims in theory/joint_discovery_threshold_proposition.md
that reference a script whose output CSV is NOT present in
results/published/. Per results/published/README.md's own rule: every
number in the paper must trace to a committed CSV.

Usage:
  python3 scripts/audit_results_published.py
"""

import os
import re

THEORY_DOC = os.path.join(os.path.dirname(__file__), "..", "theory",
                          "joint_discovery_threshold_proposition.md")
PUBLISHED_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "published")

# script name -> expected output CSV substring
SCRIPT_TO_CSV = {
    "condition_comparison_pygod.py": "condition_comparison_pygod",
    "clean_selection_degree_diagnostic.py": "clean_selection_degree",
    "severity_sweep_pygod_instrumented.py": "severity_sweep_pygod",
    "selection_bias_matrix.py": "selection_bias_matrix",
    "calibration_strategy_comparison.py": "calibration_strategy",
    "degree_baseline_check.py": "degree_baseline_check",
    "degree_sensitivity_sweep.py": "degree_sensitivity",
    "adadetect_comparison.py": "adadetect_comparison",
    "exposure_degree_confound_check.py": "exposure_degree_confound",
    "real_data_experiment.py": "real_data_experiment",
}


def main():
    with open(THEORY_DOC) as f:
        text = f.read()

    published_files = os.listdir(PUBLISHED_DIR) if os.path.isdir(PUBLISHED_DIR) else []

    print(f"Files in results/published/: {len(published_files)}")
    for f in sorted(published_files):
        print(f"  {f}")
    print()

    print("Scripts referenced in theory doc, and whether a matching CSV exists:")
    print("-" * 70)
    missing = []
    for script, csv_substr in SCRIPT_TO_CSV.items():
        mentioned = script in text
        if not mentioned:
            continue
        has_csv = any(csv_substr in f for f in published_files)
        status = "OK" if has_csv else "MISSING"
        print(f"  {script:<42}{status}")
        if not has_csv:
            missing.append(script)

    print()
    if missing:
        print(f"{len(missing)} script(s) referenced in the theory doc have NO backing CSV")
        print("in results/published/. Numbers from these are not yet citable per")
        print("results/published/README.md's own rule.")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All referenced scripts have a backing CSV.")


if __name__ == "__main__":
    main()