"""
run_full_experiment.py

The FULL experiment (not pilot-scale). Consolidates everything validated
during the pilot into one reproducible run, at publication-appropriate
seed counts instead of the smaller pilot numbers used to validate the
pipeline itself.

Phases:
1. Synthetic 3-condition experiment (clean/contaminated/adversarial),
   n_nodes=15000, at FULL seed count (default 100, up from pilot's 50).
2. Synthetic severity sweep (5 contamination levels under adversarial
   calibration), at FULL seed count (default 50, up from pilot's 20).
3. Real-data experiment on FraudAmazonDataset, at FULL seed count
   (default 30, up from pilot's 15).
4. Real-data experiment on FraudYelpDataset, same seed count as Amazon.
   NOTE: this is the first full-scale run on Yelp -- the loader was only
   syntax/logic-tested locally, never actually downloaded/run (no network
   access to DGL's servers from the dev sandbox). Watch phase 4's first
   few lines closely; if Yelp's scale or schema causes problems, the
   error will surface immediately at the data-loading step, before
   burning compute on training.

Calls the existing, already-validated scripts as subprocesses rather than
reimplementing their logic, specifically to avoid introducing new bugs in
code that has already been debugged and confirmed correct.

IMPORTANT: run with --quick first to sanity-check the entire pipeline
end-to-end with tiny seed counts (a few minutes) before committing to the
full multi-hour run. This is exactly the kind of check that would have
caught several of the pilot's bugs (calibration confounds, exchangeability
violations) in seconds instead of after a full run completed.

Usage:
  # Quick end-to-end sanity check (a few minutes, tiny seed counts):
  python3 scripts/run_full_experiment.py --quick --device cuda

  # Full experiment (multi-hour, publication seed counts):
  python3 scripts/run_full_experiment.py --device cuda

  # Skip real-data phases (e.g. if only synthetic results are needed right now):
  python3 scripts/run_full_experiment.py --skip_real_data --device cuda

  # Skip Yelp specifically (e.g. if it's known to need more debugging):
  python3 scripts/run_full_experiment.py --skip_yelp --device cuda
"""

import argparse
import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_phase(name, cmd):
    print(f"\n{'='*70}")
    print(f"PHASE: {name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.join(SCRIPT_DIR, ".."))
    elapsed = time.time() - start
    status = "SUCCEEDED" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n--- Phase '{name}' {status} in {elapsed/60:.1f} minutes ---\n")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Tiny seed counts for an end-to-end sanity check before the full run.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--skip_synthetic", action="store_true")
    parser.add_argument("--skip_severity", action="store_true")
    parser.add_argument("--skip_real_data", action="store_true")
    parser.add_argument("--skip_yelp", action="store_true",
                         help="Skip Yelp specifically (e.g. if it needs separate debugging), keeping Amazon.")
    parser.add_argument("--full_n_seeds_synthetic", type=int, default=100)
    parser.add_argument("--full_n_seeds_severity", type=int, default=50)
    parser.add_argument("--full_n_seeds_real", type=int, default=30)
    args = parser.parse_args()

    if args.quick:
        n_seeds_synthetic = 3
        n_seeds_severity = 3
        n_seeds_real = 2
        print("QUICK MODE: using tiny seed counts for an end-to-end sanity check.\n"
              "Run without --quick once this completes cleanly.")
    else:
        n_seeds_synthetic = args.full_n_seeds_synthetic
        n_seeds_severity = args.full_n_seeds_severity
        n_seeds_real = args.full_n_seeds_real

    results = {}
    overall_start = time.time()

    if not args.skip_synthetic:
        cmd = ["python3", "scripts/multi_seed_sweep.py",
               "--n_seeds", str(n_seeds_synthetic), "--alpha", str(args.alpha),
               "--device", args.device]
        results["synthetic_3condition"] = run_phase("Synthetic 3-condition experiment", cmd)

    if not args.skip_severity:
        cmd = ["python3", "scripts/severity_sweep.py",
               "--n_seeds", str(n_seeds_severity), "--alpha", str(args.alpha),
               "--device", args.device]
        results["severity_sweep"] = run_phase("Synthetic severity sweep", cmd)

    if not args.skip_real_data:
        cmd = ["python3", "scripts/real_data_experiment.py",
               "--dataset", "amazon", "--n_seeds", str(n_seeds_real),
               "--alpha", str(args.alpha), "--device", args.device]
        results["real_data_amazon"] = run_phase("Real data: Amazon", cmd)

        if not args.skip_yelp:
            cmd = ["python3", "scripts/real_data_experiment.py",
                   "--dataset", "yelp", "--n_seeds", str(n_seeds_real),
                   "--alpha", str(args.alpha), "--device", args.device]
            results["real_data_yelp"] = run_phase("Real data: Yelp (FIRST full-scale run -- watch closely)", cmd)

    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*70}")
    print("FULL EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for phase, success in results.items():
        print(f"  {phase}: {'OK' if success else 'FAILED -- check output above'}")
    print(f"\nTotal wall time: {overall_elapsed/60:.1f} minutes")
    print(f"\nRaw results saved in results/logs/ -- one CSV per phase.")

    if not all(results.values()):
        print("\nWARNING: one or more phases failed. Fix before treating results as final.")
        sys.exit(1)


if __name__ == "__main__":
    main()