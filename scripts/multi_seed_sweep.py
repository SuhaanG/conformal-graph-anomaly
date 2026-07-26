"""
multi_seed_sweep.py

Step 5: the actual H1-vs-H4 test. Runs run_single_trial() from
conformal_fdr.py across many seeds for both the "clean" and "contaminated"
calibration conditions, then compares the DISTRIBUTION of realized FDR to
the nominal alpha level.

This is the first script in the project that needs real compute (n_seeds x
2 conditions x 100 epochs of GNN training each) and is designed to run on
Google Colab with an A100. It will also run on a local M2 CPU for a quick
smaller-scale check (use --n_seeds 3 for that).

Interpretation guide (this is the actual research question):
- If mean realized FDR for "contaminated" is well above alpha, and "clean"
  stays at or below alpha -> H1 supported: clustered, propagating
  contamination breaks finite-sample FDR control on graphs.
- If both stay at or below alpha -> H4 supported: the fail-safe property
  survives graph structure; the paper becomes a certification result instead.
- If "clean" is ALSO consistently above alpha, or has zero discoveries in a
  way that looks like a distribution-mismatch artifact rather than validity
  -> the experimental design itself needs revision before either conclusion
  is trustworthy (see the note in conformal_fdr.py's smoke test about the
  clean condition's small calibration pool).

Usage (local quick check):
  python3 scripts/multi_seed_sweep.py --n_seeds 5 --alpha 0.10

Usage (Colab, full run):
  python3 scripts/multi_seed_sweep.py --n_seeds 30 --alpha 0.10 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import numpy as np
import torch

from conformal_fdr import run_single_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None,
                         help="cpu, cuda, or mps. Auto-detected if not set.")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}\n")

    all_results = []
    for condition in ["clean", "contaminated"]:
        print(f"=== Running {args.n_seeds} seeds for condition: {condition} ===")
        for seed in range(args.n_seeds):
            result = run_single_trial(
                condition, alpha=args.alpha, seed=seed,
                n_epochs=args.n_epochs, device=device,
            )
            if result is None:
                print(f"  seed {seed}: skipped (insufficient clean calibration pool)")
                continue
            all_results.append(result)
            print(f"  seed {seed}: n_discoveries={result['n_discoveries']:3d} "
                  f"realized_fdr={result['realized_fdr']:.3f} power={result['power']:.3f}")

    # Save raw results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "multi_seed_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to {csv_path}")

    # Summarize
    print("\n=== Summary (mean +/- std across seeds) ===")
    for condition in ["clean", "contaminated"]:
        subset = [r for r in all_results if r["condition"] == condition]
        if not subset:
            print(f"{condition}: no valid trials")
            continue
        fdrs = np.array([r["realized_fdr"] for r in subset])
        powers = np.array([r["power"] for r in subset])
        n_zero_discovery = sum(1 for r in subset if r["n_discoveries"] == 0)
        print(f"{condition}: "
              f"realized_fdr = {fdrs.mean():.3f} +/- {fdrs.std():.3f} "
              f"(nominal alpha = {args.alpha}), "
              f"power = {powers.mean():.3f} +/- {powers.std():.3f}, "
              f"zero-discovery trials = {n_zero_discovery}/{len(subset)}")

    print("\nInterpretation reminder: compare mean realized_fdr per condition "
          f"against nominal alpha={args.alpha}. See module docstring for what "
          "each outcome pattern means for H1 vs H4.")


if __name__ == "__main__":
    main()