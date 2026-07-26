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
from scipy import stats

from conformal_fdr import run_single_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=50)
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

    # ------------------------------------------------------------------
    # Statistical tests (this is what actually answers H1 vs H4, not eyeballing means)
    # ------------------------------------------------------------------
    print("\n=== Statistical tests ===")

    clean_subset = [r for r in all_results if r["condition"] == "clean"]
    contam_subset = [r for r in all_results if r["condition"] == "contaminated"]

    if clean_subset and contam_subset:
        clean_by_seed = {r["seed"]: r["realized_fdr"] for r in clean_subset}
        contam_by_seed = {r["seed"]: r["realized_fdr"] for r in contam_subset}
        common_seeds = sorted(set(clean_by_seed) & set(contam_by_seed))

        if len(common_seeds) >= 5:
            clean_paired = np.array([clean_by_seed[s] for s in common_seeds])
            contam_paired = np.array([contam_by_seed[s] for s in common_seeds])

            # Paired test: is contaminated FDR different from clean FDR, seed-for-seed?
            # Both conditions share the same underlying graph per seed (only the
            # calibration selection differs), so pairing by seed is valid and
            # more powerful than an unpaired comparison.
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(contam_paired, clean_paired)
                print(f"Paired Wilcoxon (contaminated vs. clean, n={len(common_seeds)} paired seeds): "
                      f"stat={wilcoxon_stat:.3f}, p={wilcoxon_p:.4f}")
            except ValueError as e:
                print(f"Paired Wilcoxon could not be computed: {e}")

            paired_ttest = stats.ttest_rel(contam_paired, clean_paired)
            print(f"Paired t-test (contaminated vs. clean): "
                  f"t={paired_ttest.statistic:.3f}, p={paired_ttest.pvalue:.4f}")

        # One-sided test: is mean realized FDR significantly ABOVE nominal alpha?
        # This is the literal H1 claim for each condition individually.
        for name, subset in [("clean", clean_subset), ("contaminated", contam_subset)]:
            fdrs = np.array([r["realized_fdr"] for r in subset])
            if len(fdrs) >= 5 and fdrs.std() > 0:
                t_stat, p_two_sided = stats.ttest_1samp(fdrs, args.alpha)
                # convert to one-sided p-value (testing mean > alpha)
                p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
                verdict = "SIGNIFICANTLY ABOVE nominal" if p_one_sided < 0.05 else "not significantly above nominal"
                print(f"One-sided test ({name} FDR > alpha={args.alpha}): "
                      f"mean={fdrs.mean():.3f}, t={t_stat:.3f}, p={p_one_sided:.4f} -> {verdict}")
            else:
                print(f"One-sided test ({name}): insufficient variance or sample size to test")

        print("\nReading guide: paired test p < 0.05 means the two conditions "
              "differ significantly for the SAME underlying graphs. One-sided "
              "test p < 0.05 for 'contaminated' (with clean NOT significant) "
              "is the strongest form of H1 support. Both non-significant "
              "supports H4 (fail-safe property survives on graphs).")
    else:
        print("Insufficient data in one or both conditions to run statistical tests.")


if __name__ == "__main__":
    main()