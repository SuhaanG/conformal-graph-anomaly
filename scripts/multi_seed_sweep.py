"""
multi_seed_sweep.py

Step 5: the actual H1-vs-H4 test. Runs run_single_trial() from
conformal_fdr.py across many seeds for three calibration conditions, then
compares the DISTRIBUTION of realized FDR to the nominal alpha level.

Conditions:
- "clean": calibration drawn only from normal nodes with zero anomalous
  neighbors (idealized baseline).
- "contaminated": calibration drawn uniformly at random from all normal
  nodes, including exposed ones (realistic average-case deployment).
- "adversarial": calibration deliberately drawn from the MOST exposed
  normal nodes available (worst case). This is the harder, more credible
  test of the fail-safe property -- random exposure averages out a lot of
  the contamination signal, but a paper claiming the guarantee survives
  contamination needs to survive the adversarial case, not just the
  average case.

Interpretation guide:
- If mean realized FDR for "contaminated" and/or "adversarial" is well
  above alpha, and "clean" stays at or below alpha -> H1 supported.
- If all three stay at or below alpha, including adversarial -> strong H4
  support: the fail-safe property survives even worst-case graph
  contamination.
- If "adversarial" breaks control but "contaminated" (random) does not,
  that is itself a meaningful, nuanced, and publishable finding: validity
  holds on average but not under systematic/worst-case contamination.

Usage (Colab, full run):
  python3 scripts/multi_seed_sweep.py --alpha 0.10 --device cuda
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

CONDITIONS = ["clean", "contaminated", "adversarial"]


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
    for condition in CONDITIONS:
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
    for condition in CONDITIONS:
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
          f"against nominal alpha={args.alpha}.")

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------
    print("\n=== Statistical tests ===")

    by_condition = {c: [r for r in all_results if r["condition"] == c] for c in CONDITIONS}

    # Pairwise comparisons (all share the same underlying graph per seed,
    # so pairing by seed is valid and more powerful than unpaired tests).
    pairs = [("contaminated", "clean"), ("adversarial", "clean"), ("adversarial", "contaminated")]
    for cond_a, cond_b in pairs:
        by_seed_a = {r["seed"]: r["realized_fdr"] for r in by_condition[cond_a]}
        by_seed_b = {r["seed"]: r["realized_fdr"] for r in by_condition[cond_b]}
        common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
        if len(common_seeds) < 5:
            print(f"{cond_a} vs {cond_b}: insufficient paired seeds ({len(common_seeds)})")
            continue
        a_paired = np.array([by_seed_a[s] for s in common_seeds])
        b_paired = np.array([by_seed_b[s] for s in common_seeds])
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(a_paired, b_paired)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
        paired_ttest = stats.ttest_rel(a_paired, b_paired)
        print(f"{cond_a} vs {cond_b} (n={len(common_seeds)} paired seeds): "
              f"Wilcoxon p={wilcoxon_p:.4f}, paired t-test t={paired_ttest.statistic:.3f} p={paired_ttest.pvalue:.4f}")

    # One-sided test: is mean realized FDR significantly ABOVE nominal alpha?
    # This is the literal H1 claim for each condition individually, and the
    # adversarial condition is the one that actually matters most here.
    for name in CONDITIONS:
        subset = by_condition[name]
        fdrs = np.array([r["realized_fdr"] for r in subset])
        if len(fdrs) >= 5 and fdrs.std() > 0:
            t_stat, p_two_sided = stats.ttest_1samp(fdrs, args.alpha)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
            verdict = "SIGNIFICANTLY ABOVE nominal" if p_one_sided < 0.05 else "not significantly above nominal"
            print(f"One-sided test ({name} FDR > alpha={args.alpha}): "
                  f"mean={fdrs.mean():.3f}, t={t_stat:.3f}, p={p_one_sided:.4f} -> {verdict}")
        else:
            print(f"One-sided test ({name}): insufficient variance or sample size to test")

    print("\nReading guide: 'adversarial SIGNIFICANTLY ABOVE nominal' with "
          "'clean' and 'contaminated' not significant is the key finding to "
          "watch for -- it would mean validity survives average-case "
          "contamination but breaks under worst-case/systematic contamination, "
          "which is a genuine, nuanced, publishable result either way.")


if __name__ == "__main__":
    main()