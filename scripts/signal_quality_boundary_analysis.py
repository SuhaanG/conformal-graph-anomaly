"""
signal_quality_boundary_analysis.py

Unifies three results that currently look like separate, disconnected
findings into one explicit, quantified claim: realized discovery power
under the conformal+BH pipeline collapses as underlying detector signal
quality degrades, REGARDLESS of what causes that degradation (contamination
severity, or a poorly-matched real dataset) -- and FDR stays controlled
throughout, including in the collapsed-power regime.

The three results being unified:
1. Severity sweep (synthetic): as p_an increases 0.002 -> 0.05 (25x),
   discovery power drops to zero by the highest severity level -- already
   run, results in results/logs/severity_sweep.csv.
2. Amazon (real, strong detector signal, AUROC~0.89): substantial,
   controlled discovery activity across seeds.
3. Reddit (real, weak detector signal, AUROC~0.58) and Tolokers (real,
   below-chance detector signal, AUROC~0.41): near-total or total
   discovery silence.

This script does NOT run any new experiments -- it reads existing CSV
results and independently-recorded AUROC values, and produces the
explicit correlation analysis a reviewer would want to see: does realized
power/discovery-rate scale with underlying signal quality (measured via
AUROC where available, or contamination severity as a proxy where AUROC
wasn't separately measured)? If yes, this becomes the paper's stated
boundary condition: "the certification result holds conditional on
non-trivial detector signal quality" -- turning three ad hoc dataset
stories into one precise, defensible scope statement.

Usage: python3 scripts/signal_quality_boundary_analysis.py
(reads results/logs/*.csv, produces a summary table and correlation stat)
"""

import os
import csv
import numpy as np
from scipy import stats


# Manually recorded AUROC values from diagnostic checks run during this
# project (not derivable from the FDR-experiment CSVs alone, since AUROC
# was checked separately before committing to full runs on each dataset).
KNOWN_AUROC = {
    "amazon": 0.8925,       # post feature-standardization fix, from Step 3/7 diagnostics
    "reddit": 0.5773,       # raw score AUROC (degree-norm found to HURT here, so raw is used)
    "tolokers": 0.4093,     # raw score AUROC (below chance -- excluded from main claim)
}


def load_severity_sweep_summary(csv_path):
    """Reads severity_sweep.csv and computes mean power + zero-discovery
    rate per severity level, to serve as the synthetic-data half of the
    signal-quality-vs-power relationship (using p_an as the signal-
    degradation axis, since AUROC wasn't separately measured per severity
    level in that experiment)."""
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found. Run severity_sweep.py first, "
              f"or point this script at the correct results path.")
        return {}

    by_severity = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_an = float(row["p_an"])
            power = float(row["power"])
            n_disc = int(row["n_discoveries"])
            by_severity.setdefault(p_an, {"powers": [], "zero_count": 0, "total": 0})
            by_severity[p_an]["powers"].append(power)
            by_severity[p_an]["total"] += 1
            if n_disc == 0:
                by_severity[p_an]["zero_count"] += 1

    summary = {}
    for p_an, d in by_severity.items():
        summary[p_an] = {
            "mean_power": np.mean(d["powers"]),
            "zero_discovery_rate": d["zero_count"] / d["total"],
        }
    return summary


def load_real_data_summary(csv_path, dataset_name):
    """Reads a real_data_experiment_*.csv and computes mean power +
    zero-discovery rate for the 'contaminated' condition (the standard,
    average-case comparison point across datasets)."""
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found for {dataset_name}.")
        return None

    powers = []
    zero_count = 0
    total = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["condition"] != "contaminated":
                continue
            powers.append(float(row["power"]))
            total += 1
            if int(row["n_discoveries"]) == 0:
                zero_count += 1

    if total == 0:
        return None
    return {
        "mean_power": np.mean(powers),
        "zero_discovery_rate": zero_count / total,
        "auroc": KNOWN_AUROC.get(dataset_name),
    }


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")

    print("=" * 70)
    print("PART 1: Synthetic severity sweep (signal degradation via contamination)")
    print("=" * 70)
    severity_summary = load_severity_sweep_summary(
        os.path.join(results_dir, "severity_sweep.csv"))
    for p_an in sorted(severity_summary.keys()):
        d = severity_summary[p_an]
        print(f"  p_an={p_an}: mean_power={d['mean_power']:.4f}, "
              f"zero_discovery_rate={d['zero_discovery_rate']:.2f}")

    print()
    print("=" * 70)
    print("PART 2: Real datasets (signal quality varies by dataset/task match)")
    print("=" * 70)
    real_results = {}
    for dataset in ["amazon", "reddit", "tolokers"]:
        csv_path = os.path.join(results_dir, f"real_data_experiment_{dataset}.csv")
        summary = load_real_data_summary(csv_path, dataset)
        if summary:
            real_results[dataset] = summary
            print(f"  {dataset}: AUROC={summary['auroc']}, "
                  f"mean_power={summary['mean_power']:.4f}, "
                  f"zero_discovery_rate={summary['zero_discovery_rate']:.2f}")
        else:
            print(f"  {dataset}: no results file found -- run real_data_experiment.py first")

    print()
    print("=" * 70)
    print("PART 3: Correlation -- does power scale with signal quality?")
    print("=" * 70)
    if len(real_results) >= 3:
        aurocs = [real_results[d]["auroc"] for d in real_results]
        powers = [real_results[d]["mean_power"] for d in real_results]
        if len(set(aurocs)) > 1:  # need variance to correlate
            corr, p_value = stats.pearsonr(aurocs, powers)
            print(f"  Pearson correlation (AUROC vs. mean power, n={len(aurocs)} datasets): "
                  f"r={corr:.3f}, p={p_value:.4f}")
            print(f"  (n=3 is too small for a reliable p-value -- report the pattern "
                  f"descriptively in the paper, not as a formal significance claim.)")
        print()
        print("  Datasets ranked by AUROC:")
        for d in sorted(real_results, key=lambda x: -real_results[x]["auroc"]):
            print(f"    {d}: AUROC={real_results[d]['auroc']:.3f} -> "
                  f"power={real_results[d]['mean_power']:.4f}, "
                  f"zero-rate={real_results[d]['zero_discovery_rate']:.2f}")
    else:
        print("  Need results from all three real datasets to compute this. "
              "Missing:", [d for d in ["amazon", "reddit", "tolokers"] if d not in real_results])

    print()
    print("=" * 70)
    print("SUGGESTED PAPER FRAMING (fill in with actual numbers from above)")
    print("=" * 70)
    print("""
  'The certification result (Section X) holds conditional on the underlying
  detector achieving non-trivial discriminative power. We characterize this
  boundary two ways: (1) synthetically, by escalating contamination severity
  until detector signal degrades, observing that discovery power falls to
  zero while FDR remains controlled (graceful degradation into silence, not
  false discovery); and (2) across real datasets spanning a range of native
  detector AUROC (Amazon: 0.89, Reddit: 0.58, Tolokers: 0.41), observing
  that discovery activity tracks signal quality directly -- strong on
  Amazon, minimal on Reddit, effectively absent on Tolokers -- while FDR
  control holds in every case. We do not claim the certification result is
  unconditional; we claim it holds precisely when the detector carries
  genuine signal, and characterize what happens when it does not.'
""")


if __name__ == "__main__":
    main()