"""
pvalue_distribution_analysis.py

BACK TO THE DRAWING BOARD, corrected version.

The previous attempt (clearance_rate_verification.py) checked only
whether anomaly scores beat the ABSOLUTE calibration ceiling (rank-1 p-value
floor). That was too crude: BH can reject at ANY rank i where p_(i) <=
alpha*i/m, not just at the strictest floor. The previous experiment's
own data showed clean counterexamples (e.g. p_an=0.01, seed 16: clearance
rate far below the predicted threshold, yet 209 discoveries -- one of the
largest in the whole sweep), disproving the oversimplified single-point
formula.

This experiment instead logs the FULL distribution of anomaly p-values
at several fixed, BH-relevant thresholds (not just the absolute floor),
alongside the actual observed discovery count and power, for every trial
across the severity sweep. This gives the raw material to find the
ACTUAL relationship empirically -- via regression/correlation across
these threshold-crossing counts vs. real outcomes -- rather than
asserting an unverified closed form a second time.

Specifically, for each trial we log:
  - n_anomalies_below_p{0.001,0.005,0.01,0.02,0.05}: how many true
    anomalies have a conformal p-value at or below each fixed level.
    These are the natural candidate predictors, since BH's rejection
    threshold at rank i is alpha*i/m -- for m~13000 and alpha=0.10,
    these fixed levels correspond to roughly rank i = level*m/alpha,
    i.e. p<=0.001 -> i~130, p<=0.01 -> i~1300, etc. -- spanning the
    range of plausible BH crossing points instead of only the extreme
    floor.
  - actual n_discoveries and power, for direct comparison.

Run on Colab:
  !python3 scripts/pvalue_distribution_analysis.py --n_seeds 20 --alpha 0.10 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import numpy as np

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detector import train_dominant
from conformal_fdr import conformal_p_values, benjamini_hochberg

# Fixed p-value thresholds to check anomaly counts against, spanning a
# wide range of plausible BH rejection ranks rather than only the
# strictest floor value.
PVALUE_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def run_trial_with_pvalue_distribution(p_an, alpha, seed, n_epochs, device, n_calib=2794):
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=p_an, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    scores, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed,
                                verbose=False, device=device)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)

    if len(normal_idx) < n_calib:
        return None

    order_by_exposure = np.argsort(-exposure)
    top_exposed = normal_idx[order_by_exposure]
    calib_idx = top_exposed[:n_calib]

    remaining_normal = np.setdiff1d(normal_idx, calib_idx)
    test_idx = np.concatenate([remaining_normal, anomaly_idx])
    test_labels = np.concatenate([
        np.zeros(len(remaining_normal), dtype=int),
        np.ones(len(anomaly_idx), dtype=int),
    ])

    calib_scores = scores[calib_idx]
    test_scores = scores[test_idx]

    p_values = conformal_p_values(calib_scores, test_scores)
    m = len(test_scores)

    # THE KEY NEW MEASUREMENT: for each fixed threshold, count how many
    # TRUE anomalies have a p-value at or below it -- the full picture,
    # not just the single strictest floor value.
    anomaly_mask = (test_labels == 1)
    anomaly_pvalues = p_values[anomaly_mask]

    threshold_counts = {}
    for thresh in PVALUE_THRESHOLDS:
        threshold_counts[f"n_anomalies_below_p{thresh}"] = int(np.sum(anomaly_pvalues <= thresh))

    # also log the median and 10th-percentile anomaly p-value directly,
    # since these summarize the whole distribution more richly than any
    # single threshold count
    median_anomaly_pvalue = float(np.median(anomaly_pvalues)) if len(anomaly_pvalues) > 0 else float("nan")
    p10_anomaly_pvalue = float(np.percentile(anomaly_pvalues, 10)) if len(anomaly_pvalues) > 0 else float("nan")

    discoveries = benjamini_hochberg(p_values, alpha)
    n_discoveries = discoveries.sum()
    realized_fdr = (np.sum(discoveries & (test_labels == 0)) / n_discoveries) if n_discoveries > 0 else 0.0
    power = (np.sum(discoveries & (test_labels == 1)) / len(anomaly_idx)) if len(anomaly_idx) > 0 else 0.0

    result = {
        "p_an": p_an, "seed": seed, "m_test": m,
        "median_anomaly_pvalue": median_anomaly_pvalue,
        "p10_anomaly_pvalue": p10_anomaly_pvalue,
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }
    result.update(threshold_counts)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    p_an_values = [0.002, 0.005, 0.01, 0.02, 0.05]
    all_results = []

    for p_an in p_an_values:
        print(f"=== p_an={p_an} ===")
        for seed in range(args.n_seeds):
            r = run_trial_with_pvalue_distribution(p_an, args.alpha, seed, args.n_epochs, device)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            counts_str = " ".join(f"p<={t}:{r[f'n_anomalies_below_p{t}']:3d}" for t in PVALUE_THRESHOLDS)
            print(f"  seed {seed}: {counts_str} | n_disc={r['n_discoveries']:3d} power={r['power']:.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pvalue_distribution_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to {csv_path}")

    print("\n=== Correlation of each threshold-count predictor against actual n_discoveries ===")
    from scipy import stats as scipy_stats
    n_disc_all = np.array([r["n_discoveries"] for r in all_results])
    for thresh in PVALUE_THRESHOLDS:
        key = f"n_anomalies_below_p{thresh}"
        vals = np.array([r[key] for r in all_results])
        if vals.std() > 0 and n_disc_all.std() > 0:
            corr, pval = scipy_stats.pearsonr(vals, n_disc_all)
            print(f"  {key}: r={corr:.4f} (p={pval:.4g})")

    print("\n=== Summary by severity level ===")
    for p_an in p_an_values:
        subset = [r for r in all_results if r["p_an"] == p_an]
        if not subset:
            continue
        mean_disc = np.mean([r["n_discoveries"] for r in subset])
        mean_power = np.mean([r["power"] for r in subset])
        mean_median_p = np.nanmean([r["median_anomaly_pvalue"] for r in subset])
        print(f"p_an={p_an}: mean_n_discoveries={mean_disc:.1f}, mean_power={mean_power:.4f}, "
              f"mean_median_anomaly_pvalue={mean_median_p:.4f}")
        for thresh in PVALUE_THRESHOLDS:
            key = f"n_anomalies_below_p{thresh}"
            mean_count = np.mean([r[key] for r in subset])
            print(f"    mean {key} = {mean_count:.1f}")

    print("\nInterpretation: find which threshold-count predictor correlates most strongly "
          "with actual n_discoveries (highest r above). That threshold level -- not the "
          "absolute floor -- is the empirically correct quantity to build a corrected, "
          "honest theoretical prediction around, rather than asserting a closed form that "
          "the previous experiment's own data contradicted.")


if __name__ == "__main__":
    main()