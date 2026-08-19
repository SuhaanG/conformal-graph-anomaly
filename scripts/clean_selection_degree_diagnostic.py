"""
clean_selection_degree_diagnostic.py

Tests a specific, falsifiable hypothesis raised by condition_comparison_pygod.py's
result: realized FDR was SIGNIFICANTLY ABOVE nominal in the CLEAN condition
(mean 0.132, d=0.837, p=0.0007) -- the one condition Proposition 1 proves has
exact exchangeability -- while contaminated and adversarial were both safely
at or below nominal. That is the reverse of what contamination-focused
concerns would predict, and it means the exchangeability violation, if real,
has nothing to do with contamination severity or adversarial selection.

THE HYPOTHESIS. "Clean" selects calibration nodes by a strict topological
filter: zero anomalous neighbors. That is not a random draw from the normal
population -- it could systematically differ in DEGREE from the broader
normal population that ends up in the test set, simply because exposure is a
ratio over neighbor count (a normal node needs at least one anomalous
neighbor among however many it has to be excluded from "clean", so very
low-degree normal nodes are structurally easier to keep exposure-zero by
chance). If dominant_pygod's scores correlate with degree AT ALL on this
generator, and clean-selection shifts the degree distribution of calibration
relative to test, that combination breaks exchangeability regardless of
contamination -- a confound, not a finding about contamination robustness.

This is EXACTLY the failure mode real_data_experiment.py's degree_normalize_
scores() was built to correct for on real data (see degree_norm_diagnostic.py)
-- but neither the original synthetic pipeline (conformal_fdr.run_single_trial)
nor condition_comparison_pygod.py applies any degree correction. If this
hypothesis holds, that is the likely fix; if it does not, something else is
going on and degree correction is not the answer.

TWO INDEPENDENT CHECKS, both need to hold for the hypothesis to explain the
result:
  1. DEGREE CONFOUND CHECK: does the clean-selected calibration set have a
     different degree distribution than the test set's normal population?
     (Kolmogorov-Smirnov two-sample test + means/medians.)
  2. SCORE-DEGREE CORRELATION CHECK: does dominant_pygod's raw score actually
     correlate with degree among normal nodes on this synthetic generator?
     (Spearman correlation, since the relationship need not be linear.)

If check 1 is null (no degree difference) or check 2 is null (score doesn't
track degree), the degree-confound hypothesis is NOT supported and the clean-
condition FDR inflation needs a different explanation -- report that plainly,
don't force the data to fit the hypothesis.

Run on Colab:
  !python scripts/clean_selection_degree_diagnostic.py --n_seeds 10 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import numpy as np
from scipy import stats

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detectors import score_nodes


def run_trial(seed, n_epochs, device, calib_frac=0.9):
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=0.002, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    scores = score_nodes("dominant_pygod", graph, features, labels=labels,
                          seed=seed, n_epochs=n_epochs, device=device)

    normal_idx = np.where(labels == 0)[0]

    degree = np.array([graph.degree(i) for i in normal_idx], dtype=float)

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)

    clean_mask = exposure == 0
    n_calib = int(round(calib_frac * clean_mask.sum()))
    if clean_mask.sum() < 20:
        return None

    rng = np.random.default_rng(seed)
    clean_normal_idx = normal_idx[clean_mask]
    calib_idx = rng.choice(clean_normal_idx, size=n_calib, replace=False)
    calib_mask = np.isin(normal_idx, calib_idx)

    calib_degree = degree[calib_mask]
    test_normal_degree = degree[~calib_mask]  # the "remaining normal" that ends up in the test set

    # CHECK 1: degree confound. Two-sample KS test, calib vs test-normal degree.
    ks_stat, ks_p = stats.ks_2samp(calib_degree, test_normal_degree)

    # CHECK 2: score-degree correlation among ALL normal nodes (not just calib/test
    # split), since this checks a property of the detector, not of the split.
    normal_scores = scores[normal_idx]
    spearman_r, spearman_p = stats.spearmanr(normal_scores, degree)

    return {
        "seed": seed,
        "n_calib": n_calib,
        "n_test_normal": len(test_normal_degree),
        "calib_degree_mean": calib_degree.mean(),
        "calib_degree_median": float(np.median(calib_degree)),
        "test_normal_degree_mean": test_normal_degree.mean(),
        "test_normal_degree_median": float(np.median(test_normal_degree)),
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "score_degree_spearman_r": spearman_r,
        "score_degree_spearman_p": spearman_p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Detector: dominant_pygod\n")
    print("Testing the degree-confound hypothesis for condition_comparison_pygod.py's")
    print("clean-condition FDR inflation (mean 0.132, d=0.837, p=0.0007).\n")

    results = []
    for seed in range(args.n_seeds):
        r = run_trial(seed, args.n_epochs, device)
        if r is None:
            print(f"  seed {seed}: skipped (insufficient clean calibration pool)")
            continue
        results.append(r)
        print(f"  seed {seed}: calib_degree_mean={r['calib_degree_mean']:.2f} "
              f"test_normal_degree_mean={r['test_normal_degree_mean']:.2f} "
              f"KS_p={r['ks_p']:.4f} "
              f"score~degree_spearman_r={r['score_degree_spearman_r']:.4f} "
              f"(p={r['score_degree_spearman_p']:.2e})")

    if not results:
        print("\nNo valid trials -- nothing to report.")
        return

    print("\n=== CHECK 1: Degree confound (calib vs test-normal population) ===")
    calib_means = np.array([r["calib_degree_mean"] for r in results])
    test_means = np.array([r["test_normal_degree_mean"] for r in results])
    ks_ps = np.array([r["ks_p"] for r in results])
    n_significant_ks = int(np.sum(ks_ps < 0.05))
    print(f"Mean calib degree across seeds: {calib_means.mean():.2f} +/- {calib_means.std():.2f}")
    print(f"Mean test-normal degree across seeds: {test_means.mean():.2f} +/- {test_means.std():.2f}")
    print(f"KS test significant (p<0.05) in {n_significant_ks}/{len(results)} seeds")
    paired_t = stats.ttest_rel(calib_means, test_means)
    print(f"Paired t-test, calib vs test-normal mean degree across seeds: "
          f"t={paired_t.statistic:.3f}, p={paired_t.pvalue:.4f}")
    if paired_t.pvalue < 0.05:
        direction = "LOWER" if calib_means.mean() < test_means.mean() else "HIGHER"
        print(f"  -> Clean-selected calibration nodes have SIGNIFICANTLY {direction} "
              f"degree than the test-set normal population. CHECK 1 SUPPORTS the "
              f"degree-confound hypothesis.")
    else:
        print(f"  -> No significant degree difference between calibration and test-normal "
              f"populations. CHECK 1 DOES NOT SUPPORT the degree-confound hypothesis.")

    print("\n=== CHECK 2: Does dominant_pygod's score correlate with degree? ===")
    spearman_rs = np.array([r["score_degree_spearman_r"] for r in results])
    spearman_ps = np.array([r["score_degree_spearman_p"] for r in results])
    n_significant_corr = int(np.sum(spearman_ps < 0.05))
    print(f"Mean Spearman r (score vs degree, normal nodes only): "
          f"{spearman_rs.mean():.4f} +/- {spearman_rs.std():.4f}")
    print(f"Significant correlation (p<0.05) in {n_significant_corr}/{len(results)} seeds")
    if n_significant_corr >= len(results) // 2 and abs(spearman_rs.mean()) > 0.1:
        print(f"  -> dominant_pygod's score meaningfully correlates with degree on this "
              f"generator. CHECK 2 SUPPORTS the degree-confound hypothesis.")
    else:
        print(f"  -> No meaningful score-degree correlation. CHECK 2 DOES NOT SUPPORT "
              f"the degree-confound hypothesis.")

    print("\n=== Verdict ===")
    check1 = paired_t.pvalue < 0.05
    check2 = (n_significant_corr >= len(results) // 2) and (abs(spearman_rs.mean()) > 0.1)
    if check1 and check2:
        print("BOTH checks support the hypothesis: clean-selection biases degree, AND the")
        print("detector's score tracks degree. The clean-condition FDR inflation is likely")
        print("a degree confound, not a genuine exchangeability violation from contamination.")
        print("Next step: test whether applying degree_normalize_scores() (already built for")
        print("real data) removes the inflation on synthetic data too.")
    elif check1 and not check2:
        print("CHECK 1 holds (degree differs) but CHECK 2 does not (score doesn't track")
        print("degree meaningfully). The degree confound exists structurally but doesn't")
        print("appear to explain the FDR inflation through THIS detector's scores. Do not")
        print("apply degree normalization on the strength of this result alone.")
    elif check2 and not check1:
        print("CHECK 2 holds (score tracks degree) but CHECK 1 does not (no degree")
        print("difference between calib and test-normal). The detector is degree-sensitive")
        print("but clean-selection doesn't create a degree confound on this generator --")
        print("the FDR inflation needs a different explanation.")
    else:
        print("NEITHER check supports the degree-confound hypothesis. Report this plainly:")
        print("the clean-condition FDR inflation is not explained by a degree confound and")
        print("needs a different investigation before it goes anywhere near the paper.")


if __name__ == "__main__":
    main()