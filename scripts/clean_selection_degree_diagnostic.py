"""
clean_selection_degree_diagnostic.py

Tests structural-covariate hypotheses for condition_comparison_pygod.py's
clean-condition FDR inflation (mean 0.132, d=0.837, p=0.0007, vs
contaminated/adversarial both safely at or below nominal -- see
theory/joint_discovery_threshold_proposition.md Part 3 for the full record).

DEGREE (checks 1-2, original hypothesis): "Clean" selects calibration nodes
by a strict topological filter, zero anomalous neighbors, which is not a
random draw and could systematically differ in degree from the broader
normal population. CONFIRMED on a prior 10-seed run: clean-selected nodes
have significantly lower degree (paired t=-24.959, p<0.0001) and
dominant_pygod's score correlates with degree (Spearman r=0.56).
condition_comparison_pygod.py's degree_matched_calib_sample() fix, built on
this finding, closed roughly HALF the FDR gap (d: 0.837 -> 0.449) without
touching scores -- real, but partial. The matching itself was imperfect in
18/20 seeds (candidate pool structurally short on high-degree members), and
a direct per-seed check found no significant correlation between match
quality and residual FDR (Spearman r=-0.161, p=0.498). Degree is a real,
partial contributor, not the whole story.

CLUSTERING COEFFICIENT (checks 3-4, new): tests whether a SECOND structural
covariate explains some of the remaining gap. Local clustering coefficient
is a natural second candidate specifically because the underlying generator
is a stochastic block model with three anomaly clusters (GraphGenConfig
n_anomaly_clusters=3) -- a normal node's local clustering structure could
differ systematically depending on its position relative to these clusters,
independent of raw degree, and reconstruction-based detectors are known to
be sensitive to local neighborhood structure (that is the entire premise of
DOMINANT's structure decoder).

FOUR CHECKS. Degree and clustering are tested independently and reported
separately -- do not average them into one combined verdict, since a real
finding for one and a null finding for the other is itself informative and
must not be blurred together:
  1. DEGREE CONFOUND (repeat of the original check, for a fresh comparison
     seed range -- results should replicate the prior 10-seed run closely;
     if they don't, that is itself worth flagging, not silently accepted).
  2. SCORE-DEGREE CORRELATION (repeat, same reasoning).
  3. CLUSTERING CONFOUND: does clean-selected calibration have a different
     local clustering coefficient distribution than the test-normal
     population? (KS test.)
  4. SCORE-CLUSTERING CORRELATION: does dominant_pygod's score correlate
     with local clustering coefficient among normal nodes? (Spearman.)

If checks 3-4 are both null, clustering does not explain the residual gap
and that should be reported as a genuine negative result, not treated as a
failed attempt to hide. If they hold, that is a second real, partial
contributor, analogous in status to degree -- likely still not the
complete explanation, and should be reported with the same caveats degree
was given (see Part 3 of the theory doc).

Run on Colab:
  !python scripts/clean_selection_degree_diagnostic.py --n_seeds 10 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import numpy as np
import networkx as nx
from scipy import stats

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detectors import score_nodes, available_detectors


def run_trial(seed, n_epochs, device, calib_frac=0.9,
              detector="dominant_pygod"):
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=0.002, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    scores = score_nodes(detector, graph, features, labels=labels,
                          seed=seed, n_epochs=n_epochs, device=device)

    normal_idx = np.where(labels == 0)[0]

    degree = np.array([graph.degree(i) for i in normal_idx], dtype=float)

    # nx.clustering computes local clustering coefficient for every node in
    # one pass (more efficient than calling it per-node in a loop); index
    # into the result dict in the same order as normal_idx.
    clustering_all = nx.clustering(graph)
    clustering = np.array([clustering_all[i] for i in normal_idx], dtype=float)

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
    test_normal_degree = degree[~calib_mask]
    calib_clustering = clustering[calib_mask]
    test_normal_clustering = clustering[~calib_mask]

    # CHECK 1: degree confound (repeat of prior run)
    ks_stat_deg, ks_p_deg = stats.ks_2samp(calib_degree, test_normal_degree)

    # CHECK 2: score-degree correlation (repeat)
    normal_scores = scores[normal_idx]
    spearman_r_deg, spearman_p_deg = stats.spearmanr(normal_scores, degree)

    # CHECK 3: clustering confound (new)
    ks_stat_clus, ks_p_clus = stats.ks_2samp(calib_clustering, test_normal_clustering)

    # CHECK 4: score-clustering correlation (new)
    spearman_r_clus, spearman_p_clus = stats.spearmanr(normal_scores, clustering)

    return {
        "seed": seed,
        "n_calib": n_calib,
        "n_test_normal": len(test_normal_degree),
        "calib_degree_mean": calib_degree.mean(),
        "test_normal_degree_mean": test_normal_degree.mean(),
        "ks_p_degree": ks_p_deg,
        "score_degree_spearman_r": spearman_r_deg,
        "score_degree_spearman_p": spearman_p_deg,
        "calib_clustering_mean": calib_clustering.mean(),
        "test_normal_clustering_mean": test_normal_clustering.mean(),
        "ks_p_clustering": ks_p_clus,
        "score_clustering_spearman_r": spearman_r_clus,
        "score_clustering_spearman_p": spearman_p_clus,
    }


def report_check(name, calib_means, test_means, ks_ps, n_trials):
    print(f"\n=== {name} confound (calib vs test-normal population) ===")
    n_significant_ks = int(np.sum(ks_ps < 0.05))
    print(f"Mean calib {name.lower()} across seeds: {calib_means.mean():.4f} +/- {calib_means.std():.4f}")
    print(f"Mean test-normal {name.lower()} across seeds: {test_means.mean():.4f} +/- {test_means.std():.4f}")
    print(f"KS test significant (p<0.05) in {n_significant_ks}/{n_trials} seeds")
    paired_t = stats.ttest_rel(calib_means, test_means)
    print(f"Paired t-test, calib vs test-normal mean {name.lower()} across seeds: "
          f"t={paired_t.statistic:.3f}, p={paired_t.pvalue:.4f}")
    holds = paired_t.pvalue < 0.05
    if holds:
        direction = "LOWER" if calib_means.mean() < test_means.mean() else "HIGHER"
        print(f"  -> Clean-selected calibration nodes have SIGNIFICANTLY {direction} "
              f"{name.lower()} than the test-set normal population.")
    else:
        print(f"  -> No significant {name.lower()} difference between calibration and "
              f"test-normal populations.")
    return holds


def report_correlation(name, spearman_rs, spearman_ps, n_trials):
    print(f"\n=== Does dominant_pygod's score correlate with {name.lower()}? ===")
    n_significant_corr = int(np.sum(spearman_ps < 0.05))
    print(f"Mean Spearman r (score vs {name.lower()}, normal nodes only): "
          f"{spearman_rs.mean():.4f} +/- {spearman_rs.std():.4f}")
    print(f"Significant correlation (p<0.05) in {n_significant_corr}/{n_trials} seeds")
    holds = (n_significant_corr >= n_trials // 2) and (abs(spearman_rs.mean()) > 0.1)
    if holds:
        print(f"  -> dominant_pygod's score meaningfully correlates with {name.lower()} "
              f"on this generator.")
    else:
        print(f"  -> No meaningful score-{name.lower()} correlation.")
    return holds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--detector", type=str, default="dominant_pygod",
                        choices=available_detectors(),
                        help="Which detector produces the scores. The degree "
                             "findings recorded in this file's docstring "
                             "(t=-24.959, Spearman r=0.56) are dominant_pygod "
                             "numbers; other detectors are expected to differ, "
                             "and how much they differ is exactly Part 4 "
                             "prediction 2.")
    args = parser.parse_args()

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Detector: dominant_pygod\n")
    print("Testing structural-covariate hypotheses for the clean-condition FDR")
    print("inflation. Degree checks repeat the prior finding for a fresh seed range;")
    print("clustering checks are new, testing a second candidate covariate.\n")

    results = []
    for seed in range(args.n_seeds):
        r = run_trial(seed, args.n_epochs, device, detector=args.detector)
        if r is None:
            print(f"  seed {seed}: skipped (insufficient clean calibration pool)")
            continue
        results.append(r)
        print(f"  seed {seed}: "
              f"deg[calib={r['calib_degree_mean']:.2f} test={r['test_normal_degree_mean']:.2f} "
              f"ks_p={r['ks_p_degree']:.4f} score_r={r['score_degree_spearman_r']:.3f}] "
              f"clus[calib={r['calib_clustering_mean']:.4f} test={r['test_normal_clustering_mean']:.4f} "
              f"ks_p={r['ks_p_clustering']:.4f} score_r={r['score_clustering_spearman_r']:.3f}]")

    if not results:
        print("\nNo valid trials -- nothing to report.")
        return

    n = len(results)
    check1 = report_check(
        "Degree",
        np.array([r["calib_degree_mean"] for r in results]),
        np.array([r["test_normal_degree_mean"] for r in results]),
        np.array([r["ks_p_degree"] for r in results]),
        n,
    )
    check2 = report_correlation(
        "degree",
        np.array([r["score_degree_spearman_r"] for r in results]),
        np.array([r["score_degree_spearman_p"] for r in results]),
        n,
    )
    check3 = report_check(
        "Clustering coefficient",
        np.array([r["calib_clustering_mean"] for r in results]),
        np.array([r["test_normal_clustering_mean"] for r in results]),
        np.array([r["ks_p_clustering"] for r in results]),
        n,
    )
    check4 = report_correlation(
        "clustering coefficient",
        np.array([r["score_clustering_spearman_r"] for r in results]),
        np.array([r["score_clustering_spearman_p"] for r in results]),
        n,
    )

    print("\n=== Verdict ===")
    print(f"Degree confound: check1={check1}, check2={check2}")
    if check1 and check2:
        print("  Degree confound REPLICATED (consistent with the prior 10-seed run and")
        print("  the partial fix already found in condition_comparison_pygod.py).")
    else:
        print("  Degree confound DID NOT REPLICATE as found previously -- investigate before")
        print("  trusting either this run or the prior one; do not silently prefer one.")

    print(f"\nClustering confound: check3={check3}, check4={check4}")
    if check3 and check4:
        print("  BOTH clustering checks hold. Clustering coefficient is a SECOND real")
        print("  candidate contributor to the residual FDR inflation degree-matching left")
        print("  unexplained. Next step: test a clustering-matched (or joint degree+")
        print("  clustering-matched) calibration sample, analogous to")
        print("  degree_matched_calib_sample(), and check whether it closes more of the")
        print("  remaining gap (target: d < 0.449, the degree-only result).")
    elif check3 and not check4:
        print("  Clustering confound exists structurally (check3) but the detector's score")
        print("  doesn't track it (check4 null) -- unlikely to explain the FDR inflation")
        print("  through this detector's scores specifically.")
    elif check4 and not check3:
        print("  Score is clustering-sensitive (check4) but clean-selection doesn't bias")
        print("  clustering (check3 null) -- clustering is not a confound on this generator")
        print("  even though the detector could in principle be sensitive to it.")
    else:
        print("  NEITHER clustering check holds. Clustering coefficient does NOT explain the")
        print("  residual gap -- report this as a genuine negative result. The remaining")
        print("  ~50% of the clean-condition FDR inflation gap is still unexplained by any")
        print("  covariate tested so far.")


if __name__ == "__main__":
    main()