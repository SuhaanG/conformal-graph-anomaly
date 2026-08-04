"""
pvalue_correlation_fixed_model.py

CORRECTED version of pairwise_correlation_sign_test.py. That experiment
tested the wrong source of randomness: it varied the TRAINING SEED
(retraining the whole model from scratch 30 times), which conflates
random initialization/optimization noise with the actual object the
PRDS condition concerns.

PRDS (positive regression dependency) is a property of conformal
P-VALUES as random variables under the randomness of the CALIBRATION
SPLIT -- i.e., for a FIXED scoring function (one trained model), if you
repeatedly resample which nodes land in calibration vs. test, are two
connected test nodes' p-values positively dependent on each other? This
is what Bates et al.'s framework actually analyzes, and what
Benjamini-Yekutieli's sufficient condition is stated in terms of.

This experiment fixes that design flaw: train ONE model (removing
init/training noise as a confound), then resample the calibration/test
split many times, computing p-values for the SAME fixed set of test
nodes across all resamples, and test whether connected pairs' p-values
are non-negatively correlated across split-resampling randomness.

Run on Colab:
  !python3 scripts/pvalue_correlation_fixed_model.py --n_resamples 50 --n_edges_sample 2000 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import numpy as np

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detector import train_dominant
from conformal_fdr import conformal_p_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_resamples", type=int, default=50,
                         help="Number of random calibration-split resamples (this is the "
                              "actual randomness source PRDS concerns).")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_edges_sample", type=int, default=2000)
    parser.add_argument("--p_an", type=float, default=0.01)
    parser.add_argument("--calib_frac", type=float, default=0.4,
                         help="Fraction of eligible normal nodes used for calibration each resample.")
    parser.add_argument("--train_seed", type=int, default=0,
                         help="Single fixed training seed -- the model is trained ONCE.")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=args.p_an, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=0,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()
    print(f"Graph generated once: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges, {labels.sum()} anomalies\n")

    print(f"Training ONE model (seed={args.train_seed}) -- this is fixed for the entire experiment...")
    scores, _ = train_dominant(graph, features, n_epochs=args.n_epochs, seed=args.train_seed,
                                verbose=False, device=device)
    print("Model trained. Now resampling calibration splits...\n")

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    # fixed test node set: use ALL normal nodes NOT eligible for any resample's
    # calibration pool as a stable comparison population is complex; instead,
    # take the SIMPLER and more standard approach: for each resample, the test
    # set is "everyone not selected for calibration this time" -- meaning the
    # test population itself varies slightly per resample. To get a clean,
    # fixed set of nodes to track p-values for across resamples, we restrict
    # to nodes that have LOW exposure (very unlikely to ever be excluded by
    # chance from the test set in a way that matters) -- specifically we track
    # a fixed random 30% subsample of normal nodes as "always-test" candidates,
    # and only resample calibration from the REMAINING 70%.
    rng_split = np.random.default_rng(42)
    shuffled = rng_split.permutation(normal_idx)
    split_point = int(0.3 * len(shuffled))
    always_test_pool = shuffled[:split_point]  # never eligible for calibration
    calib_eligible_pool = shuffled[split_point:]  # calibration drawn from here each resample

    n_calib = int(round(args.calib_frac * len(calib_eligible_pool)))
    print(f"Fixed always-test pool: {len(always_test_pool)} nodes")
    print(f"Calibration-eligible pool: {len(calib_eligible_pool)} nodes, drawing {n_calib} per resample\n")

    # sample a fixed set of edges WITHIN the always-test pool to track
    test_pool_set = set(always_test_pool.tolist())
    candidate_edges = [(u, v) for u, v in graph.edges() if u in test_pool_set and v in test_pool_set]
    print(f"Edges within the always-test pool: {len(candidate_edges)}")
    rng_edges = np.random.default_rng(0)
    n_sample = min(args.n_edges_sample, len(candidate_edges))
    sample_idx = rng_edges.choice(len(candidate_edges), size=n_sample, replace=False)
    sampled_edges = [candidate_edges[i] for i in sample_idx]
    print(f"Testing {len(sampled_edges)} sampled edges across {args.n_resamples} calibration resamples\n")

    # test set is ALWAYS the always_test_pool (fixed across resamples) plus
    # anomalies (p-values for anomalies aren't the object of interest here,
    # only null/normal p-value dependency matters for PRDS)
    test_idx = always_test_pool

    all_pvalues = np.zeros((args.n_resamples, len(test_idx)))
    rng_resample = np.random.default_rng(123)
    for r in range(args.n_resamples):
        calib_idx = rng_resample.choice(calib_eligible_pool, size=n_calib, replace=False)
        calib_scores = scores[calib_idx]
        test_scores = scores[test_idx]
        p_values = conformal_p_values(calib_scores, test_scores)
        all_pvalues[r] = p_values
        if r % 10 == 0:
            print(f"  resample {r}/{args.n_resamples}")

    # map test node id -> row index in all_pvalues
    node_to_row = {node: i for i, node in enumerate(test_idx)}

    print("\nComputing pairwise p-value correlations across calibration resamples...")
    correlations = []
    for u, v in sampled_edges:
        pvals_u = all_pvalues[:, node_to_row[u]]
        pvals_v = all_pvalues[:, node_to_row[v]]
        if pvals_u.std() > 0 and pvals_v.std() > 0:
            corr = np.corrcoef(pvals_u, pvals_v)[0, 1]
            correlations.append(corr)

    correlations = np.array(correlations)
    n_negative = np.sum(correlations < 0)
    n_total = len(correlations)

    print(f"\n=== RESULTS ===")
    print(f"Edges tested: {n_total}")
    print(f"Negative correlations: {n_negative} ({100*n_negative/n_total:.2f}%)")
    print(f"Mean correlation: {correlations.mean():.4f}")
    print(f"Median correlation: {np.median(correlations):.4f}")
    print(f"Min correlation: {correlations.min():.4f}")
    print(f"Max correlation: {correlations.max():.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pvalue_correlation_fixed_model.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["edge_index", "correlation"])
        for i, c in enumerate(correlations):
            writer.writerow([i, c])
    print(f"\nSaved raw correlations to {csv_path}")

    print("\n=== INTERPRETATION ===")
    if n_negative / n_total < 0.05:
        print(f"Only {100*n_negative/n_total:.2f}% negative correlations under the CORRECT "
              f"randomness source (calibration-split resampling, fixed model). This is real, "
              f"properly-targeted empirical support for the PRDS premise -- much stronger "
              f"evidence than the earlier (flawed) cross-seed test, since this measures the "
              f"actual randomness source the theorem concerns.")
    else:
        print(f"A meaningful fraction ({100*n_negative/n_total:.2f}%) of edges show negative "
              f"p-value correlation even under the correct randomness source. This means the "
              f"PRDS premise likely does NOT hold in general for this setting, even with the "
              f"corrected test design -- a genuine, important finding that the FDR-validity "
              f"argument needs fundamentally different technical machinery.")


if __name__ == "__main__":
    main()