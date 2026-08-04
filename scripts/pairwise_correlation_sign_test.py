"""
pairwise_correlation_sign_test.py

Directly tests the empirical premise behind the FDR-validity argument:
does the propagation mechanism ever produce NEGATIVE correlation between
connected nodes' scores, or does it only ever push connected scores
together (non-negative correlation)?

Why this matters: the FDR-control argument for adversarial calibration
relies on PRDS (positive regression dependency), which is guaranteed to
hold if the score vector satisfies MTP2 -- informally, if no two
variables are ever negatively correlated. The propagation mechanism
(averaging a node's score with its neighbors', repeated over hops)
structurally CANNOT push two nodes apart -- averaging only pulls values
toward each other. This experiment tests that claim directly: across
many trained models, do we ever observe a connected pair of nodes with
negative score correlation across seeds (i.e., as training randomness
varies, does one node's score going up ever coincide with a connected
neighbor's score going down)?

This is NOT a full proof (that would require analyzing the exact
functional form of the GNN, not just empirical correlation across
seeds), but it is real, direct, checkable evidence for or against the
premise -- and if violated even occasionally, that is an important,
honest finding that would mean the validity argument needs a different
approach.

Method: train the SAME graph across many seeds (so only training
randomness varies, not the graph itself), record each node's score
per seed, then compute the empirical correlation between every EDGE's
two endpoint scores across seeds. Report the distribution of these
pairwise correlations -- specifically, what fraction are negative.

Run on Colab:
  !python3 scripts/pairwise_correlation_sign_test.py --n_seeds 30 --n_edges_sample 2000 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import numpy as np

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detector import train_dominant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=30,
                         help="More seeds = more reliable per-pair correlation estimates.")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_edges_sample", type=int, default=2000,
                         help="Number of edges to sample and test (testing all edges is expensive).")
    parser.add_argument("--p_an", type=float, default=0.01,
                         help="Contamination level -- use a moderate value where propagation is active.")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # SAME graph across all seeds -- only training randomness varies,
    # isolating the question to "does training noise ever push connected
    # nodes' scores in opposite directions" rather than conflating this
    # with different graph realizations.
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=args.p_an, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=0,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()
    print(f"Graph generated once: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges, {labels.sum()} anomalies\n")

    # sample a fixed set of edges to test (all edges is too expensive to
    # store n_seeds x n_edges score-pairs for a graph this size)
    all_edges = list(graph.edges())
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(all_edges), size=min(args.n_edges_sample, len(all_edges)), replace=False)
    sampled_edges = [all_edges[i] for i in sample_idx]
    print(f"Testing {len(sampled_edges)} sampled edges across {args.n_seeds} training seeds\n")

    # train the SAME graph across many seeds, collect all node scores per seed
    all_scores = np.zeros((args.n_seeds, graph.number_of_nodes()))
    for seed in range(args.n_seeds):
        scores, _ = train_dominant(graph, features, n_epochs=args.n_epochs, seed=seed,
                                    verbose=False, device=device)
        all_scores[seed] = scores
        if seed % 5 == 0:
            print(f"  trained seed {seed}/{args.n_seeds}")

    print("\nComputing pairwise correlations for each sampled edge across seeds...")
    correlations = []
    for u, v in sampled_edges:
        scores_u = all_scores[:, u]
        scores_v = all_scores[:, v]
        if scores_u.std() > 0 and scores_v.std() > 0:
            corr = np.corrcoef(scores_u, scores_v)[0, 1]
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
    print(f"5th percentile: {np.percentile(correlations, 5):.4f}")
    print(f"95th percentile: {np.percentile(correlations, 95):.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pairwise_correlation_sign_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["edge_index", "correlation"])
        for i, c in enumerate(correlations):
            writer.writerow([i, c])
    print(f"\nSaved raw correlations to {csv_path}")

    print("\n=== INTERPRETATION ===")
    if n_negative == 0:
        print("ZERO negative correlations found. This is strong, direct empirical support "
              "for the non-negative-dependence premise (MTP2-like structure) underlying the "
              "PRDS argument -- consistent with the theoretical claim that propagation only "
              "pulls connected scores together, never apart.")
    elif n_negative / n_total < 0.02:
        print(f"A small fraction ({100*n_negative/n_total:.2f}%) of edges show negative "
              f"correlation. This could be sampling noise (with only {args.n_seeds} seeds, "
              f"small negative correlations near zero can appear by chance) -- worth "
              f"re-checking with more seeds before concluding the premise is violated.")
    else:
        print(f"A meaningful fraction ({100*n_negative/n_total:.2f}%) of edges show negative "
              f"correlation. This is an IMPORTANT, HONEST finding: it means the simple "
              f"non-negative-dependence argument does NOT cleanly hold, and the FDR-validity "
              f"proof needs a different technical approach than the one attempted -- this "
              f"redirects the theoretical work rather than completing it.")


if __name__ == "__main__":
    main()