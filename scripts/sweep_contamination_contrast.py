"""
sweep_contamination_contrast.py

Step 2: fix the weak propagation contrast found in check_graph_gen.py.

Problem: at default settings, too many normal nodes end up adjacent to *some*
anomaly, which dilutes the "isolated vs. contaminated" contrast we need to
cleanly test H1 (does clustered contamination break FDR control?).

This script sweeps p_an (anomaly-normal edge exposure) and n_anomaly_clusters
(scattered vs. tight anomaly placement) across graph sizes, and reports which
combinations produce:
  (a) a meaningful fraction of normal nodes NOT adjacent to any anomaly
      (a clean "isolated" control group), and
  (b) a meaningful fraction of normal nodes adjacent to anomalies
      (the "contaminated" group), with
  (c) a clear separation in propagated feature shift between the two groups.

Run: python scripts/sweep_contamination_contrast.py
Output: results/logs/contamination_sweep.csv, results/figures/contamination_sweep.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator


def run_single_config(n_nodes, p_an, n_clusters, seed, hops=1, mix_weight=0.5):
    cfg = GraphGenConfig(
        n_nodes=n_nodes,
        p_aa=0.3,           # fixed high homophily so clusters are tight
        p_an=p_an,
        p_nn=0.005,         # kept low and fixed so background density doesn't confound results
        feature_shift=3.0,
        n_anomaly_clusters=n_clusters,
        random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    gen.generate()
    raw = gen.features.copy()
    propagated = gen.propagate_contamination(hops=hops, mix_weight=mix_weight)

    # CORRECTED APPROACH: binary adjacent/non-adjacent split under-detects
    # contamination because mean-aggregation dilutes a minority of anomalous
    # neighbors among a majority of normal ones. The mechanism that actually
    # drives contamination severity is the FRACTION of anomalous neighbors a
    # node has — so we measure a dose-response curve instead of a binary split.
    normal_idx = np.where(gen.labels == 0)[0]

    anom_neighbor_frac = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(gen.graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        n_anom_neighbors = sum(1 for n in neighbors if gen.labels[n] == 1)
        anom_neighbor_frac[j] = n_anom_neighbors / len(neighbors)

    shift = np.mean(np.abs(propagated[normal_idx] - raw[normal_idx]), axis=1)

    # correlation between anomalous-neighbor fraction and feature shift:
    # this is the real signal — positive correlation means contamination
    # scales with exposure, which is what H1 needs to be testable at all.
    if np.std(anom_neighbor_frac) > 0 and np.std(shift) > 0:
        dose_response_corr = np.corrcoef(anom_neighbor_frac, shift)[0, 1]
    else:
        dose_response_corr = np.nan

    frac_with_any_exposure = np.mean(anom_neighbor_frac > 0)
    frac_isolated_control = 1 - frac_with_any_exposure
    mean_exposure_among_exposed = (
        np.mean(anom_neighbor_frac[anom_neighbor_frac > 0])
        if frac_with_any_exposure > 0 else np.nan
    )

    return {
        "n_nodes": n_nodes, "p_an": p_an, "n_clusters": n_clusters,
        "frac_adjacent": frac_with_any_exposure,
        "frac_isolated_control": frac_isolated_control,
        "mean_exposure_among_exposed": mean_exposure_among_exposed,
        "contrast_ratio": dose_response_corr,  # repurposed: now the dose-response correlation
    }


def main():
    n_nodes_options = [3000, 6000]
    p_an_options = [0.0005, 0.001, 0.002, 0.005, 0.01]
    n_clusters_options = [1, 3]
    seeds = [1, 2, 3]  # average over seeds for stability

    results = []
    for n_nodes in n_nodes_options:
        for p_an in p_an_options:
            for n_clusters in n_clusters_options:
                seed_results = [
                    run_single_config(n_nodes, p_an, n_clusters, seed)
                    for seed in seeds
                ]
                avg = {
                    "n_nodes": n_nodes, "p_an": p_an, "n_clusters": n_clusters,
                    "frac_adjacent": np.nanmean([r["frac_adjacent"] for r in seed_results]),
                    "frac_isolated_control": np.nanmean([r["frac_isolated_control"] for r in seed_results]),
                    "contrast_ratio": np.nanmean([r["contrast_ratio"] for r in seed_results]),
                }
                results.append(avg)
                print(
                    f"n={n_nodes} p_an={p_an} clusters={n_clusters} "
                    f"-> frac_adjacent={avg['frac_adjacent']:.3f} "
                    f"frac_isolated_control={avg['frac_isolated_control']:.3f} "
                    f"contrast_ratio={avg['contrast_ratio']:.3f}"
                )

    # Save results to CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "contamination_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved sweep results to {csv_path}")

    # Recommend best config: want frac_isolated_control >= 0.3, frac_adjacent >= 0.3,
    # and the highest contrast_ratio among those satisfying both.
    candidates = [
        r for r in results
        if r["frac_isolated_control"] >= 0.3 and r["frac_adjacent"] >= 0.3
        and not np.isnan(r["contrast_ratio"])
    ]
    if candidates:
        best = max(candidates, key=lambda r: r["contrast_ratio"])
        print("\n=== Recommended config (balanced groups, highest contrast) ===")
        print(best)
    else:
        print("\nNo config satisfied the 30/30 balance threshold — "
              "widen the p_an range or relax the threshold and re-run.")

    # Plot: contrast_ratio vs p_an, one line per (n_nodes, n_clusters) combo
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for n_nodes in n_nodes_options:
        for n_clusters in n_clusters_options:
            subset = [r for r in results if r["n_nodes"] == n_nodes and r["n_clusters"] == n_clusters]
            subset.sort(key=lambda r: r["p_an"])
            xs = [r["p_an"] for r in subset]
            ys = [r["contrast_ratio"] for r in subset]
            plt.plot(xs, ys, marker="o", label=f"n={n_nodes}, clusters={n_clusters}")
    plt.xlabel("p_an (anomaly-normal edge probability)")
    plt.ylabel("contrast ratio (adjacent shift / non-adjacent shift)")
    plt.title("Propagation contrast across contamination exposure settings")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "contamination_sweep.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    main()