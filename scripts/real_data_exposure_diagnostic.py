"""
real_data_exposure_diagnostic.py

Checks whether the calibration-contamination mechanism this project's real-data
results are built on is actually operating on Amazon/Reddit/Tolokers, or
whether it has the same problem found on synthetic data. See
DETECTOR_DIAGNOSTIC.md for the full synthetic-data writeup this extends.

WHAT WAS FOUND ON SYNTHETIC DATA (2026-08-16). While smoke-testing an
AdaDetect baseline, one result looked implausibly clean. Chasing it found:
  - A normal node's exposure to anomalous neighbors has ~zero correlation
    with its score (|r| < 0.016, never significant, across three severity
    levels, n=3000, 3 seeds each).
  - Root cause: the GCN encoder's final layer is dead -- ReLU on the final
    embedding forces Z >= 0, and since ~99.7% of node pairs are non-edges
    pulling the reconstructed sigmoid(z_i . z_j) toward 0, the loss is
    minimized at Z = 0 exactly. Confirmed across every synthetic size/seed
    tested, including n=15,000, the size used throughout the paper.
  - With Z = 0, the structure decoder outputs a constant (0.25*n per node
    regardless of edges) and cannot affect ranking. The detector's ~0.91 AUROC
    is coming entirely from raw feature magnitude, not from anything
    graph-structural.
  - Two candidate fixes were tested and both fail: removing the final ReLU
    revives the mechanism (exposure correlation -> +0.35) but inverts
    detection (AUROC -> 0.12, because dense anomaly clusters are *easier* to
    reconstruct); wiring up the already-written-but-unused
    propagate_contamination() does not restore the correlation either
    (variance reduction from mostly-normal neighbors swamps the effect).

WHY THIS SCRIPT MATTERS. If the same zero-correlation pattern shows up on real
data, the paper's real-data results have the identical problem: the
clean/contaminated/adversarial conditions are not actually testing different
things, because a node's exposure doesn't reach its score. If instead real
data shows a genuine, significant exposure-score correlation, the results
could still be valid -- fraudulent accounts and their neighbors may share raw
FEATURES in a way synthetic data doesn't construct -- but the paper's stated
mechanism (message-passing propagation) would need rewriting to match what is
actually happening (a feature-similarity effect, not a graph-convolution one).

Deliberately uses this repo's UNMODIFIED train_dominant/normalize_adj -- the
exact code path that produced the paper's existing Amazon/Reddit numbers --
so this tests the pipeline the results actually came from, not a hypothetical
fixed version.

Run on the GPU box (Amazon/Reddit/Tolokers are ~11-12K nodes; a few seconds
per model on this code path, no special hardware needed):
  python3 scripts/real_data_exposure_diagnostic.py --datasets amazon reddit tolokers --n_seeds 3 --device cuda

Yelp is deliberately excluded from the default list: at n=45,954 the
unmodified dense normalize_adj costs roughly an hour PER MODEL. Pass
--datasets yelp explicitly only if you're prepared to wait, or use the
sparse-propagation path (not yet in this branch) instead.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv
import numpy as np
import networkx as nx
import torch
from scipy import stats

from detector import train_dominant, normalize_adj
from real_data_experiment import load_any_dataset, degree_normalize_scores, DEGREE_NORM_BY_DATASET

# Reference numbers from the synthetic-data diagnostic, printed alongside real
# results so the comparison doesn't require cross-referencing another file.
SYNTHETIC_REFERENCE = {
    0.002: {"mean_exposure": 0.021, "pearson_r": 0.0154, "pearson_p": 0.434},
    0.010: {"mean_exposure": 0.096, "pearson_r": 0.0027, "pearson_p": 0.354},
    0.050: {"mean_exposure": 0.342, "pearson_r": -0.0064, "pearson_p": 0.574},
}


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney U), no sklearn dependency. Verified
    against scipy.stats.mannwhitneyu including heavy-tie and degenerate cases."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / counts)[inv]
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def compute_exposure(graph, normal_idx, labels):
    """Fraction of each normal node's neighbors that are anomalous. Same
    definition used throughout this project (baseline_comparison.py,
    real_data_experiment.py, the severity sweep)."""
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if neighbors:
            exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)
    return exposure


def encoder_embedding(graph, features, model, device):
    """Rebuilds Z, the GNN's internal node representation, the same way
    train_dominant built it internally -- so the same dead-layer check that
    was run on synthetic data can be run here. Uses the frozen, unmodified
    normalize_adj deliberately: this must be the exact matrix training used,
    or the embedding would not be the one the model actually produced."""
    n = graph.number_of_nodes()
    A = nx.to_numpy_array(graph, nodelist=range(n))
    A_norm = normalize_adj(A).to(device)
    X = torch.tensor(features, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        Z = model.encode(A_norm, X)
    return Z.cpu().numpy()


def diagnose_dataset(dataset_name, n_seeds, n_epochs, device):
    print(f"\n{'=' * 70}\n{dataset_name.upper()}\n{'=' * 70}")
    try:
        graph, features, labels = load_any_dataset(dataset_name)
    except Exception as e:
        print(f"  SKIPPED -- could not load ({type(e).__name__}: {e})")
        print(f"  (likely a missing library for this dataset's loader -- "
              f"dgl for amazon/yelp, torch_geometric for tolokers, pygod for weibo/reddit)")
        return []

    use_degree_norm = DEGREE_NORM_BY_DATASET.get(dataset_name, True)
    normal_idx = np.where(labels == 0)[0]
    print(f"  n_nodes={graph.number_of_nodes():,}  n_anomalies={int(labels.sum()):,} "
          f"({labels.mean():.4f})  degree_norm={use_degree_norm}")

    rows = []
    for seed in range(n_seeds):
        raw_scores, model = train_dominant(graph, features, n_epochs=n_epochs,
                                           seed=seed, verbose=False, device=device)
        final_scores = degree_normalize_scores(graph, raw_scores) if use_degree_norm else raw_scores

        Z = encoder_embedding(graph, features, model, device)
        z_all_zero = bool((Z == 0).all())
        z_std = float(Z.std())

        exposure = compute_exposure(graph, normal_idx, labels)
        norm_scores = final_scores[normal_idx]
        r, p = stats.pearsonr(exposure, norm_scores)
        rho, _ = stats.spearmanr(exposure, norm_scores)
        gap = (float(norm_scores[exposure > 0].mean() - norm_scores[exposure == 0].mean())
               if (exposure > 0).any() and (exposure == 0).any() else float("nan"))

        row = dict(dataset=dataset_name, seed=seed, n_nodes=graph.number_of_nodes(),
                  auroc=auroc(final_scores, labels), z_all_zero=z_all_zero, z_std=z_std,
                  mean_exposure=float(exposure.mean()), pearson_r=float(r), pearson_p=float(p),
                  spearman_r=float(rho), exposure_gap=gap)
        rows.append(row)
        print(f"  seed {seed}: AUROC={row['auroc']:.4f}  Z_all_zero={z_all_zero}  "
              f"Z_std={z_std:.6f}  mean_exposure={row['mean_exposure']:.4f}  "
              f"exposure_r={r:+.4f} (p={p:.4f})  gap={gap:+.4f}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["amazon", "reddit", "tolokers"],
                        help="yelp is excluded from the default list -- unmodified "
                             "normalize_adj costs ~1hr/model at that scale.")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    all_rows = []
    for name in args.datasets:
        all_rows.extend(diagnose_dataset(name, args.n_seeds, args.n_epochs, device))

    if not all_rows:
        print("\nNo results -- every dataset was skipped. Check library availability.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "real_data_exposure_diagnostic.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {csv_path}")

    print(f"\n{'=' * 70}\nSUMMARY (real data vs. the synthetic finding)\n{'=' * 70}")
    print(f"{'dataset':10} {'mean AUROC':>11} {'Z dead?':>8} {'mean exposure_r':>16} {'mean p':>8}")
    for p_an, ref in SYNTHETIC_REFERENCE.items():
        print(f"{'(synthetic':10} {'':>11} {'':>8} {ref['pearson_r']:+16.4f} {ref['pearson_p']:8.4f}"
              f"  <- p_an={p_an}, mean_exposure={ref['mean_exposure']:.3f})")
    print()
    for name in args.datasets:
        rows = [r for r in all_rows if r["dataset"] == name]
        if not rows:
            continue
        a = np.array([r["auroc"] for r in rows])
        z = all(r["z_all_zero"] for r in rows)
        er = np.array([r["pearson_r"] for r in rows])
        ep = np.array([r["pearson_p"] for r in rows])
        print(f"{name:10} {a.mean():11.4f} {str(z):>8} {er.mean():+16.4f} {ep.mean():8.4f}")

    print("""
How to read this:
  - If Z_all_zero is True and exposure_r is near 0 (|r| roughly < 0.05, p not
    significant), same bug as synthetic: contamination is not reaching the
    scores on real data either, and the three calibration conditions are not
    testing what the paper says they test.
  - If exposure_r is clearly non-zero and significant (e.g. |r| > 0.1,
    p < 0.05), even with Z_all_zero=True, then real fraud accounts and their
    neighbors genuinely share raw FEATURES (not a graph-message-passing
    effect). The real-data numbers could still be valid, but the paper's
    stated mechanism needs to be rewritten to match what is actually
    happening.
""")


if __name__ == "__main__":
    main()
