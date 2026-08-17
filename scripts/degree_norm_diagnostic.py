"""
degree_norm_diagnostic.py

Decides DEGREE_NORM_BY_DATASET entries by measurement instead of assumption.

Degree normalization (score / log1p(degree)) was introduced to correct
high-degree "hub" accounts on Amazon, whose reconstruction error is inflated by
connectivity rather than anomalousness. It is NOT universally helpful: on
sparse Reddit it actively inverted the signal (AUROC 0.577 raw -> 0.452
normalized, i.e. below chance), which is why DEGREE_NORM_BY_DATASET exists at
all rather than a single global setting.

The dict currently has no "weibo" key, so `.get(dataset, True)` silently
defaults Weibo to True. That default may well be right -- Weibo is dense
(~8,405 nodes, ~408K edges) -- but "probably right" is exactly the reasoning
that produced the Reddit regression. This script measures it.

Reports, per dataset, the detector's AUROC with degree normalization ON vs OFF,
averaged over seeds, plus the graph statistics that explain the difference. The
winning setting goes into DEGREE_NORM_BY_DATASET with the measured numbers
recorded in a comment, matching how amazon/reddit were decided.

Run on Colab (needs the dataset's library -- dgl / pygod / torch-geometric):
  python3 scripts/degree_norm_diagnostic.py --dataset weibo --n_seeds 5 --device cuda
  python3 scripts/degree_norm_diagnostic.py --dataset yelp  --n_seeds 3 --device cuda --use_sparse_prop
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np

from detector import train_dominant
from real_data_experiment import (
    SUPPORTED_DATASETS,
    DEGREE_NORM_BY_DATASET,
    load_any_dataset,
    degree_normalize_scores,
)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), implemented in numpy so this script
    has no sklearn dependency -- sklearn is not in requirements.txt and is not
    importable on every dev machine here."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties, so tied scores don't bias the statistic
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / counts)[inv]
    r_pos = ranks[labels == 1].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def describe_graph(graph, labels):
    degrees = np.array([graph.degree(i) for i in range(graph.number_of_nodes())], dtype=float)
    normal_idx = np.where(labels == 0)[0]

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        nbrs = list(graph.neighbors(i))
        if nbrs:
            exposure[j] = sum(1 for n in nbrs if labels[n] == 1) / len(nbrs)

    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "anomaly_rate": float(labels.mean()),
        "mean_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "max_degree": float(degrees.max()),
        # The clean-condition guard in run_real_data_trial returns None for
        # ALL THREE conditions when this is < 20, which crashes main() on
        # all_results[0]. Surface it here rather than discovering it mid-run.
        "n_zero_exposure_normals": int((exposure == 0).sum()),
        "n_normals": len(normal_idx),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="weibo",
                        choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_sparse_prop", action="store_true",
                        help="Large-graph path; required for yelp (n=45,954).")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print(f"Loading {args.dataset}...")
    graph, features, labels = load_any_dataset(args.dataset)

    stats = describe_graph(graph, labels)
    print(f"\n=== {args.dataset} graph statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")

    if stats["n_zero_exposure_normals"] < 20:
        print(f"\n  WARNING: only {stats['n_zero_exposure_normals']} normal nodes have zero")
        print("  anomalous neighbors. run_real_data_trial returns None for ALL conditions")
        print("  below 20, which crashes main() on all_results[0]. Report clean as N/A")
        print("  for this dataset, citing the pool size.")

    raw_aurocs, norm_aurocs = [], []
    for seed in range(args.n_seeds):
        scores, _ = train_dominant(graph, features, n_epochs=args.n_epochs,
                                   seed=seed, verbose=False, device=device,
                                   use_sparse_prop=args.use_sparse_prop)
        raw_aurocs.append(auroc(scores, labels))
        norm_aurocs.append(auroc(degree_normalize_scores(graph, scores), labels))
        print(f"  seed {seed}: raw={raw_aurocs[-1]:.4f}  degree_norm={norm_aurocs[-1]:.4f}")

    raw = np.array(raw_aurocs)
    norm = np.array(norm_aurocs)
    print(f"\n=== AUROC over {args.n_seeds} seeds ===")
    print(f"  raw scores        : {raw.mean():.4f} +/- {raw.std():.4f}")
    print(f"  degree-normalized : {norm.mean():.4f} +/- {norm.std():.4f}")

    current = DEGREE_NORM_BY_DATASET.get(args.dataset, None)
    recommended = bool(norm.mean() > raw.mean())
    print(f"\n  RECOMMENDED DEGREE_NORM_BY_DATASET['{args.dataset}'] = {recommended}")
    if current is None:
        print(f"  (currently ABSENT from the dict -- silently defaulting to True)")
    elif current != recommended:
        print(f"  (currently {current} -- MEASUREMENT DISAGREES, update it)")
    else:
        print(f"  (currently {current} -- matches, no change needed)")

    if max(raw.mean(), norm.mean()) < 0.6:
        print("\n  WARNING: best AUROC < 0.6. A detector this weak produces few or no")
        print("  discoveries, so a controlled FDR on this dataset is a weak test of the")
        print("  guarantee rather than a confirmation -- report it the way Reddit is.")

    print(f"\nPaste into scripts/real_data_experiment.py DEGREE_NORM_BY_DATASET:")
    print(f'  "{args.dataset}": {recommended},  '
          f'# measured: raw={raw.mean():.3f}, degree_norm={norm.mean():.3f} '
          f'({args.n_seeds} seeds)')


if __name__ == "__main__":
    main()
