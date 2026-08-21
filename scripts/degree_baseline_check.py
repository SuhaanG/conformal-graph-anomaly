"""
degree_baseline_check.py

THE HIGHEST-VALUE REMAINING EXPERIMENT, and the cheapest.

THE CLAIM TO TEST. Reconstruction-based graph anomaly detectors may be, to a
large extent, measuring node degree rather than anomalousness. If a trivial
degree-only baseline matches or beats trained GNN detectors on the standard
benchmarks, then reported AUROC on those benchmarks is not evidence of anomaly
detection, and a large body of comparative results is confounded.

WHY THIS IS WORTH RUNNING. Two independent observations point at it:

  1. On the synthetic generator, sweeping anomaly-anomaly density with
     FEATURES HELD IDENTICAL moved dominant_pygod's AUROC from 0.0793 to
     1.0000, crossing 0.5 exactly where the anomaly/normal expected-degree
     ratio crosses 1.0 (theory doc Part 7).
  2. On amazon, dominant_pygod's score-degree Spearman is +0.918 while its
     AUROC is ~0.97. A score that is near-monotone in degree cannot have an
     AUROC far from degree's own.

Neither establishes the claim. This does, or refutes it.

WHAT IS COMPARED. Four label-free baselines that require no training at all,
against every detector in src/detectors.py, on every real dataset:

    degree          raw node degree
    log_degree      log1p(degree)  -- same ranking, included as a sanity check
                    that AUROC is rank-based and therefore identical
    neg_degree      -degree, for datasets where anomalies are LOW degree
    random          a random score, as the true chance floor

AUROC is rank-based, so degree and log_degree MUST agree exactly. If they do
not, the AUROC implementation is wrong -- that is a deliberate internal check.

HOW TO READ IT. For each dataset the script reports the best degree-based
baseline against each trained detector, and the gap. A detector that cannot
beat `degree` by a clear margin is not demonstrably doing anything a lookup
table could not do on that dataset.

INTERPRET WITH CARE, IN BOTH DIRECTIONS. A degree baseline scoring well does
NOT prove the detectors are worthless -- on some graphs degree may genuinely
be anomalous (fraud rings, bots). What it establishes is that AUROC on those
benchmarks cannot distinguish a detector from a degree lookup, so AUROC alone
does not support claims of anomaly detection there. That is a claim about the
EVALUATION, which is the defensible one.

Run:
  python scripts/degree_baseline_check.py --device cuda
  python scripts/degree_baseline_check.py --datasets amazon reddit --device cuda
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv

import numpy as np
import torch
from scipy import stats

from detectors import available_detectors, score_nodes
from real_data_experiment import load_any_dataset, SUPPORTED_DATASETS

DEFAULT_DATASETS = ["amazon", "reddit", "tolokers", "weibo"]


def auroc(scores, labels):
    """Mann-Whitney AUROC with CORRECT TIE HANDLING.

    Uses scipy rankdata (method='average'), not argsort-of-argsort. This
    matters enormously here and nowhere else in the project: DEGREE IS HEAVILY
    TIED -- thousands of nodes share a degree -- and argsort breaks ties in
    arbitrary index order, which silently inflates or deflates the baseline's
    AUROC depending on how the label vector happens to be ordered. A tied score
    must contribute exactly 0.5, and only average ranks give that.

    Verified against perfect (1.0), inverted (0.0), and all-tied (0.5) cases.
    The argsort version returns 1.0 on the all-tied case, which is how the bug
    was caught."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]), method="average")
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS,
                    choices=sorted(SUPPORTED_DATASETS))
    ap.add_argument("--detectors", type=str, nargs="+",
                    default=available_detectors(), choices=available_detectors())
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n_epochs", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--use_sparse_prop", action="store_true")
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {args.device}")
    print("Comparing trained GNN detectors against untrained degree baselines.\n")

    rows = []
    for ds in args.datasets:
        print(f"### {ds}")
        graph, features, labels = load_any_dataset(ds)
        n = graph.number_of_nodes()
        deg = np.array([graph.degree(int(i)) for i in range(n)], dtype=float)

        au_deg = auroc(deg, labels)
        au_logdeg = auroc(np.log1p(deg), labels)
        au_negdeg = auroc(-deg, labels)
        rng = np.random.default_rng(0)
        au_rand = auroc(rng.random(n), labels)

        # AUROC is rank-based; a strictly monotone transform cannot change it.
        # If this fires, the AUROC implementation is broken, not the finding.
        assert abs(au_deg - au_logdeg) < 1e-9, (
            f"degree and log_degree AUROC differ ({au_deg} vs {au_logdeg}); "
            f"the AUROC implementation is not rank-based")

        best_free = max(au_deg, au_negdeg)
        best_name = "degree" if au_deg >= au_negdeg else "neg_degree"
        print(f"    {n} nodes, {int(labels.sum())} anomalies "
              f"({labels.mean():.4f})")
        print(f"    degree AUROC={au_deg:.4f}   neg_degree={au_negdeg:.4f}   "
              f"random={au_rand:.4f}")
        print(f"    best training-free baseline: {best_name} = {best_free:.4f}")

        for det in args.detectors:
            aus = []
            for seed in range(args.n_seeds):
                try:
                    s = score_nodes(det, graph, features, labels=labels, seed=seed,
                                    n_epochs=args.n_epochs, device=args.device,
                                    use_sparse_prop=args.use_sparse_prop)
                except Exception as e:
                    print(f"      {det:<16} FAILED {type(e).__name__}: {str(e)[:60]}")
                    aus = []
                    break
                aus.append(auroc(s, labels))
                if seed == 0:
                    sd = stats.spearmanr(s, deg).statistic
            if not aus:
                continue
            au_det = float(np.mean(aus))
            gap = au_det - best_free
            verdict = ("beats degree" if gap > 0.02 else
                       "TIED WITH DEGREE" if gap > -0.02 else
                       "LOSES TO DEGREE")
            print(f"      {det:<16} AUROC={au_det:.4f}  "
                  f"vs baseline {gap:+.4f}   sdeg={sd:+.3f}   {verdict}")
            rows.append({
                "dataset": ds, "detector": det, "n_nodes": n,
                "anomaly_rate": float(labels.mean()),
                "detector_auroc": au_det,
                "degree_auroc": au_deg, "neg_degree_auroc": au_negdeg,
                "random_auroc": au_rand,
                "best_free_baseline": best_name, "best_free_auroc": best_free,
                "gap_vs_baseline": gap,
                "spearman_score_degree": float(sd),
            })
        print()

    if not rows:
        print("Nothing completed.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "degree_baseline_check.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} rows to {out}\n")

    print("=" * 78)
    print("CAN A TRAINED DETECTOR BEAT AN UNTRAINED DEGREE LOOKUP?")
    print("=" * 78)
    n_tied = sum(1 for r in rows if r["gap_vs_baseline"] <= 0.02)
    print(f"  {n_tied} of {len(rows)} detector-dataset cells fail to beat the")
    print(f"  best training-free degree baseline by more than 0.02 AUROC.\n")

    for ds in args.datasets:
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            continue
        b = sub[0]["best_free_auroc"]
        best_det = max(sub, key=lambda r: r["detector_auroc"])
        print(f"  {ds:<10} best free baseline {b:.4f} "
              f"({sub[0]['best_free_baseline']})   "
              f"best detector {best_det['detector_auroc']:.4f} "
              f"({best_det['detector']})   gap {best_det['gap_vs_baseline']:+.4f}")

    print()
    frac = n_tied / len(rows)
    if frac >= 0.5:
        print("  RESULT: on most cells, AUROC does not separate a trained GNN from")
        print("  a degree lookup table. Reported AUROC on these benchmarks cannot")
        print("  be read as evidence of anomaly detection. This is a claim about")
        print("  the EVALUATION, not about whether the detectors could ever work.")
    elif frac > 0:
        print("  RESULT: MIXED. Some cells are indistinguishable from a degree")
        print("  lookup and others are not. Report per-dataset -- the split is")
        print("  more informative than any aggregate.")
    else:
        print("  RESULT: detectors beat the degree baseline everywhere. The")
        print("  degree-artifact concern does NOT generalise from synthetic to")
        print("  these real benchmarks. Drop the claim.")


if __name__ == "__main__":
    main()
