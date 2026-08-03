"""
score_separation_mechanism_analysis.py

Building on the validated finding: n_anomalies_below_p0.001 predicts
actual discovery counts strongly (r=0.807, p=4e-24). This experiment
tests the NEXT link in the causal chain: is that tail-count itself
tightly governed by a simple, measurable detector-quality statistic --
specifically, the standardized separation between the anomaly and
normal score distributions (a d-prime-style statistic: (mean_anomaly -
mean_normal) / pooled_std)?

If this link is tight (high r, ideally >0.85-0.90), we have a full
mechanistic chain that could plausibly be formalized with an extreme-
value/order-statistics argument:

  score separation (raw detector quality)
    -> n_anomalies_below_p0.001 (extreme-tail count, r=0.807 already confirmed)
    -> n_discoveries (actual outcome)

rather than a single unexplained empirical correlation. This is what
would let a formal proposition be stated in terms of measurable
distributional quantities (as flagged as the key remaining gap in
theory/joint_discovery_threshold_proposition.md's "what remains to
formalize" section).

If this link is NOT tight, that itself is an important, honest finding:
it would mean detector quality alone doesn't determine the tail count,
and something else (e.g. calibration set composition specifically,
independent of overall detector quality) plays a bigger role than
expected -- which would redirect where the theoretical effort should
focus.

Run on Colab:
  !python3 scripts/score_separation_mechanism_analysis.py --n_seeds 20 --alpha 0.10 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import numpy as np
from scipy import stats

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detector import train_dominant
from conformal_fdr import conformal_p_values, benjamini_hochberg


def run_trial_with_separation_stat(p_an, alpha, seed, n_epochs, device, n_calib=2794):
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

    # THE KEY NEW MEASUREMENT: d-prime-style score separation between
    # anomaly and normal populations IN THE TEST SET (the same population
    # the tail-count and discovery-count statistics are computed over).
    anomaly_mask = (test_labels == 1)
    normal_mask = (test_labels == 0)
    anomaly_scores = test_scores[anomaly_mask]
    normal_scores = test_scores[normal_mask]

    mean_diff = anomaly_scores.mean() - normal_scores.mean()
    pooled_std = np.sqrt((anomaly_scores.var() + normal_scores.var()) / 2)
    d_prime = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Also compute AUROC directly for comparison (a second, standard
    # detector-quality metric, to check whether d-prime or AUROC is the
    # better predictor).
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(test_labels, test_scores)
    except Exception:
        auroc = float("nan")

    p_values = conformal_p_values(calib_scores, test_scores)
    anomaly_pvalues = p_values[anomaly_mask]
    n_below_p001 = int(np.sum(anomaly_pvalues <= 0.001))

    discoveries = benjamini_hochberg(p_values, alpha)
    n_discoveries = discoveries.sum()
    power = (np.sum(discoveries & (test_labels == 1)) / len(anomaly_idx)) if len(anomaly_idx) > 0 else 0.0

    return {
        "p_an": p_an, "seed": seed,
        "d_prime": d_prime, "auroc": auroc,
        "n_anomalies_below_p001": n_below_p001,
        "n_discoveries": int(n_discoveries), "power": power,
    }


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
            r = run_trial_with_separation_stat(p_an, args.alpha, seed, args.n_epochs, device)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            print(f"  seed {seed}: d_prime={r['d_prime']:.3f} auroc={r['auroc']:.3f} "
                  f"n_below_p001={r['n_anomalies_below_p001']:3d} n_disc={r['n_discoveries']:3d}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "score_separation_mechanism.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to {csv_path}")

    d_primes = np.array([r["d_prime"] for r in all_results])
    aurocs = np.array([r["auroc"] for r in all_results])
    n_below = np.array([r["n_anomalies_below_p001"] for r in all_results])
    n_disc = np.array([r["n_discoveries"] for r in all_results])

    print("\n=== THE KEY TEST: does a simple detector-quality statistic predict the tail count? ===")
    corr_dprime, p_dprime = stats.pearsonr(d_primes, n_below)
    print(f"d_prime vs. n_anomalies_below_p001: r={corr_dprime:.4f} (p={p_dprime:.4g})")
    corr_auroc, p_auroc = stats.pearsonr(aurocs, n_below)
    print(f"AUROC vs. n_anomalies_below_p001:   r={corr_auroc:.4f} (p={p_auroc:.4g})")

    print("\n=== For reference: does the same statistic predict discoveries directly? ===")
    corr_dprime_disc, p_dprime_disc = stats.pearsonr(d_primes, n_disc)
    print(f"d_prime vs. n_discoveries: r={corr_dprime_disc:.4f} (p={p_dprime_disc:.4g})")
    corr_auroc_disc, p_auroc_disc = stats.pearsonr(aurocs, n_disc)
    print(f"AUROC vs. n_discoveries:   r={corr_auroc_disc:.4f} (p={p_auroc_disc:.4g})")

    print("\nInterpretation: if d_prime (or AUROC) correlates with n_anomalies_below_p001 "
          "at r>0.85-0.90, that closes the mechanistic chain from raw, measurable detector "
          "quality all the way to discovery outcomes -- a real candidate for formalization "
          "via extreme-value/order-statistics theory. If the correlation is weaker than the "
          "already-confirmed n_anomalies_below_p001 -> n_discoveries link (r=0.807), that "
          "means detector quality alone does not fully explain the tail count, and something "
          "else (e.g. calibration composition specifically) matters independently -- an "
          "honest finding that redirects rather than completes the theoretical picture.")


if __name__ == "__main__":
    main()