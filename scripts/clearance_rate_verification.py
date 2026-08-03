"""
clearance_rate_verification.py

Directly tests the theoretical proposition's prediction: does the
clearance rate c (fraction of true anomalies whose score exceeds the
ENTIRE calibration set) actually cross the predicted threshold
(1/(alpha*(n_calib+1)*pi_1) = 0.0716 for this setup) exactly between
p_an=0.02 and p_an=0.05, where discovery power was observed to collapse
to zero in the original severity sweep?

This is a real, targeted measurement -- not previously logged. The
original severity_sweep.py only recorded aggregate power/FDR per trial;
it never measured c directly. This script re-runs the same trials
(same graph generator, same detector, same adversarial calibration
selection) and additionally computes and logs c explicitly, seed by
seed, at every severity level.

If c crosses ~0.0716 between p_an=0.02 and p_an=0.05, that is direct,
quantitative confirmation of the proposition's mechanism -- not just
consistency, but a verified numerical prediction.

Run on Colab:
  !python3 scripts/clearance_rate_verification.py --n_seeds 20 --alpha 0.10 --device cuda
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


def run_trial_with_clearance_rate(p_an, alpha, seed, n_epochs, device, n_calib=2794):
    """Same trial as severity_sweep.py's run_severity_trial, but additionally
    computes and returns the clearance rate c: the fraction of TRUE
    anomalies in the test set whose score exceeds every calibration score
    (i.e., achieves the conformal p-value floor of 1/(n_calib+1))."""
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
    mean_calib_exposure = exposure[order_by_exposure][:n_calib].mean()

    remaining_normal = np.setdiff1d(normal_idx, calib_idx)
    test_idx = np.concatenate([remaining_normal, anomaly_idx])
    test_labels = np.concatenate([
        np.zeros(len(remaining_normal), dtype=int),
        np.ones(len(anomaly_idx), dtype=int),
    ])

    calib_scores = scores[calib_idx]
    test_scores = scores[test_idx]

    # THE KEY NEW MEASUREMENT: clearance rate c.
    # A test point achieves the p-value floor 1/(n_calib+1) exactly when
    # its score exceeds every single calibration score (count_ge == 0 in
    # the conformal_p_values formula's terms).
    calib_max = calib_scores.max()
    anomaly_test_mask = (test_labels == 1)
    anomaly_test_scores = test_scores[anomaly_test_mask]
    n_anomalies_clearing_ceiling = np.sum(anomaly_test_scores > calib_max)
    clearance_rate_c = n_anomalies_clearing_ceiling / len(anomaly_test_scores) if len(anomaly_test_scores) > 0 else 0.0

    p_values = conformal_p_values(calib_scores, test_scores)
    discoveries = benjamini_hochberg(p_values, alpha)

    n_discoveries = discoveries.sum()
    if n_discoveries == 0:
        realized_fdr = 0.0
    else:
        realized_fdr = np.sum(discoveries & (test_labels == 0)) / n_discoveries

    n_true_found = np.sum(discoveries & (test_labels == 1))
    power = n_true_found / len(anomaly_idx) if len(anomaly_idx) > 0 else 0.0

    return {
        "p_an": p_an, "seed": seed, "mean_calib_exposure": mean_calib_exposure,
        "clearance_rate_c": clearance_rate_c,
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
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

    # predicted threshold, computed from the proposition:
    # c* = 1 / (alpha * (n_calib+1) * pi_1)
    n_calib = 2794
    pi1 = 0.05  # synthetic anomaly rate
    predicted_c_threshold = 1 / (args.alpha * (n_calib + 1) * pi1)
    print(f"PREDICTED clearance-rate threshold for ANY discovery: c* = {predicted_c_threshold:.4f}")
    print(f"(i.e. need at least {predicted_c_threshold*100:.2f}% of true anomalies to beat the entire calibration set)\n")

    p_an_values = [0.002, 0.005, 0.01, 0.02, 0.05]
    all_results = []

    for p_an in p_an_values:
        print(f"=== p_an={p_an} ===")
        for seed in range(args.n_seeds):
            r = run_trial_with_clearance_rate(p_an, args.alpha, seed, args.n_epochs, device, n_calib=n_calib)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            print(f"  seed {seed}: clearance_rate_c={r['clearance_rate_c']:.4f} "
                  f"n_discoveries={r['n_discoveries']:3d} power={r['power']:.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "clearance_rate_verification.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to {csv_path}")

    print("\n=== VERIFICATION: does mean clearance rate cross the predicted threshold ===")
    print(f"Predicted threshold c* = {predicted_c_threshold:.4f}\n")
    for p_an in p_an_values:
        subset = [r for r in all_results if r["p_an"] == p_an]
        if not subset:
            continue
        mean_c = np.mean([r["clearance_rate_c"] for r in subset])
        mean_power = np.mean([r["power"] for r in subset])
        above_threshold = mean_c >= predicted_c_threshold
        print(f"p_an={p_an}: mean_clearance_rate_c={mean_c:.4f} "
              f"({'ABOVE' if above_threshold else 'BELOW'} predicted threshold), "
              f"mean_power={mean_power:.4f}")

    print("\nInterpretation: if mean_clearance_rate_c crosses BELOW the predicted threshold "
          "at exactly the same p_an where mean_power collapses to ~0, that is direct "
          "quantitative confirmation of the proposition's mechanism -- not just a "
          "plausibility argument, an actual verified numerical prediction.")


if __name__ == "__main__":
    main()