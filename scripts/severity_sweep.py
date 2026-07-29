"""
severity_sweep.py

Step 6: escalate contamination SEVERITY (not just calibration selection
strategy) under the adversarial condition, to determine whether the
fail-safe property found in Step 5 holds across a range of contamination
magnitudes or only at the specific p_an=0.002 setting tested so far.

Without this, a reviewer's obvious objection to "adversarial calibration
doesn't break FDR control" is: you never tested a severe enough contamination
regime to find the break. This sweep answers that directly by escalating
p_an (anomaly-normal edge probability, which controls how much anomalous
signal reaches each normal node) across an order of magnitude, always using
the adversarial (worst-case, most-exposed-nodes-in-calibration) condition,
since that's the hardest test established in Step 5.

If realized FDR stays near or below nominal across the ENTIRE range -> strong,
defensible H4 certification result: the guarantee survives contamination
severity, not just contamination selection strategy.

If realized FDR climbs above nominal at some p_an threshold -> that threshold
IS the finding: a characterized breaking point, which is itself a strong,
specific, testable contribution (arguably stronger than either H1 or H4 in
isolation, since it gives a precise boundary condition).

Run on Colab with GPU (many training runs across p_an values x seeds):
  python3 scripts/severity_sweep.py --n_seeds 20 --alpha 0.10 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import numpy as np
import torch
from scipy import stats

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detector import train_dominant
from conformal_fdr import conformal_p_values, benjamini_hochberg


def run_severity_trial(p_an, alpha, seed, n_epochs, device, n_calib=2794):
    """Same pipeline as run_single_trial's 'adversarial' condition, but with
    p_an as a free parameter instead of fixed, so we can sweep contamination
    severity directly.

    n_calib is now a FIXED constant (2794, matching calib_frac=0.9 at the
    baseline severity level p_an=0.002) rather than derived from the
    clean-calibration pool each time. UPDATED from the original 1242
    (calib_frac=0.4) after a baseline-comparison test showed the smaller
    calibration size was capping power: zero-discovery rate dropped from
    ~60% to ~10-15% and power roughly doubled at calib_frac=0.9, with FDR
    remaining controlled at both sizes. This sweep only runs the
    adversarial condition (top-exposed nodes), not the clean condition."""
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=p_an, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    scores, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed, verbose=False, device=device)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)

    if len(normal_idx) < n_calib:
        return None  # graph too small / too contaminated for even this fixed calib size

    # adversarial: worst-case, most-exposed nodes
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

    p_values = conformal_p_values(calib_scores, test_scores)
    discoveries = benjamini_hochberg(p_values, alpha)

    n_discoveries = discoveries.sum()
    if n_discoveries == 0:
        realized_fdr = 0.0
    else:
        false_discoveries = np.sum(discoveries & (test_labels == 0))
        realized_fdr = false_discoveries / n_discoveries

    n_true_anomalies_found = np.sum(discoveries & (test_labels == 1))
    power = n_true_anomalies_found / len(anomaly_idx) if len(anomaly_idx) > 0 else 0.0

    return {
        "p_an": p_an,
        "seed": seed,
        "n_calib": n_calib,
        "mean_calib_exposure": mean_calib_exposure,
        "n_discoveries": int(n_discoveries),
        "realized_fdr": realized_fdr,
        "power": power,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}\n")

    # escalating severity: 0.002 (validated baseline) up to 0.05 (25x baseline)
    p_an_values = [0.002, 0.005, 0.01, 0.02, 0.05]

    all_results = []
    for p_an in p_an_values:
        print(f"=== p_an={p_an} ===")
        for seed in range(args.n_seeds):
            result = run_severity_trial(p_an, args.alpha, seed, args.n_epochs, device)
            if result is None:
                print(f"  seed {seed}: skipped (clean pool exhausted at this severity)")
                continue
            all_results.append(result)
            print(f"  seed {seed}: mean_calib_exposure={result['mean_calib_exposure']:.3f} "
                  f"n_discoveries={result['n_discoveries']:3d} realized_fdr={result['realized_fdr']:.3f} "
                  f"power={result['power']:.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "severity_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to {csv_path}")

    print("\n=== Summary by severity level ===")
    for p_an in p_an_values:
        subset = [r for r in all_results if r["p_an"] == p_an]
        if not subset:
            print(f"p_an={p_an}: no valid trials (clean pool likely exhausted)")
            continue
        fdrs = np.array([r["realized_fdr"] for r in subset])
        exposures = np.array([r["mean_calib_exposure"] for r in subset])
        t_stat, p_two_sided = stats.ttest_1samp(fdrs, args.alpha) if fdrs.std() > 0 else (float("nan"), float("nan"))
        p_one_sided = (p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2) if not np.isnan(p_two_sided) else float("nan")
        verdict = ("SIGNIFICANTLY ABOVE nominal" if (not np.isnan(p_one_sided) and p_one_sided < 0.05)
                   else "not significantly above nominal")
        print(f"p_an={p_an}: mean_calib_exposure={exposures.mean():.3f}, "
              f"realized_fdr={fdrs.mean():.3f}+/-{fdrs.std():.3f} (nominal={args.alpha}), "
              f"one-sided p={p_one_sided:.4f} -> {verdict}")

    print("\nInterpretation: if FDR stays controlled across ALL severity levels up to "
          "p_an=0.05 (25x the original baseline), that is a strong, defensible H4 "
          "certification claim. The first p_an where the verdict flips to "
          "'SIGNIFICANTLY ABOVE nominal' is the characterized breaking point -- "
          "report it explicitly as the boundary condition of the guarantee.")


if __name__ == "__main__":
    main()