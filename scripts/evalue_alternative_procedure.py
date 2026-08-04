"""
evalue_alternative_procedure.py

PIVOT, grounded in the original literature audit (not a third guess).
The last two experiments tested whether conformal p-values satisfy PRDS
under graph-induced dependence -- results were mixed (16.65% negative
pairwise p-value correlations even under the corrected test design),
meaning the p-value+BH pipeline's FDR guarantee may not be cleanly
provable via the standard route.

Rather than continuing to search for a proof that p-values satisfy a
condition our own data shows is shaky, this switches to a DIFFERENT
statistical object that sidesteps the problem: e-values.

Key fact (Vovk & Wang 2021; Wang & Ramdas 2022, JRSS-B): e-values merge
and control FDR (via the e-BH procedure) validly under ARBITRARY,
UNKNOWN dependence -- no PRDS, no positive dependence, no independence
assumption needed at all. This was flagged as relevant machinery in the
project's original literature audit and is being used now for the first
time.

Construction:
1. p-to-e calibrator (Vovk & Wang): given any valid (super-uniform)
   p-value p, e(p) = kappa * p^(kappa-1) is a valid e-value for any
   kappa in (0,1), REGARDLESS of dependence structure among the
   p-values. We use kappa=0.5: e(p) = 0.5 / sqrt(p).
2. e-BH (Wang & Ramdas): sort e-values descending. Reject the top k
   where k = max{i : e_(i) >= m/(alpha*i)}. This controls FDR at level
   alpha under ARBITRARY dependence among the e-values.

This experiment runs BOTH procedures (standard p-value+BH, and the new
e-value+e-BH) on the SAME trained models and calibration splits across
the severity sweep, to directly compare:
  (a) does e-BH also control FDR (expected: yes, unconditionally)
  (b) how much power is lost by switching to the dependence-robust
      procedure (expected: some loss, since e-values are a more
      conservative/general-purpose tool -- the question is how much)

If the power cost is small, e-BH becomes the honest, provably-valid
procedure to report in the paper INSTEAD of the p-value+BH pipeline,
sidestepping the whole PRDS question rather than needing to resolve it.

Run on Colab:
  !python3 scripts/evalue_alternative_procedure.py --n_seeds 20 --alpha 0.10 --device cuda
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


def p_to_e_calibrator(p_values: np.ndarray, kappa: float = 0.5) -> np.ndarray:
    """Vovk & Wang (2021) universal p-to-e calibrator. Valid for ANY
    super-uniform p-value (which conformal p-values are, by construction,
    regardless of dependence structure) -- produces a valid e-value with
    E[e] <= 1 under the null, no additional assumptions needed."""
    p_values = np.clip(p_values, 1e-300, 1.0)  # avoid division issues at p=0
    return kappa * np.power(p_values, kappa - 1)


def e_bh(e_values: np.ndarray, alpha: float) -> np.ndarray:
    """Wang & Ramdas (2022) e-BH procedure. Controls FDR at level alpha
    under ARBITRARY, UNKNOWN dependence among the e-values -- no PRDS or
    independence assumption required, unlike standard BH on p-values."""
    m = len(e_values)
    order = np.argsort(-e_values)  # descending
    sorted_e = e_values[order]
    thresholds = m / (alpha * np.arange(1, m + 1))
    meets = sorted_e >= thresholds
    if not np.any(meets):
        return np.zeros(m, dtype=bool)
    max_idx = np.max(np.where(meets)[0])
    reject_sorted = np.zeros(m, dtype=bool)
    reject_sorted[: max_idx + 1] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def run_comparison_trial(p_an, alpha, seed, n_epochs, device, n_calib=2794):
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
    calib_idx = normal_idx[order_by_exposure][:n_calib]  # adversarial condition

    remaining_normal = np.setdiff1d(normal_idx, calib_idx)
    test_idx = np.concatenate([remaining_normal, anomaly_idx])
    test_labels = np.concatenate([
        np.zeros(len(remaining_normal), dtype=int),
        np.ones(len(anomaly_idx), dtype=int),
    ])

    calib_scores = scores[calib_idx]
    test_scores = scores[test_idx]
    p_values = conformal_p_values(calib_scores, test_scores)

    # Procedure A: standard p-value + BH (the current pipeline)
    discoveries_bh = benjamini_hochberg(p_values, alpha)
    n_disc_bh = discoveries_bh.sum()
    fdr_bh = (np.sum(discoveries_bh & (test_labels == 0)) / n_disc_bh) if n_disc_bh > 0 else 0.0
    power_bh = (np.sum(discoveries_bh & (test_labels == 1)) / len(anomaly_idx)) if len(anomaly_idx) > 0 else 0.0

    # Procedure B: e-value + e-BH (dependence-robust)
    e_values = p_to_e_calibrator(p_values)
    discoveries_ebh = e_bh(e_values, alpha)
    n_disc_ebh = discoveries_ebh.sum()
    fdr_ebh = (np.sum(discoveries_ebh & (test_labels == 0)) / n_disc_ebh) if n_disc_ebh > 0 else 0.0
    power_ebh = (np.sum(discoveries_ebh & (test_labels == 1)) / len(anomaly_idx)) if len(anomaly_idx) > 0 else 0.0

    return {
        "p_an": p_an, "seed": seed,
        "n_disc_bh": int(n_disc_bh), "fdr_bh": fdr_bh, "power_bh": power_bh,
        "n_disc_ebh": int(n_disc_ebh), "fdr_ebh": fdr_ebh, "power_ebh": power_ebh,
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

    p_an_values = [0.002, 0.01, 0.05]  # sample across the severity range, not all 5 levels, to save compute
    all_results = []

    for p_an in p_an_values:
        print(f"=== p_an={p_an} ===")
        for seed in range(args.n_seeds):
            r = run_comparison_trial(p_an, args.alpha, seed, args.n_epochs, device)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            print(f"  seed {seed}: BH[n={r['n_disc_bh']:3d} fdr={r['fdr_bh']:.3f} pow={r['power_bh']:.3f}] "
                  f"eBH[n={r['n_disc_ebh']:3d} fdr={r['fdr_ebh']:.3f} pow={r['power_ebh']:.3f}]")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "evalue_alternative_procedure.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to {csv_path}")

    print("\n=== Summary by severity level ===")
    for p_an in p_an_values:
        subset = [r for r in all_results if r["p_an"] == p_an]
        if not subset:
            continue
        mean_fdr_bh = np.mean([r["fdr_bh"] for r in subset])
        mean_power_bh = np.mean([r["power_bh"] for r in subset])
        mean_fdr_ebh = np.mean([r["fdr_ebh"] for r in subset])
        mean_power_ebh = np.mean([r["power_ebh"] for r in subset])
        print(f"p_an={p_an}:")
        print(f"  BH:  mean_fdr={mean_fdr_bh:.4f}, mean_power={mean_power_bh:.4f}")
        print(f"  eBH: mean_fdr={mean_fdr_ebh:.4f}, mean_power={mean_power_ebh:.4f}")
        power_retained = (mean_power_ebh / mean_power_bh * 100) if mean_power_bh > 0 else float("nan")
        print(f"  Power retained by eBH vs BH: {power_retained:.1f}%")

    print("\nInterpretation: e-BH's FDR should stay controlled REGARDLESS of dependence "
          "structure (this is the whole point -- no PRDS needed). The key practical question "
          "is the power cost: if eBH retains most of BH's power, it becomes the honest, "
          "provably-valid procedure to report instead of chasing a PRDS proof for standard BH.")


if __name__ == "__main__":
    main()