"""
verify_ocgnn_cola_hypothesis.py

Reads selection_bias_matrix.csv output. Groups weibo detectors by
gamma_at_bh. Reports whether cola (non-reconstruction, non-one-class)
lands near ocgnn (near-valid) or near the 4 reconstruction detectors
(gamma~7).

Run after: selection_bias_matrix.py --datasets weibo
             --detectors cola ocgnn dominant_pygod gae anomalydae dominant_ours

Usage:
  python3 scripts/verify_ocgnn_cola_hypothesis.py --csv results/logs/selection_bias_matrix.csv
"""

import argparse
import csv
import numpy as np


RECONSTRUCTION = {"dominant_ours", "dominant_pygod", "gae", "anomalydae"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    weibo = [r for r in rows if r["dataset"] == "weibo"]
    if not weibo:
        print("No weibo rows found in this CSV.")
        return

    detectors = sorted(set(r["detector"] for r in weibo))
    print(f"{'detector':<16}{'gamma_at_bh':>14}{'realized_fdr':>14}{'n_seeds':>10}")
    print("-" * 54)

    summary = {}
    for det in detectors:
        sub = [r for r in weibo if r["detector"] == det]
        gammas = [float(r["gamma_at_bh"]) for r in sub if r["gamma_at_bh"] not in ("", "nan")]
        fdrs = [float(r["realized_fdr"]) for r in sub]
        g_mean = np.mean(gammas) if gammas else float("nan")
        f_mean = np.mean(fdrs)
        summary[det] = g_mean
        print(f"{det:<16}{g_mean:>14.3f}{f_mean:>14.3f}{len(sub):>10}")

    print()
    if "cola" not in summary:
        print("cola not in this CSV -- nothing to compare.")
        return

    recon_gammas = [summary[d] for d in summary if d in RECONSTRUCTION and not np.isnan(summary[d])]
    ocgnn_gamma = summary.get("ocgnn", float("nan"))
    cola_gamma = summary["cola"]

    recon_mean = np.mean(recon_gammas) if recon_gammas else float("nan")

    print(f"reconstruction-detector mean gamma_at_bh: {recon_mean:.3f}")
    print(f"ocgnn gamma_at_bh:                        {ocgnn_gamma:.3f}")
    print(f"cola gamma_at_bh:                         {cola_gamma:.3f}")
    print()

    dist_to_recon = abs(cola_gamma - recon_mean)
    dist_to_ocgnn = abs(cola_gamma - ocgnn_gamma)

    if dist_to_ocgnn < dist_to_recon:
        print("VERDICT: cola groups with ocgnn (near-valid).")
        print("Supports: the discriminator is 'reconstruction objective, yes or no.'")
    else:
        print("VERDICT: cola groups with the reconstruction detectors (gamma~7).")
        print("Refutes the reconstruction-vs-not hypothesis.")
        print("Narrows toward something specific to one-class scoring in ocgnn.")


if __name__ == "__main__":
    main()