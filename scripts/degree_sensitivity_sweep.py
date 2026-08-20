"""
degree_sensitivity_sweep.py

THE DECISIVE TEST for theory/joint_discovery_threshold_proposition.md Part 4,
replacing prediction 2 (the detector x dataset matrix).

WHY THE MATRIX WAS NOT ENOUGH. selection_bias_matrix.py varies degree
sensitivity by swapping detectors, which is observational: detector identity
moves alongside the proposed cause, so architecture, training dynamics and
score scale all vary at the same time. Worse, the coverage was degenerate --
across all 100 trials of the first run, EVERY cell with sdeg > 0.5 was
dominant_pygod. Remove that one detector and the high end of the dose-response
curve vanishes, which is why the result collapsed from rho=+0.81 (p=0.0149,
n=8) to rho=+0.50 (p=0.3910, n=5).

WHAT THIS DOES INSTEAD. Manipulate the cause directly, holding everything else
fixed. degree_normalize_scores divides by log1p(degree); this generalises the
exponent:

    score_beta = score / log1p(degree + eps) ** beta

    beta < 0    amplifies degree sensitivity
    beta = 0    raw scores, untouched
    beta = 1    the existing correction in real_data_experiment.py
    beta > 1    over-corrects into negative degree dependence

Sweeping beta on ONE detector and ONE graph gives a continuous dose-response
curve in Spearman(score, degree) with no cross-detector confound: same
architecture, same trained weights, same graph, same calibration frame. Only
the score transform moves.

THE PREDICTION, AND WHAT WOULD FALSIFY IT. If Part 4's mechanism is real,
gamma at the BH operating point should rise monotonically as beta decreases
(more degree sensitivity -> more selection tilt -> more anti-conservative null
p-values), and should pass through gamma ~ 1 near whichever beta makes the
score degree-neutral. If gamma is flat in beta, or moves independently of
sdeg, Part 4 is dead -- unambiguously, not for lack of power.

WHERE TO RUN IT. amazon and tolokers: assumption (A1) holds strongly there
(Kendall tau ~ -0.92, 25/25 seeds) and the clean filter is genuinely harsh
(2% and 9% of normals retained). Weibo is a poor choice -- (A1) FAILS there
(tau = -0.02, 0/25 seeds), so Part 4 predicts nothing on that graph and a null
result would be uninformative.

The detector is trained ONCE per (dataset, seed) and reused across every beta,
which is what makes the comparison exact rather than merely matched -- and also
what makes the sweep cheap.

Run:
  python scripts/degree_sensitivity_sweep.py --dataset amazon --n_seeds 5 --device cuda
  python scripts/degree_sensitivity_sweep.py --dataset tolokers --n_seeds 5 --device cuda

Output: results/logs/degree_sensitivity_sweep_{dataset}_{detector}.csv
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
from conformal_fdr import conformal_p_values, benjamini_hochberg
from selection_bias import (
    anticonservativeness,
    exchangeable_null_pvalues,
    bh_threshold_from_rejections,
    left_tail_gamma,
    score_degree_dependence,
    empirical_clean_probability,
)
from real_data_experiment import load_any_dataset, SUPPORTED_DATASETS

DEFAULT_BETAS = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def apply_beta(scores, degrees, beta):
    """score / log1p(degree)**beta. beta=0 returns scores unchanged.

    Matches real_data_experiment.degree_normalize_scores at beta=1 (same
    log1p(deg + 1e-8) denominator), so the sweep passes through the existing
    correction rather than through something merely similar.
    """
    if beta == 0.0:
        return np.asarray(scores, dtype=float).copy()
    denom = np.log1p(np.asarray(degrees, dtype=float) + 1e-8)
    denom = np.where(denom <= 0, 1e-8, denom)
    return np.asarray(scores, dtype=float) / (denom ** beta)


def build_clean_frame(graph, labels, scores, seed, trim_pct=0.01):
    """Replicates run_real_data_trial's CLEAN frame, verbatim in call order.

    Copied rather than imported because the sweep needs the frame built ONCE
    from the raw scores and then held fixed while beta varies -- if the frame
    were rebuilt per beta, the trimming and eligibility would shift underneath
    the comparison and beta would be changing two things at once.

    NOTE the deliberate choice: eligibility/trimming use the RAW scores, so
    every beta shares one identical calibration and test partition. Only the
    scores fed to the conformal step change.
    """
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        nb = list(graph.neighbors(int(i)))
        if nb:
            exposure[j] = sum(1 for n in nb if labels[n] == 1) / len(nb)

    rng = np.random.default_rng(seed)
    normal_scores_all = scores[normal_idx]
    cutoff = np.percentile(normal_scores_all, 100 * (1 - trim_pct))
    eligible_normal_idx = normal_idx[normal_scores_all <= cutoff]
    eligible_mask = np.isin(normal_idx, eligible_normal_idx)
    clean_pool = normal_idx[(exposure == 0) & eligible_mask]
    if len(clean_pool) < 20:
        return None

    calib_idx = clean_pool
    remaining = np.setdiff1d(eligible_normal_idx, calib_idx)
    if len(remaining) > 5000:
        remaining = rng.choice(remaining, size=5000, replace=False)
    test_idx = np.concatenate([remaining, anomaly_idx])
    test_labels = np.concatenate([np.zeros(len(remaining), dtype=int),
                                  np.ones(len(anomaly_idx), dtype=int)])
    return calib_idx, test_idx, test_labels, normal_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="amazon",
                    choices=sorted(SUPPORTED_DATASETS))
    ap.add_argument("--detector", type=str, default="dominant_pygod",
                    choices=available_detectors())
    ap.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n_epochs", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--use_sparse_prop", action="store_true")
    args = ap.parse_args()

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    print(f"{args.dataset} / {args.detector} / betas={args.betas} / "
          f"{args.n_seeds} seeds")
    print("Detector trained ONCE per seed; only the score transform varies.\n")

    graph, features, labels = load_any_dataset(args.dataset)
    print(f"{graph.number_of_nodes()} nodes, {int(labels.sum())} anomalies\n")

    rows = []
    for seed in range(args.n_seeds):
        scores_raw = score_nodes(args.detector, graph, features, labels=labels,
                                 seed=seed, n_epochs=args.n_epochs,
                                 device=args.device,
                                 use_sparse_prop=args.use_sparse_prop)
        frame = build_clean_frame(graph, labels, scores_raw, seed)
        if frame is None:
            print(f"  seed {seed}: skipped (clean pool < 20)")
            continue
        calib_idx, test_idx, test_labels, normal_idx = frame
        deg_all = np.array([graph.degree(int(i)) for i in range(graph.number_of_nodes())],
                           dtype=float)
        n_calib, m = len(calib_idx), len(test_idx)

        # (A1) is a property of the FRAME, not of beta -- report it once.
        q = empirical_clean_probability(deg_all[normal_idx],
                                        np.isin(normal_idx, calib_idx), n_bins=12)

        print(f"  seed {seed}: n_calib={n_calib} m={m} "
              f"A1_monotone={q['is_monotone']} tau={q['kendall_tau']:+.3f}")

        for beta in args.betas:
            s_b = apply_beta(scores_raw, deg_all, beta)
            dep = score_degree_dependence(s_b[normal_idx], deg_all[normal_idx])
            p = conformal_p_values(s_b[calib_idx], s_b[test_idx])
            rej = benjamini_hochberg(p, args.alpha)
            k = int(rej.sum())
            fdr = float(np.sum(rej & (test_labels == 0)) / k) if k else 0.0
            power = float(np.sum(rej & (test_labels == 1)) / max(1, int(test_labels.sum())))

            null_p = p[test_labels == 0]
            bh_t = bh_threshold_from_rejections(k, m, args.alpha)
            obs = anticonservativeness(null_p, n_calib, bh_threshold=bh_t)
            tail = left_tail_gamma(null_p)
            null = exchangeable_null_pvalues(obs, n_calib, len(null_p),
                                             bh_threshold=bh_t,
                                             min_rank=obs["min_rank_used"],
                                             n_sim=args.n_sim, seed=777 + seed)
            row = {
                "dataset": args.dataset, "detector": args.detector,
                "seed": seed, "beta": beta, "alpha": args.alpha,
                "n_calib": n_calib, "m_test": m, "n_null": len(null_p),
                "spearman_score_degree": dep["spearman_r"],
                "n_discoveries": k, "realized_fdr": fdr, "power": power,
                "gamma_hat": obs["gamma_hat"], "gamma_at_bh": obs["gamma_at_bh"],
                "mean_p": obs["mean_p"], "ks_uniform": obs["ks_uniform"],
                "mean_p_null_p": null["mean_p_null_p"],
                "q_is_monotone": q["is_monotone"],
                "q_kendall_tau": q["kendall_tau"],
            }
            row.update(tail)
            rows.append(row)
            print(f"      beta={beta:>+5.2f}  sdeg={dep['spearman_r']:+.4f}  "
                  f"g_t0.01={tail['gamma_t0.01']:>7.2f}  "
                  f"mean_p={obs['mean_p']:.4f}  disc={k:<6d} fdr={fdr:.3f}")

    if not rows:
        print("\nNo trials completed.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir,
                       f"degree_sensitivity_sweep_{args.dataset}_{args.detector}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}")

    # ---- the dose-response curve ----
    print("\n" + "=" * 78)
    print("DOSE-RESPONSE: does gamma track degree sensitivity within one detector?")
    print("=" * 78)
    print(f"{'beta':>7} {'sdeg':>9} {'g_t0.01':>9} {'g_t0.05':>9} {'mean_p':>8} "
          f"{'disc':>7} {'fdr':>7}")
    xs, ys = [], []
    for beta in args.betas:
        sub = [r for r in rows if r["beta"] == beta]
        if not sub:
            continue
        f_ = lambda k: float(np.nanmean([r[k] for r in sub]))
        sd, g1 = f_("spearman_score_degree"), f_("gamma_t0.01")
        print(f"{beta:>+7.2f} {sd:>+9.4f} {g1:>9.2f} {f_('gamma_t0.05'):>9.2f} "
              f"{f_('mean_p'):>8.4f} {f_('n_discoveries'):>7.0f} "
              f"{f_('realized_fdr'):>7.3f}")
        if np.isfinite(sd) and np.isfinite(g1):
            xs.append(sd); ys.append(g1)

    if len(xs) >= 5:
        rho, p = stats.spearmanr(xs, ys)
        print(f"\nSpearman(sdeg, gamma_t0.01) along the beta curve: "
              f"rho={rho:+.4f} p={p:.4f}  (n={len(xs)} beta levels)")
        print()
        if rho > 0 and p < 0.05:
            print("  SUPPORTS Part 4. Degree sensitivity was manipulated directly,")
            print("  with architecture, weights, graph and calibration frame held")
            print("  fixed, and the violation tracked it. This is a causal design,")
            print("  not the observational cross-detector one.")
        elif p >= 0.05:
            print("  NOT SUPPORTED. gamma does not track degree sensitivity even")
            print("  when it is manipulated directly on a fixed frame. That is the")
            print("  cleanest evidence available against Part 4's mechanism --")
            print("  revise the theory, do not defend it.")
        else:
            print("  WRONG SIGN. gamma moves OPPOSITE to the prediction. Part 4 as")
            print("  stated is wrong.")
    else:
        print(f"\nOnly {len(xs)} usable beta levels -- too few to fit a curve.")


if __name__ == "__main__":
    main()
