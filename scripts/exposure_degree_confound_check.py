"""
exposure_degree_confound_check.py

Follow-up to real_data_exposure_diagnostic.py. That script found significant
exposure-score correlations on real data (Amazon +0.270, Tolokers +0.189,
Reddit +0.022) -- which would be good news for the paper, since it suggests the
contamination mechanism operates on real data even though it does not on
synthetic. But those results contain an internal contradiction that has to be
resolved before they can be believed:

    dataset    degree_norm  exposure_r   gap(exposed - unexposed)
    amazon     True         +0.2697      -109.32     <- signs disagree
    tolokers   True         +0.1889      -587.81     <- signs disagree
    reddit     False        +0.0220        +2.00     <- signs agree

A positive correlation says "more exposure -> higher score". A large negative
gap says exposed nodes score much LOWER on average. Both cannot describe the
same clean monotone effect, and the contradiction appears in exactly and only
the two datasets where degree normalization is enabled.

THE SUSPECTED CONFOUND. Exposure and degree are mechanically linked. On Amazon
only 267 of 11,123 normal nodes have zero anomalous neighbors -- with average
degree ~740 and a 6.87% anomaly rate, a node avoids all anomalies essentially
only by having very few neighbors. So `exposure == 0` largely means "low
degree". Meanwhile degree_normalize_scores divides by log1p(degree), which
INFLATES low-degree nodes' scores (and for a genuinely isolated node,
log1p(0 + 1e-8) ~ 1e-8, inflating its score by ~1e8). Both the negative gap
and the positive correlation could therefore be measuring DEGREE rather than
contamination.

WHAT THIS SCRIPT DOES. Separates the two explanations directly:
  1. Quantifies the exposure-degree relationship (the confound itself).
  2. Recomputes the exposure-score correlation on RAW scores, before degree
     normalization -- removing the inflation mechanism entirely.
  3. Computes the PARTIAL correlation of exposure vs score controlling for
     degree (and separately for log-degree, matching the normalization's own
     functional form). This is the decisive number.
  4. Profiles the exposure==0 subgroup (size, degree, isolated-node count) to
     show whether the negative gap is driven by a small, atypical, low-degree
     group.
  5. Reports Spearman alongside Pearson throughout, since the raw scores are
     known to be heavy-tailed (the README documents Amazon normals with
     mean~1502 but max~7596).

HOW TO READ THE RESULT:
  - If the partial correlation stays clearly non-zero and significant after
    controlling for degree, the contamination effect is real on real data.
    The paper's real-data results survive, and only the stated MECHANISM needs
    rewriting (feature similarity among fraud neighborhoods, not GNN message
    passing -- since the synthetic diagnostic showed the message-passing route
    is not what produces it).
  - If the partial correlation collapses toward zero, the raw +0.270 was a
    degree artifact, real data has the same problem as synthetic, and the
    three calibration conditions are not testing what the paper says.

Run in the dgl311 env (CPU is fine at these graph sizes):
  ~/envs/dgl311/bin/python scripts/exposure_degree_confound_check.py \
      --datasets amazon reddit tolokers --n_seeds 3 --device cpu
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

from detector import train_dominant
from real_data_experiment import load_any_dataset, degree_normalize_scores, DEGREE_NORM_BY_DATASET


def compute_exposure(graph, normal_idx, labels):
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if neighbors:
            exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)
    return exposure


def partial_corr(x, y, z):
    """Pearson correlation of x and y with the linear effect of z removed from
    both. Equivalent to correlating the residuals of x~z and y~z."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if np.std(z) == 0:
        r, p = stats.pearsonr(x, y)
        return r, p
    rxy, _ = stats.pearsonr(x, y)
    rxz, _ = stats.pearsonr(x, z)
    ryz, _ = stats.pearsonr(y, z)
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0:
        return float("nan"), float("nan")
    r = (rxy - rxz * ryz) / denom
    # t-test on the partial correlation, dof = n - 3 (two variables + control)
    n = len(x)
    dof = n - 3
    if dof <= 0 or abs(r) >= 1.0:
        return float(r), float("nan")
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), dof)
    return float(r), float(p)


def analyze(dataset_name, n_seeds, n_epochs, device):
    print(f"\n{'=' * 74}\n{dataset_name.upper()}\n{'=' * 74}")
    try:
        graph, features, labels = load_any_dataset(dataset_name)
    except Exception as e:
        print(f"  SKIPPED ({type(e).__name__}: {e})")
        return []

    use_degree_norm = DEGREE_NORM_BY_DATASET.get(dataset_name, True)
    normal_idx = np.where(labels == 0)[0]
    degrees = np.array([graph.degree(int(i)) for i in normal_idx], dtype=float)
    exposure = compute_exposure(graph, normal_idx, labels)

    # --- the confound itself, independent of any model ---
    n_zero_exp = int((exposure == 0).sum())
    n_isolated = int((degrees == 0).sum())
    r_ed, p_ed = stats.pearsonr(exposure, degrees)
    rho_ed, _ = stats.spearmanr(exposure, degrees)
    deg_zero = degrees[exposure == 0]
    deg_pos = degrees[exposure > 0]

    print(f"  n_nodes={graph.number_of_nodes():,}  n_normals={len(normal_idx):,}  "
          f"n_anomalies={int(labels.sum()):,}  degree_norm={use_degree_norm}")
    print(f"  normal-node degree: min={degrees.min():.0f} median={np.median(degrees):.0f} "
          f"max={degrees.max():.0f}  isolated(deg=0)={n_isolated}")
    print(f"  exposure==0 group : {n_zero_exp:,} nodes ({n_zero_exp / len(normal_idx):.2%} of normals)")
    if n_zero_exp and len(deg_pos):
        print(f"     their degree   : median={np.median(deg_zero):.0f}  vs  "
              f"exposed median={np.median(deg_pos):.0f}   <- if these differ a lot, "
              f"the gap is a degree effect")
    print(f"  CONFOUND exposure~degree: pearson={r_ed:+.4f} (p={p_ed:.4g})  spearman={rho_ed:+.4f}")

    rows = []
    for seed in range(n_seeds):
        raw_scores, _ = train_dominant(graph, features, n_epochs=n_epochs,
                                       seed=seed, verbose=False, device=device)
        norm_scores = degree_normalize_scores(graph, raw_scores)

        s_raw = raw_scores[normal_idx]
        s_norm = norm_scores[normal_idx]
        s_used = s_norm if use_degree_norm else s_raw

        r_raw, p_raw = stats.pearsonr(exposure, s_raw)
        rho_raw, prho_raw = stats.spearmanr(exposure, s_raw)
        r_used, p_used = stats.pearsonr(exposure, s_used)

        pr_deg, pp_deg = partial_corr(exposure, s_used, degrees)
        pr_logdeg, pp_logdeg = partial_corr(exposure, s_used, np.log1p(degrees))
        pr_raw_deg, pp_raw_deg = partial_corr(exposure, s_raw, degrees)

        rows.append(dict(
            dataset=dataset_name, seed=seed, degree_norm=use_degree_norm,
            n_zero_exposure=n_zero_exp, n_isolated=n_isolated,
            exposure_degree_r=r_ed,
            r_used=r_used, p_used=p_used,
            r_raw=r_raw, p_raw=p_raw, spearman_raw=rho_raw, p_spearman_raw=prho_raw,
            partial_r_ctrl_degree=pr_deg, partial_p_ctrl_degree=pp_deg,
            partial_r_ctrl_logdegree=pr_logdeg, partial_p_ctrl_logdegree=pp_logdeg,
            partial_r_raw_ctrl_degree=pr_raw_deg, partial_p_raw_ctrl_degree=pp_raw_deg,
        ))
        print(f"  seed {seed}: r(as-used)={r_used:+.4f}  r(RAW score)={r_raw:+.4f} "
              f"(p={p_raw:.4g})  spearman(RAW)={rho_raw:+.4f}")
        print(f"          PARTIAL r | degree = {pr_deg:+.4f} (p={pp_deg:.4g})   "
              f"PARTIAL r | log-degree = {pr_logdeg:+.4f} (p={pp_logdeg:.4g})")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["amazon", "reddit", "tolokers"])
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    all_rows = []
    for name in args.datasets:
        all_rows.extend(analyze(name, args.n_seeds, args.n_epochs, device))

    if not all_rows:
        print("\nNo results.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "exposure_degree_confound_check.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {csv_path}")

    print(f"\n{'=' * 74}\nSUMMARY -- does the correlation survive controlling for degree?\n{'=' * 74}")
    print(f"{'dataset':10} {'exp~deg':>9} {'r(as-used)':>11} {'r(RAW)':>9} "
          f"{'PARTIAL|deg':>12} {'PARTIAL|logdeg':>15}")
    for name in args.datasets:
        rows = [r for r in all_rows if r["dataset"] == name]
        if not rows:
            continue
        g = lambda k: np.mean([r[k] for r in rows])
        print(f"{name:10} {g('exposure_degree_r'):+9.4f} {g('r_used'):+11.4f} "
              f"{g('r_raw'):+9.4f} {g('partial_r_ctrl_degree'):+12.4f} "
              f"{g('partial_r_ctrl_logdegree'):+15.4f}")

    print("""
Reading the two rightmost columns (the decisive ones):
  - Partial r stays clearly non-zero and significant  -> the contamination
    effect on real data is REAL, not a degree artifact. The paper's real-data
    results survive; only the stated mechanism needs rewriting (feature
    similarity in fraud neighborhoods, not GNN message passing).
  - Partial r collapses toward zero                   -> the raw correlation
    was degree, not contamination. Real data has the same problem as
    synthetic, and the three calibration conditions are not testing what the
    paper claims.

Also compare r(RAW) against r(as-used): a large difference means degree
normalization itself is manufacturing or destroying the apparent effect.
""")


if __name__ == "__main__":
    main()
