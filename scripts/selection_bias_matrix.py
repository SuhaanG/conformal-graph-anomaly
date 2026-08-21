"""
selection_bias_matrix.py

THE FALSIFICATION TEST for theory/joint_discovery_threshold_proposition.md
Part 4. Nothing else in this repo can kill that theorem; this can.

THE CLAIM. Part 4 says the clean-condition FDR inflation is caused by
selection: "zero anomalous neighbors" is a topological filter that selects
low-degree nodes, degree-sensitive scores inherit the tilt, null conformal
p-values end up stochastically dominated by Uniform, and BH controls only at
gamma*alpha.

THE TEST. If that mechanism is real, the size of the violation must scale with
how degree-sensitive each detector actually is. So across a detector x dataset
matrix, measure both axes per cell:

    x-axis   Spearman(score, degree) among normal nodes  -- the cause
    y-axis   gamma statistics under the clean condition  -- the effect

and check whether they correlate ACROSS cells. This varies the proposed cause
while holding the graph and the selection filter fixed, which is the only way
to distinguish "degree sensitivity causes the inflation" from "dominant_pygod
happens to do both."

WHY REAL GRAPHS ONLY. Every number in Part 3 came from synthetic data, and
"does this reproduce on real graphs?" is the single easiest reason to reject
the paper. Running the matrix on amazon/reddit/tolokers/weibo answers the
falsification question and the replication question in one pass.

DEGREE NORMALIZATION IS OFF BY DEFAULT, AND THAT IS DELIBERATE.
degree_normalize_scores() divides by log1p(degree), i.e. it directly suppresses
the x-axis of this experiment. Running the matrix with it on would partially
control away the very quantity being measured. --degree_norm on is still
exposed, because it doubles as a control: if the mechanism is real, turning
normalization ON should reduce BOTH Spearman and gamma together.

INTERPRETING THE OUTPUT. gamma is NOT compared against 1.0 -- see
src/selection_bias.py, which documents why a valid procedure does not produce
gamma = 1 when every test point shares one calibration draw. Each statistic is
calibrated against a simulated exchangeable null at the same n_calib/n_null,
and `*_null_p` is the number to read.

Run:
  python scripts/selection_bias_matrix.py --n_seeds 5 --device cuda
  python scripts/selection_bias_matrix.py --datasets amazon reddit --detectors dominant_pygod gae

Output: results/logs/selection_bias_matrix.csv (one row per detector x dataset
x seed), plus a cross-cell correlation report printed at the end.
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

from detectors import available_detectors
from selection_bias import (
    anticonservativeness,
    exchangeable_null_pvalues,
    bh_threshold_from_rejections,
    score_degree_dependence,
    empirical_clean_probability,
    default_min_rank,
    left_tail_gamma,
)
from real_data_experiment import (
    load_any_dataset,
    run_real_data_trial,
    SUPPORTED_DATASETS,
)

# Yelp is excluded from the default set: it needs --use_sparse_prop and runs
# ~65 min per call through the frozen normalization path. Pass it explicitly
# with --use_sparse_prop if there is time.
DEFAULT_DATASETS = ["amazon", "reddit", "tolokers", "weibo"]

# The statistics whose cross-cell correlation with score-degree dependence is
# the actual test. gamma_at_bh is carried for interpretation but excluded from
# the headline: its exchangeable-null spread is [0.647, 1.440] against an
# effect near 1.32, so it cannot separate (measured; see selection_bias.py).
# gamma_t0.01 FIRST: it is the only one of these measured where BH actually
# cuts. gamma_hat and ks_uniform are sup statistics over ranks >= n_calib//4,
# which on weibo was 112x further into the bulk than BH's realized threshold --
# the mismatch that produced a false confirmation on the first run. They are
# kept as supporting evidence about the bulk, not as the test.
HEADLINE_STATS = ["gamma_t0.01", "mean_p", "ks_uniform", "gamma_hat"]


def run_cell(dataset, detector, seed, args, cached_graph):
    """One (dataset, detector, seed) cell under the CLEAN condition.

    Returns a row dict, or None if the clean pool was too small for the frame
    to be built (run_real_data_trial's own guard, which fires when fewer than
    20 normal nodes have zero anomalous neighbors).
    """
    graph, features, labels = cached_graph

    diag = {}
    trial = run_real_data_trial(
        graph, features, labels,
        contamination_condition="clean",
        alpha=args.alpha, seed=seed, n_epochs=args.n_epochs, device=args.device,
        use_degree_norm=args.degree_norm == "on",
        use_sparse_prop=args.use_sparse_prop,
        detector=detector,
        diagnostics_out=diag,
    )
    if trial is None or not diag:
        return None

    scores = diag["scores"]
    normal_idx = diag["normal_idx"]
    degrees_normal = np.array([graph.degree(int(i)) for i in normal_idx], dtype=float)

    # --- x-axis: how degree-sensitive is this detector, on this graph? ---
    dep = score_degree_dependence(scores[normal_idx], degrees_normal)

    # --- y-axis: how far from exchangeable is the clean calibration set? ---
    n_calib = diag["n_calib"]
    null_p = diag["null_p_values"]
    n_null = len(null_p)
    bh_t = bh_threshold_from_rejections(
        diag["n_discoveries"], m_test=len(diag["test_idx"]), alpha=args.alpha)

    obs = anticonservativeness(null_p, n_calib, bh_threshold=bh_t)
    # Measured at the BH operating point, NOT at gamma_hat's ranks. See
    # left_tail_gamma's docstring: on the first run gamma_hat sat 112x further
    # into the bulk than BH's realized threshold on weibo, which is what
    # produced a false 'consistent with Part 4' verdict.
    tail = left_tail_gamma(null_p)
    null = exchangeable_null_pvalues(
        obs, n_calib, n_null, bh_threshold=bh_t,
        min_rank=obs["min_rank_used"], n_sim=args.n_sim, seed=10_000 + seed)

    # --- (A1): is q(d) = P(clean-eligible | degree) actually non-increasing? ---
    # Part 4 needs only monotonicity, but the (1-pi)^d closed form assumes
    # independent attachment, which is false on these graphs. So measure it.
    clean_eligible = np.isin(normal_idx, diag["calib_idx"])
    q = empirical_clean_probability(degrees_normal, clean_eligible,
                                    n_bins=args.n_degree_bins)

    row = {
        "dataset": dataset,
        "detector": detector,
        "seed": seed,
        "alpha": args.alpha,
        "degree_norm": args.degree_norm,
        "n_calib": n_calib,
        "n_null": n_null,
        "m_test": trial["m_test"],
        "n_discoveries": trial["n_discoveries"],
        "realized_fdr": trial["realized_fdr"],
        "power": trial["power"],
        # x-axis
        "spearman_score_degree": dep["spearman_r"],
        "spearman_p": dep["spearman_p"],
        # y-axis
        "min_rank_used": obs["min_rank_used"],
        "gamma_hat": obs["gamma_hat"],
        "gamma_at_bh": obs["gamma_at_bh"],
        "mean_p": obs["mean_p"],
        "ks_uniform": obs["ks_uniform"],
        # null calibration
        "n_sim": null["n_sim"],
        # (A1)
        "q_kendall_tau": q["kendall_tau"],
        "q_kendall_p": q["kendall_p"],
        "q_is_monotone": q["is_monotone"],
        "q_first_bin": float(q["q"][0]) if len(q["q"]) else np.nan,
        "q_last_bin": float(q["q"][-1]) if len(q["q"]) else np.nan,
    }
    for k in ("gamma_hat", "gamma_at_bh", "mean_p", "ks_uniform"):
        row[f"{k}_null_mean"] = null[f"{k}_null_mean"]
        row[f"{k}_null_p"] = null[f"{k}_null_p"]
    row.update(tail)
    return row


def block_permutation_p(x, y, blocks, n_perm=10000, seed=0, negative=False):
    """Spearman p-value that respects the crossed design.

    The 20 cells are 5 detectors x 4 datasets, so they are NOT 20 independent
    units, and scipy's Spearman p-value -- which assumes they are -- is
    anti-conservative here. This permutes y only WITHIN blocks (detector, or
    dataset), which preserves the block structure under the null and gives a
    valid p-value for "does the association survive once detector identity, or
    dataset identity, is held fixed?"

    negative=True when the expected association is negative (mean_p).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    blocks = np.asarray(blocks)
    obs = stats.spearmanr(x, y).statistic
    if not np.isfinite(obs):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx_by_block = [np.where(blocks == b)[0] for b in np.unique(blocks)]
    count = 0
    for _ in range(n_perm):
        yp = y.copy()
        for ix in idx_by_block:
            yp[ix] = rng.permutation(y[ix])
        r = stats.spearmanr(x, yp).statistic
        if np.isfinite(r) and ((r <= obs) if negative else (r >= obs)):
            count += 1
    return float(obs), float((count + 1) / (n_perm + 1))


def within_group_report(cell_rows, group_key, other_key, stat, negative):
    """Correlation computed WITHIN each level of group_key, across other_key.

    This is the unit of analysis that is not confounded by group identity. The
    pooled n=20 correlation can be driven by detectors differing on both axes
    for reasons unrelated to the mechanism; correlating within a detector,
    across datasets, cannot be.

    Reports each group's rho plus an exact sign test over the groups, which is
    the honest aggregate given only a handful of points per group.
    """
    rhos = {}
    for g in sorted({k[0] if group_key == "dataset" else k[1] for k in cell_rows}):
        pts = [(v["spearman_score_degree"], v[stat]) for k, v in cell_rows.items()
               if (k[0] if group_key == "dataset" else k[1]) == g]
        pts = [(a, b) for a, b in pts if np.isfinite(a) and np.isfinite(b)]
        if len(pts) < 3:
            rhos[g] = np.nan
            continue
        a, b = zip(*pts)
        rhos[g] = stats.spearmanr(a, b).statistic
    finite = [r for r in rhos.values() if np.isfinite(r)]
    n_right = sum((r < 0) if negative else (r > 0) for r in finite)
    # Exact one-sided sign test against p=0.5.
    sign_p = (stats.binomtest(n_right, len(finite), 0.5, alternative="greater").pvalue
              if finite else np.nan)
    return rhos, n_right, len(finite), sign_p


def report(rows, args):
    """Cross-cell correlation between the proposed cause and the effect."""
    print("\n" + "=" * 78)
    print("PART 4 PREDICTION 2: does gamma track score-degree dependence?")
    print("=" * 78)

    if not rows:
        print("No cells completed. Nothing to test.")
        return

    # Aggregate to one point per (dataset, detector) cell -- seeds within a
    # cell are not independent replicates of the CAUSE (same detector, same
    # graph), so correlating at seed level would inflate n and manufacture
    # significance.
    cells = {}
    for r in rows:
        cells.setdefault((r["dataset"], r["detector"]), []).append(r)

    print(f"\n{len(cells)} cells, {len(rows)} trials "
          f"(degree_norm={args.degree_norm})\n")
    print(f"{'dataset':<10} {'detector':<16} {'spearman':>9} {'mean_p':>8} "
          f"{'ks':>7} {'gamma':>7} {'disc':>6} {'fdr':>6}")
    print("-" * 78)

    xs, ys = [], {s: [] for s in HEADLINE_STATS}
    for (ds, det), rs in sorted(cells.items()):
        def avg(k):
            v = [r[k] for r in rs if np.isfinite(r[k])]
            return float(np.mean(v)) if v else np.nan
        sp = avg("spearman_score_degree")
        print(f"{ds:<10} {det:<16} {sp:>9.4f} {avg('mean_p'):>8.4f} "
              f"{avg('ks_uniform'):>7.4f} {avg('gamma_hat'):>7.4f} "
              f"{avg('n_discoveries'):>6.0f} {avg('realized_fdr'):>6.3f}")
        if np.isfinite(sp):
            xs.append(sp)
            for s in HEADLINE_STATS:
                ys[s].append(avg(s))

    n = len(xs)
    print(f"\nCross-cell rank correlation against Spearman(score, degree), n={n}")
    if n < 5:
        print(f"  n={n} is too small to correlate. Part 4 prediction 2 is "
              f"UNTESTED, not supported.")
        return

    print(f"  {'statistic':<14} {'rho':>8} {'p':>10}   expected direction")
    verdicts = []
    for s in HEADLINE_STATS:
        y = np.array(ys[s], dtype=float)
        x = np.array(xs, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            print(f"  {s:<14} {'--':>8} {'--':>10}   insufficient finite values")
            continue
        rho, pv = stats.spearmanr(x[ok], y[ok])
        # mean_p falls as anti-conservativeness rises; the others rise.
        want_negative = (s == "mean_p")
        direction = "negative" if want_negative else "positive"
        agrees = (rho < 0) if want_negative else (rho > 0)
        verdicts.append(bool(agrees and pv < 0.05))
        print(f"  {s:<14} {rho:>8.4f} {pv:>10.4f}   {direction}"
              f"{'  <-- agrees' if agrees else '  <-- WRONG SIGN'}")

    # --- The pooled correlation above is NOT sufficient on its own. ---
    # 5 detectors x 4 datasets are not 20 independent units, so the p-values
    # printed above are anti-conservative. And a block of conservative cells at
    # one end plus one extreme detector at the other can carry a rank
    # correlation on their own. Both checks below were added after a review
    # found the pooled result did not survive them.
    cellmap = {}
    for (ds, det), rs in cells.items():
        cellmap[(ds, det)] = {
            k: float(np.mean([r[k] for r in rs if np.isfinite(r[k])]))
            if any(np.isfinite(r[k]) for r in rs) else np.nan
            for k in ["spearman_score_degree"] + HEADLINE_STATS}

    print()
    print("-" * 78)
    print("BLOCK PERMUTATION (valid under the crossed design)")
    print("-" * 78)
    keys = sorted(cellmap)
    xv = [cellmap[k]["spearman_score_degree"] for k in keys]
    print(f"  {'statistic':<14} {'rho':>8} {'perm p':>9} {'perm p':>9}")
    print(f"  {'':<14} {'':>8} {'(w/in det)':>9} {'(w/in data)':>9}")
    for st in HEADLINE_STATS:
        yv = [cellmap[k][st] for k in keys]
        neg = (st == "mean_p")
        rho, p_det = block_permutation_p(xv, yv, [k[1] for k in keys],
                                         n_perm=args.n_perm, negative=neg)
        _, p_ds = block_permutation_p(xv, yv, [k[0] for k in keys],
                                      n_perm=args.n_perm, negative=neg)
        print(f"  {st:<14} {rho:>8.4f} {p_det:>9.4f} {p_ds:>9.4f}")

    print()
    print("-" * 78)
    print("WITHIN-DETECTOR (not confounded by detector identity)")
    print("-" * 78)
    within_ok = []
    for st in HEADLINE_STATS:
        neg = (st == "mean_p")
        rhos, n_right, n_tot, sign_p = within_group_report(
            cellmap, "detector", "dataset", st, neg)
        detail = "  ".join(f"{g}={r:+.2f}" if np.isfinite(r) else f"{g}=na"
                           for g, r in sorted(rhos.items()))
        print(f"  {st:<14} {n_right}/{n_tot} right-signed, sign-test p={sign_p:.4f}")
        print(f"      {detail}")
        within_ok.append(bool(np.isfinite(sign_p) and sign_p < 0.05))

    print()
    if all(verdicts) and verdicts and all(within_ok):
        print("  VERDICT: consistent with Part 4, and it survives both the block")
        print("  permutation and the within-detector check. Part 4 may be written")
        print("  up as Theorem 2.")
    elif all(verdicts) and verdicts:
        print("  VERDICT: pooled correlation agrees, but it does NOT fully survive")
        print("  the block permutation / within-detector checks. That is the exact")
        print("  objection a referee will raise. Report the within-detector result")
        print("  as primary and the pooled n=20 as support -- not the reverse --")
        print("  and do not call this a confirmed theorem yet.")
    elif any(verdicts):
        print("  VERDICT: MIXED. Some statistics track, others do not. Report this")
        print("  honestly -- a partial result is not a confirmation, and which")
        print("  statistics disagree is itself informative.")
    else:
        print("  VERDICT: NOT SUPPORTED. gamma does not track score-degree")
        print("  dependence across cells. Part 4's mechanism as stated is wrong.")
        print("  Revise the theory doc -- do not defend it. The selection effect")
        print("  may still be real; the DEGREE explanation for it is what failed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS,
                        choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--detectors", type=str, nargs="+",
                        default=available_detectors(), choices=available_detectors())
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_sim", type=int, default=200,
                        help="Exchangeable-null simulations per cell. 200 gives a "
                             "resolution floor of ~0.005 on null_p; raise it only if "
                             "a cell needs to separate p=0.01 from p=0.001.")
    parser.add_argument("--n_degree_bins", type=int, default=12)
    parser.add_argument("--n_perm", type=int, default=10000,
                        help="Permutations for the block permutation test.")
    parser.add_argument("--degree_norm", type=str, default="off",
                        choices=["on", "off"],
                        help="OFF by default and deliberately so: "
                             "degree_normalize_scores() divides by log1p(degree), "
                             "which suppresses the x-axis this experiment measures. "
                             "Turning it ON is a control -- if the mechanism is real, "
                             "it should lower Spearman and gamma together.")
    parser.add_argument("--use_sparse_prop", action="store_true",
                        help="Required for yelp (n=45,954).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    print(f"Matrix: {len(args.detectors)} detectors x {len(args.datasets)} datasets "
          f"x {args.n_seeds} seeds = "
          f"{len(args.detectors) * len(args.datasets) * args.n_seeds} trials")
    print(f"Condition: clean (the one Proposition 1 proves is exchangeable)")
    print(f"Degree normalization: {args.degree_norm}\n")

    rows = []
    for dataset in args.datasets:
        print(f"### loading {dataset}")
        cached = load_any_dataset(dataset)
        n_nodes = cached[0].number_of_nodes()
        print(f"    {n_nodes} nodes, {int(cached[2].sum())} anomalies "
              f"({cached[2].mean():.4f})")

        for detector in args.detectors:
            for seed in range(args.n_seeds):
                try:
                    row = run_cell(dataset, detector, seed, args, cached)
                except Exception as e:
                    # One detector failing on one graph must not lose the other
                    # 19 cells -- record it and continue.
                    print(f"    {detector:<16} seed {seed}: FAILED "
                          f"{type(e).__name__}: {str(e)[:70]}")
                    continue
                if row is None:
                    print(f"    {detector:<16} seed {seed}: skipped "
                          f"(clean pool < 20)")
                    continue
                rows.append(row)
                print(f"    {detector:<16} seed {seed}: "
                      f"spearman={row['spearman_score_degree']:+.4f} "
                      f"mean_p={row['mean_p']:.4f} "
                      f"(null_p={row['mean_p_null_p']:.3f}) "
                      f"ks={row['ks_uniform']:.4f} "
                      f"disc={row['n_discoveries']}")

    if not rows:
        print("\nNo cells completed successfully.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "selection_bias_matrix.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}")

    report(rows, args)


if __name__ == "__main__":
    main()
