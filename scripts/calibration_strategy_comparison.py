"""
calibration_strategy_comparison.py

THE CENTRAL EXPERIMENT. Tests whether the exchangeability failure is caused by
the calibration SELECTION RULE itself, by varying only that rule and holding
literally everything else fixed.

THE ARGUMENT BEING TESTED. Under the "clean" condition the frame is:

    calibration  C = {normal : exposure == 0}      (all of them)
    test-normals T = {normal : exposure  > 0}      (all of them)

C and T are DISJOINT IN EXPOSURE by construction. Conformal validity requires
S|C ==d S|T, so if the score depends on exposure at all, validity fails by
design. Contrapositive: clean calibration is valid only if the score is
insensitive to exposure -- exactly the case where contamination was never a
problem. The strategy is valid when it is unnecessary and invalid when it is
needed.

Degree is not a separate mechanism, it is a second channel to the same
failure: P(exposure == 0 | degree) decreases in degree, so filtering on
exposure also filters on degree, and a degree-sensitive score breaks even when
exposure-sensitivity is zero. That is why amazon and tolokers (harsh filters,
2% and 9% of normals retained) fail through degree, while weibo (70% retained,
(A1) flat at tau=-0.02) fails through exposure instead.

WHAT THIS SCRIPT FIXES ABOUT THE EXISTING COMPARISON. real_data_experiment.py
already runs clean vs contaminated, but it is CONFOUNDED: clean uses
n_calib = len(clean_pool) while contaminated uses min(4000, n_eligible). A
different n_calib means a different p-value resolution floor 1/(n+1) and a
different bh_min_rank, which is the exact confound that invalidated this
project's original Method B/C. Here:

  - the TEST SET is drawn once and held IDENTICAL across strategies
  - n_calib is IDENTICAL across strategies
  - therefore m, the floor, and bh_min_rank are identical
  - the ONLY thing that varies is which normals become calibration

STRATEGIES

  clean          calibration drawn from {exposure == 0}. The "safe" choice,
                 and the one predicted to break.
  random         calibration drawn from ALL eligible normals. Exchangeable
                 with the test set by construction. Predicted gamma ~ 1.
                 This is the control the project has never run at matched n.
  exposed_only   calibration drawn from {exposure > 0}. Distinguishes "any
                 exposure-based filter shifts the distribution" from "removing
                 contamination specifically helps". If clean and exposed_only
                 BOTH break, the problem is the filtering, not the direction.

PREDICTIONS, stated before running:
  1. random gives gamma ~ 1 and FDR ~ alpha.
  2. clean gives gamma > 1 and FDR >> alpha.
  3. The size of the clean-vs-random gap tracks how much the score responds to
     exposure and degree.
  4. exposed_only is ALSO broken (in the opposite direction, conservative),
     because it is equally a filter.
  5. TRUE contamination (actual anomalies in calibration) is SAFE: high-scoring
     anomalies in calibration make test anomalies face stiffer competition, so
     p-values rise, power falls, and FDR stays at or below nominal. If 5 holds
     alongside 2, the paper's finding is sharp -- the failure mode everyone
     guards against is benign, and the standard guard is what breaks FDR.

If (1) fails -- if random calibration is also broken -- then the problem is
not the selection rule and this whole line is wrong.

NOTE THIS IS A CONSERVATIVE VERSION OF THE ORIGINAL COMPARISON. In
real_data_experiment.py's clean condition the test-normals are ALL exposed
(they are exactly the complement of the clean pool), so calibration and test
are maximally separated in exposure. Here the test set is drawn at random from
all eligible normals, so it is a MIXTURE of exposed and unexposed. The shift
between calibration and test is therefore SMALLER than in the pipeline, and
any effect measured here understates the pipeline's.

That is deliberate. Holding the test population fixed at the real normal
population is what makes "which calibration rule gives valid inference about
the population you actually care about" a well-posed question. If the effect
survives this weaker version, it is not an artifact of the pipeline's
particular partition.

Run:
  python scripts/calibration_strategy_comparison.py --dataset amazon --n_seeds 5 --device cuda
  python scripts/calibration_strategy_comparison.py --dataset weibo  --n_seeds 5 --device cuda

Output: results/logs/calibration_strategy_{dataset}_{detector}.csv
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
    adaptive_t_grid,
    score_degree_dependence,
)
from real_data_experiment import load_any_dataset, SUPPORTED_DATASETS

# "true_contam_XX" injects ACTUAL ANOMALIES into calibration at rate XX%.
# This is what "contaminated calibration" means in Bates et al. 2023 and
# AdaDetect, and it is the experiment this project has never run: every
# condition in real_data_experiment.py draws calibration from labels == 0, so
# no anomaly has ever entered a calibration set here. See theory doc Part 6.
# random_full is NOT in the matched frame, deliberately. Matched n isolates the
# mechanism but handicaps random to the clean pool's size (n=225 on amazon),
# where BH needs 125 floor-tied points to reject anything and NO strategy can
# discover. In deployment only `clean` is actually limited to its pool -- random
# can use every eligible normal. Reported separately with in_matched_frame=False.
STRATEGIES = ["clean", "random", "exposed_only", "true_contam_05", "true_contam_10"]
UNMATCHED = ["random_full"]

TRUE_CONTAM_RATES = {"true_contam_05": 0.05, "true_contam_10": 0.10}


def compute_exposure(graph, labels, normal_idx):
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        nb = list(graph.neighbors(int(i)))
        if nb:
            exposure[j] = sum(1 for n in nb if labels[n] == 1) / len(nb)
    return exposure


def standardized_gap(a, b):
    """Cohen's d between calibration and test-normal scores.

    This is the quantity conformal validity actually depends on -- not the
    global score-covariate correlation the beta sweep used as its x-axis, which
    turned out to be the wrong summary (gamma crossed 1 at sdeg=+0.67, not 0).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / max(1, len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def run_seed(graph, features, labels, seed, args):
    scores = score_nodes(args.detector, graph, features, labels=labels, seed=seed,
                         n_epochs=args.n_epochs, device=args.device,
                         use_sparse_prop=args.use_sparse_prop)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    degrees = np.array([graph.degree(int(i)) for i in range(graph.number_of_nodes())],
                       dtype=float)
    exposure = compute_exposure(graph, labels, normal_idx)

    rng = np.random.default_rng(seed)

    # Trim exactly as the frozen pipeline does, so eligibility matches.
    ns = scores[normal_idx]
    cutoff = np.percentile(ns, 100 * (1 - args.trim_pct))
    eligible = normal_idx[ns <= cutoff]
    elig_mask = np.isin(normal_idx, eligible)

    unexposed = normal_idx[(exposure == 0) & elig_mask]
    exposed = normal_idx[(exposure > 0) & elig_mask]

    # --- ONE test set, drawn once, shared by every strategy ---
    # Sampled from ALL eligible normals so it is not itself exposure-filtered;
    # a test set drawn only from exposed nodes would build the shift into the
    # comparison and guarantee the result.
    n_test_normal = min(args.n_test_normal, len(eligible) // 2)
    test_normal = rng.choice(eligible, size=n_test_normal, replace=False)
    test_idx = np.concatenate([test_normal, anomaly_idx])
    test_labels = np.concatenate([np.zeros(len(test_normal), dtype=int),
                                  np.ones(len(anomaly_idx), dtype=int)])

    # --- calibration pools, disjoint from the test set ---
    pools = {
        "clean": np.setdiff1d(unexposed, test_normal),
        "random": np.setdiff1d(eligible, test_normal),
        "exposed_only": np.setdiff1d(exposed, test_normal),
    }
    # n_calib is the binding constraint: the smallest pool caps everyone, so
    # every strategy gets the SAME n and therefore the same p-value floor.
    n_calib = min(len(p) for p in pools.values())
    if n_calib < 50:
        return None, f"smallest calibration pool is {n_calib} (<50)"

    dep_deg = score_degree_dependence(scores[normal_idx], degrees[normal_idx])
    ok = np.isfinite(scores[normal_idx]) & np.isfinite(exposure)
    r_exp, p_exp = (stats.spearmanr(scores[normal_idx][ok], exposure[ok])
                    if ok.sum() > 3 else (np.nan, np.nan))

    # Anomalies not already spent on the test set are the injection source.
    # The test set keeps every anomaly (that is the frozen pipeline's design),
    # so injection draws WITH the test anomalies excluded would leave nothing.
    # Instead inject copies-by-index from the anomaly pool: calibration and test
    # then share some anomalies, which is exactly the transductive setting the
    # rest of the pipeline already assumes.
    anom_pool = anomaly_idx

    t_grid = adaptive_t_grid(n_calib)
    out = []
    for strat in STRATEGIES + UNMATCHED:
        if strat == "random_full":
            n_here = min(args.n_calib_full, len(pools["random"]))
            calib_idx = rng.choice(pools["random"], size=n_here, replace=False)
        elif strat in TRUE_CONTAM_RATES:
            eps = TRUE_CONTAM_RATES[strat]
            n_anom = int(round(eps * n_calib))
            n_norm = n_calib - n_anom
            if n_anom < 1 or n_anom > len(anom_pool):
                continue
            calib_idx = np.concatenate([
                rng.choice(pools["random"], size=n_norm, replace=False),
                rng.choice(anom_pool, size=n_anom, replace=False)])
        else:
            calib_idx = rng.choice(pools[strat], size=n_calib, replace=False)

        n_here = len(calib_idx)
        grid_here = t_grid if strat not in UNMATCHED else adaptive_t_grid(n_here)
        p = conformal_p_values(scores[calib_idx], scores[test_idx])
        rej = benjamini_hochberg(p, args.alpha)
        k = int(rej.sum())
        fdr = float(np.sum(rej & (test_labels == 0)) / k) if k else 0.0
        power = float(np.sum(rej & (test_labels == 1)) / max(1, int(test_labels.sum())))

        null_p = p[test_labels == 0]
        bh_t = bh_threshold_from_rejections(k, len(test_idx), args.alpha)
        obs = anticonservativeness(null_p, n_here, bh_threshold=bh_t)
        tail = left_tail_gamma(null_p, t_grid=grid_here)
        null = exchangeable_null_pvalues(obs, n_here, len(null_p),
                                         bh_threshold=bh_t,
                                         min_rank=obs["min_rank_used"],
                                         n_sim=args.n_sim, seed=4242 + seed)

        row = {
            "dataset": args.dataset, "detector": args.detector, "seed": seed,
            "strategy": strat, "alpha": args.alpha,
            "n_calib": n_here, "m_test": len(test_idx), "n_null": len(null_p),
            "in_matched_frame": strat not in UNMATCHED,
            "bh_min_rank": float(len(test_idx) / (args.alpha * (n_here + 1))),
            "calib_frac_anomalous": float(np.mean(labels[calib_idx] == 1)),
            "calib_mean_exposure": float(
                exposure[np.isin(normal_idx, calib_idx)].mean())
                if np.any(np.isin(normal_idx, calib_idx)) else float("nan"),
            "calib_mean_degree": float(degrees[calib_idx].mean()),
            "test_mean_degree": float(degrees[test_normal].mean()),
            "score_gap_cohens_d": standardized_gap(scores[calib_idx], scores[test_normal]),
            "ks_calib_vs_test": float(stats.ks_2samp(scores[calib_idx],
                                                     scores[test_normal]).statistic),
            "spearman_score_degree": dep_deg["spearman_r"],
            "spearman_score_exposure": float(r_exp),
            "spearman_score_exposure_p": float(p_exp),
            "gamma_t_lo": tail[f"gamma_t{grid_here[0]:g}"],
            "t_lo": grid_here[0],
            "gamma_hat": obs["gamma_hat"], "mean_p": obs["mean_p"],
            "ks_uniform": obs["ks_uniform"],
            "mean_p_null_p": null["mean_p_null_p"],
            "n_discoveries": k, "realized_fdr": fdr, "power": power,
        }
        out.append(row)

    # The whole comparison rests on these being identical across strategies:
    # a different n_calib means a different p-value floor 1/(n+1) and a
    # different bh_min_rank, which is precisely the confound that invalidated
    # this project's original Method B/C. Assert rather than trust.
    matched = [r for r in out if r["in_matched_frame"]]
    for key in ("n_calib", "m_test", "n_null", "t_lo"):
        vals = {r[key] for r in matched}
        if len(vals) != 1:
            raise AssertionError(
                f"seed {seed}: {key} differs across strategies ({vals}). "
                f"The frame is not matched and the comparison is invalid.")
    return out, None


def report(rows, args):
    print()
    print("=" * 78)
    print("DOES THE CALIBRATION SELECTION RULE CAUSE THE FAILURE?")
    print("=" * 78)
    print("n_calib, test set, m and the p-value floor are IDENTICAL across")
    print("strategies. Only the rule choosing calibration nodes differs.\n")

    print(f"{'strategy':<14}{'calib_exp':>10}{'calib_deg':>10}{'test_deg':>9}"
          f"{'gap(d)':>8}{'gamma':>8}{'mean_p':>8}{'disc':>7}{'fdr':>7}{'power':>7}")
    print("-" * 78)
    agg = {}
    for strat in STRATEGIES + UNMATCHED:
        sub = [r for r in rows if r["strategy"] == strat]
        if not sub:
            continue
        f_ = lambda k: float(np.nanmean([r[k] for r in sub]))
        agg[strat] = {k: f_(k) for k in
                      ("gamma_t_lo", "mean_p", "realized_fdr", "power",
                       "score_gap_cohens_d", "n_discoveries", "n_calib",
                       "bh_min_rank")}
        print(f"{strat:<14}{f_('calib_mean_exposure'):>10.4f}"
              f"{f_('calib_mean_degree'):>10.1f}{f_('test_mean_degree'):>9.1f}"
              f"{f_('score_gap_cohens_d'):>8.3f}{f_('gamma_t_lo'):>8.2f}"
              f"{f_('mean_p'):>8.4f}{f_('n_discoveries'):>7.0f}"
              f"{f_('realized_fdr'):>7.3f}{f_('power'):>7.3f}")

    sd = float(np.nanmean([r["spearman_score_degree"] for r in rows]))
    se = float(np.nanmean([r["spearman_score_exposure"] for r in rows]))
    print(f"\nscore-degree Spearman = {sd:+.4f}   "
          f"score-exposure Spearman = {se:+.4f}")

    if "random" not in agg or "clean" not in agg:
        print("\nMissing a strategy; cannot conclude.")
        return

    g_rand, g_clean = agg["random"]["gamma_t_lo"], agg["clean"]["gamma_t_lo"]
    f_rand, f_clean = agg["random"]["realized_fdr"], agg["clean"]["realized_fdr"]

    print()
    print("-" * 78)
    print(f"  random calibration:  gamma={g_rand:.2f}  FDR={f_rand:.3f}")
    print(f"  clean  calibration:  gamma={g_clean:.2f}  FDR={f_clean:.3f}")
    print()

    d_rand = agg["random"]["n_discoveries"]
    d_clean = agg["clean"]["n_discoveries"]

    # A valid-looking FDR from zero discoveries is meaningless. BH scores a
    # zero-discovery trial as FDR=0, which is how degree normalization once
    # "fixed" the problem (Part 3) and how the beta sweep once "fixed" it
    # (Part 5). Separate the exchangeability claim from the remedy claim.
    exch_ok = 0.5 <= g_rand <= 2.0
    clean_broken = (g_clean > 2.0) or (f_clean > args.alpha * 2)

    print("  --- exchangeability (does the selection rule cause the violation?) ---")
    if exch_ok and clean_broken:
        print("  SUPPORTED. At the SAME n_calib, same test set and same p-value")
        print("  floor, random calibration is exchangeable and the exposure-filtered")
        print("  rule is not. gamma is computed from null p-values and does not")
        print("  depend on discovery counts, so this holds regardless of power.")
    elif not exch_ok:
        print("  REFUTED. Random calibration is ALSO non-exchangeable, so the")
        print("  selection rule is not the cause. Look at the detector and the")
        print("  score distribution before continuing this line.")
    else:
        print("  INCONCLUSIVE. Random is exchangeable but clean is not clearly")
        print("  broken here -- check how much the filter actually bites.")

    print()
    print("  --- remedy (is random calibration actually USABLE?) ---")
    if d_rand < 1:
        print(f"  NOT ESTABLISHED. random produced {d_rand:.0f} discoveries, so its")
        print(f"  FDR of {f_rand:.3f} is the zero-discovery artifact, not FDR control.")
        print(f"  At n_calib={agg['random'].get('n_calib', float('nan')):.0f}, BH needs "
              f"~{agg['random'].get('bh_min_rank', float('nan')):.0f} floor-tied points")
        print("  before it can reject anything, so NO strategy could discover here.")
        print("  The matched-n design isolates the mechanism but handicaps random:")
        print("  only `clean` is genuinely limited to its pool. See random_full.")
    elif f_rand <= args.alpha * 1.5:
        print(f"  SUPPORTED. random discovers ({d_rand:.0f}) AND controls FDR")
        print(f"  ({f_rand:.3f} vs nominal {args.alpha}). Accepting contamination into")
        print("  calibration is both safer and usable.")
    else:
        print(f"  FAILED. random discovers but does not control FDR ({f_rand:.3f}).")

    if "random_full" in agg:
        rf = agg["random_full"]
        print()
        print("  --- random_full (unmatched n: what random achieves in deployment) ---")
        print(f"  n_calib={rf.get('n_calib', float('nan')):.0f} "
              f"gamma={rf['gamma_t_lo']:.2f} disc={rf['n_discoveries']:.0f} "
              f"FDR={rf['realized_fdr']:.3f} power={rf['power']:.3f}")
        if rf["n_discoveries"] >= 1 and rf["realized_fdr"] <= args.alpha * 1.5:
            print("  Random calibration at its natural size is valid AND useful.")
            print("  That is the deployable recommendation: do not filter.")
        elif rf["n_discoveries"] < 1:
            print("  Still no discoveries even unhandicapped -- the limit is the")
            print("  detector or the graph, not the calibration rule.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="amazon",
                    choices=sorted(SUPPORTED_DATASETS))
    ap.add_argument("--detector", type=str, default="dominant_pygod",
                    choices=available_detectors())
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n_epochs", type=int, default=100)
    ap.add_argument("--n_test_normal", type=int, default=2000)
    ap.add_argument("--trim_pct", type=float, default=0.01)
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--n_calib_full", type=int, default=4000,
                    help="Calibration size for the unmatched random_full arm, "
                         "which measures what random calibration achieves when "
                         "it is NOT handicapped to the clean pool's size.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--use_sparse_prop", action="store_true")
    args = ap.parse_args()

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    print(f"{args.dataset} / {args.detector} / {args.n_seeds} seeds")
    print(f"Strategies: {STRATEGIES}")
    print("Matched frame: same n_calib, same test set, same p-value floor.\n")

    graph, features, labels = load_any_dataset(args.dataset)
    print(f"{graph.number_of_nodes()} nodes, {int(labels.sum())} anomalies\n")

    rows = []
    for seed in range(args.n_seeds):
        res, err = run_seed(graph, features, labels, seed, args)
        if res is None:
            print(f"  seed {seed}: skipped ({err})")
            continue
        rows.extend(res)
        for r in res:
            print(f"  seed {seed} {r['strategy']:<13} n_cal={r['n_calib']:<5} "
                  f"gap_d={r['score_gap_cohens_d']:+.3f} "
                  f"gamma={r['gamma_t_lo']:>6.2f} disc={r['n_discoveries']:<5} "
                  f"fdr={r['realized_fdr']:.3f}")

    if not rows:
        print("\nNo seeds completed.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir,
                       f"calibration_strategy_{args.dataset}_{args.detector}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}")

    report(rows, args)


if __name__ == "__main__":
    main()
