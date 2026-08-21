"""
condition_comparison_pygod.py

Retest of multi_seed_sweep.py (the paper's original Table II: clean vs
contaminated vs adversarial calibration on synthetic graphs) under a CORRECT
detector, to answer a question the severity sweep raised but could not
resolve on its own.

WHY THIS SCRIPT EXISTS. severity_sweep_pygod_instrumented.py escalates
contamination severity but only ever runs the adversarial condition (that
was true of the original severity_sweep.py too -- see its docstring). Its
result: pooled across all 100 trials, realized FDR is significantly ABOVE
the nominal target (mean 0.109 vs alpha=0.10, one-sided t=3.142, p=0.0011,
Cohen's d=0.314), worst at the highest severity level (d=1.343, 95% of
trials above target). That is the OPPOSITE of the original paper's claim
(FDR always at or below nominal). But because only the adversarial condition
was tested, it is not possible to tell from that run alone whether this is:

  (a) a genuine contamination effect that the old, structure-blind, broken
      detector was incapable of exposing, since it barely used the graph
      at all, or
  (b) a property of adversarial-selected calibration in general, unrelated
      to contamination severity specifically, that would show up even at
      the lowest severity level tested.

This script disambiguates by running all three conditions (clean,
contaminated, adversarial) at a SINGLE fixed severity (p_an=0.002, the
paper's baseline), under dominant_pygod. If FDR inflation shows up ONLY in
adversarial and not in clean/contaminated even at this single baseline
severity, that points toward (b) -- a general adversarial-selection
property. If it does not show up here at all (i.e. this run reproduces the
original paper's "all three conditions safely at or below nominal" result),
that points toward the severity sweep's inflation being specific to escalated
p_an, i.e. (a).

METHODOLOGY. This deliberately replicates conformal_fdr.py's run_single_trial
logic exactly (same calib_frac=0.9-of-clean-pool sizing, same graph config,
same three condition definitions), rather than reusing run_single_trial
itself, because that function is FROZEN (hardcodes the broken train_dominant)
and per repo convention frozen code is not edited in place -- new variants
are added alongside it instead. Every non-detector line below is intentionally
identical to run_single_trial's, so results are structurally comparable to
the original multi_seed_sweep.csv, not just similar in spirit.

Rank-level logging (per severity_sweep_pygod_instrumented.py's approach) is
included here too, since clean/contaminated conditions -- unlike the
adversarial severity sweep, where AUROC=1.0 made the floor condition trivially
succeed every time -- may actually produce cases where the floor fails but a
larger rank succeeds, giving the extended proposition (theory/joint_discovery
_threshold_proposition.md Part 2) a real synthetic test case for the first
time, not just the real-data one already in PAPER_REFRAME_HANDOFF.md.

Run on Colab (per repo convention):
  !python scripts/condition_comparison_pygod.py --n_seeds 20 --alpha 0.10 --device cuda

Outputs:
  results/logs/condition_comparison_pygod.csv        -- trial-level
  results/logs/condition_comparison_pygod_ranks.csv  -- rank-grid, long format
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
from detectors import score_nodes, available_detectors
from conformal_fdr import conformal_p_values, benjamini_hochberg
from real_data_experiment import degree_normalize_scores
from weighted_conformal import (
    weighted_conformal_p_values,
    estimate_selection_propensity,
    calibration_weights_from_propensity,
)

CONDITIONS = ["clean", "contaminated", "adversarial"]


def degree_matched_calib_sample(rng, candidate_idx, candidate_degree, target_degree,
                                 n_calib, n_bins=10):
    """Draws up to n_calib items from candidate_idx so the SAMPLE's degree
    distribution matches target_degree's distribution, instead of taking a
    uniform draw from candidate_idx directly (which is what produced the
    degree-biased clean calibration set found by
    clean_selection_degree_diagnostic.py: clean-selected nodes had
    significantly lower degree than the test-set normal population,
    t=-24.959, p<0.0001).

    Unlike degree_normalize_scores() (which rescales every score and was
    tested in condition_comparison_pygod.py --use_degree_norm -- it collapsed
    AUROC from 1.0 to ~0.90 and power from 1.0 to ~0.002, and made conditional
    FDR on the rare trials that still fired WORSE, 40-44% vs the original
    13.2%, not better), this function never touches scores. It only changes
    which nodes get selected into calibration, addressing the selection bias
    directly rather than distorting the detector's signal to compensate for it.

    Bins are defined by target_degree's quantiles (n_bins equal-mass bins).
    For each bin, draws min(bin's target proportion * n_calib, candidates
    available in that bin) from candidate_idx, without replacement. If bin
    shortfalls leave the total sample below n_calib (candidate_idx simply
    lacks enough high-degree members to fully match, since it's drawn from
    a low-degree-biased pool by construction), a top-up pass fills remaining
    slots from whatever unselected candidates remain, so the final sample
    size still equals n_calib when the candidate pool is large enough overall
    -- but the achieved match will be imperfect in that case, which is
    reported via the returned KS statistic, not hidden.

    Returns: (selected_idx, achieved_ks_stat, achieved_ks_p) -- the KS test
    compares the SELECTED sample's degree against target_degree, so the
    caller can see how well matching actually worked, not just assume it did."""
    candidate_idx = np.asarray(candidate_idx)
    candidate_degree = np.asarray(candidate_degree, dtype=float)
    target_degree = np.asarray(target_degree, dtype=float)

    bin_edges = np.quantile(target_degree, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-9   # ensure the minimum value falls inside bin 1, not on the boundary
    bin_edges[-1] += 1e-9

    target_bin = np.digitize(target_degree, bin_edges) - 1
    target_bin = np.clip(target_bin, 0, n_bins - 1)
    target_counts = np.bincount(target_bin, minlength=n_bins)
    target_props = target_counts / target_counts.sum()

    cand_bin = np.digitize(candidate_degree, bin_edges) - 1
    cand_bin = np.clip(cand_bin, 0, n_bins - 1)

    selected = []
    used_mask = np.zeros(len(candidate_idx), dtype=bool)
    for b in range(n_bins):
        desired = int(round(target_props[b] * n_calib))
        available_mask = (cand_bin == b) & (~used_mask)
        available_positions = np.where(available_mask)[0]
        take = min(desired, len(available_positions))
        if take > 0:
            chosen = rng.choice(available_positions, size=take, replace=False)
            selected.extend(chosen.tolist())
            used_mask[chosen] = True

    # top-up pass: if bin shortfalls left us short of n_calib, fill from
    # whatever candidates remain unselected, regardless of bin
    if len(selected) < n_calib:
        remaining_positions = np.where(~used_mask)[0]
        shortfall = n_calib - len(selected)
        take = min(shortfall, len(remaining_positions))
        if take > 0:
            topup = rng.choice(remaining_positions, size=take, replace=False)
            selected.extend(topup.tolist())

    selected_idx = candidate_idx[np.array(selected, dtype=int)]
    selected_degree = candidate_degree[np.array(selected, dtype=int)]
    ks_stat, ks_p = stats.ks_2samp(selected_degree, target_degree)
    return selected_idx, ks_stat, ks_p


def compute_ranks(calib_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Identical to severity_sweep_pygod_instrumented.py's version (unit-tested
    there against conformal_p_values). Duplicated rather than imported since
    scripts/ has no shared-utility module in this repo and adding one now is
    out of scope for this task."""
    n_calib = len(calib_scores)
    sorted_calib = np.sort(calib_scores)
    count_lt = np.searchsorted(sorted_calib, test_scores, side="left")
    count_ge = n_calib - count_lt
    return count_ge + 1


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Identical to severity_sweep_pygod_instrumented.py's version."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks_per_value = np.zeros(len(counts))
    np.add.at(sum_ranks_per_value, inv, ranks)
    avg_rank = (sum_ranks_per_value / counts)[inv]
    rank_sum_pos = avg_rank[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def rank_grid(n_calib: int, n_points: int = 25) -> np.ndarray:
    """Identical to severity_sweep_pygod_instrumented.py's version."""
    return np.unique(np.round(np.geomspace(1, n_calib + 1, n_points)).astype(int))


def run_condition_trial(condition, alpha, seed, n_epochs, device,
                         calib_frac=0.9, n_rank_points=25, use_degree_norm=False,
                         degree_matched_calib=False, n_degree_bins=10,
                         use_weighted_conformal=False,
                         detector="dominant_pygod"):
    """Structurally identical to conformal_fdr.run_single_trial, with the
    single substitution of a src/detectors.py scorer for the frozen
    train_dominant.

    detector defaults to dominant_pygod, which is what produced the
    clean-condition FDR inflation (mean 0.132, d=0.837, p=0.0007). It is
    parameterised because Part 4 of the theory doc predicts the inflation
    should scale with each detector's own score-degree dependence, and testing
    that requires varying the detector while holding graph and selection filter
    fixed. See theory/joint_discovery_threshold_proposition.md Part 4,
    prediction 2.

    Three independent fixes for the clean condition, all off by default and
    mutually exclusive in practice (only one should be True at a time --
    combining them is not tested and not meaningful):

    use_degree_norm=False. Tested True in a prior run: it collapsed AUROC
    from 1.0 to ~0.90 and power from 1.0 to ~0.002, and made conditional
    FDR on the rare trials that still fired WORSE (40-44%, not better) --
    too blunt an instrument, kept here only for reproducibility.

    degree_matched_calib=False. Replaces the "clean" condition's uniform
    draw with degree_matched_calib_sample(). Tested True in a prior run:
    closed about half the gap (d: 0.837 -> 0.449), capped by the candidate
    pool structurally lacking high-degree members (see theory doc Part 4's
    corollary on why this ceiling is provable, not just observed).

    use_weighted_conformal=False. Applies src/weighted_conformal.py's
    inverse-propensity weighting to the SAME "clean" calibration draw
    uniform sampling would produce -- the third fix, and the one the
    theory doc's Theorem (selection-induced non-exchangeability) actually
    predicts should work, as opposed to the other two which were
    engineering attempts to work around it. Not yet tested against this
    synthetic setup; weighted_conformal.py's own self-test validates the
    mechanism on an idealized simulation, and calibration_strategy_
    comparison.py's "weighted" strategy tests it on real data -- this is
    the third and last place it needs to be checked.

    Only "clean" is affected by any of the three -- "contaminated" already
    draws uniformly from the full normal pool (no selection bias to
    correct), and "adversarial" is a deliberate worst-case selection, not
    a candidate for any of these fixes."""
    assert condition in CONDITIONS
    assert sum([use_degree_norm, degree_matched_calib, use_weighted_conformal]) <= 1, (
        "at most one of use_degree_norm, degree_matched_calib, "
        "use_weighted_conformal should be set -- combining them is untested"
    )

    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=0.002, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    scores = score_nodes(detector, graph, features, labels=labels,
                          seed=seed, n_epochs=n_epochs, device=device)

    if use_degree_norm:
        scores = degree_normalize_scores(graph, scores)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    degree = np.array([graph.degree(i) for i in normal_idx], dtype=float)

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)

    rng = np.random.default_rng(seed)

    clean_mask = exposure == 0
    clean_pool = normal_idx[clean_mask]
    n_calib = int(round(calib_frac * len(clean_pool)))

    if len(clean_pool) < 20:
        return None, None

    achieved_ks_p = None
    if condition == "clean":
        eligible_calib_pool = clean_pool
        if degree_matched_calib:
            calib_idx, achieved_ks_stat, achieved_ks_p = degree_matched_calib_sample(
                rng, clean_pool, degree[clean_mask], degree, n_calib, n_bins=n_degree_bins,
            )
        else:
            calib_idx = rng.choice(eligible_calib_pool, size=n_calib, replace=False)
    elif condition == "contaminated":
        eligible_calib_pool = normal_idx
        calib_idx = rng.choice(eligible_calib_pool, size=n_calib, replace=False)
    else:  # adversarial
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

    if use_weighted_conformal and condition == "clean":
        q_hat_pool = estimate_selection_propensity(degree, clean_mask, n_bins=n_degree_bins)
        pool_pos = {int(idx): pos for pos, idx in enumerate(normal_idx)}
        q_hat_calib = np.array([q_hat_pool[pool_pos[int(i)]] for i in calib_idx])
        calib_weights = calibration_weights_from_propensity(q_hat_calib)
        p_values = weighted_conformal_p_values(calib_scores, calib_weights,
                                                test_scores, test_weight=1.0)
    else:
        p_values = conformal_p_values(calib_scores, test_scores)
    discoveries = benjamini_hochberg(p_values, alpha)

    n_discoveries = int(discoveries.sum())
    if n_discoveries == 0:
        realized_fdr = 0.0
    else:
        false_discoveries = int(np.sum(discoveries & (test_labels == 0)))
        realized_fdr = false_discoveries / n_discoveries

    n_true_anomalies_found = int(np.sum(discoveries & (test_labels == 1)))
    power = n_true_anomalies_found / len(anomaly_idx) if len(anomaly_idx) > 0 else 0.0

    trial_auroc = auroc(test_scores, test_labels)

    m = len(test_idx)
    m_1 = int(test_labels.sum())
    pi_1 = m_1 / m if m > 0 else float("nan")

    trial_row = {
        "condition": condition,
        "seed": seed,
        "alpha": alpha,
        "n_calib": n_calib,
        "m_test": m,
        "m_1": m_1,
        "pi_1": pi_1,
        "auroc": trial_auroc,
        "n_discoveries": n_discoveries,
        "realized_fdr": realized_fdr,
        "power": power,
        "degree_match_ks_p": achieved_ks_p if achieved_ks_p is not None else float("nan"),
    }

    ranks = compute_ranks(calib_scores, test_scores)
    anomaly_mask = test_labels == 1
    rank_rows = []
    for r in rank_grid(n_calib, n_rank_points):
        n1_r = int(np.sum(ranks[anomaly_mask] <= r))
        n0_r = int(np.sum(ranks[~anomaly_mask] <= r))
        n_r = n1_r + n0_r
        bh_threshold = (m / alpha) * (r / (n_calib + 1))
        rank_rows.append({
            "condition": condition,
            "seed": seed,
            "r": int(r),
            "n1_r": n1_r,
            "n0_r": n0_r,
            "n_r": n_r,
            "c_r": n1_r / m_1 if m_1 > 0 else float("nan"),
            "bh_threshold_at_r": bh_threshold,
            "dagger_satisfied": bool(n_r >= bh_threshold),
        })

    return trial_row, rank_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_rank_points", type=int, default=25)
    parser.add_argument("--detector", type=str, default="dominant_pygod",
                        choices=available_detectors(),
                        help="Which detector produces the scores. Defaults to "
                             "dominant_pygod, which is what found the clean-condition "
                             "FDR inflation (0.132 vs nominal 0.10) and which writes to "
                             "the original unsuffixed filename. Other detectors get a "
                             "_<detector> suffix so runs never overwrite each other. "
                             "Varying this is Part 4 prediction 2: the inflation should "
                             "scale with each detector's own score-degree dependence, "
                             "and a detector with weak degree sensitivity should show "
                             "little or none.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_degree_norm", action="store_true",
                         help="Apply real_data_experiment.degree_normalize_scores() before "
                              "calibration. Off by default. TESTED AND NOT RECOMMENDED: "
                              "collapsed AUROC 1.0->0.90, power 1.0->0.002, and made "
                              "conditional FDR worse (40-44%% vs 13.2%%) on a prior run. "
                              "Kept for reproducibility, not as a suggested fix.")
    parser.add_argument("--degree_matched_calib", action="store_true",
                         help="For the 'clean' condition only, draw calibration via "
                              "degree_matched_calib_sample() instead of uniform sampling from "
                              "the zero-exposure pool, so calibration's degree distribution "
                              "matches the full normal population's rather than being "
                              "structurally biased low. Does not touch scores, unlike "
                              "--use_degree_norm.")
    parser.add_argument("--n_degree_bins", type=int, default=10,
                         help="Used with --degree_matched_calib or --use_weighted_conformal.")
    parser.add_argument("--use_weighted_conformal", action="store_true",
                         help="For the 'clean' condition only, apply "
                              "src/weighted_conformal.py's inverse-propensity weighting to "
                              "the standard uniform draw. The fix Theorem selection-induced-"
                              "non-exchangeability actually predicts, as opposed to "
                              "--use_degree_norm or --degree_matched_calib which were "
                              "engineering workarounds. Mutually exclusive with those two.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Detector: {args.detector} (see src/detectors.py)\n")

    all_trials = []
    all_ranks = []
    for condition in CONDITIONS:
        print(f"=== Running {args.n_seeds} seeds for condition: {condition} ===")
        for seed in range(args.n_seeds):
            trial_row, rank_rows = run_condition_trial(
                condition, args.alpha, seed, args.n_epochs, device,
                n_rank_points=args.n_rank_points, use_degree_norm=args.use_degree_norm,
                degree_matched_calib=args.degree_matched_calib, n_degree_bins=args.n_degree_bins,
                use_weighted_conformal=args.use_weighted_conformal,
                detector=args.detector,
            )
            if trial_row is None:
                print(f"  seed {seed}: skipped (insufficient clean calibration pool)")
                continue
            all_trials.append(trial_row)
            all_ranks.extend(rank_rows)
            any_dagger = any(r["dagger_satisfied"] for r in rank_rows)
            floor_dagger = rank_rows[0]["dagger_satisfied"]
            ks_extra = (f" degree_match_ks_p={trial_row['degree_match_ks_p']:.4f}"
                        if not np.isnan(trial_row['degree_match_ks_p']) else "")
            print(f"  seed {seed}: auroc={trial_row['auroc']:.4f} "
                  f"n_discoveries={trial_row['n_discoveries']:4d} "
                  f"realized_fdr={trial_row['realized_fdr']:.3f} "
                  f"power={trial_row['power']:.3f} | "
                  f"floor_predicts={floor_dagger} any_r_predicts={any_dagger} "
                  f"observed={trial_row['n_discoveries'] > 0}{ks_extra}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    if args.use_degree_norm:
        suffix = "_degreenorm"
    elif args.degree_matched_calib:
        suffix = "_degreematched"
    elif args.use_weighted_conformal:
        suffix = "_weighted"
    else:
        suffix = ""
    # The default detector keeps the original filename, so the run that found
    # the 0.132 clean-condition inflation is still written where every
    # reference to it expects. Only non-default detectors get a suffix.
    if args.detector != "dominant_pygod":
        suffix += f"_{args.detector}"

    trial_csv = os.path.join(out_dir, f"condition_comparison_pygod{suffix}.csv")
    with open(trial_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_trials[0].keys()))
        writer.writeheader()
        writer.writerows(all_trials)
    print(f"\nSaved trial-level results to {trial_csv}")

    rank_csv = os.path.join(out_dir, f"condition_comparison_pygod{suffix}_ranks.csv")
    with open(rank_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_ranks[0].keys()))
        writer.writeheader()
        writer.writerows(all_ranks)
    print(f"Saved rank-grid results to {rank_csv}")

    print("\n=== Summary (mean +/- std across seeds) ===")
    for condition in CONDITIONS:
        subset = [r for r in all_trials if r["condition"] == condition]
        if not subset:
            print(f"{condition}: no valid trials")
            continue
        fdrs = np.array([r["realized_fdr"] for r in subset])
        powers = np.array([r["power"] for r in subset])
        aurocs = np.array([r["auroc"] for r in subset])
        n_zero_discovery = sum(1 for r in subset if r["n_discoveries"] == 0)
        print(f"{condition}: auroc={aurocs.mean():.4f}+/-{aurocs.std():.4f}, "
              f"realized_fdr={fdrs.mean():.3f}+/-{fdrs.std():.3f} "
              f"(nominal alpha={args.alpha}), power={powers.mean():.3f}+/-{powers.std():.3f}, "
              f"zero-discovery trials={n_zero_discovery}/{len(subset)}")

    print("\n=== Statistical tests ===")
    by_condition = {c: [r for r in all_trials if r["condition"] == c] for c in CONDITIONS}

    pairs = [("contaminated", "clean"), ("adversarial", "clean"), ("adversarial", "contaminated")]
    for cond_a, cond_b in pairs:
        by_seed_a = {r["seed"]: r["realized_fdr"] for r in by_condition[cond_a]}
        by_seed_b = {r["seed"]: r["realized_fdr"] for r in by_condition[cond_b]}
        common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
        if len(common_seeds) < 5:
            print(f"{cond_a} vs {cond_b}: insufficient paired seeds ({len(common_seeds)})")
            continue
        a_paired = np.array([by_seed_a[s] for s in common_seeds])
        b_paired = np.array([by_seed_b[s] for s in common_seeds])
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(a_paired, b_paired)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
        paired_ttest = stats.ttest_rel(a_paired, b_paired)
        print(f"{cond_a} vs {cond_b} (n={len(common_seeds)} paired seeds): "
              f"Wilcoxon p={wilcoxon_p:.4f}, paired t={paired_ttest.statistic:.3f} p={paired_ttest.pvalue:.4f}")

    for name in CONDITIONS:
        subset = by_condition[name]
        fdrs = np.array([r["realized_fdr"] for r in subset])
        if len(fdrs) >= 5 and fdrs.std() > 0:
            t_stat, p_two_sided = stats.ttest_1samp(fdrs, args.alpha)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
            d = (fdrs.mean() - args.alpha) / fdrs.std(ddof=1)
            verdict = "SIGNIFICANTLY ABOVE nominal" if p_one_sided < 0.05 else "not significantly above nominal"
            print(f"One-sided test ({name} FDR > alpha={args.alpha}): "
                  f"mean={fdrs.mean():.3f}, t={t_stat:.3f}, p={p_one_sided:.4f}, d={d:.3f} -> {verdict}")
        else:
            print(f"One-sided test ({name}): insufficient variance or sample size to test")

    print("\nReading guide, per the question this script was built to answer:")
    print("  - If ONLY adversarial shows significant FDR inflation here (clean and")
    print("    contaminated do not), that matches severity_sweep_pygod's finding")
    print("    and points toward a general adversarial-selection property, not a")
    print("    severity-specific contamination effect.")
    print("  - If NONE of the three conditions show inflation here (all near or")
    print("    below nominal, similar to the ORIGINAL Table II), the inflation seen")
    print("    in severity_sweep_pygod is specific to escalated p_an and does not")
    print("    appear at baseline severity -- suggesting a real severity-driven")
    print("    contamination effect worth investigating further.")
    print("  - If clean/contaminated ALSO show inflation, that would suggest the")
    print("    inflation is not about contamination or adversarial selection at all,")
    print("    but some other property of dominant_pygod on this synthetic")
    print("    generator -- report this plainly rather than reaching for either")
    print("    of the above explanations.")


if __name__ == "__main__":
    main()