"""
severity_sweep_pygod_instrumented.py

Retest of severity_sweep.py under a CORRECT detector, with rank-level logging
added. Two separate problems motivate this, and one run answers both.

PROBLEM 1 -- "fails into silence" is suspect. severity_sweep.py's headline
result (power collapses to zero at high contamination severity while FDR
stays controlled) was produced by dominant_ours, whose encoder is known to be
broken (see DETECTOR_DIAGNOSTIC.md): a dead ReLU on the final embedding layer
forces the structure decoder to a constant, so ~0.92 AUROC came entirely from
raw feature magnitude, not the graph. Under a correct detector (dominant_pygod
via src/detectors.py), AUROC on the same synthetic graphs RISES with severity
(0.9865 -> 1.0000) instead of falling, so the power collapse this paper's Fig.
1 depends on may be an artifact of the broken detector degrading as the graph
densifies, not a property of the calibration procedure. This script settles
that by rerunning the identical sweep under dominant_pygod. See
PAPER_REFRAME_HANDOFF.md section 4.8: "Do not carry this claim into the
rewrite without re-testing it."

PROBLEM 2 -- the extended (rank-indexed) discovery proposition in
theory/joint_discovery_threshold_proposition.md Part 2 has never been checked
against real data. Its central object is c(r), the fraction of anomalies
clearing rank r, for r ranging across the WHOLE calibration size, not just
r=1 (the floor). severity_sweep.py never logged per-trial ranks, only
aggregate power/FDR, so there is currently no data anywhere in this repo that
could confirm or refute the extended proposition. This script logs, for a
grid of candidate ranks r, the anomaly clearance count N_1(r), the normal
clearance count N_0(r), and whether the exact BH crossing condition (dagger)
from the theory doc is satisfied at that r -- which is exactly what's needed
to check the extended proposition's prediction against the actual discovery
outcome for every trial.

WHAT THIS DOES NOT CHANGE. graph_gen.py and conformal_fdr.py are not
modified; this script imports them as-is. The only substitution relative to
severity_sweep.py is scoring via src/detectors.py's score_nodes("dominant_pygod",
...) in place of the frozen train_dominant. n_calib is held at the same fixed
value (2794) used in the original sweep so the two runs are comparable.

Run on the GPU box (per repo README section 11):
  ~/envs/dgl311/bin/python scripts/severity_sweep_pygod_instrumented.py \
      --n_seeds 20 --alpha 0.10 --device cuda

Outputs:
  results/logs/severity_sweep_pygod.csv        -- trial-level, same schema
                                                     as severity_sweep.csv
                                                     plus an auroc column
  results/logs/severity_sweep_pygod_ranks.csv  -- long format, one row per
                                                     (trial, candidate rank r):
                                                     N1(r), N0(r), N(r),
                                                     whether (dagger) holds
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
from detectors import score_nodes
from conformal_fdr import conformal_p_values, benjamini_hochberg


def compute_ranks(calib_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """r(v) = |{u in calib : S(u) >= S(v)}| + 1, matching conformal_fdr.py's
    conformal_p_values definition exactly (count_ge = np.sum(calib_scores >= s)),
    but vectorized via searchsorted for speed. p_values from conformal_p_values
    equal ranks / (n_calib + 1) exactly, since neither function randomizes ties
    (conformal_p_values' docstring claims a randomized tie-break; the body has
    none -- noted here rather than silently relied upon)."""
    n_calib = len(calib_scores)
    sorted_calib = np.sort(calib_scores)
    # count of calib scores < test_score, via searchsorted(side="left")
    count_lt = np.searchsorted(sorted_calib, test_scores, side="left")
    count_ge = n_calib - count_lt
    return count_ge + 1


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard rank-based AUROC (Mann-Whitney U / (n_pos*n_neg)), independent
    of any detector-specific code, so it means the same thing regardless of
    which detector produced `scores`."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average-rank tie correction
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks_per_value = np.zeros(len(counts))
    np.add.at(sum_ranks_per_value, inv, ranks)
    avg_rank = (sum_ranks_per_value / counts)[inv]
    rank_sum_pos = avg_rank[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def rank_grid(n_calib: int, n_points: int = 25) -> np.ndarray:
    """Geometric grid of candidate ranks from 1 to n_calib+1, capturing both
    the floor (r=1, what the OLD proposition checks) and the bulk of the
    distribution (larger r, what the EXTENDED proposition adds). Geometric
    rather than linear spacing because the interesting behavior (per the
    falsification case in the theory doc, r=1 failing while some larger r
    succeeds) can occur at any scale, and a linear grid over-samples the
    large-r end at the expense of resolution near the floor."""
    grid = np.unique(np.round(np.geomspace(1, n_calib + 1, n_points)).astype(int))
    return grid


def run_severity_trial(p_an, alpha, seed, n_epochs, device, n_calib=2794, n_rank_points=25):
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=p_an, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    scores = score_nodes("dominant_pygod", graph, features, labels=labels,
                          seed=seed, n_epochs=n_epochs, device=device)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)

    if len(normal_idx) < n_calib:
        return None, None  # graph too small / too contaminated for this fixed calib size

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
        "p_an": p_an,
        "seed": seed,
        "n_calib": n_calib,
        "m_test": m,
        "m_1": m_1,
        "pi_1": pi_1,
        "mean_calib_exposure": mean_calib_exposure,
        "auroc": trial_auroc,
        "n_discoveries": n_discoveries,
        "realized_fdr": realized_fdr,
        "power": power,
    }

    # --- rank-indexed logging for the extended proposition ---
    ranks = compute_ranks(calib_scores, test_scores)
    anomaly_mask = test_labels == 1
    rank_rows = []
    for r in rank_grid(n_calib, n_rank_points):
        n1_r = int(np.sum(ranks[anomaly_mask] <= r))
        n0_r = int(np.sum(ranks[~anomaly_mask] <= r))
        n_r = n1_r + n0_r
        bh_threshold = (m / alpha) * (r / (n_calib + 1))
        rank_rows.append({
            "p_an": p_an,
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
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Detector: dominant_pygod (correct implementation, per src/detectors.py)\n")

    p_an_values = [0.002, 0.005, 0.01, 0.02, 0.05]

    all_trials = []
    all_ranks = []
    for p_an in p_an_values:
        print(f"=== p_an={p_an} ===")
        for seed in range(args.n_seeds):
            trial_row, rank_rows = run_severity_trial(
                p_an, args.alpha, seed, args.n_epochs, device,
                n_rank_points=args.n_rank_points,
            )
            if trial_row is None:
                print(f"  seed {seed}: skipped (clean pool exhausted at this severity)")
                continue
            all_trials.append(trial_row)
            all_ranks.extend(rank_rows)
            any_dagger = any(r["dagger_satisfied"] for r in rank_rows)
            floor_dagger = rank_rows[0]["dagger_satisfied"]  # smallest r in grid, closest to r=1
            print(f"  seed {seed}: auroc={trial_row['auroc']:.4f} "
                  f"mean_calib_exposure={trial_row['mean_calib_exposure']:.3f} "
                  f"n_discoveries={trial_row['n_discoveries']:4d} "
                  f"realized_fdr={trial_row['realized_fdr']:.3f} "
                  f"power={trial_row['power']:.3f} | "
                  f"floor_predicts_discovery={floor_dagger} "
                  f"any_r_predicts_discovery={any_dagger} "
                  f"observed_discovery={trial_row['n_discoveries'] > 0}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)

    trial_csv = os.path.join(out_dir, "severity_sweep_pygod.csv")
    with open(trial_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_trials[0].keys()))
        writer.writeheader()
        writer.writerows(all_trials)
    print(f"\nSaved trial-level results to {trial_csv}")

    rank_csv = os.path.join(out_dir, "severity_sweep_pygod_ranks.csv")
    with open(rank_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_ranks[0].keys()))
        writer.writeheader()
        writer.writerows(all_ranks)
    print(f"Saved rank-grid results to {rank_csv}")

    print("\n=== Summary by severity level (PyGOD detector) ===")
    for p_an in p_an_values:
        subset = [r for r in all_trials if r["p_an"] == p_an]
        if not subset:
            print(f"p_an={p_an}: no valid trials")
            continue
        fdrs = np.array([r["realized_fdr"] for r in subset])
        aurocs = np.array([r["auroc"] for r in subset])
        powers = np.array([r["power"] for r in subset])
        exposures = np.array([r["mean_calib_exposure"] for r in subset])
        t_stat, p_two_sided = stats.ttest_1samp(fdrs, args.alpha) if fdrs.std() > 0 else (float("nan"), float("nan"))
        p_one_sided = (p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2) if not np.isnan(p_two_sided) else float("nan")
        verdict = ("SIGNIFICANTLY ABOVE nominal" if (not np.isnan(p_one_sided) and p_one_sided < 0.05)
                   else "not significantly above nominal")
        print(f"p_an={p_an}: auroc={aurocs.mean():.4f}+/-{aurocs.std():.4f}, "
              f"mean_calib_exposure={exposures.mean():.3f}, "
              f"realized_fdr={fdrs.mean():.3f}+/-{fdrs.std():.3f}, "
              f"power={powers.mean():.3f}+/-{powers.std():.3f} -> {verdict}")

    print("\nCOMPARE against the original severity_sweep.csv (dominant_ours) by hand:")
    print("  - If AUROC here RISES with p_an (as in the pygod_exposure_check.py")
    print("    diagnostic: 0.9865 -> 1.0000) while power still collapses to zero,")
    print("    'fails into silence' survives under a correct detector -- keep it.")
    print("  - If power stays nonzero as severity increases (tracking the rising")
    print("    AUROC instead of collapsing), 'fails into silence' was an artifact")
    print("    of the broken encoder degrading -- drop the claim per")
    print("    PAPER_REFRAME_HANDOFF.md section 4.8.")
    print("\nUse severity_sweep_pygod_ranks.csv to check the extended proposition:")
    print("  for each trial, does ANY r in the grid satisfy dagger_satisfied=True")
    print("  exactly when n_discoveries > 0 in the matching trial row? Mismatches")
    print("  (dagger predicts discovery but none observed, or vice versa) are the")
    print("  proposition's failure cases and should be reported, not discarded.")


if __name__ == "__main__":
    main()