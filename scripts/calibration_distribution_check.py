"""
calibration_distribution_check.py

Answers the paper's question directly instead of through a proxy.

WHY THIS EXISTS. Two earlier diagnostics measured the correlation between a
normal node's anomaly-neighbor exposure and its score, and used that as a proxy
for "is the calibration set contaminated?". That proxy has real problems, found
by auditing them:
  - It ran on the UNTRIMMED normal set, while the actual pipeline trims the top
    1% of normal scores from both calibration eligibility and the test set.
    On Amazon those trimmed nodes are exactly the documented extreme hubs
    (normal scores mean ~1502, max ~7596), so the proxy may have been dominated
    by points the pipeline discards.
  - Linear partial correlation was used to control for a MULTIPLICATIVE
    confound (degree_normalize_scores divides by log1p(degree)). Controlling
    for degree vs log-degree gave materially different answers (+0.198 vs
    +0.057 on Amazon), which means the control was not clean.
  - A correlation is not the quantity conformal p-values depend on anyway.

WHAT ACTUALLY MATTERS. A conformal p-value is
    p_j = (#{i in calib : S_i >= T_j} + 1) / (n_calib + 1)
so the only thing about the calibration set that reaches the p-values is its
SCORE DISTRIBUTION -- specifically the upper tail, and above all the maximum,
since a test point attains the p-value floor 1/(n_calib+1) exactly when it
exceeds every calibration score. A small mean correlation could still shift
that tail materially; a large one could leave it untouched. So this script
measures the distribution itself.

WHAT IT DOES. Replicates run_real_data_trial's frame VERBATIM (same trimming,
same calibration sizing, same RNG call order, same condition logic -- copied,
not re-derived), then for each of clean / contaminated / adversarial reports:
  - the calibration score distribution (quantiles + max)
  - two-sample KS tests between every pair of conditions
  - the clearance rate c: fraction of true anomalies scoring above the whole
    calibration set. This is the quantity theory/joint_discovery_threshold_
    proposition.md predicts governs whether ANY discovery occurs.
  - the resulting p-value floor, BH feasibility rank, and actual outcomes
    (discoveries, realized FDR, power)

HOW TO READ IT.
  - If the three conditions produce materially different calibration
    distributions (KS significant, visibly different max / upper quantiles)
    then the conditions ARE testing something real, whatever mechanism drives
    it, and the paper's experimental design is sound even if its stated
    mechanism needs correcting.
  - If the distributions are near-identical, the conditions are not testing
    different things and the clean/contaminated/adversarial comparison cannot
    support a contamination-robustness claim.

ONE CONFOUND THAT MUST BE READ CAREFULLY. By design (documented as
"decoupled calibration sizing" in real_data_experiment.py), the clean
condition uses whatever its zero-exposure pool provides while contaminated and
adversarial use min(4000, eligible). On Amazon that is 267 vs 4000. Different
n_calib means a different p-value floor, so clean is NOT directly comparable to
the other two on discovery outcomes. The distribution comparison is still
meaningful (KS is scale-free in sample size, though its power is not), but any
outcome difference involving clean is partly a sample-size artifact. Columns
are reported so this is visible rather than buried.

Trains once per seed and evaluates all three conditions on those same scores --
equivalent to the frozen code, which retrains per condition with the same seed
and therefore produces identical scores, just without paying 3x.

Run in the dgl311 env (CPU is fine at these sizes):
  ~/envs/dgl311/bin/python scripts/calibration_distribution_check.py \
      --datasets amazon reddit tolokers --n_seeds 3 --device cpu
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv
import math
import numpy as np
import torch
from scipy import stats

from detector import train_dominant
from conformal_fdr import conformal_p_values, benjamini_hochberg
from real_data_experiment import load_any_dataset, degree_normalize_scores, DEGREE_NORM_BY_DATASET

CONDITIONS = ["clean", "contaminated", "adversarial"]
QUANTILES = [50, 75, 90, 95, 99]


def compute_exposure(graph, normal_idx, labels):
    """Verbatim from run_real_data_trial."""
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)
    return exposure


def build_frame(scores, labels, normal_idx, anomaly_idx, exposure, condition, seed,
                trim_pct=0.01, max_normal_test=5000, n_calib_cap=4000):
    """Verbatim copy of run_real_data_trial's frame construction -- same trim,
    same sizing, same RNG object and call order. Copied deliberately rather
    than refactored: this must reproduce the frozen behaviour exactly or the
    numbers do not describe the pipeline that produced the paper's results."""
    rng = np.random.default_rng(seed)

    normal_scores_all = scores[normal_idx]
    score_cutoff = np.percentile(normal_scores_all, 100 * (1 - trim_pct))
    eligible_normal_idx = normal_idx[normal_scores_all <= score_cutoff]
    eligible_mask = np.isin(normal_idx, eligible_normal_idx)
    clean_pool = normal_idx[(exposure == 0) & eligible_mask]

    if len(clean_pool) < 20:
        return None

    if condition == "clean":
        calib_idx = clean_pool
    else:
        n_calib = min(n_calib_cap, len(eligible_normal_idx))
        if condition == "contaminated":
            calib_idx = rng.choice(eligible_normal_idx, size=n_calib, replace=False)
        elif condition == "adversarial":
            eligible_exposure = exposure[eligible_mask]
            order = np.argsort(-eligible_exposure)
            calib_idx = eligible_normal_idx[order][:n_calib]
        else:
            raise ValueError(f"unknown condition {condition!r}")

    remaining_normal = np.setdiff1d(eligible_normal_idx, calib_idx)
    if len(remaining_normal) > max_normal_test:
        remaining_normal = rng.choice(remaining_normal, size=max_normal_test, replace=False)

    test_idx = np.concatenate([remaining_normal, anomaly_idx])
    test_labels = np.concatenate([
        np.zeros(len(remaining_normal), dtype=int),
        np.ones(len(anomaly_idx), dtype=int),
    ])

    # calibration exposure, for the sanity check that the conditions differ in
    # the way they are supposed to (0 for clean, base rate for contaminated,
    # high for adversarial). If these do NOT separate, the selection logic
    # itself is broken and nothing downstream means anything.
    pos_of = {int(v): j for j, v in enumerate(normal_idx)}
    calib_exposure = np.array([exposure[pos_of[int(v)]] for v in calib_idx])

    return dict(calib_idx=calib_idx, test_idx=test_idx, test_labels=test_labels,
                calib_exposure=calib_exposure)


def evaluate(scores, frame, labels, alpha, anomaly_idx):
    calib_scores = scores[frame["calib_idx"]]
    test_scores = scores[frame["test_idx"]]
    test_labels = frame["test_labels"]

    n_calib = len(calib_scores)
    m = len(test_scores)
    calib_max = float(calib_scores.max())

    # Clearance rate: fraction of TRUE anomalies beating the entire calibration
    # set, i.e. attaining the p-value floor. This is the quantity the discovery-
    # threshold proposition is stated in terms of.
    anom_test = test_scores[test_labels == 1]
    norm_test = test_scores[test_labels == 0]
    clearance_anom = float((anom_test > calib_max).mean()) if len(anom_test) else float("nan")
    clearance_norm = float((norm_test > calib_max).mean()) if len(norm_test) else float("nan")

    p_values = conformal_p_values(calib_scores, test_scores)
    discoveries = benjamini_hochberg(p_values, alpha)
    n_disc = int(discoveries.sum())
    realized_fdr = float(np.sum(discoveries & (test_labels == 0)) / n_disc) if n_disc else 0.0
    power = float(np.sum(discoveries & (test_labels == 1)) / len(anomaly_idx)) if len(anomaly_idx) else 0.0

    row = dict(
        n_calib=n_calib, m_test=m,
        calib_mean_exposure=float(frame["calib_exposure"].mean()),
        calib_mean=float(calib_scores.mean()), calib_std=float(calib_scores.std()),
        calib_max=calib_max,
        p_floor=1.0 / (n_calib + 1),
        bh_min_rank=int(math.ceil(m / (alpha * (n_calib + 1)))),
        clearance_anomalies=clearance_anom, clearance_normals=clearance_norm,
        n_discoveries=n_disc, realized_fdr=realized_fdr, power=power,
    )
    for q in QUANTILES:
        row[f"calib_q{q}"] = float(np.percentile(calib_scores, q))
    return row, calib_scores


def run_dataset(dataset_name, n_seeds, n_epochs, alpha, device):
    print(f"\n{'=' * 78}\n{dataset_name.upper()}\n{'=' * 78}")
    try:
        graph, features, labels = load_any_dataset(dataset_name)
    except Exception as e:
        print(f"  SKIPPED ({type(e).__name__}: {e})")
        return []

    use_degree_norm = DEGREE_NORM_BY_DATASET.get(dataset_name, True)
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    exposure = compute_exposure(graph, normal_idx, labels)
    print(f"  n_nodes={graph.number_of_nodes():,}  n_anomalies={len(anomaly_idx):,}  "
          f"degree_norm={use_degree_norm}")

    rows = []
    for seed in range(n_seeds):
        raw, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed,
                                verbose=False, device=device)
        scores = degree_normalize_scores(graph, raw) if use_degree_norm else raw

        per_condition_scores = {}
        print(f"\n  --- seed {seed} ---")
        for cond in CONDITIONS:
            frame = build_frame(scores, labels, normal_idx, anomaly_idx, exposure, cond, seed)
            if frame is None:
                print(f"    {cond:13s}: SKIPPED (clean pool < 20)")
                continue
            row, calib_scores = evaluate(scores, frame, labels, alpha, anomaly_idx)
            row.update(dataset=dataset_name, seed=seed, condition=cond)
            rows.append(row)
            per_condition_scores[cond] = calib_scores
            print(f"    {cond:13s}: n_cal={row['n_calib']:5d}  calib_exposure={row['calib_mean_exposure']:.4f}  "
                  f"q95={row['calib_q95']:12.4f}  max={row['calib_max']:12.4f}  "
                  f"clear_anom={row['clearance_anomalies']:.4f}  n_disc={row['n_discoveries']:4d}  "
                  f"power={row['power']:.4f}")

        # Pairwise KS tests on the calibration score distributions themselves.
        # This is the direct question: are these three sets of numbers actually
        # different from one another?
        names = list(per_condition_scores)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ks, p = stats.ks_2samp(per_condition_scores[a], per_condition_scores[b])
                verdict = "DIFFERENT" if p < 0.05 else "indistinguishable"
                print(f"      KS {a:12s} vs {b:12s}: D={ks:.4f}  p={p:.3g}  -> {verdict}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["amazon", "reddit", "tolokers"])
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  alpha: {args.alpha}")

    all_rows = []
    for name in args.datasets:
        all_rows.extend(run_dataset(name, args.n_seeds, args.n_epochs, args.alpha, device))

    if not all_rows:
        print("\nNo results.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "calibration_distribution_check.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {csv_path}")

    print(f"\n{'=' * 78}\nSUMMARY -- do the three conditions produce different calibration sets?\n{'=' * 78}")
    print(f"{'dataset':10} {'condition':13} {'n_cal':>6} {'calib_exp':>10} {'q95':>13} "
          f"{'max':>13} {'clear_anom':>11} {'power':>8}")
    for name in args.datasets:
        for cond in CONDITIONS:
            sub = [r for r in all_rows if r["dataset"] == name and r["condition"] == cond]
            if not sub:
                continue
            g = lambda k: float(np.mean([r[k] for r in sub]))
            print(f"{name:10} {cond:13} {g('n_calib'):6.0f} {g('calib_mean_exposure'):10.4f} "
                  f"{g('calib_q95'):13.4f} {g('calib_max'):13.4f} "
                  f"{g('clearance_anomalies'):11.4f} {g('power'):8.4f}")
        print()

    print("""
Reading this:
  1. calib_exp MUST separate across conditions (clean ~0 < contaminated ~base
     rate < adversarial high). If it does not, the selection logic is broken
     and nothing else here is interpretable.
  2. Given (1) holds, compare q95 / max / clear_anom ACROSS conditions within a
     dataset. Those are what conformal p-values actually depend on.
       - materially different -> the conditions test something real, and the
         paper's design stands even if the stated mechanism must change.
       - near-identical       -> the conditions are not testing different
         things, and the contamination-robustness claim is unsupported.
  3. Any comparison INVOLVING clean is confounded by n_calib (clean uses its
     natural pool, the others min(4000, eligible)), which changes the p-value
     floor. contaminated vs adversarial is the clean comparison -- same
     n_calib by construction, differing only in which nodes were selected.
""")


if __name__ == "__main__":
    main()
