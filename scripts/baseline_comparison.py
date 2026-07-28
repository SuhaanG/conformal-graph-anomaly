"""
baseline_comparison.py

Step 8: baseline comparisons -- REVISED after a substantive, correct
critique of the original version. Documenting the correction explicitly
rather than silently rewriting, since the flaw and the fix are both
instructive.

WHAT WAS WRONG WITH THE ORIGINAL VERSION:
The original Method B (ensemble) and Method C (naive) baselines flagged a
FIXED number of nodes (k = oracle anomaly rate * n_nodes) with no
FDR-targeting mechanism at all. Since k was set to exactly equal the true
number of anomalies (750 = 750), the identity FDR = 1 - power held
EXACTLY and TAUTOLOGICALLY for every single trial -- this was verified
numerically, not just argued. Reporting "the baseline's FDR is ~49%" was
therefore not an empirical finding about the baseline being poorly
calibrated; it was an arithmetic consequence of the comparison's design.
A method with no FDR target cannot "fail" to hit one, any more than a
ruler can be faulted for reporting the wrong temperature. Additionally,
the oracle anomaly rate was originally framed as "generous" to the
baseline; it is actually what FORCES the tautology -- true generosity
would have let the baseline choose its own threshold.

THE FIX: Method B is redesigned to also target FDR via the SAME
conformal p-value + Benjamini-Hochberg machinery as Method A, applied to
ensemble-averaged scores instead of single-seed scores. This makes the
comparison mechanism-vs-mechanism (does averaging scores across seeds
before conformal calibration help, hurt, or not matter?) instead of
mechanism-vs-no-mechanism. Method C (naive fixed-rate thresholding) is
KEPT but explicitly relabeled as a descriptive reference point only --
its FDR/power numbers are reported, but no claim is made that it "should"
hit any FDR target, and no t-test treats it as a competing calibrated
procedure.

ALSO FIXED: honest reporting for Method A distinguishing MARGINAL FDR
(the actual BH guarantee, averaged over seeds including zero-discovery
seeds, where empty sets legitimately count as 0) from CONDITIONAL FDR
(the realized false-discovery proportion only on seeds where a discovery
was actually made) -- these can differ substantially and reporting only
the marginal number without disclosing the zero-discovery rate is
misleading by omission, even though the marginal number itself is not
incorrect.

ALSO ADDED: a calibration-size check. The critique identified that
conformal p-values have a resolution floor of 1/(n_calib+1), and BH can
only reject at rank i if that floor is <= alpha*i/m. With n_calib~1242
and m~13758, this floor requires i >~ 111 before ANY rejection is
possible -- explaining the observed all-or-nothing (0 or 120+) discovery
pattern exactly. This script now also runs a LARGER calibration size
variant to test whether this is fixable, per the critique's own
recommendation that this is "the single highest-value fix available."

Run on Colab/H200:
  python3 scripts/baseline_comparison.py --n_seeds 20 --n_ensemble 5 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import numpy as np
from scipy import stats

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detector import train_dominant
from conformal_fdr import conformal_p_values, benjamini_hochberg


def _build_calib_test_split(graph, labels, exposure, normal_idx, anomaly_idx,
                             condition, calib_frac, seed):
    """Shared calibration/test split logic used by both Method A and the
    redesigned Method B, so both apply IDENTICAL conformal machinery to
    whatever scores they're given -- the only difference between methods
    should be the SCORES (single-seed vs. ensemble-averaged), not the
    statistical procedure around them."""
    clean_pool = normal_idx[exposure == 0]
    if len(clean_pool) < 20:
        return None, None, None
    n_calib = int(round(calib_frac * len(clean_pool)))

    rng = np.random.default_rng(seed)
    if condition == "contaminated":
        calib_idx = rng.choice(normal_idx, size=n_calib, replace=False)
    else:  # adversarial
        order = np.argsort(-exposure)
        calib_idx = normal_idx[order][:n_calib]

    remaining_normal = np.setdiff1d(normal_idx, calib_idx)
    test_idx = np.concatenate([remaining_normal, anomaly_idx])
    test_labels = np.concatenate([
        np.zeros(len(remaining_normal), dtype=int),
        np.ones(len(anomaly_idx), dtype=int),
    ])
    return calib_idx, test_idx, test_labels


def _compute_exposure(graph, normal_idx, labels):
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)
    return exposure


def run_our_method_trial(alpha, seed, n_epochs, device, condition="contaminated", calib_frac=0.4):
    """Method A: our conformal+BH pipeline, single seed."""
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=0.002, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    scores, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed,
                                verbose=False, device=device)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    exposure = _compute_exposure(graph, normal_idx, labels)

    calib_idx, test_idx, test_labels = _build_calib_test_split(
        graph, labels, exposure, normal_idx, anomaly_idx, condition, calib_frac, seed)
    if calib_idx is None:
        return None

    calib_scores = scores[calib_idx]
    test_scores = scores[test_idx]
    p_values = conformal_p_values(calib_scores, test_scores)
    discoveries = benjamini_hochberg(p_values, alpha)

    n_discoveries = discoveries.sum()
    realized_fdr = (np.sum(discoveries & (test_labels == 0)) / n_discoveries) if n_discoveries > 0 else 0.0
    power = (np.sum(discoveries & (test_labels == 1)) / len(anomaly_idx)) if len(anomaly_idx) > 0 else 0.0

    method_name = f"our_conformal_bh_{condition}_calib{calib_frac}"
    return {
        "method": method_name, "seed": seed, "n_ensemble": 1, "calib_frac": calib_frac,
        "n_calib": len(calib_idx),
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }


def run_ensemble_conformal_trial(alpha, seed, n_ensemble, n_epochs, device,
                                  condition="contaminated", calib_frac=0.4):
    """REDESIGNED Method B: train n_ensemble models, average their raw
    scores, then apply the EXACT SAME conformal p-value + BH procedure as
    Method A to those averaged scores. This isolates the actual question
    the baseline was meant to answer -- does averaging scores across
    seeds before conformal calibration help, hurt, or not matter -- as a
    fair mechanism-vs-mechanism comparison instead of the original
    mechanism-vs-no-mechanism design."""
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=0.002, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    all_scores = []
    for e in range(n_ensemble):
        ensemble_seed = seed * 1000 + e
        s, _ = train_dominant(graph, features, n_epochs=n_epochs,
                               seed=ensemble_seed, verbose=False, device=device)
        all_scores.append(s)
    avg_scores = np.mean(all_scores, axis=0)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    exposure = _compute_exposure(graph, normal_idx, labels)

    calib_idx, test_idx, test_labels = _build_calib_test_split(
        graph, labels, exposure, normal_idx, anomaly_idx, condition, calib_frac, seed)
    if calib_idx is None:
        return None

    calib_scores = avg_scores[calib_idx]
    test_scores = avg_scores[test_idx]
    p_values = conformal_p_values(calib_scores, test_scores)
    discoveries = benjamini_hochberg(p_values, alpha)

    n_discoveries = discoveries.sum()
    realized_fdr = (np.sum(discoveries & (test_labels == 0)) / n_discoveries) if n_discoveries > 0 else 0.0
    power = (np.sum(discoveries & (test_labels == 1)) / len(anomaly_idx)) if len(anomaly_idx) > 0 else 0.0

    return {
        "method": f"ensemble_conformal_bh_{condition}", "seed": seed, "n_ensemble": n_ensemble,
        "calib_frac": calib_frac, "n_calib": len(calib_idx),
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }


def run_naive_threshold_trial(seed, n_epochs, device, top_k_frac=None):
    """Method C, KEPT as a descriptive reference point only. NOT a
    calibrated FDR-targeting procedure -- no t-test compares it to Method
    A as if it were a competing FDR method, per the critique. Its
    FDR = 1 - power identity is an ARITHMETIC ARTIFACT of setting
    k = true anomaly count, not a finding about the method's quality."""
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=0.002, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    if top_k_frac is None:
        top_k_frac = labels.mean()

    scores, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed,
                                verbose=False, device=device)

    n_nodes = len(labels)
    k = int(round(top_k_frac * n_nodes))
    top_k_idx = np.argsort(-scores)[:k]
    flagged = np.zeros(n_nodes, dtype=bool)
    flagged[top_k_idx] = True

    n_discoveries = flagged.sum()
    realized_fdr = (np.sum(flagged & (labels == 0)) / n_discoveries) if n_discoveries > 0 else 0.0
    power = (np.sum(flagged & (labels == 1)) / labels.sum()) if labels.sum() > 0 else 0.0

    return {
        "method": "naive_threshold_DESCRIPTIVE_ONLY", "seed": seed, "n_ensemble": 1,
        "calib_frac": None, "n_calib": None,
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }


def summarize_with_conditional(all_results, method, alpha):
    subset = [r for r in all_results if r["method"] == method]
    if not subset:
        return
    fdrs = np.array([r["realized_fdr"] for r in subset])
    powers = np.array([r["power"] for r in subset])
    n_disc = np.array([r["n_discoveries"] for r in subset])
    n_zero = int(np.sum(n_disc == 0))
    n_total = len(subset)

    print(f"{method}:")
    print(f"  marginal (all {n_total} seeds): realized_fdr={fdrs.mean():.3f}+/-{fdrs.std():.3f} "
          f"(nominal={alpha}), power={powers.mean():.3f}+/-{powers.std():.3f}")
    print(f"  zero-discovery seeds: {n_zero}/{n_total} ({100*n_zero/n_total:.0f}%)")
    if n_zero < n_total:
        cond_fdrs = fdrs[n_disc > 0]
        print(f"  CONDITIONAL on nonzero discovery ({n_total - n_zero} seeds): "
              f"realized_fdr={cond_fdrs.mean():.3f}+/-{cond_fdrs.std():.3f}")
        if cond_fdrs.mean() > alpha:
            print(f"  NOTE: conditional FDR ({cond_fdrs.mean():.3f}) exceeds nominal ({alpha}) -- "
              f"this is still consistent with valid marginal FDR control (BH's guarantee is an "
              f"average over runs including empty sets), but should be reported honestly rather "
              f"than only citing the marginal number.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--n_ensemble", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--large_calib_frac", type=float, default=0.9,
                         help="Larger calibration fraction to test the resolution-floor fix.")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    all_results = []

    for condition in ["contaminated", "adversarial"]:
        print(f"=== Method A: our conformal+BH, {condition}, calib_frac=0.4 ({args.n_seeds} seeds) ===")
        for seed in range(args.n_seeds):
            r = run_our_method_trial(args.alpha, seed, args.n_epochs, device,
                                      condition=condition, calib_frac=0.4)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            print(f"  seed {seed}: n_calib={r['n_calib']} n_discoveries={r['n_discoveries']:3d} "
                  f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")

        print(f"\n=== RESOLUTION-FLOOR CHECK: Method A, {condition}, "
              f"LARGER calib_frac={args.large_calib_frac} ({args.n_seeds} seeds) ===")
        for seed in range(args.n_seeds):
            r = run_our_method_trial(args.alpha, seed, args.n_epochs, device,
                                      condition=condition, calib_frac=args.large_calib_frac)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            print(f"  seed {seed}: n_calib={r['n_calib']} n_discoveries={r['n_discoveries']:3d} "
                  f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")

        print(f"\n=== Method B (REDESIGNED): ensemble-averaged scores -> conformal+BH, "
              f"{condition} ({args.n_seeds} trials, {args.n_ensemble} models each) ===")
        for seed in range(args.n_seeds):
            r = run_ensemble_conformal_trial(args.alpha, seed, args.n_ensemble, args.n_epochs,
                                              device, condition=condition, calib_frac=0.4)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            print(f"  seed {seed}: n_discoveries={r['n_discoveries']:3d} "
                  f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")

    print(f"\n=== Method C (DESCRIPTIVE ONLY, not FDR-targeting): naive fixed-rate threshold "
          f"({args.n_seeds} seeds) ===")
    for seed in range(args.n_seeds):
        r = run_naive_threshold_trial(seed, args.n_epochs, device)
        all_results.append(r)
        print(f"  seed {seed}: n_discoveries={r['n_discoveries']:3d} "
              f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "baseline_comparison_v2.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to {csv_path}")

    print("\n=== Summary (marginal + conditional FDR for every method) ===")
    method_names = sorted(set(r["method"] for r in all_results))
    for method in method_names:
        summarize_with_conditional(all_results, method, args.alpha)
        print()

    print("=== Paired t-tests: mechanism-vs-mechanism comparisons only ===")
    for condition in ["contaminated", "adversarial"]:
        ours = [r["realized_fdr"] for r in all_results
                if r["method"] == f"our_conformal_bh_{condition}_calib0.4"]
        ensemble = [r["realized_fdr"] for r in all_results
                    if r["method"] == f"ensemble_conformal_bh_{condition}"]
        if len(ours) == len(ensemble) and len(ours) >= 5:
            t = stats.ttest_rel(ensemble, ours)
            print(f"{condition}: ensemble-conformal vs. single-seed-conformal: "
                  f"t={t.statistic:.3f}, p={t.pvalue:.4f}")

    print("\nHONEST interpretation: this comparison now tests whether averaging scores across "
          "seeds BEFORE conformal calibration changes FDR/power, holding the FDR-targeting "
          "procedure itself constant. Method C's numbers are reported for reference but its "
          "FDR = 1 - power identity is an artifact of matching k to the true anomaly count, "
          "not a finding -- do not cite it as 'the naive baseline fails to control FDR.' Check "
          "the resolution-floor section: if the larger calib_frac run shows meaningfully fewer "
          "zero-discovery seeds and/or better power at a similar realized FDR, that confirms "
          "calibration size was capping power and should become the new default going forward.")


if __name__ == "__main__":
    main()