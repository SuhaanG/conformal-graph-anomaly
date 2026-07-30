"""
baseline_comparison_real_data.py

Extends the corrected baseline comparison (single-seed conformal+BH vs.
ensemble-averaged conformal+BH, same mechanism, same procedure) from
synthetic data to real data (FraudAmazonDataset), to test whether the
"ensembling adds no measurable value" finding generalizes beyond
synthetic constructions.

Reuses the validated real-data fixes from real_data_experiment.py
directly (feature standardization, degree-normalized scoring, symmetric
trimming of extreme-score outlier nodes from both calibration AND test
sets) rather than duplicating that logic -- those fixes are real-data
specific and were hard-won through several rounds of debugging.

Run on Colab (needs dgl):
  python3 scripts/baseline_comparison_real_data.py --dataset amazon --n_seeds 20 --n_ensemble 5 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv
import numpy as np
from scipy import stats

from detector import train_dominant
from conformal_fdr import conformal_p_values, benjamini_hochberg
from real_data_experiment import load_fraud_graph, degree_normalize_scores


def _prepare_calib_test(graph, labels, scores, condition, seed, trim_pct=0.01):
    """Shared logic (mirrors run_real_data_trial's validated approach):
    symmetric trimming of extreme-score outliers from calibration
    eligibility AND the test set, then condition-specific calibration
    selection (contaminated=random, adversarial=worst-case-exposure)."""
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)

    normal_scores_all = scores[normal_idx]
    score_cutoff = np.percentile(normal_scores_all, 100 * (1 - trim_pct))
    eligible_normal_idx = normal_idx[normal_scores_all <= score_cutoff]
    eligible_mask = np.isin(normal_idx, eligible_normal_idx)

    clean_pool = normal_idx[(exposure == 0) & eligible_mask]
    if len(clean_pool) < 20:
        return None, None, None

    rng = np.random.default_rng(seed)
    if condition == "contaminated":
        n_calib = min(4000, len(eligible_normal_idx))
        calib_idx = rng.choice(eligible_normal_idx, size=n_calib, replace=False)
    else:  # adversarial
        n_calib = min(4000, len(eligible_normal_idx))
        eligible_exposure = exposure[eligible_mask]
        order = np.argsort(-eligible_exposure)
        calib_idx = eligible_normal_idx[order][:n_calib]

    remaining_normal = np.setdiff1d(eligible_normal_idx, calib_idx)
    max_normal_test = 5000
    if len(remaining_normal) > max_normal_test:
        remaining_normal = rng.choice(remaining_normal, size=max_normal_test, replace=False)

    test_idx = np.concatenate([remaining_normal, anomaly_idx])
    test_labels = np.concatenate([
        np.zeros(len(remaining_normal), dtype=int),
        np.ones(len(anomaly_idx), dtype=int),
    ])
    return calib_idx, test_idx, test_labels


def run_single_seed_trial(graph, features, labels, condition, alpha, seed, n_epochs, device):
    """Method A on real data: single model, degree-normalized, trimmed, conformal+BH."""
    scores, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed,
                                verbose=False, device=device)
    scores = degree_normalize_scores(graph, scores)

    calib_idx, test_idx, test_labels = _prepare_calib_test(graph, labels, scores, condition, seed)
    if calib_idx is None:
        return None

    p_values = conformal_p_values(scores[calib_idx], scores[test_idx])
    discoveries = benjamini_hochberg(p_values, alpha)

    n_discoveries = discoveries.sum()
    realized_fdr = (np.sum(discoveries & (test_labels == 0)) / n_discoveries) if n_discoveries > 0 else 0.0
    power = (np.sum(discoveries & (test_labels == 1)) / len(np.where(labels == 1)[0])) if labels.sum() > 0 else 0.0

    return {
        "method": f"single_seed_{condition}", "seed": seed, "n_ensemble": 1,
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }


def run_ensemble_trial(graph, features, labels, condition, alpha, seed, n_ensemble, n_epochs, device):
    """Method B on real data: ensemble-averaged scores, SAME degree-norm +
    trim + conformal+BH procedure as Method A -- mechanism-vs-mechanism,
    per the corrected comparison design."""
    all_scores = []
    for e in range(n_ensemble):
        ensemble_seed = seed * 1000 + e
        s, _ = train_dominant(graph, features, n_epochs=n_epochs,
                               seed=ensemble_seed, verbose=False, device=device)
        all_scores.append(degree_normalize_scores(graph, s))
    avg_scores = np.mean(all_scores, axis=0)

    calib_idx, test_idx, test_labels = _prepare_calib_test(graph, labels, avg_scores, condition, seed)
    if calib_idx is None:
        return None

    p_values = conformal_p_values(avg_scores[calib_idx], avg_scores[test_idx])
    discoveries = benjamini_hochberg(p_values, alpha)

    n_discoveries = discoveries.sum()
    realized_fdr = (np.sum(discoveries & (test_labels == 0)) / n_discoveries) if n_discoveries > 0 else 0.0
    power = (np.sum(discoveries & (test_labels == 1)) / len(np.where(labels == 1)[0])) if labels.sum() > 0 else 0.0

    return {
        "method": f"ensemble_{condition}", "seed": seed, "n_ensemble": n_ensemble,
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }


def summarize(all_results, method, alpha):
    subset = [r for r in all_results if r["method"] == method]
    if not subset:
        return
    fdrs = np.array([r["realized_fdr"] for r in subset])
    powers = np.array([r["power"] for r in subset])
    n_disc = np.array([r["n_discoveries"] for r in subset])
    n_zero = int(np.sum(n_disc == 0))
    n_total = len(subset)
    print(f"{method}: marginal realized_fdr={fdrs.mean():.3f}+/-{fdrs.std():.3f} "
          f"(nominal={alpha}), power={powers.mean():.3f}+/-{powers.std():.3f}, "
          f"zero-discovery={n_zero}/{n_total}")
    if n_zero < n_total:
        cond = fdrs[n_disc > 0]
        print(f"  conditional (nonzero only, {n_total-n_zero} seeds): "
              f"realized_fdr={cond.mean():.3f}+/-{cond.std():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon", choices=["amazon"],
                         help="Yelp excluded: documented scalability limitation (see README).")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--n_ensemble", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.dataset}...")
    graph, features, labels = load_fraud_graph(args.dataset)
    print(f"Graph: {graph.number_of_nodes()} nodes, {labels.sum()} fraud "
          f"({labels.mean():.4f} rate)\n")

    all_results = []
    for condition in ["contaminated", "adversarial"]:
        print(f"=== Single-seed, {condition} ({args.n_seeds} seeds) ===")
        for seed in range(args.n_seeds):
            r = run_single_seed_trial(graph, features, labels, condition,
                                       args.alpha, seed, args.n_epochs, device)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            print(f"  seed {seed}: n_discoveries={r['n_discoveries']:3d} "
                  f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")

        print(f"\n=== Ensemble ({args.n_ensemble} models), {condition} ({args.n_seeds} trials) ===")
        for seed in range(args.n_seeds):
            r = run_ensemble_trial(graph, features, labels, condition, args.alpha,
                                    seed, args.n_ensemble, args.n_epochs, device)
            if r is None:
                print(f"  seed {seed}: skipped")
                continue
            all_results.append(r)
            print(f"  seed {seed}: n_discoveries={r['n_discoveries']:3d} "
                  f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")
        print()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"baseline_comparison_real_{args.dataset}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved raw results to {csv_path}\n")

    print("=== Summary ===")
    for condition in ["contaminated", "adversarial"]:
        summarize(all_results, f"single_seed_{condition}", args.alpha)
        summarize(all_results, f"ensemble_{condition}", args.alpha)
        print()

    print("=== Paired t-tests (mechanism-vs-mechanism) ===")
    for condition in ["contaminated", "adversarial"]:
        single = [r["realized_fdr"] for r in all_results if r["method"] == f"single_seed_{condition}"]
        ens = [r["realized_fdr"] for r in all_results if r["method"] == f"ensemble_{condition}"]
        if len(single) == len(ens) and len(single) >= 5:
            t = stats.ttest_rel(ens, single)
            print(f"{condition}: ensemble vs. single-seed: t={t.statistic:.3f}, p={t.pvalue:.4f}")

    print("\nInterpretation: tests whether the synthetic-data finding (ensembling adds no "
          "measurable value once the conformal+BH procedure is held constant) generalizes "
          "to real fraud data.")


if __name__ == "__main__":
    main()