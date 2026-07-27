"""
baseline_comparison.py

Step 8: the ensemble-averaging baseline. This is the single most important
comparison a TCYB reviewer would demand -- flagged explicitly in the
original literature audit as the "free baseline" that could kill the
paper's framing if it dominates: if simply averaging raw anomaly scores
across N training seeds achieves comparable false-discovery control and
power WITHOUT any conformal p-value / BH machinery, then the conformal
apparatus this whole project is built around adds no real value beyond a
much simpler ensembling trick.

What this tests, precisely:
- Method A (our method): conformal p-values from a SINGLE seed's scores,
  calibrated against a held-out set, thresholded via Benjamini-Hochberg.
  This is what multi_seed_sweep.py / severity_sweep.py / real_data_
  experiment.py all measure.
- Method B (the baseline): train N models across N seeds, AVERAGE their
  raw anomaly scores per node, then flag the top-k nodes by averaged score
  using a simple fixed threshold (e.g. top X% by score) -- no conformal
  calibration, no BH, no formal guarantee. This is the "free ensemble"
  a reviewer will point to.

Since Method B has NO formal FDR guarantee, the fair comparison is: does
Method B, even without a guarantee, achieve LOWER OR SIMILAR realized FDR
in practice compared to Method A, for similar or better power? If yes,
Method A's conformal machinery needs a different selling point (the formal
guarantee itself, not just better empirical numbers) -- if no, Method A
has a clear empirical edge in addition to the theoretical guarantee.

Run on Colab:
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


def run_ensemble_baseline_trial(alpha, seed, n_ensemble, n_epochs, device, top_k_frac=None):
    """Method B: train n_ensemble models across different seeds (derived
    deterministically from the trial seed), average their raw scores, and
    flag the top-k nodes by averaged score. top_k_frac defaults to the
    anomaly rate (0.05) -- the most favorable, "oracle-informed" choice for
    the baseline, since in practice you wouldn't know the true anomaly rate;
    this is deliberately generous to the baseline to make the comparison fair."""
    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=0.002, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    if top_k_frac is None:
        top_k_frac = labels.mean()  # oracle anomaly rate, generous to baseline

    all_scores = []
    for e in range(n_ensemble):
        ensemble_seed = seed * 1000 + e  # distinct, reproducible seed per ensemble member
        scores, _ = train_dominant(graph, features, n_epochs=n_epochs,
                                    seed=ensemble_seed, verbose=False, device=device)
        all_scores.append(scores)

    avg_scores = np.mean(all_scores, axis=0)
    n_nodes = len(labels)
    k = int(round(top_k_frac * n_nodes))
    top_k_idx = np.argsort(-avg_scores)[:k]

    flagged = np.zeros(n_nodes, dtype=bool)
    flagged[top_k_idx] = True

    n_discoveries = flagged.sum()
    realized_fdr = (np.sum(flagged & (labels == 0)) / n_discoveries) if n_discoveries > 0 else 0.0
    power = (np.sum(flagged & (labels == 1)) / labels.sum()) if labels.sum() > 0 else 0.0

    return {
        "method": "ensemble_baseline", "seed": seed, "n_ensemble": n_ensemble,
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }


def run_our_method_trial(alpha, seed, n_epochs, device, calib_frac=0.4):
    """Method A: our conformal+BH pipeline, single seed, 'contaminated'
    (random calibration) condition -- the standard, average-case setting,
    for a fair like-for-like comparison against the ensemble baseline
    (which also uses no special adversarial selection)."""
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
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)

    clean_pool = normal_idx[exposure == 0]
    if len(clean_pool) < 20:
        return None
    n_calib = int(round(calib_frac * len(clean_pool)))

    rng = np.random.default_rng(seed)
    calib_idx = rng.choice(normal_idx, size=n_calib, replace=False)
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

    n_discoveries = discoveries.sum()
    realized_fdr = (np.sum(discoveries & (test_labels == 0)) / n_discoveries) if n_discoveries > 0 else 0.0
    power = (np.sum(discoveries & (test_labels == 1)) / len(anomaly_idx)) if len(anomaly_idx) > 0 else 0.0

    return {
        "method": "our_conformal_bh", "seed": seed, "n_ensemble": 1,
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }


def run_naive_threshold_trial(alpha, seed, n_epochs, device, top_k_frac=None):
    """Method C: the simplest possible baseline. Single seed, single model,
    NO conformal p-values, NO BH, NO ensembling -- just threshold the raw
    anomaly scores at the oracle anomaly rate. This isolates whether the
    conformal+BH machinery adds value over doing nothing statistically
    sophisticated at all, as opposed to Method B (ensemble baseline) which
    isolates whether SEED-DERANDOMIZATION specifically adds value. Together,
    A vs C tests the whole apparatus; A vs B tests just the ensembling piece."""
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
        "method": "naive_threshold", "seed": seed, "n_ensemble": 1,
        "n_discoveries": int(n_discoveries), "realized_fdr": realized_fdr, "power": power,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--n_ensemble", type=int, default=5,
                         help="Number of models to ensemble for the baseline.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    all_results = []

    print(f"=== Method A: our conformal+BH pipeline ({args.n_seeds} seeds) ===")
    for seed in range(args.n_seeds):
        r = run_our_method_trial(args.alpha, seed, args.n_epochs, device)
        if r is None:
            print(f"  seed {seed}: skipped")
            continue
        all_results.append(r)
        print(f"  seed {seed}: n_discoveries={r['n_discoveries']:3d} "
              f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")

    print(f"\n=== Method B: ensemble-averaging baseline "
          f"({args.n_seeds} trials, {args.n_ensemble} models averaged each) ===")
    for seed in range(args.n_seeds):
        r = run_ensemble_baseline_trial(args.alpha, seed, args.n_ensemble, args.n_epochs, device)
        all_results.append(r)
        print(f"  seed {seed}: n_discoveries={r['n_discoveries']:3d} "
              f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")

    print(f"\n=== Method C: naive single-seed threshold baseline ({args.n_seeds} seeds) ===")
    for seed in range(args.n_seeds):
        r = run_naive_threshold_trial(args.alpha, seed, args.n_epochs, device)
        all_results.append(r)
        print(f"  seed {seed}: n_discoveries={r['n_discoveries']:3d} "
              f"realized_fdr={r['realized_fdr']:.3f} power={r['power']:.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "baseline_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved raw results to {csv_path}")

    print("\n=== Summary ===")
    for method in ["our_conformal_bh", "ensemble_baseline", "naive_threshold"]:
        subset = [r for r in all_results if r["method"] == method]
        if not subset:
            continue
        fdrs = np.array([r["realized_fdr"] for r in subset])
        powers = np.array([r["power"] for r in subset])
        print(f"{method}: realized_fdr={fdrs.mean():.3f}+/-{fdrs.std():.3f} "
              f"(nominal={args.alpha}), power={powers.mean():.3f}+/-{powers.std():.3f}")

    ours = [r["realized_fdr"] for r in all_results if r["method"] == "our_conformal_bh"]
    baseline_b = [r["realized_fdr"] for r in all_results if r["method"] == "ensemble_baseline"]
    baseline_c = [r["realized_fdr"] for r in all_results if r["method"] == "naive_threshold"]
    if len(ours) == len(baseline_b) and len(ours) >= 5:
        ttest_b = stats.ttest_rel(baseline_b, ours)
        print(f"\nPaired t-test (Method B ensemble baseline vs. ours, same seeds): "
              f"t={ttest_b.statistic:.3f}, p={ttest_b.pvalue:.4f}")
    if len(ours) == len(baseline_c) and len(ours) >= 5:
        ttest_c = stats.ttest_rel(baseline_c, ours)
        print(f"Paired t-test (Method C naive threshold vs. ours, same seeds): "
              f"t={ttest_c.statistic:.3f}, p={ttest_c.pvalue:.4f}")

    print("\nInterpretation: if the ensemble baseline's FDR is comparable to or lower than "
          "ours WITHOUT any formal guarantee, our method's selling point must rest on the "
          "guarantee itself (worst-case protection) rather than better typical-case numbers. "
          "If the baseline's FDR is meaningfully higher or more variable, that is a real "
          "empirical advantage for the conformal approach, not just a theoretical one. "
          "Method C (naive threshold) is expected to have the least controlled FDR of the "
          "three -- if it doesn't, that's an important finding: it would mean the conformal "
          "machinery adds no measurable value at all, only a theoretical guarantee.")


if __name__ == "__main__":
    main()