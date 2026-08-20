"""
synthetic_difficulty_sweep.py

Fixes a real weakness and tests a prediction at the same time.

THE WEAKNESS. Every synthetic result in this project was produced at
feature_shift=1.0 with feature_dim=16 and feature_scale=1.0, which is a
Mahalanobis separation of 1.0*sqrt(16) = 4.0 sigma between normals and
anomalies. A working detector hits AUROC 1.0000 +/- 0.0000 there. FDR claims
made in a perfect-separation regime are close to vacuous, and a referee will
say so. The generator is not broken -- it was simply never calibrated to a
regime where detection is non-trivial.

THE PREDICTION. The selection effect should get WORSE as the task gets harder,
and the synthetic number in the paper (clean FDR 0.132) should therefore be an
UNDERSTATEMENT rather than an exaggeration.

Reasoning: FDR = V/(V+S). Selection filtering inflates V by shifting the null
p-values. At AUROC 1.0 the true anomalies occupy every top rank, so S is huge
and swamps the inflated V -- the violation is present but hidden. As AUROC
falls, S shrinks, the shifted nulls compete directly with the anomalies, and
the same violation produces a much larger FDR. This is the same V-versus-S
confound identified on real data (theory doc Part 5), where weibo at power 0.12
showed FDR 0.65 from a smaller gamma than amazon's.

If instead the clean-vs-random gap SHRINKS as AUROC falls, the reasoning above
is wrong and the synthetic result depends on perfect separation -- which would
be worth knowing before it goes in a paper.

WHAT IT DOES. Sweeps feature_shift, and at each level reports the detector's
AUROC alongside a matched clean-vs-random calibration comparison (same
n_calib, same test set, same p-value floor -- the frame invariants are
asserted). Pick the shift whose AUROC lands in a realistic band (0.75-0.90)
and use it for future synthetic runs.

DOES NOT CHANGE ANY DEFAULT. GraphGenConfig is untouched, so every existing
synthetic result still reproduces. This only measures what the alternatives
would look like.

Run:
  python scripts/synthetic_difficulty_sweep.py --n_seeds 5 --device cuda
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv

import numpy as np
import torch

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detectors import available_detectors, score_nodes
from conformal_fdr import conformal_p_values, benjamini_hochberg
from selection_bias import (
    anticonservativeness, bh_threshold_from_rejections,
    left_tail_gamma, adaptive_t_grid,
)

# 1.0 is the value every existing synthetic result used (4.0 sigma separation).
# The lower end is included to find where detection becomes non-trivial.
DEFAULT_SHIFTS = [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]


def auroc(scores, labels):
    """Mann-Whitney AUROC. numpy only, to keep this path sklearn-free."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def run_one(shift, seed, args):
    cfg = GraphGenConfig(n_nodes=args.n_nodes, p_aa=0.3, p_an=0.002, p_nn=0.005,
                         feature_shift=shift, n_anomaly_clusters=3,
                         random_state=seed)
    graph, features, labels = ContaminatedGraphGenerator(cfg).generate()
    scores = score_nodes(args.detector, graph, features, labels=labels, seed=seed,
                         n_epochs=args.n_epochs, device=args.device)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    au = auroc(scores, labels)

    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        nb = list(graph.neighbors(int(i)))
        if nb:
            exposure[j] = sum(1 for n in nb if labels[n] == 1) / len(nb)

    rng = np.random.default_rng(seed)
    ns = scores[normal_idx]
    cutoff = np.percentile(ns, 100 * (1 - args.trim_pct))
    eligible = normal_idx[ns <= cutoff]
    elig_mask = np.isin(normal_idx, eligible)
    unexposed = normal_idx[(exposure == 0) & elig_mask]

    n_test_normal = min(args.n_test_normal, len(eligible) // 2)
    test_normal = rng.choice(eligible, size=n_test_normal, replace=False)
    test_idx = np.concatenate([test_normal, anomaly_idx])
    test_labels = np.concatenate([np.zeros(len(test_normal), dtype=int),
                                  np.ones(len(anomaly_idx), dtype=int)])

    pools = {"clean": np.setdiff1d(unexposed, test_normal),
             "random": np.setdiff1d(eligible, test_normal)}
    n_calib = min(len(p) for p in pools.values())
    if n_calib < 50:
        return None

    t_grid = adaptive_t_grid(n_calib)
    out = []
    for strat, pool in pools.items():
        calib_idx = rng.choice(pool, size=n_calib, replace=False)
        p = conformal_p_values(scores[calib_idx], scores[test_idx])
        rej = benjamini_hochberg(p, args.alpha)
        k = int(rej.sum())
        fdr = float(np.sum(rej & (test_labels == 0)) / k) if k else 0.0
        power = float(np.sum(rej & (test_labels == 1)) / max(1, int(test_labels.sum())))
        null_p = p[test_labels == 0]
        bh_t = bh_threshold_from_rejections(k, len(test_idx), args.alpha)
        obs = anticonservativeness(null_p, n_calib, bh_threshold=bh_t)
        tail = left_tail_gamma(null_p, t_grid=t_grid)
        out.append({
            "feature_shift": shift, "seed": seed, "strategy": strat,
            "auroc": au, "n_calib": n_calib, "m_test": len(test_idx),
            "gamma_t_lo": tail[f"gamma_t{t_grid[0]:g}"], "t_lo": t_grid[0],
            "mean_p": obs["mean_p"], "n_discoveries": k,
            "realized_fdr": fdr, "power": power,
        })
    # Same invariant as calibration_strategy_comparison: unmatched frames make
    # the clean-vs-random comparison meaningless.
    for key in ("n_calib", "m_test", "t_lo"):
        if len({r[key] for r in out}) != 1:
            raise AssertionError(f"shift={shift} seed={seed}: {key} unmatched")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=float, nargs="+", default=DEFAULT_SHIFTS)
    ap.add_argument("--detector", type=str, default="dominant_pygod",
                    choices=available_detectors())
    ap.add_argument("--n_nodes", type=int, default=15000)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n_epochs", type=int, default=100)
    ap.add_argument("--n_test_normal", type=int, default=2000)
    ap.add_argument("--trim_pct", type=float, default=0.01)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {args.device}   detector: {args.detector}")
    print(f"feature_shift sweep: {args.shifts}")
    print("Existing synthetic results all used feature_shift=1.0 (AUROC 1.000).\n")

    rows = []
    for shift in args.shifts:
        for seed in range(args.n_seeds):
            r = run_one(shift, seed, args)
            if r is None:
                print(f"  shift={shift} seed={seed}: skipped (pool too small)")
                continue
            rows.extend(r)
        sub = [r for r in rows if r["feature_shift"] == shift]
        if sub:
            print(f"  shift={shift:<5} AUROC={np.mean([r['auroc'] for r in sub]):.4f}")

    if not rows:
        print("Nothing completed.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"synthetic_difficulty_{args.detector}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}")

    print()
    print("=" * 78)
    print("DOES THE SELECTION EFFECT GROW AS THE TASK GETS HARDER?")
    print("=" * 78)
    print(f"{'shift':>7} {'AUROC':>7} | {'clean FDR':>10} {'rand FDR':>9} "
          f"{'gap':>7} | {'clean g':>8} {'rand g':>7} | {'clean pow':>9}")
    print("-" * 78)
    gaps = []
    for shift in args.shifts:
        c = [r for r in rows if r["feature_shift"] == shift and r["strategy"] == "clean"]
        q = [r for r in rows if r["feature_shift"] == shift and r["strategy"] == "random"]
        if not c or not q:
            continue
        au = np.mean([r["auroc"] for r in c])
        cf, qf = np.mean([r["realized_fdr"] for r in c]), np.mean([r["realized_fdr"] for r in q])
        cg, qg = np.nanmean([r["gamma_t_lo"] for r in c]), np.nanmean([r["gamma_t_lo"] for r in q])
        cp = np.mean([r["power"] for r in c])
        print(f"{shift:>7.2f} {au:>7.4f} | {cf:>10.3f} {qf:>9.3f} {cf-qf:>7.3f} | "
              f"{cg:>8.2f} {qg:>7.2f} | {cp:>9.3f}")
        gaps.append((au, cf - qf))

    print()
    usable = [g for g in gaps if np.isfinite(g[0]) and np.isfinite(g[1])]
    if len(usable) < 3:
        print("  Too few usable levels to judge the trend.")
        return
    easy = [g[1] for g in usable if g[0] >= 0.95]
    hard = [g[1] for g in usable if g[0] < 0.95]
    if easy and hard:
        print(f"  mean clean-vs-random FDR gap at AUROC >= 0.95: {np.mean(easy):+.3f}")
        print(f"  mean clean-vs-random FDR gap at AUROC <  0.95: {np.mean(hard):+.3f}")
        if np.mean(hard) > np.mean(easy):
            print()
            print("  The selection effect is LARGER on harder tasks. The paper's")
            print("  synthetic number (clean FDR 0.132 at AUROC 1.000) therefore")
            print("  UNDERSTATES the problem, and the perfect-separation regime was")
            print("  hiding it rather than manufacturing it.")
        else:
            print()
            print("  The effect SHRINKS on harder tasks. The synthetic result depends")
            print("  on near-perfect separation -- report it with that caveat, and do")
            print("  not generalise it to realistic detection regimes.")
    print()
    print("  For future synthetic runs pick the shift whose AUROC lands in")
    print("  0.75-0.90. GraphGenConfig's default is deliberately unchanged so")
    print("  every existing result still reproduces.")


if __name__ == "__main__":
    main()
