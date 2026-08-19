"""
adadetect_comparison.py

Head-to-head against AdaDetect (Marandon et al., 2024), the closest
FDR-controlling novelty-detection method in the literature.

This closes the most serious gap in the paper's evaluation: every baseline so
far (ensemble averaging, naive top-k) is a variant of our own procedure, so a
reviewer can fairly ask what our method buys over prior work. AdaDetect is the
right comparator because it targets the same object we do -- a discovery set
with finite-sample FDR control -- and differs on exactly one axis: its score
ADAPTS to the observed test mixture, where ours is a fixed reconstruction
error.

WHAT MAKES THE COMPARISON FAIR (see src/adadetect.py for the full argument):

  * Every arm shares ONE calibration/test partition, built from the DOMINANT
    score before AdaDetect's score exists, so n_calib, m, the p-value floor
    1/(n_calib+1), and the BH feasibility rank ceil(m/(alpha*(n_calib+1))) are
    identical across arms. A naive AdaDetect that split the reference pool
    would halve n_calib and lose on power arithmetically -- the same tautology
    this project already had to retract once in baseline_comparison.py.
  * D_train is carved from normals the frozen protocol ALREADY discards (the
    max_normal_test=5000 cap), so on the real-data path calib_idx and test_idx
    are byte-identical to the published runs.
  * Every arm routes through adadetect.evaluate_arm, the single call site for
    conformal_p_values + benjamini_hochberg.

PRE-REGISTERED PRIMARY COMPARISON: adadetect_embed vs ours_matched. The other
variants are exploratory -- score1d is a correctness control (it must
reproduce our arm), feat is under-informed by construction, embed_score grants
AdaDetect strictly more information than we have.

Run on Colab:
  python3 scripts/adadetect_comparison.py --dataset synthetic --n_seeds 20 --device cuda
  python3 scripts/adadetect_comparison.py --dataset amazon --n_seeds 20 --device cuda
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv
import numpy as np
from scipy import stats

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from adadetect import (
    VARIANTS,
    build_matched_frame,
    build_covariates,
    encoder_embedding,
    adadetect_scores,
    evaluate_arm,
    frame_floor_stats,
)
from detectors import score_nodes, available_detectors

SYNTHETIC = "synthetic"

# Matches the configuration every other synthetic experiment in this repo uses.
SYNTH_CFG = dict(n_nodes=15000, p_aa=0.3, p_an=0.002, p_nn=0.005,
                 feature_shift=1.0, n_anomaly_clusters=3)

ROW_FIELDS = [
    "dataset", "condition", "method", "variant", "classifier", "method_key",
    "seed", "alpha", "n_calib", "n_train", "n_test", "n_test_null", "n_test_alt",
    "p_floor", "bh_min_rank", "in_matched_frame", "frame_exact_match_frozen",
    "jitter_scale", "frac_unique_test_scores", "max_tied_block",
    "n_discoveries", "realized_fdr", "power", "score_auroc", "clf_partition_auc",
]


def _row(**kw):
    """Every row must carry every field -- csv.DictWriter derives fieldnames
    from the first row, so a missing key on a later row raises."""
    row = {f: "" for f in ROW_FIELDS}
    row.update(kw)
    return row


def load_graph(dataset, seed, use_degree_norm_default=True):
    """Returns (graph, features, labels, frame_kind, use_degree_norm)."""
    if dataset == SYNTHETIC:
        cfg = GraphGenConfig(random_state=seed, **SYNTH_CFG)
        graph, features, labels = ContaminatedGraphGenerator(cfg).generate()
        return graph, features, labels, "synthetic", False

    from real_data_experiment import load_any_dataset, DEGREE_NORM_BY_DATASET
    graph, features, labels = load_any_dataset(dataset)
    use_degree_norm = DEGREE_NORM_BY_DATASET.get(dataset, use_degree_norm_default)
    return graph, features, labels, "trimmed_real", use_degree_norm


def run_seed(dataset, condition, seed, args, cached):
    """One (condition, seed): train DOMINANT once, then evaluate every arm on
    the same frame. Sharing the trained model across arms is what makes the
    cross-arm pairing exact enough to license a paired test."""
    key = (dataset, seed)
    if key not in cached:
        graph, features, labels, frame_kind, use_degree_norm = load_graph(dataset, seed)
        scores, model = score_nodes(args.detector, graph, features, labels=labels,
                                    n_epochs=args.n_epochs, seed=seed,
                                    device=args.device, use_sparse_prop=args.use_sparse_prop,
                                    return_model=True)
        if use_degree_norm:
            from real_data_experiment import degree_normalize_scores
            scores = degree_normalize_scores(graph, scores)
        embedding = encoder_embedding(graph, features, model, device=args.device,
                                      use_sparse_prop=args.use_sparse_prop)
        degrees = np.array([graph.degree(i) for i in range(graph.number_of_nodes())],
                           dtype=float)
        cached.clear()  # only ever hold one graph -- these are large
        cached[key] = (graph, features, labels, scores, embedding, degrees, frame_kind)

    graph, features, labels, scores, embedding, degrees, frame_kind = cached[key]

    frame = build_matched_frame(
        graph, labels, scores, condition, seed,
        frame=frame_kind, n_train_target=args.n_train_target)
    if frame is None:
        return []

    ci, ti, tri, tl = (frame["calib_idx"], frame["test_idx"],
                       frame["train_idx"], frame["test_labels"])
    floors = frame_floor_stats(frame["n_calib"], frame["n_test"], args.alpha)

    def base(**kw):
        return _row(
            dataset=dataset, condition=condition, seed=seed, alpha=args.alpha,
            n_calib=frame["n_calib"], n_train=frame["n_train"],
            n_test=frame["n_test"], n_test_null=frame["n_test_null"],
            n_test_alt=frame["n_test_alt"],
            p_floor=floors["p_floor"], bh_min_rank=floors["bh_min_rank"],
            in_matched_frame=True,
            frame_exact_match_frozen=frame["frame_exact_match_frozen"],
            jitter_scale=args.jitter_scale, **kw)

    rng_jitter = np.random.default_rng((seed, 11))
    rows = []

    ours = evaluate_arm(scores[ci], scores[ti], tl, args.alpha,
                        rng_jitter=rng_jitter, jitter_scale=args.jitter_scale,
                        n_anomaly_total=frame["n_anomaly_total"])
    rows.append(base(method="ours_matched", variant="none", classifier="none",
                     method_key=f"ours_matched_none_{condition}",
                     **{k: v for k, v in ours.items() if not k.startswith("_")}))

    for variant in args.variants:
        X = build_covariates(variant, features=features, scores=scores,
                             embedding=embedding, degrees=degrees)
        cs, ts, partition_auc = adadetect_scores(
            X[tri], X[ti], X[ci], classifier=args.classifier, seed=seed,
            device=args.device)
        res = evaluate_arm(cs, ts, tl, args.alpha,
                           rng_jitter=np.random.default_rng((seed, 11)),
                           jitter_scale=args.jitter_scale,
                           n_anomaly_total=frame["n_anomaly_total"])
        rows.append(base(method="adadetect", variant=variant,
                         classifier=args.classifier,
                         method_key=f"adadetect_{variant}_{condition}",
                         clf_partition_auc=partition_auc,
                         **{k: v for k, v in res.items() if not k.startswith("_")}))

    return rows


def assert_frame_invariant(rows):
    """The single most important check a reviewer can run. Within one
    (dataset, condition, seed), every matched-frame arm must share n_calib,
    n_test, p_floor and bh_min_rank -- otherwise the arms are not comparable
    and any power difference is partly an artifact of the floor."""
    groups = {}
    for r in rows:
        if r["in_matched_frame"] is not True:
            continue
        groups.setdefault((r["dataset"], r["condition"], r["seed"]), []).append(r)

    for key, group in groups.items():
        for field in ("n_calib", "n_test", "p_floor", "bh_min_rank"):
            values = {r[field] for r in group}
            if len(values) > 1:
                raise AssertionError(
                    f"FRAME INVARIANT VIOLATED at {key}: {field} differs across arms "
                    f"({values}). The resolution floor is not matched, so the "
                    f"comparison is confounded. Do not report these numbers.")
    return len(groups)


def summarize(rows, method_key, alpha):
    subset = [r for r in rows if r["method_key"] == method_key]
    if not subset:
        return None
    fdr = np.array([r["realized_fdr"] for r in subset])
    power = np.array([r["power"] for r in subset])
    ndisc = np.array([r["n_discoveries"] for r in subset])
    n_zero = int((ndisc == 0).sum())

    print(f"{method_key}:")
    print(f"    marginal FDR = {fdr.mean():.4f} +/- {fdr.std():.4f} (nominal {alpha}), "
          f"power = {power.mean():.4f} +/- {power.std():.4f}, "
          f"zero-discovery = {n_zero}/{len(subset)}")
    if n_zero < len(subset):
        cond = fdr[ndisc > 0]
        flag = "  <-- ABOVE NOMINAL" if cond.mean() > alpha else ""
        print(f"    conditional FDR (nonzero only, n={len(cond)}) = "
              f"{cond.mean():.4f} +/- {cond.std():.4f}{flag}")
    return {"fdr": fdr, "power": power, "ndisc": ndisc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=SYNTHETIC)
    parser.add_argument("--detector", type=str, default="dominant_pygod",
                        choices=available_detectors(),
                        help="Which detector trains the scores every arm is built from. "
                             "Defaults to dominant_pygod (the correct implementation), NOT "
                             "dominant_ours (the frozen, broken detector -- see "
                             "DETECTOR_DIAGNOSTIC.md) despite the latter being adadetect.py's "
                             "original historical default. Pass --detector dominant_ours "
                             "explicitly if you need to reproduce a prior broken-detector run "
                             "for comparison; do not rely on this flag's default silently "
                             "changing old results if this script is rerun without re-reading "
                             "this help text.")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["contaminated", "adversarial"])
    parser.add_argument("--variants", type=str, nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--classifier", type=str, default="logreg",
                        choices=["logreg", "gbm", "rf"])
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_train_target", type=int, default=2000)
    parser.add_argument("--jitter_scale", type=float, default=0.0,
                        help="Randomized tie-breaking. Keep 0.0 for primary runs: "
                             "any nonzero value perturbs the DOMINANT scores and "
                             "breaks bit-exact agreement with the frozen results. "
                             "Use ~1e-9 only for the tree-ensemble tie ablation.")
    parser.add_argument("--use_sparse_prop", action="store_true",
                        help="Large-graph path; required for yelp (n=45,954).")
    args = parser.parse_args()

    import torch
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    print(f"Dataset: {args.dataset} | variants: {args.variants} | "
          f"classifier: {args.classifier} | jitter: {args.jitter_scale}\n")

    if args.jitter_scale != 0.0:
        print("  NOTE: jitter is ON, so 'ours_matched' will NOT reproduce the frozen\n"
              "  published numbers bit-exactly. This is expected for the tie ablation\n"
              "  only; report it separately from the primary table.\n")

    all_rows, cached = [], {}
    for condition in args.conditions:
        print(f"=== {condition} ===")
        for seed in range(args.n_seeds):
            rows = run_seed(args.dataset, condition, seed, args, cached)
            if not rows:
                print(f"  seed {seed}: skipped (frame infeasible)")
                continue
            all_rows.extend(rows)
            ours = next(r for r in rows if r["method"] == "ours_matched")
            parts = " ".join(
                f"{r['variant'][:5]}={r['n_discoveries']:4d}"
                for r in rows if r["method"] == "adadetect")
            print(f"  seed {seed}: n_cal={ours['n_calib']} m={ours['n_test']} "
                  f"min_rank={ours['bh_min_rank']:3d} | ours={ours['n_discoveries']:4d} {parts}")
        print()

    if not all_rows:
        print("No results -- every frame was infeasible. Check clean-pool size.")
        return

    n_groups = assert_frame_invariant(all_rows)
    print(f"Frame invariant holds across all {n_groups} (dataset, condition, seed) groups.\n")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"adadetect_comparison_{args.dataset}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows to {csv_path}\n")

    print("=== Summary ===")
    for condition in args.conditions:
        summarize(all_rows, f"ours_matched_none_{condition}", args.alpha)
        for v in args.variants:
            summarize(all_rows, f"adadetect_{v}_{condition}", args.alpha)
        print()

    print("=== Paired tests: ours_matched vs each AdaDetect variant ===")
    print("(pre-registered primary comparison is 'embed'; the rest are exploratory)\n")
    for condition in args.conditions:
        ours = [r for r in all_rows if r["method_key"] == f"ours_matched_none_{condition}"]
        for v in args.variants:
            ada = [r for r in all_rows if r["method_key"] == f"adadetect_{v}_{condition}"]
            if len(ada) != len(ours) or len(ours) < 5:
                continue
            o_fdr = np.array([r["realized_fdr"] for r in ours])
            a_fdr = np.array([r["realized_fdr"] for r in ada])
            o_pow = np.array([r["power"] for r in ours])
            a_pow = np.array([r["power"] for r in ada])

            t_fdr = stats.ttest_rel(a_fdr, o_fdr)
            tag = " [PRIMARY]" if v == "embed" else ""
            print(f"{condition} / {v}{tag}:")
            print(f"    FDR   ttest_rel t={t_fdr.statistic:+.3f} p={t_fdr.pvalue:.4f}")
            # Power is bounded and zero-inflated when zero-discovery seeds are
            # common, so Wilcoxon is the primary test; t-test reported for
            # continuity with the repo's existing tables.
            if np.any(a_pow - o_pow):
                w = stats.wilcoxon(a_pow, o_pow)
                t_pow = stats.ttest_rel(a_pow, o_pow)
                print(f"    power wilcoxon W={w.statistic:.1f} p={w.pvalue:.4f} | "
                      f"ttest_rel t={t_pow.statistic:+.3f} p={t_pow.pvalue:.4f}")
            else:
                print(f"    power identical across all seeds (no test applicable)")
        print()


if __name__ == "__main__":
    main()