"""
conformal_fdr.py

Step 4: the core experiment. Tests H1 (does clustered, propagating
contamination break finite-sample FDR control?) against H4 (does the
fail-safe property survive on graphs?).

Pipeline:
1. Train the DOMINANT detector on the full graph (transductive — standard
   for GNN conformal prediction, and what makes the exchangeability argument
   from the literature audit apply at all).
2. Split NORMAL nodes into a calibration set and a test set. Anomalous nodes
   are never in calibration (semi-supervised novelty detection setup, same
   as AdaDetect / Bates et al.) — but we test two calibration conditions:
     - "clean" calibration: calibration nodes drawn only from the region with
       zero anomalous neighbors (the isolated-control group from Step 2).
     - "contaminated" calibration: calibration nodes drawn from anywhere,
       including nodes with high anomalous-neighbor exposure.
3. Compute conformal p-values for test nodes (mix of normal + anomalous)
   using the calibration scores.
4. Apply Benjamini-Hochberg at nominal FDR level alpha to get a discovery set.
5. Measure REALIZED FDR against ground truth labels.
6. Repeat across many seeds to get a distribution, not a single point estimate
   — single-run FDR numbers are exactly the kind of unstable result this
   whole project exists to interrogate.

This is deliberately kept as simple, standard conformal-FDR machinery
(Bates et al. / AdaDetect style p-values + BH) so that if something breaks,
it's attributable to the contamination condition, not to a nonstandard or
buggy conformal implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detector import train_dominant


def conformal_p_values(calib_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Standard conformal p-value: for each test score, the fraction of
    calibration scores that are >= it (plus a randomized tie-break term,
    plus the +1 correction for finite-sample validity), following
    Bates et al. (2023) / the standard split-conformal outlier-testing recipe.

    Higher anomaly score = more anomalous, so we test whether the test score
    is unusually LARGE relative to calibration (one-sided, upper tail).
    """
    n_calib = len(calib_scores)
    p_values = np.zeros(len(test_scores))
    for i, s in enumerate(test_scores):
        # count calibration scores >= test score (conservative direction)
        count_ge = np.sum(calib_scores >= s)
        p_values[i] = (count_ge + 1) / (n_calib + 1)
    return p_values


def benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Returns a boolean mask of rejected (discovered) hypotheses at level alpha."""
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    below = sorted_p <= thresholds
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_idx = np.max(np.where(below)[0])
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[: max_idx + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject


def run_single_trial(
    contamination_condition: str,  # "clean" or "contaminated"
    alpha: float = 0.10,
    seed: int = 0,
    n_epochs: int = 100,
    calib_frac: float = 0.4,
    device: str = "cpu",
    score_alpha: float = 0.5,  # weight on attribute vs structure loss in the detector
):
    assert contamination_condition in ("clean", "contaminated")

    cfg = GraphGenConfig(
        n_nodes=15000, p_aa=0.3, p_an=0.01, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=seed,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    scores, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed, verbose=False, device=device, alpha=score_alpha)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    # compute anomalous-neighbor exposure for normal nodes (same metric as Step 2)
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)

    rng = np.random.default_rng(seed)

    rng = np.random.default_rng(seed)

    # Hold calibration SET SIZE constant across conditions. Otherwise a smaller
    # clean-calibration pool produces coarser conformal p-values (minimum
    # possible p-value is 1/(n_calib+1)), which alone could suppress
    # discoveries regardless of contamination -- a confound unrelated to H1.
    clean_pool = normal_idx[exposure == 0]
    n_calib_fixed = int(round(calib_frac * len(clean_pool)))  # bottleneck = smaller pool

    if len(clean_pool) < 20:
        return None  # not enough clean nodes to calibrate on at this seed — skip

    if contamination_condition == "clean":
        eligible_calib_pool = clean_pool
    else:
        eligible_calib_pool = normal_idx  # anywhere, including exposed nodes

    n_calib = n_calib_fixed
    calib_idx = rng.choice(eligible_calib_pool, size=n_calib, replace=False)

    # test set: remaining normal nodes (not used for calibration) + all anomalies
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
    if n_discoveries == 0:
        realized_fdr = 0.0
    else:
        false_discoveries = np.sum(discoveries & (test_labels == 0))
        realized_fdr = false_discoveries / n_discoveries

    n_true_anomalies_found = np.sum(discoveries & (test_labels == 1))
    power = n_true_anomalies_found / len(anomaly_idx) if len(anomaly_idx) > 0 else 0.0

    return {
        "condition": contamination_condition,
        "seed": seed,
        "alpha": alpha,
        "n_calib": n_calib,
        "n_discoveries": int(n_discoveries),
        "realized_fdr": realized_fdr,
        "power": power,
    }


if __name__ == "__main__":
    print("Running single-trial smoke test for both conditions (seed=0)...\n")
    for condition in ["clean", "contaminated"]:
        result = run_single_trial(condition, seed=0)
        print(f"[{condition}] {result}")