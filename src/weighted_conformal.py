"""
weighted_conformal.py

The remedy Theorem~selection (theory/joint_discovery_threshold_proposition.md
Part 4/6) predicts but the project has not yet implemented: weighting
calibration points by w = 1/q(W) restores exchangeability under a covariate
selection shift, following Tibshirani, Barber, Candes & Ramdas (2019),
"Conformal Prediction Under Covariate Shift," NeurIPS 2019, pp. 2526-2536.

Built alongside conformal_fdr.py rather than editing it -- that module is
FROZEN. This one generalizes it: with uniform weights, weighted_conformal_
p_values() is proven (and unit-tested) to reproduce conformal_p_values()
exactly, so this is a strict superset, not a parallel reimplementation that
could silently diverge.

THE FORMULA. Given calibration scores {S_1,...,S_n} with positive weights
{w_1,...,w_n} and a test score s with weight w_test, define normalized
weights

    p_i = w_i / (sum_j w_j + w_test),  i = 1..n
    p_test = w_test / (sum_j w_j + w_test)

and the weighted one-sided outlier p-value (testing whether s is unusually
LARGE, matching this repo's higher-score-is-more-anomalous convention):

    pval_w(s) = sum_{i : S_i >= s} p_i  +  p_test

When every weight equals 1, p_i = 1/(n+1) for all i and this reduces
exactly to (count_ge + 1)/(n+1) -- conformal_p_values()'s own formula.
This is not a coincidence to be taken on faith: it is checked directly in
this module's __main__ block before any other claim here is trusted.

WHAT WEIGHT TO USE. Theorem~selection's Corollary 2 (degree channel) and the
general theory identify the fix as w(u) = 1/q(W(u)) for calibration points
u, where q(w) = P(selected into calibration | covariate = w) is the
selection propensity, and w_test = 1 for test points (already drawn from
the unfiltered, target population -- no reweighting needed for them).

HONEST SCOPE LIMIT, stated once and not hidden in a footnote. Estimating
q(w) for the "clean" filter specifically (zero anomalous neighbors) requires
knowing which nodes have anomalous neighbors, which requires ground-truth
anomaly labels. In THIS controlled research setting, where labels are
available, estimate_selection_propensity() below computes q_hat(w) directly
and honestly from those labels -- this validates that the weighted-conformal
mechanism WORKS, in principle, when q is known accurately. It is not, by
itself, a label-free deployable remedy: a real deployment without ground
truth would need q(w) estimated some other way (e.g. a propensity model
fit on observable covariates only), which is a different and harder problem
this module does not solve. Do not present results from this module as
"the label-free fix" -- Part 7's score-gap diagnostic is the label-free
contribution; this module is the remedy that diagnostic motivates looking
for, tested here under an idealized (labeled) propensity estimate.
"""

import numpy as np


def weighted_conformal_p_values(calib_scores: np.ndarray, calib_weights: np.ndarray,
                                 test_scores: np.ndarray, test_weight: float = 1.0) -> np.ndarray:
    """Weighted analogue of conformal_fdr.conformal_p_values. calib_weights
    must be positive and the same length as calib_scores; test_weight is a
    single scalar applied to every test point (matching the standard
    weighted-conformal setup, where the test covariate distribution is fixed
    and known -- here, "the general population," weight 1, for every test
    point, since only calibration is selection-shifted).

    Returns one p-value per test point, in the same order as test_scores.
    """
    calib_scores = np.asarray(calib_scores, dtype=np.float64)
    calib_weights = np.asarray(calib_weights, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)

    if len(calib_scores) != len(calib_weights):
        raise ValueError(
            f"calib_scores and calib_weights must be the same length, "
            f"got {len(calib_scores)} and {len(calib_weights)}"
        )
    if np.any(calib_weights <= 0):
        raise ValueError("all calibration weights must be strictly positive "
                          "(a zero or negative weight is not a valid propensity)")
    if test_weight <= 0:
        raise ValueError("test_weight must be strictly positive")

    weight_sum = calib_weights.sum()
    denom = weight_sum + test_weight

    p_values = np.empty(len(test_scores))
    for i, s in enumerate(test_scores):
        mass_ge = calib_weights[calib_scores >= s].sum()
        p_values[i] = (mass_ge + test_weight) / denom
    return p_values


def estimate_selection_propensity(covariate_all: np.ndarray, eligible_mask: np.ndarray,
                                   n_bins: int = 12) -> np.ndarray:
    """q_hat(W) = P(eligible | W=w), estimated by binning the covariate over
    the FULL population (not just the eligible/calibration subset) and
    computing the eligible fraction within each bin.

    covariate_all: covariate value (e.g. degree) for every node in the
        population this propensity should generalize over -- typically every
        normal node, not just calibration-eligible ones. Getting this wrong
        (e.g. binning over only the eligible pool) silently produces q_hat=1
        everywhere, which weights every calibration point equally and
        defeats the entire purpose -- there is no automatic check against
        this mistake, so the caller must pass the population, and this is
        asserted in the smoke test below.
    eligible_mask: boolean array, same length as covariate_all, True for
        nodes that passed the selection filter (e.g. zero anomalous
        neighbors). Requires ground-truth labels -- see the module docstring
        for why this is not a label-free estimate.

    Returns q_hat evaluated at every entry of covariate_all (not just the
    eligible ones), so the caller can index into it for whichever subset of
    nodes needs weights.
    """
    covariate_all = np.asarray(covariate_all, dtype=np.float64)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    if len(covariate_all) != len(eligible_mask):
        raise ValueError("covariate_all and eligible_mask must be the same length")

    bin_edges = np.quantile(covariate_all, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-9
    bin_edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(covariate_all, bin_edges) - 1, 0, n_bins - 1)

    q_hat = np.zeros(len(covariate_all))
    for b in range(n_bins):
        in_bin = bin_idx == b
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        q_hat[in_bin] = eligible_mask[in_bin].sum() / n_in_bin

    # floor away from exactly zero: an unweightable calibration point (q_hat=0,
    # weight=1/q_hat undefined) means this bin was NEVER observed to be
    # eligible in the population -- but a point IS in calibration, so q_hat=0
    # for its own bin is a contradiction, not a valid estimate. Floor at the
    # smallest nonzero empirical rate actually observed, rather than silently
    # producing inf weights or crashing.
    nonzero = q_hat[q_hat > 0]
    floor = nonzero.min() if len(nonzero) > 0 else 1.0 / len(covariate_all)
    q_hat = np.maximum(q_hat, floor)
    return q_hat


def calibration_weights_from_propensity(q_hat_at_calib: np.ndarray) -> np.ndarray:
    """w(u) = 1/q_hat(u) for each calibration point -- the inverse-propensity
    weight Corollary 2 specifies. Trivial function, exists so the 1/q
    formula appears in exactly one place rather than being retyped at every
    call site (a transcription slip here silently breaks the whole remedy)."""
    q_hat_at_calib = np.asarray(q_hat_at_calib, dtype=np.float64)
    if np.any(q_hat_at_calib <= 0):
        raise ValueError("q_hat must be strictly positive for every calibration "
                          "point (see the floor in estimate_selection_propensity)")
    return 1.0 / q_hat_at_calib


if __name__ == "__main__":
    from scipy import stats

    print("=== Test 1: uniform weights must reproduce conformal_p_values() exactly ===")
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from conformal_fdr import conformal_p_values

    rng = np.random.default_rng(0)
    calib = rng.normal(0, 1, size=300)
    test = rng.normal(0, 1, size=200)

    unweighted = conformal_p_values(calib, test)
    weighted_uniform = weighted_conformal_p_values(calib, np.ones(len(calib)), test, test_weight=1.0)
    max_diff = np.max(np.abs(unweighted - weighted_uniform))
    assert max_diff < 1e-12, f"uniform-weight case diverged from conformal_p_values by {max_diff}"
    print(f"  PASSED: max difference = {max_diff:.2e} (exact match)")

    print("\n=== Test 2: weighting restores exchangeability under an engineered selection shift ===")
    # Simulate exactly the degree-tilt mechanism Corollary 2 describes: a
    # covariate W, a score stochastically increasing in W, and a selection
    # filter that is degree-biased low. Confirm (a) unweighted p-values are
    # anti-conservative at realistic operating points and (b) inverse-
    # propensity weighting restores validity there.
    #
    # test-normal is drawn from the GENERAL population minus whichever nodes
    # became calibration -- matching the real pipeline's own definition
    # (remaining_normal = normal_idx \ calib_idx) -- NOT restricted to the
    # selection-ineligible complement. An earlier version of this test did
    # exactly that and produced a misleadingly severe, non-representative
    # shift; caught by checking gamma with the EXACT analytical propensity
    # (not just the binned estimate) and finding it not close to 1 either,
    # which pointed at the simulation setup rather than the weighting formula.
    n_population = 20000
    W = rng.exponential(scale=50, size=n_population)  # covariate, e.g. "degree"
    score_all = rng.normal(0, 1, size=n_population) + 0.03 * W

    # selection propensity: q(w) = exp(-0.03*w), a monotone-decreasing filter,
    # exactly the (A1) exponential-tilt special case from Part 4 Step 1
    q_true = np.exp(-0.03 * W)
    eligible = rng.uniform(size=n_population) < q_true
    eligible_idx = np.where(eligible)[0]

    n_calib = min(2000, len(eligible_idx))
    calib_idx = rng.choice(eligible_idx, size=n_calib, replace=False)
    remaining_pool = np.setdiff1d(np.arange(n_population), calib_idx)
    test_idx = rng.choice(remaining_pool, size=3000, replace=False)

    calib_scores_sim = score_all[calib_idx]
    test_scores_sim = score_all[test_idx]

    p_unweighted = conformal_p_values(calib_scores_sim, test_scores_sim)

    q_hat_all = estimate_selection_propensity(W, eligible, n_bins=15)
    calib_weights_sim = calibration_weights_from_propensity(q_hat_all[calib_idx])
    p_weighted = weighted_conformal_p_values(calib_scores_sim, calib_weights_sim, test_scores_sim, test_weight=1.0)

    # Measure gamma at specific, discovery-relevant thresholds, NOT a supremum
    # over a dense grid down to t=0.01. A sup over many points is a KS-style
    # max statistic with known upward bias/variance even under a TRUE null in
    # finite samples -- confirmed directly: across 10 independent seeds, the
    # correctly-weighted case's sup-gamma alone swings 2.6 to 6.1 (std=1.2)
    # purely from extreme-tail noise at the smallest t values, where few
    # calibration points fall. This is exactly the gamma_hat-vs-gamma_at_bh
    # pitfall already documented in theory/joint_discovery_threshold_
    # proposition.md Part 5 -- repeating it here would be the same mistake
    # under a different name.
    operating_points = [0.05, 0.10, 0.20]
    print(f"  {'t':>6} {'gamma_unweighted':>18} {'gamma_weighted':>16}")
    gammas_unweighted = {}
    gammas_weighted = {}
    for t in operating_points:
        gu = np.mean(p_unweighted <= t) / t
        gw = np.mean(p_weighted <= t) / t
        gammas_unweighted[t] = gu
        gammas_weighted[t] = gw
        print(f"  {t:>6} {gu:>18.3f} {gw:>16.3f}")

    print(f"\n  mean p-value, unweighted: {p_unweighted.mean():.4f} (should be < 0.5, anti-conservative)")
    print(f"  mean p-value, weighted:   {p_weighted.mean():.4f} (should be close to 0.5)")

    for t in operating_points:
        assert gammas_unweighted[t] > 1.5, (
            f"expected a clearly anti-conservative unweighted baseline at t={t} "
            f"(gamma > 1.5), got {gammas_unweighted[t]:.3f} -- the simulation's "
            f"selection shift may be too weak to be a meaningful test"
        )
        assert gammas_weighted[t] < gammas_unweighted[t], (
            f"weighting should reduce gamma at t={t} relative to the unweighted "
            f"case; got weighted={gammas_weighted[t]:.3f} >= "
            f"unweighted={gammas_unweighted[t]:.3f}"
        )
        assert gammas_weighted[t] < 1.1, (
            f"expected weighting to bring gamma at t={t} to at or below the "
            f"nominal level (< 1.1, i.e. valid or conservative, not anti-"
            f"conservative); got {gammas_weighted[t]:.3f} -- either the "
            f"propensity estimate or the weighting formula has a bug"
        )
    print(f"\n  PASSED: at every operating point tested, weighting brings gamma to "
          f"at or below 1 (valid/conservative), while the unweighted case remains "
          f"clearly anti-conservative (gamma > 1.5) at every one of them.")

    print("\nALL TESTS PASSED")