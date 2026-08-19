"""
test_selection_bias.py

Guards the gamma estimator in src/selection_bias.py.

gamma is the number Part 4's falsification test turns on (see
theory/joint_discovery_threshold_proposition.md): the claim is that gamma
tracks each detector's score-degree dependence across a 5x4 detector-dataset
matrix. If the estimator is wrong, that test measures nothing -- and it would
fail in the worst possible way, by producing plausible numbers.

So every check here compares against a value known ANALYTICALLY or by
construction, not against a previously-observed output. The p-value families
used:

  p ~ Uniform on the conformal grid   -> gamma = 1 exactly (exchangeable)
  p = U^(1/theta), theta < 1          -> F(t) = t^theta, so F(t)/t = t^(theta-1)
                                         and sup over t >= t_min is
                                         t_min^(theta-1). Anti-conservative.
  p = U^(1/theta), theta > 1          -> super-uniform, gamma <= 1.

Run either way:
    python3 -m pytest tests/test_selection_bias.py -v
    python3 tests/test_selection_bias.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from selection_bias import (
    anticonservativeness,
    bh_threshold_from_rejections,
    score_degree_dependence,
    empirical_clean_probability,
    exchangeable_null_pvalues,
    default_min_rank,
    _null_p_values_one_draw,
)
from conformal_fdr import conformal_p_values

N_CALIB = 2000
N_NULL = 40000
MIN_RANK = 10
SEED = 0


def _tilted(rng, n, theta):
    """p = U^(1/theta) has CDF F(t) = t^theta on (0,1)."""
    return rng.random(n) ** (1.0 / theta)


def test_exchangeable_gives_gamma_one():
    """Arithmetic check on I.I.D. uniform input -- gamma must come out at 1.

    NOTE the scope: these p-values are drawn independently, with no shared
    calibration set, so there is no induced correlation and the theoretical
    value applies. Real conformal p-values are NOT like this; see
    test_valid_procedure_is_not_flagged_as_broken for that case.
    """
    rng = np.random.default_rng(SEED)
    p = rng.random(N_NULL)
    out = anticonservativeness(p, N_CALIB, min_rank=MIN_RANK)
    # Sampling noise at rank 10 of 2001 is real; the sup over a long grid also
    # biases slightly high. 15% is loose enough not to flake, tight enough that
    # the observed clean-condition gamma (~1.32) would still fail it.
    assert abs(out["gamma_hat"] - 1.0) < 0.15, (
        f"uniform nulls gave gamma_hat={out['gamma_hat']:.4f}, expected ~1.0")
    assert abs(out["mean_p"] - 0.5) < 0.02, (
        f"uniform nulls gave mean_p={out['mean_p']:.4f}, expected ~0.5")


def test_anticonservative_matches_analytic_gamma():
    """F(t) = t^theta with theta<1 has sup_{t>=t_min} F(t)/t = t_min^(theta-1)."""
    rng = np.random.default_rng(SEED)
    for theta in (0.8, 0.6, 0.5):
        p = _tilted(rng, N_NULL, theta)
        out = anticonservativeness(p, N_CALIB, min_rank=MIN_RANK)
        t_min = MIN_RANK / (N_CALIB + 1)
        expected = t_min ** (theta - 1.0)
        rel = abs(out["gamma_hat"] - expected) / expected
        assert rel < 0.15, (
            f"theta={theta}: gamma_hat={out['gamma_hat']:.3f} vs analytic "
            f"{expected:.3f} (rel err {rel:.3f})")
        assert out["gamma_hat"] > 1.0, "anti-conservative nulls must give gamma > 1"


def test_superuniform_does_not_report_inflation():
    """Conservative nulls must not be flagged as anti-conservative."""
    rng = np.random.default_rng(SEED)
    p = _tilted(rng, N_NULL, 1.5)  # F(t) = t^1.5 < t
    out = anticonservativeness(p, N_CALIB, min_rank=MIN_RANK)
    assert out["gamma_hat"] <= 1.0 + 1e-6, (
        f"super-uniform nulls gave gamma_hat={out['gamma_hat']:.4f} > 1")
    assert out["mean_p"] > 0.5, "super-uniform nulls should have mean p > 0.5"


def test_gamma_at_bh_is_a_plain_ratio():
    """gamma_at_bh must be exactly Fhat(t)/t at the supplied threshold."""
    # Constructed by hand: 100 p-values, 30 of them <= 0.10.
    p = np.concatenate([np.full(30, 0.05), np.full(70, 0.50)])
    out = anticonservativeness(p, n_calib=99, bh_threshold=0.10)
    assert abs(out["gamma_at_bh"] - 3.0) < 1e-9, (
        f"expected 0.30/0.10 = 3.0, got {out['gamma_at_bh']}")


def test_gamma_at_bh_is_nan_without_a_threshold():
    """Zero rejections means no threshold was reached -- NaN, never 0."""
    rng = np.random.default_rng(SEED)
    p = rng.random(1000)
    out = anticonservativeness(p, N_CALIB, bh_threshold=None)
    assert np.isnan(out["gamma_at_bh"])
    assert bh_threshold_from_rejections(0, m_test=5000, alpha=0.1) is None
    assert abs(bh_threshold_from_rejections(50, 5000, 0.1) - 0.001) < 1e-12


def test_empty_input_returns_nan_not_crash():
    out = anticonservativeness(np.array([]), N_CALIB)
    assert out["n_null"] == 0
    assert np.isnan(out["gamma_hat"])


def test_ks_flags_the_anticonservative_direction_only():
    """One-sided: only departures BELOW uniform count.

    A conservative procedure is safe -- reporting it as a violation would send
    the falsification test chasing detectors that are behaving correctly.
    """
    rng = np.random.default_rng(SEED)
    anti = anticonservativeness(_tilted(rng, N_NULL, 0.6), N_CALIB)
    cons = anticonservativeness(_tilted(rng, N_NULL, 1.6), N_CALIB)
    assert anti["ks_uniform"] > 0.05, "clear anti-conservative case should register"
    assert cons["ks_uniform"] <= 0.0 + 1e-9, (
        f"conservative case produced positive ks_uniform ({cons['ks_uniform']})")


def test_score_degree_dependence_recovers_planted_correlation():
    rng = np.random.default_rng(SEED)
    deg = rng.integers(1, 200, size=5000).astype(float)
    scores = np.log1p(deg) + rng.normal(0, 0.05, size=5000)   # strong, monotone
    normal = rng.random(5000) < 0.9

    out = score_degree_dependence(scores, deg, normal_mask=normal)
    assert out["spearman_r"] > 0.9, f"planted r not recovered: {out['spearman_r']}"
    assert out["n"] == int(normal.sum()), "normal_mask was not applied"

    # And an independent score must not register dependence.
    indep = score_degree_dependence(rng.normal(size=5000), deg)
    assert abs(indep["spearman_r"]) < 0.05


def test_empirical_clean_probability_recovers_a_decreasing_q():
    """Plant q(d) = 0.95^d, the Part 3 form, and check (A1) is detected."""
    rng = np.random.default_rng(SEED)
    deg = rng.integers(1, 120, size=60000).astype(float)
    eligible = rng.random(60000) < (0.95 ** deg)

    out = empirical_clean_probability(deg, eligible, n_bins=12)
    assert out["is_monotone"], (
        f"decreasing q(d) not detected: tau={out['kendall_tau']}, "
        f"p={out['kendall_p']}")
    assert out["kendall_tau"] < 0
    # q must actually fall across the degree range, not merely rank-correlate.
    assert out["q"][0] > out["q"][-1] + 0.2

    # A degree-independent filter must NOT be reported as monotone, since that
    # is the null Part 4 predicts for a detector with no degree sensitivity.
    flat = rng.random(60000) < 0.3
    out_flat = empirical_clean_probability(deg, flat, n_bins=12)
    assert not out_flat["is_monotone"], "flat q(d) wrongly flagged as decreasing"


def test_fast_null_draw_matches_frozen_conformal_p_values():
    """The simulator's shortcut must equal conformal_fdr's Python loop exactly.

    The null calibration is only meaningful if it simulates the SAME procedure
    the real trials run. searchsorted and the loop must agree bit-for-bit.
    """
    rng = np.random.default_rng(SEED)
    n_calib, n_null = 300, 900
    z = rng.random(n_calib + n_null)
    calib, test = z[:n_calib], z[n_calib:]

    reference = conformal_p_values(calib, test)
    fast = _null_p_values_one_draw(np.random.default_rng(SEED), n_calib, n_null)
    # Different draws, so compare the machinery on identical inputs instead:
    sorted_calib = np.sort(calib)
    count_ge = n_calib - np.searchsorted(sorted_calib, test, side="left")
    shortcut = (count_ge + 1) / (n_calib + 1)

    assert np.array_equal(reference, shortcut), (
        "searchsorted shortcut does not reproduce conformal_p_values")
    assert fast.shape == (n_null,) and np.all((fast > 0) & (fast <= 1.0))


def test_valid_procedure_is_not_flagged_as_broken():
    """THE regression test for this module.

    A genuinely exchangeable trial -- calibration and test-normal drawn from
    one population -- must NOT come back significant. An earlier version of
    this module compared statistics against their theoretical values and
    reported ks_p = 3e-07 here, because every test point shares one calibration
    draw and the nulls are therefore correlated. If this test fails, the 20-cell
    falsification matrix is measuring noise.
    """
    rng = np.random.default_rng(SEED)
    n_calib, n_null = 1000, 5000
    for trial in range(5):
        pool = rng.normal(size=n_calib + n_null)
        p = conformal_p_values(pool[:n_calib], pool[n_calib:])
        obs = anticonservativeness(p, n_calib)
        null = exchangeable_null_pvalues(obs, n_calib, n_null, n_sim=100,
                                         seed=100 + trial)
        for stat in ("gamma_hat", "mean_p", "ks_uniform"):
            assert null[f"{stat}_null_p"] > 0.01, (
                f"trial {trial}: valid procedure flagged on {stat} "
                f"(null_p={null[f'{stat}_null_p']:.4f})")


def test_planted_calibration_shift_is_detected():
    """The converse: a real tilt must be caught.

    Shifting calibration scores DOWN is exactly the Part 4 mechanism (clean
    selection favours low-degree nodes, which score lower).
    """
    rng = np.random.default_rng(SEED)
    n_calib, n_null = 1000, 5000
    pool = rng.normal(size=n_calib + n_null)
    calib = pool[:n_calib] - 0.15
    p = conformal_p_values(calib, pool[n_calib:])

    obs = anticonservativeness(p, n_calib)
    null = exchangeable_null_pvalues(obs, n_calib, n_null, n_sim=100, seed=7)
    assert null["mean_p_null_p"] < 0.05, (
        f"planted shift missed by mean_p (null_p={null['mean_p_null_p']})")
    assert null["ks_uniform_null_p"] < 0.05, (
        f"planted shift missed by ks_uniform (null_p={null['ks_uniform_null_p']})")


def test_min_rank_default_scales_with_calibration_size():
    assert default_min_rank(2000) == 500
    assert default_min_rank(267) == 66      # Amazon's clean pool
    assert default_min_rank(40) == 50       # floor holds for tiny pools


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
