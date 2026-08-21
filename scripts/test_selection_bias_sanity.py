"""
test_selection_bias_sanity.py

Independent check of selection_bias.py's anticonservativeness() on known
cases: a true exchangeable null (gamma should be ~1) and an engineered
anti-conservative case (gamma should be clearly > 1). Not a replacement
for exchangeable_null_pvalues() -- a quick sanity gate before trusting
gamma numbers from a real run.

Usage:
  python3 scripts/test_selection_bias_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from selection_bias import anticonservativeness, bh_threshold_from_rejections


def main():
    rng = np.random.default_rng(0)
    n_calib = 2000
    grid = np.arange(1, n_calib + 2) / (n_calib + 1)

    # Large n_null: small-threshold noise (few events at tiny t) otherwise
    # swamps the check -- confirmed directly: at n_null=5000, bh_t=0.001
    # expects only ~5 events, and a single draw gave gamma_at_bh=1.20 from
    # noise alone, not a bug. 200000 fixes this.
    print("=== Exchangeable null (should give gamma ~ 1) ===")
    p_null = rng.choice(grid, size=200000, replace=True)
    bh_t = bh_threshold_from_rejections(20000, 200000, 0.10)
    r = anticonservativeness(p_null, n_calib, bh_threshold=bh_t)
    print(f"mean_p={r['mean_p']:.3f}  gamma_at_bh={r['gamma_at_bh']:.3f}")
    assert abs(r["mean_p"] - 0.5) < 0.02, "mean_p should be ~0.5 under exchangeability"
    assert abs(r["gamma_at_bh"] - 1.0) < 0.10, "gamma_at_bh should be ~1.0 under exchangeability"
    print("PASSED")

    print("\n=== Anti-conservative case (should give gamma > 1) ===")
    p_bad = rng.beta(0.5, 2.0, size=200000)
    p_bad = np.clip(np.round(p_bad * n_calib) / n_calib, grid.min(), grid.max())
    bh_t2 = bh_threshold_from_rejections(20000, 200000, 0.10)
    r2 = anticonservativeness(p_bad, n_calib, bh_threshold=bh_t2)
    print(f"mean_p={r2['mean_p']:.3f}  gamma_at_bh={r2['gamma_at_bh']:.3f}")
    assert r2["mean_p"] < 0.4, "mean_p should be clearly below 0.5"
    assert r2["gamma_at_bh"] > 1.5, "gamma_at_bh should be clearly above 1"
    print("PASSED")

    print("\nALL TESTS PASSED")


def test_null_pvalue_formula_matches_conformal_fdr():
    """_null_p_values_one_draw's formula is claimed (in its own docstring) to
    match conformal_fdr.conformal_p_values exactly. Verify directly rather
    than trust the docstring."""
    from conformal_fdr import conformal_p_values
    rng = np.random.default_rng(0)
    calib = rng.random(500)
    test = rng.random(300)

    p_ref = conformal_p_values(calib, test)
    n_calib = 500
    calib_sorted = np.sort(calib)
    count_ge = n_calib - np.searchsorted(calib_sorted, test, side="left")
    p_alt = (count_ge + 1) / (n_calib + 1)

    max_diff = np.max(np.abs(p_ref - p_alt))
    print(f"\n=== _null_p_values_one_draw formula vs conformal_p_values ===")
    print(f"max diff: {max_diff}")
    assert max_diff < 1e-12, "formulas diverge -- the docstring's claim is wrong"
    print("PASSED")


def test_exchangeable_null_pvalues_end_to_end():
    """Feed exchangeable_null_pvalues() an observed statistic drawn from a
    TRUE exchangeable null and confirm null_p is not systematically small
    (it should look like a random draw from its own reference distribution,
    not always flag as significant)."""
    from selection_bias import exchangeable_null_pvalues, anticonservativeness, bh_threshold_from_rejections

    rng = np.random.default_rng(1)
    n_calib, n_null = 500, 2000
    grid = np.arange(1, n_calib + 2) / (n_calib + 1)
    p_obs = rng.choice(grid, size=n_null, replace=True)
    bh_t = bh_threshold_from_rejections(200, n_null, 0.10)
    observed = anticonservativeness(p_obs, n_calib, bh_threshold=bh_t)

    result = exchangeable_null_pvalues(observed, n_calib, n_null,
                                       bh_threshold=bh_t, n_sim=200, seed=2)
    print(f"\n=== exchangeable_null_pvalues on a TRUE null observation ===")
    print(f"gamma_at_bh_null_p={result['gamma_at_bh_null_p']:.3f} (should not be tiny)")
    assert result["gamma_at_bh_null_p"] > 0.02, (
        "a true null observation being flagged as extremely significant "
        "(null_p < 0.02) suggests a bug in the null-simulation machinery"
    )
    print("PASSED")


if __name__ == "__main__":
    main()
    test_null_pvalue_formula_matches_conformal_fdr()
    test_exchangeable_null_pvalues_end_to_end()