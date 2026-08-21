"""
selection_bias.py

Measures how far a calibration set departs from exchangeability with the
test-normal population, and what that costs BH.

WHY THIS EXISTS. condition_comparison_pygod.py found realized FDR
significantly ABOVE nominal under the CLEAN condition specifically (0.132 vs
alpha=0.10, d=0.837, p=0.0007) -- the one condition Proposition 1 proves is
exchangeable. theory/joint_discovery_threshold_proposition.md Part 4 explains
why: "clean" (zero anomalous neighbors) is a topological filter that selects on
degree, degree-sensitive scores inherit that tilt, and the null p-values end up
stochastically DOMINATED by Uniform rather than super-uniform. BH's guarantee
then holds only at gamma*alpha, for an anti-conservativeness factor

    gamma = sup_t P(p <= t) / t   >= 1

This module estimates gamma. It is the quantity Part 4's falsification test
turns on: gamma should track each detector's own score-degree dependence.

DELIBERATELY NOT IN conformal_fdr.py. That module is imported by 7+ scripts
which produced committed results; it is frozen. Nothing here modifies it -- the
p-value convention below is copied to MATCH it (higher score = more anomalous,
p = (#{calib >= test} + 1)/(n_calib + 1)), not to replace it.

DISCRETE SUPPORT MATTERS. Conformal p-values live on the grid
{1/(n+1), ..., (n+1)/(n+1)}, so gamma is evaluated on that grid exactly rather
than over a continuum. Under exact exchangeability P(p <= r/(n+1)) = r/(n+1),
so the estimator is a direct ratio against a known truth, with no asymptotics.

DO NOT COMPARE THESE STATISTICS AGAINST 1.0. This was measured, not assumed:
every test point in a trial shares ONE calibration draw, so the null p-values
are correlated, and a perfectly valid procedure does not produce gamma = 1.
On an exchangeable simulation with n_calib=2000, n_null=10000:

    min_rank=10   gamma_hat null mean 1.21, p95 1.57
    min_rank=500  gamma_hat null mean 1.04, p95 1.09

The clean-condition effect we are chasing sits near gamma = 1.32 -- INSIDE the
min_rank=10 null band. A naive threshold at 1.0 would have reported the effect
as real in cases where nothing was wrong, and a small min_rank cannot see the
effect at all. Hence two design decisions:

  1. min_rank defaults to max(50, n_calib // 4), not a small constant.
  2. Every statistic is reported against a SIMULATED exchangeable null via
     exchangeable_null_pvalues(), not against its theoretical value.

Relative discriminating power, measured on a planted calibration shift
(null 5th-95th percentile vs. mean under the effect):

    mean_p        null [0.489, 0.511]   effect 0.472   separates
    ks_uniform    null [0.004, 0.031]   effect 0.050   separates
    gamma_hat     null [1.003, 1.088]   effect 1.147   separates (min_rank=500)
    gamma_at_bh   null [0.647, 1.440]   effect 1.316   DOES NOT SEPARATE

gamma_at_bh is the theoretically meaningful quantity -- it is where BH actually
cut -- but it is evaluated at a single threshold with few points beneath it, so
its variance is enormous. Report it for interpretation; do NOT use it as the
primary statistic. mean_p is the most stable.

THE ENDPOINT TRAP (fixed, but worth understanding before changing this code).
Both sup-type statistics are evaluated over a RESTRICTED window, capped above
at n_calib//2. The cap is load-bearing, not tidying. At the final grid point
t = 1 we have F_hat(1) = 1 identically, so F_hat(t)/t = 1 and F_hat(t) - t = 0
there for EVERY possible input. A sup taken over a window containing t = 1 is
therefore floored: gamma_hat returns exactly 1.0000 and ks_uniform exactly
0.000000 for ANY conservative input, regardless of how conservative it is.

That is not a cosmetic issue. In the first 20-cell matrix run, roughly half the
cells were conservative detectors, and all of them reported those two identical
boundary values. A rank correlation computed over a block of exact ties at an
extreme is measuring the tie structure as much as the effect. With the window
capped, the same inputs now separate properly:

    family              gamma_hat   mean_p    ks_uniform
    very conservative      0.2502   0.7513     -0.234375
    conservative           0.5044   0.6658     -0.185292
    uniform                1.0233   0.4984     +0.006898
    anti-conservative      1.7487   0.3734     +0.190761

Any matrix produced BEFORE this fix has degenerate gamma_hat and ks_uniform
columns and must be regenerated. mean_p was never affected -- it is a plain
mean with no sup and no endpoint -- so mean_p results from an older run remain
valid, which matters because it is also the most stable of the three.
"""

import numpy as np
from scipy import stats


def _achievable_grid(n_calib: int) -> np.ndarray:
    """The p-values a conformal procedure with this calibration size can emit."""
    return np.arange(1, n_calib + 2) / (n_calib + 1)


def default_min_rank(n_calib: int) -> int:
    """Smallest rank the sup should consider.

    Measured, not guessed: at rank 10 of 2001 the exchangeable null already has
    a p95 of 1.57, which swamps the ~1.32 effect. Scaling with n_calib keeps the
    noise floor roughly constant as calibration size changes across datasets --
    Amazon's clean pool is 267 nodes while synthetic runs ~2800, and a fixed
    constant would mean something completely different in those two regimes.
    """
    return max(50, n_calib // 4)


def anticonservativeness(null_p_values: np.ndarray,
                         n_calib: int,
                         bh_threshold: float = None,
                         min_rank: int = None) -> dict:
    """Estimate the anti-conservativeness factor gamma from NULL p-values only.

    Args:
        null_p_values: conformal p-values of test points known to be NORMAL.
            Passing anomalies in here invalidates everything -- gamma is a
            statement about the null distribution.
        n_calib: calibration set size, which fixes the achievable p-value grid.
        bh_threshold: the realized BH rejection threshold alpha*k/m from the
            trial. Pass None (or use the k=0 case) to skip gamma_at_bh.
        min_rank: smallest rank r to include in the sup. Below this the ratio
            is driven by a handful of points and is mostly noise. None (the
            default) uses default_min_rank(n_calib); see the module docstring
            for the measurement that set it.

    Returns a dict with:
        gamma_hat      sup over r >= min_rank of Fhat(r/(n+1)) * (n+1)/r.
                       The headline robustness number.
        gamma_at_bh    Fhat(t)/t at the realized BH threshold. This is the
                       value that actually governs FDR inflation, since it is
                       where BH made its cut. NaN if bh_threshold is None.
        gamma_argmax_t the t attaining gamma_hat, for diagnosing whether the
                       sup sits in the tail (real) or at the boundary (noise).
        ks_uniform     one-sided KS statistic sup(Fhat(t) - t), i.e. the
                       anti-conservative direction only. The best-behaved of
                       the sup-type statistics.
        mean_p         mean null p-value. 0.5 under exchangeability; below 0.5
                       is anti-conservative. Blunt, but the MOST STABLE of the
                       four and the one to lead with.
        n_null         number of null p-values used.
        min_rank_used  the resolved min_rank, since it is now data-dependent.

    Interpretation requires exchangeable_null_pvalues(); see the module
    docstring. None of these should be compared against their theoretical
    values directly, because a shared calibration draw correlates the nulls.
    """
    p = np.asarray(null_p_values, dtype=float)
    p = p[np.isfinite(p)]
    n_null = len(p)

    if min_rank is None:
        min_rank = default_min_rank(n_calib)

    out = {
        "gamma_hat": np.nan,
        "gamma_at_bh": np.nan,
        "gamma_argmax_t": np.nan,
        "ks_uniform": np.nan,
        "mean_p": np.nan,
        "n_null": n_null,
        "min_rank_used": int(min_rank),
    }
    if n_null == 0:
        return out

    out["mean_p"] = float(p.mean())

    grid = _achievable_grid(n_calib)
    # Fhat on the achievable grid. searchsorted with side="right" counts
    # p <= t, which is the direction the BH rule uses.
    counts = np.searchsorted(np.sort(p), grid, side="right")
    f_hat = counts / n_null

    # --- gamma_hat: sup of the ratio, restricted at BOTH ends ---
    # The upper restriction is not cosmetic. At the last grid point t=1 we have
    # F_hat(1) = 1 identically, so F_hat(t)/t = 1 there for EVERY input. A sup
    # taken over a range including t=1 therefore has a hard floor of 1, and any
    # conservative detector returns exactly 1.0000 -- a degenerate value shared
    # by every conservative cell. That produced a block of exact ties at the
    # boundary in the 20-cell matrix, which is precisely the kind of structure
    # that manufactures a rank correlation. Capping at n_calib//2 keeps the sup
    # inside the region where BH actually operates and lets conservative inputs
    # score genuinely below 1.
    lo = min(max(min_rank, 1), len(grid)) - 1
    hi = max(lo + 1, len(grid) // 2)
    ratio = f_hat[lo:hi] / grid[lo:hi]
    if len(ratio):
        j = int(np.argmax(ratio))
        out["gamma_hat"] = float(ratio[j])
        out["gamma_argmax_t"] = float(grid[lo:hi][j])

    # --- gamma_at_bh: the ratio where BH actually cut ---
    if bh_threshold is not None and np.isfinite(bh_threshold) and bh_threshold > 0:
        f_at = float(np.mean(p <= bh_threshold))
        out["gamma_at_bh"] = f_at / float(bh_threshold)

    # --- one-sided KS against Uniform, anti-conservative direction ---
    # Compare against the DISCRETE uniform on the grid, not continuous U(0,1):
    # a correctly-calibrated conformal p-value is discrete, and testing it
    # against a continuous reference would flag valid procedures as invalid.
    #
    # No analytic p-value is attached. The usual exp(-2 n D^2) assumes i.i.d.
    # observations; these are not, because they share a calibration set. That
    # formula returned p = 3e-07 on a VERIFIED-exchangeable simulation. Use
    # exchangeable_null_pvalues() instead.
    # Same endpoint trap as gamma_hat: f_hat - grid is exactly 0 at t=1 for
    # every input, so an unrestricted max is floored at 0 and every
    # conservative detector returns exactly 0.000000. Restricted to the same
    # window so conservative inputs go properly negative.
    out["ks_uniform"] = float(np.max(f_hat[lo:hi] - grid[lo:hi]))

    return out


def _null_p_values_one_draw(rng, n_calib, n_null):
    """One exchangeable trial's null conformal p-values, computed fast.

    Matches conformal_fdr.conformal_p_values exactly -- p = (#{calib >= t}+1)
    /(n+1) -- but via searchsorted instead of that module's Python loop, since
    the null needs hundreds of trials. Asserted equivalent in tests.

    The statistics are rank-based, hence distribution-free, so drawing from
    Uniform costs no generality: only the relative ordering of calibration and
    test scores enters.
    """
    z = rng.random(n_calib + n_null)
    calib = np.sort(z[:n_calib])
    test = z[n_calib:]
    count_ge = n_calib - np.searchsorted(calib, test, side="left")
    return (count_ge + 1) / (n_calib + 1)


def exchangeable_null_pvalues(observed: dict,
                              n_calib: int,
                              n_null: int,
                              bh_threshold: float = None,
                              min_rank: int = None,
                              n_sim: int = 200,
                              seed: int = 0) -> dict:
    """Calibrate observed statistics against a simulated exchangeable null.

    This is the piece that makes the numbers interpretable. It simulates n_sim
    trials in which exchangeability HOLDS BY CONSTRUCTION, at the same n_calib
    and n_null as the real trial, and reports where each observed statistic
    falls in that null.

    Returns, for each statistic, `<stat>_null_mean`, `<stat>_null_p95` (or p05
    for mean_p, which departs downward) and `<stat>_null_p`: the one-sided
    proportion of null draws at least as extreme as observed. A null_p below
    0.05 means the departure exceeds what a valid procedure produces at this
    calibration size.

    n_sim=200 gives a resolution floor of 1/201 ~ 0.005 on null_p; raise it if
    a cell needs to distinguish p=0.01 from p=0.001.
    """
    if min_rank is None:
        min_rank = default_min_rank(n_calib)

    rng = np.random.default_rng(seed)
    keys = ("gamma_hat", "gamma_at_bh", "mean_p", "ks_uniform")
    draws = {k: np.full(n_sim, np.nan) for k in keys}

    for b in range(n_sim):
        p = _null_p_values_one_draw(rng, n_calib, n_null)
        s = anticonservativeness(p, n_calib, bh_threshold=bh_threshold,
                                 min_rank=min_rank)
        for k in keys:
            draws[k][b] = s[k]

    out = {"n_sim": n_sim}
    for k in keys:
        d = draws[k][np.isfinite(draws[k])]
        obs = observed.get(k, np.nan)
        if len(d) == 0 or not np.isfinite(obs):
            out[f"{k}_null_mean"] = np.nan
            out[f"{k}_null_p95"] = np.nan
            out[f"{k}_null_p"] = np.nan
            continue
        out[f"{k}_null_mean"] = float(d.mean())
        if k == "mean_p":
            # Anti-conservativeness pushes mean_p DOWN, so the extreme tail is
            # the lower one. Getting this backwards would report every valid
            # run as broken.
            out[f"{k}_null_p95"] = float(np.quantile(d, 0.05))
            out[f"{k}_null_p"] = float((d <= obs).mean())
        else:
            out[f"{k}_null_p95"] = float(np.quantile(d, 0.95))
            out[f"{k}_null_p"] = float((d >= obs).mean())
    return out


def adaptive_t_grid(n_calib: int, min_ranks: int = 10, n_points: int = 6,
                    t_max: float = 0.20) -> tuple:
    """t values that are actually MEASURABLE at this calibration size.

    Conformal p-values live on {1/(n+1), ..., 1}, so a threshold t only
    resolves floor(t*(n+1)) distinct ranks. A fixed grid ignores this and
    silently degenerates on small calibration sets: on amazon n_calib=267, so
    t=0.01 covers calibration ranks 1-2 ONLY. That is what produced
    gamma_t0.01 values of exactly 0.00 in the first beta sweep -- not a valid
    procedure, just a statistic with nothing to measure.

    Requiring at least min_ranks achievable ranks fixes it: t >= min_ranks/(n+1).
    """
    t_lo = max(min_ranks / (n_calib + 1.0), 1e-4)
    if t_lo >= t_max:
        return (round(t_lo, 5),)
    return tuple(round(float(t), 5)
                 for t in np.geomspace(t_lo, t_max, n_points))


def left_tail_gamma(null_p_values: np.ndarray, t_grid=(0.001, 0.002, 0.005,
                                                          0.01, 0.02, 0.05)) -> dict:
    """gamma = Fhat(t)/t evaluated in the LEFT TAIL, where BH actually cuts.

    WHY THIS EXISTS, and why gamma_hat is not enough. gamma_hat takes a sup over
    ranks r >= min_rank = n_calib//4. That controls variance, but it measures a
    region BH never visits. Measured on the first 20-cell run: on weibo,
    n_calib ~ 5659 so min_rank ~ 1250, while BH's realized threshold was
    t = 0.0022, i.e. calibration rank 11. gamma_hat looked 112x further into the
    bulk than the procedure it describes, reported 1.07 ("no violation"), while
    the true anti-conservativeness at BH's cut point was 6.94.

    That single mismatch is what produced a false "consistent with Part 4"
    verdict: the degree correlation is rho=+0.81 (p=0.0007) measured at
    gamma_hat's ranks and rho=+0.46 (p=0.13) measured where BH cuts.

    Variance is controlled the right way here -- by POOLING null p-values across
    seeds before calling this, so the same small t has more data behind it --
    rather than by moving t somewhere quieter and less relevant.

    Returns gamma at each t plus the count behind it, since a gamma computed
    from three points is not worth reading and the caller must be able to see
    that.
    """
    p = np.asarray(null_p_values, dtype=float)
    p = p[np.isfinite(p)]
    out = {}
    if len(p) == 0:
        for t in t_grid:
            out[f"gamma_t{t:g}"] = np.nan
            out[f"n_below_t{t:g}"] = 0
        return out
    for t in t_grid:
        n_below = int(np.sum(p <= t))
        out[f"gamma_t{t:g}"] = float((n_below / len(p)) / t)
        out[f"n_below_t{t:g}"] = n_below
    return out


def bh_threshold_from_rejections(n_rejections: int, m_test: int, alpha: float):
    """The p-value threshold BH effectively applied: alpha * k / m.

    Returns None when k = 0, since no threshold was reached and gamma_at_bh is
    undefined rather than zero. Callers should record the NaN, not impute.
    """
    if n_rejections <= 0:
        return None
    return alpha * n_rejections / m_test


def score_degree_dependence(scores: np.ndarray,
                            degrees: np.ndarray,
                            normal_mask: np.ndarray = None) -> dict:
    """Spearman(score, degree) among NORMAL nodes -- the x-axis of Part 4's
    falsification test.

    Restricted to normals on purpose: anomalies are the thing being detected,
    so including them measures detector accuracy, not the degree sensitivity
    that assumption (A2) is about.
    """
    s = np.asarray(scores, dtype=float)
    d = np.asarray(degrees, dtype=float)
    if normal_mask is not None:
        m = np.asarray(normal_mask, dtype=bool)
        s, d = s[m], d[m]
    ok = np.isfinite(s) & np.isfinite(d)
    s, d = s[ok], d[ok]
    if len(s) < 3 or np.all(d == d[0]):
        return {"spearman_r": np.nan, "spearman_p": np.nan, "n": len(s)}
    r, pv = stats.spearmanr(s, d)
    return {"spearman_r": float(r), "spearman_p": float(pv), "n": int(len(s))}


def empirical_clean_probability(degrees: np.ndarray,
                                eligible_mask: np.ndarray,
                                n_bins: int = 12) -> dict:
    """Measure q(d) = P(calibration-eligible | degree d) from data.

    Part 4 assumption (A1) requires only that q be NON-INCREASING in degree.
    The closed form q(d) = (1-pi)^d used in Part 3 assumes independent
    attachment, which is false on Amazon/Weibo/Yelp -- so on real graphs q must
    be MEASURED, and this is the function that does it.

    Bins are quantile-based so each carries comparable weight; on the heavy-
    tailed degree distributions here, equal-width bins would put almost every
    node in the first bin.

    Returns bin centers, q per bin, counts, and `is_monotone` -- whether the
    measured q is non-increasing (Kendall tau <= 0 with p < 0.05 counts as
    supporting (A1)).
    """
    d = np.asarray(degrees, dtype=float)
    e = np.asarray(eligible_mask, dtype=bool)

    edges = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return {"degree_bin": np.array([]), "q": np.array([]),
                "count": np.array([]), "is_monotone": False,
                "kendall_tau": np.nan, "kendall_p": np.nan}

    idx = np.clip(np.digitize(d, edges[1:-1], right=False), 0, len(edges) - 2)
    centers, qs, counts = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        centers.append(float(d[sel].mean()))
        qs.append(float(e[sel].mean()))
        counts.append(int(sel.sum()))

    centers = np.array(centers)
    qs = np.array(qs)
    counts = np.array(counts)

    tau, tau_p = (np.nan, np.nan)
    if len(centers) >= 3:
        tau, tau_p = stats.kendalltau(centers, qs)

    return {
        "degree_bin": centers,
        "q": qs,
        "count": counts,
        "kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
        "kendall_p": float(tau_p) if np.isfinite(tau_p) else np.nan,
        # (A1) is a claim about direction, so a positive tau is the thing that
        # would falsify it. Non-significant negative tau is weak support, not
        # a violation -- reported as such rather than collapsed to a bool alone.
        "is_monotone": bool(np.isfinite(tau) and tau <= 0 and tau_p < 0.05),
    }
