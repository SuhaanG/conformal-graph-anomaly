# Proposition: The Joint Discovery-Threshold Condition

## Status
Derived from first principles (conformal p-value mechanics + Benjamini-
Hochberg rejection rule), then verified against three independent real
datasets it was NOT fit to. This is a candidate centerpiece for the
paper's theoretical section -- ready for review/refinement, not a
finished, formally verified theorem. Bring this to the advisor as a
starting point.

## Setup

Let n = n_calib (calibration set size), m = test set size, m_1 = number
of true anomalies in the test set, pi_1 = m_1/m (test-set anomaly
prevalence), and alpha = nominal FDR target.

Conformal p-value for test point j: p_j = (|{i in calib : S_i >= T_j}| + 1)
/ (n + 1), where S are calibration scores and T are test scores.

**Fact 1 (resolution floor).** The minimum achievable p-value is
1/(n+1), attained exactly when a test point's score exceeds every
calibration score.

**Fact 2 (BH rejection rule).** Benjamini-Hochberg rejects the k
smallest p-values, where k = max{i : p_(i) <= alpha*i/m}. In particular,
if k points are tied at the floor p-value 1/(n+1), those points are
jointly rejectable only if 1/(n+1) <= alpha*k/m, i.e.
**k >= m / (alpha*(n+1))**.

## Proposition

Define the **clearance rate** c = (fraction of true anomalies whose
score exceeds every calibration score), a quantity that should increase
monotonically with detector quality (AUROC) and decrease as calibration
contamination pushes calibration scores upward. The expected number of
anomalies achieving the p-value floor is m_1 * c = m * pi_1 * c.

Substituting into Fact 2's threshold: at least one discovery occurs
(under the simplifying assumption that normal/null test points do not
also achieve the floor value, which holds when the detector has any
real separation) when

**pi_1 * c >= 1 / (alpha * (n_calib + 1))**              (*)

This says: whether the procedure discovers ANYTHING is jointly governed
by the anomaly base rate (pi_1) and detector quality (via c) -- NOT by
detector quality (AUROC) alone. A low base rate can prevent discovery
even with a good detector; a high base rate can permit discovery even
with a poor detector, provided enough anomalies clear the ceiling by
chance alone.

## Empirical verification (post-hoc, not fit)

For each dataset, compute the REQUIRED clearance rate c* = 1 /
(alpha*(n_calib+1)*pi_1) -- the minimum clearance rate needed to satisfy
(*). Compare against whether discovery was actually observed.

| Dataset  | pi_1   | n_calib | Required c* | AUROC  | Observed |
|----------|--------|---------|-------------|--------|----------|
| Amazon   | 0.0687 | 4000    | 0.0364      | 0.8925 | Discovered (power ~0.08, every seed) |
| Reddit   | 0.0333 | 4000    | 0.0750      | 0.5773 | NOT discovered (0/20 seeds) |
| Tolokers | 0.2182 | 4000    | 0.0115      | 0.4093 | Discovered (power ~0.008, every seed) |

The pattern matches exactly, and non-trivially: Reddit has BETTER raw
AUROC than Tolokers (0.577 vs 0.409, the latter below chance) but
FAILED to discover anything, while Tolokers succeeded. AUROC alone
predicts the opposite ordering. The required-c* framing correctly
resolves this: Reddit's low base rate (3.3%) sets a much higher bar
(7.5% clearance needed) than Tolokers' high base rate (21.8%) sets
(1.15% needed) -- low enough that even a below-chance detector clears it
by chance alone.

**This is the key claim for the paper:** discovery activity is not
simply "detector quality is good enough" -- it is a joint condition
between base rate and detector quality, mediated by calibration size,
and derivable directly from the mechanics of the procedure itself.

## What remains to formalize (next steps, in order of difficulty)

1. **Define c rigorously and relate it to AUROC.** Currently c is
   described qualitatively ("increases with AUROC"). A cleaner version
   would express c as a function of the score distributions of
   normal/anomalous populations directly (e.g., via order statistics of
   the calibration set's maximum), which would let this proposition be
   stated purely in terms of measurable distributional quantities
   rather than "AUROC" as an informal proxy.
2. **Verify the severity-sweep prediction directly**, rather than only
   arguing consistency. This requires re-running a small instrumented
   version of the pipeline that logs the ACTUAL clearance rate c at
   each severity level (currently not logged -- only aggregate
   power/FDR were recorded), then checking whether c crosses the
   predicted threshold (0.0716 for that setup) exactly between
   p_an=0.02 and p_an=0.05, matching where power collapsed to zero.
   This is a cheap, targeted analysis script, not a new experiment --
   should be built next.
3. **State and prove the graceful-degradation direction formally**:
   under adversarial (worst-case) calibration selection, as contamination
   magnitude increases, calibration scores increase (stochastically),
   which mechanically decreases c toward zero -- giving a clean
   monotonicity argument for why the system fails into silence rather
   than into false discovery (the "adversarial" condition's calibration
   scores rising can only WEAKLY increase p-values for test points,
   never manufacture artificially low ones, unlike the asymmetric-
   trimming bug found earlier in this project). This direction (validity
   is preserved) is likely easier to formalize than the discovery-
   threshold condition above, and is closer to a proper theorem than a
   proposition.
4. **Connect explicitly to the literature audit's Proposition 2/3
   sketch** (see theory/theoretical_characterization_draft.md from
   earlier in this project) -- this document's Fact 1/Fact 2 mechanics
   are the concrete instantiation of what that earlier scaffold left
   abstract.