# Proposition: The Joint Discovery-Threshold Condition

## Status
Section 1 (floor-only) is the original derivation, verified against real data.
Section 2 (extended, rank-indexed) is new: it generalizes the argument beyond
points tied at the resolution floor, which is the direction the bidirectional
reading of the old proposition was falsified on (see PAPER_REFRAME_HANDOFF.md
section 5.5). Section 2's main result is stated in expectation, not as a
finite-sample guarantee -- see "What remains" at the bottom before treating it
as more than that.

---

## Part 1: Floor-only condition

### Setup

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

### Proposition (floor-only, sufficient condition)

Define the **clearance rate** c = c(1) = (fraction of true anomalies whose
score exceeds every calibration score). Discovery occurs (under the
simplifying assumption that no normal test point also achieves the floor
value) when

**pi_1 * c >= 1 / (alpha * (n_calib + 1))**              (*)

### Empirical verification (corrected pi_1)

The original verification table used the graph-wide anomaly rate for pi_1.
This is wrong: pi_1 is defined as test-set prevalence, m_1/m, and the
pipeline enriches the test set relative to the graph as a whole (calibration
removes normal nodes, and the pipeline caps test-set normals at 5000 while
keeping every anomaly). Corrected:

| Dataset  | pi_1 (graph-wide, WRONG) | pi_1 (test-set, CORRECT) | n_calib | Required c* (corrected) | AUROC  | Observed |
|----------|--------------------------|---------------------------|---------|--------------------------|--------|----------|
| Amazon   | 0.0687                   | 0.1410                    | 4000    | 0.0177                   | 0.7542 (measured; NOT 0.893 as in old draft) | Discovered |
| Reddit   | 0.0333                   | 0.0682                    | 4000    | 0.0366                   | 0.5773 | NOT discovered (0/20 seeds) |
| Tolokers | 0.2182                   | 0.3391                    | 4000    | 0.0074                   | 0.4390 (measured; NOT 0.4093 as in old draft) | Discovered |

Ordering argument survives unchanged (Tolokers < Amazon < Reddit required
clearance either way). Only the numbers change. AUROC values corrected per
PAPER_REFRAME_HANDOFF.md section 6.4 item 4.

The pattern still matches non-trivially: Reddit has BETTER raw AUROC than
Tolokers (0.577 vs 0.439, still below chance) but FAILED to discover
anything, while Tolokers succeeded. AUROC alone predicts the opposite
ordering; the required-c* framing resolves it via base rate.

---

## Part 2: Extended condition (beyond the floor)

### Why Part 1 is not sufficient on its own

Part 1 only governs test points tied exactly at the p-value floor. BH can
reject at any rank where the p-value falls below a rank-dependent threshold
that grows with rank, so a nonempty discovery set does not require any point
to reach the floor at all. This was caught empirically: under a correctly
implemented detector (`dominant_pygod`), Amazon under the clean condition
has n_calib=267, m=5821, required clearance count 218, and an OBSERVED
clearance count of only 134 -- Part 1 predicts NO discovery. The actual
result was 3,420 discoveries, off by a factor of ~25, driven by rejections
at ranks well above the floor. Part 1's bidirectional reading (using it to
also predict absence of discovery) is falsified by this. It remains valid
as a one-directional sufficient condition.

### Rank-indexed clearance

For a test point v with score S(v), define its calibration rank
r(v) = |{u in calib : S(u) >= S(v)}| + 1, so p(v) = r(v)/(n+1) and
r(v) in {1, ..., n+1}. For integer r in {1, ..., n+1}, define

    c(r) = (# true anomalies with r(v) <= r) / m_1

the fraction of anomalies whose p-value is at most r/(n+1). This
generalizes c = c(1) from Part 1. c(r) is nondecreasing, c(n+1) = 1.

### Lemma (exact null contribution under exchangeability)

If calib and a normal test point v are exchangeable (same condition
Proposition 1 in the paper already invokes for the clean condition), then
for any integer r in {1, ..., n+1}:

    P(r(v) <= r) = r / (n+1)    exactly

This follows because the combined set of n+1 scores (calib union {v}) is
exchangeable among its members, so v's score is marginally equally likely
to occupy any of the n+1 rank positions.

Let N_0(r) = (# normal test points with r(v) <= r). By linearity of
expectation over the m - m_1 normal test points (no independence needed
across them, only the marginal exchangeability above):

    E[N_0(r)] = (m - m_1) * r / (n+1)          exact identity      (**)

### The generalized discovery condition

Writing N(r) = (# test points, any label, with r(v) <= r), the discovery
set is nonempty iff there exists r in {1, ..., n+1} with

    N(r) >= (m/alpha) * r/(n+1)                                    (dagger)

This is an exact restatement of Fact 2, not an approximation.

**Proposition (generalized discovery condition).** Let calib satisfy the
exchangeability condition above. Define kappa(alpha, pi_1) = 1/alpha -
(1 - pi_1). If there exists integer r in {1,...,n+1} such that

    pi_1 * c(r) >= kappa(alpha, pi_1) * r / (n+1)                  (ddagger)

then, in expectation over the exchangeable randomness governing normal
test-point ranks, N(r) meets the threshold required by (dagger) at that r.

**Proof.** N(r) = m_1*c(r) + N_0(r). Taking expectations and substituting
(**): E[N(r)] = m_1*c(r) + (m-m_1)*r/(n+1). (dagger) holds in expectation
at rank r when this is >= (m/alpha)*r/(n+1); rearranging and dividing by m
gives (ddagger). QED.

**Corollary (recovers Part 1).** If additionally N_0(1) = 0 exactly (not
just in expectation), (ddagger) at r=1 reduces to pi_1*c(1) >=
1/(alpha*(n+1)), i.e. exactly (*) from Part 1.

### What this does and does not establish

Does: gives a sufficient (in-expectation) condition for discovery at ANY
rank, not just the floor, and formally explains why the Part-1 bidirectional
reading fails (a failure at r=1 leaves every r>1 open).

Does not: this is NOT a finite-sample, high-probability guarantee. (**) is
an exact expectation, but N_0(r) is a sum of dependent (negatively
associated, since ranks are computed against a shared calibration set, not
independent) indicator variables, and no concentration bound has been
derived here. A finite-sample version (discovery with probability >= 1-delta)
would need such a bound -- standard concentration inequalities for negatively
associated sums are the natural tool, but this has not been carried out.
Do not present Part 2 as more than an in-expectation result until this gap
is closed.

## What remains to formalize (next steps, in order of difficulty)

1. **Numerically verify (ddagger) against real rank data.** Compute c(r)
   directly from calibration_distribution_check.py output (or a small
   instrumented rerun that logs per-node ranks, not just aggregate
   power/FDR) for each of the 15 real-data cells, and check whether the
   best-case r satisfying (ddagger) predicts the observed discovery count
   direction correctly. This is the next concrete step, not a new
   experiment -- the rank data already exists implicitly in every trial,
   it just is not currently logged.
2. **Derive the finite-sample version.** Bound N_0(r) via a concentration
   inequality for negatively associated sums (see "What this does and does
   not establish" above). This is the piece that would turn the generalized
   discovery condition into an actual theorem rather than an in-expectation
   statement.
3. **Verify the severity-sweep prediction directly**, logging actual c(r)
   at each severity level under a CORRECT detector (dominant_pygod), not
   the broken one. Note: PAPER_REFRAME_HANDOFF.md section 4.8 flags that
   the "fails into silence" finding itself is suspect under the broken
   detector and needs retesting before this step is meaningful.
4. **Connect to the literature audit's Proposition 2/3 sketch** -- NOTE:
   `theory/theoretical_characterization_draft.md`, referenced by the old
   version of this document, does not exist in this repo. Do not cite it
   until/unless it is actually written.

## Confirm before publishing

- `m` values in any results table must be read directly from `m_test` in
  `results/logs/calibration_distribution_check.csv`, not reconstructed by
  hand.
- AUROC values used anywhere (Amazon, Tolokers especially) must match the
  measured values in this document (0.7542, 0.4390), not the old draft's
  incorrect 0.893 / 0.4093.