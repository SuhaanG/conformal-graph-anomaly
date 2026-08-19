# Proposition: The Joint Discovery-Threshold Condition

## Status
Section 1 (floor-only) is the original derivation, verified against real data.
Section 2 (extended, rank-indexed) is new: it generalizes the argument beyond
points tied at the resolution floor, which is the direction the bidirectional
reading of the old proposition was falsified on (see PAPER_REFRAME_HANDOFF.md
section 5.5). Section 2's main result is stated in expectation, not as a
finite-sample guarantee -- see "What remains" at the bottom before treating it
as more than that.

Infrastructure to numerically check (ddagger) now exists:
`scripts/severity_sweep_pygod_instrumented.py`,
`scripts/condition_comparison_pygod.py`, and
`scripts/real_data_experiment.py --log_ranks` all produce rank-grid CSVs, and
`scripts/verify_extended_proposition.py` checks any of them against observed
discovery outcomes. The severity sweep has been run once (synthetic,
adversarial-only, AUROC=1.0 at every level) and every trial had the floor
condition alone already succeed -- that run did NOT stress-test the
extension, since the interesting case (floor fails, larger r succeeds) never
arose. The interesting case is still only confirmed in the Amazon/clean
numbers already cited below (from PAPER_REFRAME_HANDOFF.md, not from a
rank-logged rerun). Running `real_data_experiment.py --log_ranks` on Amazon
and checking it with `verify_extended_proposition.py` is the next concrete
step, not a new one to design.

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

1. **Numerically verify (ddagger) against real rank data.** Infrastructure
   for this now exists (`real_data_experiment.py --log_ranks` +
   `verify_extended_proposition.py`, see Status above) but has not yet been
   RUN on the real datasets. Run `real_data_experiment.py --dataset amazon
   --detector dominant_pygod --log_ranks` (and similarly for reddit,
   tolokers, weibo), then `verify_extended_proposition.py --all`, and check
   whether the soundness report comes back clean (no predicted-but-not-
   observed discoveries) and whether the Amazon/clean case is correctly
   flagged as "floor fails, extended succeeds."
2. **Derive the finite-sample version.** Bound N_0(r) via a concentration
   inequality for negatively associated sums (see "What this does and does
   not establish" above). This is the piece that would turn the generalized
   discovery condition into an actual theorem rather than an in-expectation
   statement.
3. **Verify the severity-sweep prediction directly**, logging actual c(r)
   at each severity level under a CORRECT detector (dominant_pygod), not
   the broken one. DONE ONCE: `severity_sweep_pygod_instrumented.py` ran
   this on synthetic data (adversarial condition, all 5 severity levels).
   Result: AUROC=1.0 and power=1.0 at every level, so "fails into silence"
   is confirmed dead (drop it from the paper), but the floor condition
   alone succeeded in 100% of trials, meaning this run gave the extension
   no case where it was actually needed. A separate, unresolved finding
   came out of the same run: pooled realized FDR across all 100 trials was
   significantly ABOVE nominal (mean 0.109 vs alpha=0.10, one-sided
   t=3.142, p=0.0011, Cohen's d=0.314), worst at the highest severity level
   (d=1.343). `condition_comparison_pygod.py` was built specifically to
   determine whether this is a general adversarial-selection property or a
   severity-specific contamination effect -- it has not been run yet.
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

---

## Part 3: Empirical tension with Proposition 1's clean-condition guarantee

**This section documents a real, unresolved conflict between Proposition 1
and the synthetic data. It must be reflected in the paper as a caveat, not
omitted.**

Proposition 1 (paper Section III) claims exact marginal FDR control for the
clean condition, citing Bates et al. 2023 + the CF-GNN exchangeability
result (permutation invariance of the scoring function under transductive
training). `condition_comparison_pygod.py` tested this directly on
synthetic graphs (20 seeds, correct detector, baseline severity p_an=0.002)
and found:

    Clean condition: realized FDR = 0.132 +/- 0.037, d=0.837 vs nominal
    0.10, one-sided p=0.0007 -- SIGNIFICANTLY ABOVE nominal.

This is the OPPOSITE of what Proposition 1 predicts, and the opposite of
what contaminated/adversarial showed in the same run (both safely at or
below nominal, d=-0.349 and d=-0.530 respectively). The condition with the
proven guarantee is the one that violated it; the two conditions without a
proof held up fine.

### Investigation: degree confound

`clean_selection_degree_diagnostic.py` (10 seeds) confirmed two things,
both required for a degree-confound explanation to hold:

1. Clean-selected calibration nodes have significantly LOWER degree than
   the test-set normal population (paired t=-24.959, p<0.0001, 10/10 seeds
   individually significant by KS test). Mean degree: calib 71.32 +/- 0.18
   vs test-normal 73.10 +/- 0.07 -- small in absolute terms but extremely
   consistent.
2. dominant_pygod's score correlates with degree among normal nodes
   (Spearman r=0.56 +/- 0.01, significant in 10/10 seeds).

### Two fixes tried

**degree_normalize_scores() (score / log(1+degree)), already validated on
real data.** Tested via `--use_degree_norm`. Result: made things WORSE, not
better. AUROC collapsed 1.0 -> ~0.90, power collapsed 1.0 -> ~0.002 (19/20
trials produced zero discoveries in every condition), and on the rare
trials that still fired, CONDITIONAL FDR was 40-44%, roughly 4x the
original 13.2%. The marginal FDR number looked good (0.02) only because
BH counts a zero-discovery trial as contributing FDR=0 -- a vacuous
improvement, not a real one. Do not use this fix; kept in the codebase only
for reproducibility of this comparison.

**degree_matched_calib_sample(), selection-level fix instead of a score-level
one.** Tested via `--degree_matched_calib`. Draws the clean calibration set
via degree-stratified sampling so its degree distribution matches the full
normal population's, without touching scores. Result: PARTIAL improvement.
FDR dropped to 0.116 +/- 0.034, d=0.449 (down from 0.837), p=0.0295 (down
from 0.0007) -- roughly half the gap to nominal closed, and AUROC/power
stayed at a perfect 1.000 since scores were never altered.

Still not resolved: (a) still significantly above nominal at p=0.0295, (b)
the matching itself was imperfect in 18/20 seeds (KS test against the
target distribution still significant, p<0.05, meaning the clean-selected
candidate pool structurally lacks enough high-degree members to fully
close the gap regardless of sampling strategy), and (c) a direct per-seed
check -- does better matching THIS seed predict lower FDR THIS seed --
found NO significant correlation (Spearman r=-0.161, p=0.498, n=20). That
last point matters: the aggregate improvement is real, but the per-seed
evidence does not cleanly establish that degree-matching quality is the
mechanism driving the improvement, as opposed to some other effect of the
intervention or noise at this sample size.

### Honest verdict

Degree confound is a REAL, PARTIAL contributor to the clean-condition FDR
inflation -- the degree-matched intervention improved things without
breaking signal, which a pure coincidence would be unlikely to do this
consistently. But roughly half the effect remains unexplained, and the
per-seed correlation check does not confirm the mechanism as cleanly as
the aggregate numbers suggest. Do not claim this is solved.

### What this means for the paper

Proposition 1 as currently stated is not empirically supported by this
run. Options, not yet decided:
1. Add an explicit caveat to Proposition 1's discussion: the exchangeability
   precondition assumes calibration selection is independent of covariates
   the score is sensitive to (e.g. degree), which may not hold when
   selection is topological (zero anomalous neighbors) and the detector is
   degree-sensitive. State this as a known boundary condition, not hide it.
2. Report the degree-confound investigation itself (Part 3 of this
   document) as a genuine empirical contribution: even the
   theoretically-guaranteed condition can show real exchangeability
   violations from selection mechanisms existing theory doesn't
   anticipate, partially but not fully attributable to a measurable
   covariate (degree).
3. Do NOT claim Proposition 1 is "confirmed by experiment" anywhere in the
   paper's Results section without this caveat attached -- Table II-style
   reporting of the clean condition must show the actual FDR (0.132 or
   0.116, not assumed-controlled) and cite this section.

### Second covariate tested: local clustering coefficient (null result)

Since degree-matching only closed roughly half the gap, a second candidate
covariate -- local clustering coefficient, motivated by the generator's
3-anomaly-cluster SBM structure and DOMINANT's known sensitivity to local
neighborhood structure -- was tested via the same two-check methodology
(`clean_selection_degree_diagnostic.py`, 10 seeds):

    Clustering confound (calib vs test-normal): KS significant in 0/10
    seeds, paired t=-0.279, p=0.7869. NO confound.
    Score-clustering correlation: mean Spearman r=0.0044 +/- 0.0058,
    significant in 0/10 seeds. NO correlation.

Both checks null, cleanly. Note the underlying reason this covariate had
little to explain in the first place: local clustering coefficient is
nearly flat across the entire population on this generator (0.0049-0.0051
for both calib and test-normal), meaning the graph has almost no local
triangle structure at this density -- there wasn't much variance available
for clustering to account for. This is a property of the generator, not a
flaw in the check.

### Why degree-matching could only ever close part of the gap: a structural
### explanation, not an open question

For a node with degree d, under roughly independent random attachment at
the generator's ~5% anomaly rate, P(zero anomalous neighbors) scales
approximately as (1 - 0.05)^d -- decreasing exponentially in degree:

    degree=10:  P(clean) ~ 0.60
    degree=30:  P(clean) ~ 0.21
    degree=50:  P(clean) ~ 0.08
    degree=71:  P(clean) ~ 0.026   (~ the observed calib mean degree)
    degree=100: P(clean) ~ 0.006
    degree=150: P(clean) ~ 0.0005

The "clean" condition's defining filter (EXACTLY zero anomalous neighbors)
excludes high-degree nodes almost by mathematical necessity, not as an
artifact of sampling that better matching could fully correct. The
clean-eligible pool is combinatorially thin on high-degree members --
`degree_matched_calib_sample()` can only draw from what actually exists in
that pool, and at degree 100+ there is very little left to draw. This is
why the fix landed at "closes about half the gap" rather than "fully
resolves it," and why no amount of additional matching sophistication
(finer bins, importance weighting) is expected to close much more of the
remainder -- the limiting factor is the composition of the candidate pool
itself, not the matching algorithm.

### Status: investigation concluded, not fully resolved

Two covariates tested (degree: real, partial, mechanically explained
ceiling around 50% correction; clustering: clean null, explained by low
variance in the covariate itself on this generator). No further covariate
hunting is planned -- diminishing returns are expected given the structural
argument above, which applies to any covariate correlated with degree, not
just degree itself. The clean-condition FDR inflation is a REAL,
PARTIALLY-EXPLAINED finding: report it as such in the paper (Proposition 1
caveat + this section as a Discussion/Threats-to-Validity subsection), not
as fully solved and not as fully mysterious.