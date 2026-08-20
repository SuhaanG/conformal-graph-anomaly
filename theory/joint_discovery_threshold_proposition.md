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

Part 3 (empirical tension with Proposition 1) and Part 4 (its formalization)
were added after the degree-confound investigation. **Part 4 is now the
recommended primary theoretical contribution**, ahead of Part 2's finite-sample
gap: it is a shorter proof, it rests on standard stochastic-order results, and
it explains a finding we actually measured rather than sharpening a condition
reviewers already read as an observation.

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
---

## Part 4: Selection-induced exchangeability failure (theorem sketch)

**This is the formalization of Part 3. Part 3 established the clean-condition
FDR inflation empirically and gave a structural argument for why the
degree-matched fix capped at ~50%. That structural argument is not merely an
explanation of a limitation -- it is the load-bearing step of a theorem, and
it should be developed as one. Recommended as the paper's central theoretical
contribution, ahead of item 2 in "What remains" (the finite-sample
concentration bound for Part 2).**

### The claim, informally

Constructing a "clean" calibration set by a topological filter is not a
neutral operation. The filter is a covariate filter: it selects on degree,
because low-degree nodes are combinatorially more likely to pass it. If the
detector's score is degree-sensitive -- and reconstruction-based graph
detectors are, measurably (Spearman r = 0.56, Part 3) -- then calibration
scores are stochastically smaller than test-normal scores, the conformal
p-values of normal test points are stochastically smaller than Uniform, and
BH's FDR guarantee fails upward. The very step taken to guarantee validity is
what breaks it.

### Setup and assumptions

Let D be the degree of a uniformly drawn normal node, with pmf f(d) over the
population of normal nodes. Let F denote the calibration eligibility event
(for the clean condition: "zero anomalous neighbors"). Calibration nodes are
drawn uniformly from {normal nodes : F holds}; test-normal nodes are drawn
from the normal population without conditioning on F.

**(A1) Monotone selection.** q(d) := P(F | D = d) is non-increasing in d.

**(A2) Monotone score.** The conditional score distribution S | D = d is
stochastically increasing in d: d1 <= d2 implies (S | d1) <=_st (S | d2).

**(A3) Continuity.** Scores are continuous, so ties have probability zero.
(In practice enforced by the existing jitter path; ties weaken the strictness
of the conclusion, not its direction.)

Note that (A1) is far weaker than the (1-pi)^d form used in Part 3. The
theorem needs only monotonicity. The exponential form is a worked special
case supplying the quantitative rate; it should NOT be assumed on real
graphs, where q(d) must be measured rather than derived, since independent
attachment does not hold there.

### Step 1: Selection is a likelihood-ratio tilt on degree

The eligible pool has degree pmf

    f_F(d) = f(d) q(d) / Z,     Z = sum_d f(d) q(d)

so the likelihood ratio f_F(d)/f(d) = q(d)/Z is non-increasing in d by (A1).
A non-increasing likelihood ratio is the definition of the likelihood-ratio
order, hence

    D_F <=_lr D    and therefore    D_F <=_st D

(likelihood-ratio order implies the usual stochastic order; Shaked &
Shanthikumar, Stochastic Orders, Thm 1.C.1).

**Special case.** Under independent attachment at neighbor-anomaly rate pi,
q(d) = (1-pi)^d = exp(-lambda d) with lambda = -log(1-pi) > 0. Then f_F is an
exponential tilt of f with negative parameter, and the mean-degree gap is

    E[D] - E[D_F] = lambda * Var(D) + O(lambda^2)

for small lambda. This is the closed form behind Part 3's table, and it
predicts the sign and rough magnitude of the observed calibration-vs-test
mean degree gap (71.32 vs 73.10) without fitting anything.

### Step 2: The ordering transfers to scores

By (A2), the map d -> (S | d) is a stochastically monotone kernel. The usual
stochastic order is closed under such kernels, so D_F <=_st D gives

    S_cal <=_st S_test-normal

strictly, unless q is constant on the support of f (no selection) or the
score is degree-independent ((A2) holding with equality).

### Step 3: Stochastically small calibration scores inflate rejection rates

Let v be a normal test point with score S_v, and let the conformal p-value be
p_v = (|{i in cal : S_i >= S_v}| + 1)/(n+1).

Couple S_v with S_v' ~ F_cal so that S_v >= S_v' almost surely (possible by
Step 2 and Strassen's theorem). The calibration rank is non-increasing in the
test score, so r(S_v) <= r(S_v') pointwise, hence for every t,

    P(p_v <= t) >= P(p_v' <= t) = floor(t(n+1))/(n+1)

The right-hand side is the exact exchangeable (valid) rate. So the null
p-values are stochastically DOMINATED by Uniform -- anti-conservative -- with
the inequality strict whenever Step 2 is strict.

### Step 4: Consequence for BH

BH controls FDR at alpha*m_0/m when null p-values are (super-)uniform. Define
the anti-conservativeness factor

    gamma := sup_{t in (0,1]} P(p_v <= t) / t   >= 1

Step 3 gives gamma > 1 strictly under selection. Running BH at level alpha
then controls FDR only at level gamma * alpha * m_0/m: the realized FDR can
exceed nominal by up to a factor gamma. The observed 0.132 against a nominal
0.10 corresponds to gamma ~ 1.32, which is the number to predict from a
measured q(d) and the measured score-degree dependence. That prediction --
theory-predicted gamma versus empirically realized gamma -- is the experiment
that would turn this from a sketch into a validated theorem, and it needs no
new infrastructure beyond what condition_comparison_pygod.py and
clean_selection_degree_diagnostic.py already log.

### Corollary (why any reweighting fix has a ceiling)

degree_matched_calib_sample() reweights within the eligible pool. Any such
scheme produces a degree distribution absolutely continuous with respect to
f_F, so it can never place mass where f_F has none. The achievable total
variation distance to the target f is bounded below:

    inf over g << f_F of TV(g, f)  >=  sum over {d : f_F(d) = 0} of f(d)

and more practically, matching in a region where f_F(d)/f(d) is tiny requires
importance weights of order 1/q(d), whose variance explodes exactly where the
pool is thin. This is a proof, not a conjecture, that the fix caps out -- and
it explains Part 3's "closes about half the gap, and finer bins will not
help" without appeal to implementation detail.

### What this predicts that is testable and not yet tested

1. **Weighted conformal is the principled fix.** Weighting calibration points
   by w(d) proportional to 1/q(d) restores exchangeability in the weighted
   conformal framework (Tibshirani, Barber, Candes & Ramdas 2019), up to the
   variance blow-up above. This should beat degree_matched_calib_sample(), by
   a predictable margin. Not yet implemented.
2. **The effect should scale with measured score-degree dependence.** Across
   the five detectors already in the matrix, gamma should track each
   detector's own Spearman(score, degree). A detector with weak degree
   dependence should show little or no clean-condition inflation. This is a
   cross-detector prediction the existing multi-detector infrastructure can
   test directly, and it is the single most falsifiable claim here.
3. **The effect should vanish under a degree-independent filter.** Selecting
   calibration uniformly at random (accepting contamination) should give
   gamma ~ 1. Part 3 already reports contaminated and adversarial safely at
   or below nominal, which is consistent -- but that was not framed as a test
   of this prediction and should be re-read as one.

### Honest status

Steps 1-3 are rigorous given (A1)-(A3) and rest on standard stochastic-order
results, not new machinery. Step 4's gamma is a valid upper bound but not yet
a sharp characterization. Prediction 2 is the one that would most
convincingly establish the mechanism, because it varies the proposed cause
across detectors while holding the graph and the selection filter fixed; if
gamma does not track score-degree dependence across detectors, the mechanism
as stated is wrong and this section must be revised, not defended.

None of Part 4 has been numerically checked yet. Do not present it as
established until at least prediction 2 has been run.


---

## Part 5: The falsification test ran. Verdict is UNDERPOWERED, not falsified.

**Status: Part 4 is neither confirmed nor refuted. Within the regime where its
own stated precondition holds, the prediction is supported (rho=+0.81,
p=0.0149), but that rests on a single detector and collapses to p=0.39 without
it. Do not write Theorem 2 yet. The decisive experiment is cheap and is
specified below.**

Run: `selection_bias_matrix.py`, 5 detectors x 4 real graphs x 5 seeds, clean
condition, commit b320128. Results in `results/published/`.

### The script's printed verdict was wrong, and the error was ours

It reported `VERDICT: consistent with Part 4`, with all three statistics
agreeing at p<0.001. That does not hold, because `gamma_hat` was measuring the
wrong region of the distribution.

`gamma_hat` sups over calibration ranks r >= n_calib//4. BH does not cut there.
On weibo n_calib ~ 5636, so min_rank ~ 1409, while BH's realized threshold
t = alpha*k/m = 0.0044 corresponds to calibration rank ~25. The statistic
described the bulk while BH operated in the extreme left tail. The min_rank
default was introduced to control variance -- an exchangeable null at
min_rank=10 has p95 1.57, which would swamp the effect -- and it does that, but
it bought the variance reduction by measuring somewhere the procedure never
goes.

Recomputed at the BH operating point, using the CSV's real n_null and m_test:

    Spearman(sdeg, gamma_hat) over discovering cells   rho=+0.813  p=0.0007
    Spearman(sdeg, gamma@BH)  over discovering cells   rho=+0.423  p=0.1497

### Weibo is not a counterexample -- it falls outside the assumptions

The apparent refutation was that weibo shows FDR 0.61-0.65 for detectors whose
score-degree correlation is NEGATIVE, which Part 4 says should be conservative
and show no violation.

But (A1) -- q(d) = P(clean-eligible | degree) non-increasing -- **fails on
weibo and only on weibo**. The runner measured this directly:

| dataset | clean pool | % of normals | (A1) monotone | Kendall tau |
|---|---|---|---|---|
| amazon | 265 | 2% | 25/25 seeds | -0.92 |
| reddit | 9778 | 92% | 25/25 seeds | -0.91 |
| tolokers | 801 | 9% | 25/25 seeds | -0.92 |
| **weibo** | 5636 | 70% | **0/25 seeds** | **-0.02** |

On weibo the clean filter does not select on degree at all. Part 4 therefore
predicts nothing there -- no tilt, hence no degree-mediated violation -- and
that is what is observed. Testing the theory on weibo was testing it outside
its own scope.

Two facts make this a real point in the theory's favour rather than a
post-hoc rescue:

  - (A1) was written into Part 4 in commit ad50360, which is an ANCESTOR of the
    run commit b320128. The criterion was pre-registered.
  - `q_is_monotone` was computed by the runner during the run and is a column
    in the output CSV. It was not derived afterwards to fit the result.

The criterion was stated in advance, measured automatically, and it
discriminated correctly. Restricted to the three datasets where (A1) holds,
measured at the BH operating point:

    n=8 cells    Spearman(sdeg, gamma@BH)   rho=+0.8095   p=0.0149

### Why this still is not a confirmation

    all cells where A1 holds        n=8   rho=+0.8095   p=0.0149
    drop dominant_pygod             n=5   rho=+0.5000   p=0.3910

The result rests on one detector. The cause is a coverage problem rather than a
subtle statistical one: across all 100 trials, **every cell with sdeg > 0.5 is
dominant_pygod.** Per-detector sdeg ranges:

    dominant_pygod  [+0.44, +0.92]
    ocgnn           [-0.22, +0.67]
    gae             [-0.20, +0.41]
    anomalydae      [-0.31, +0.38]
    dominant_ours   [-0.23, +0.14]

Removing dominant_pygod deletes the entire high end of the dose-response curve,
so n=5 has no power to detect anything. This is the objection a referee will
raise; it is correct as stated, and it is fixable.

### The decisive experiment: sweep degree sensitivity within one detector

Relying on five detectors that happen to differ in degree sensitivity is a weak
design. Detector identity varies alongside the proposed cause, and only one
detector reaches the high end. Manipulate the cause directly instead.

`degree_normalize_scores` divides by log1p(degree). Generalise the exponent:

    score_beta = score / log1p(degree) ** beta,    beta in [-0.5, 1.5]

beta < 0 AMPLIFIES degree sensitivity; beta = 0 leaves scores untouched;
beta = 1 is the existing correction; beta > 1 over-corrects into negative
dependence. Sweeping beta on ONE detector and ONE graph gives a continuous
dose-response curve in sdeg with:

  - no cross-detector confound (same architecture, same weights, same graph,
    same calibration frame -- only the score transform moves)
  - dense coverage of the whole sdeg range, including above 0.5
  - as many points as compute allows, instead of 5
  - a built-in internal control: gamma should rise and fall with sdeg as beta
    varies, and should cross gamma ~ 1 near whatever beta makes the score
    degree-neutral

Run it on amazon and tolokers, where (A1) holds strongly (tau ~ -0.92) and the
clean filter is genuinely harsh (2% and 9% of normals retained). If gamma
tracks sdeg along that curve, Part 4 has real support from a proper causal
design rather than an observational one. If it does not, Part 4 is dead, and
unambiguously so rather than underpowered.

This subsumes prediction 2 and should replace it as the primary test.

### Separately: what breaks weibo is unexplained

Weibo shows median FDR 0.634 with no degree tilt and, for three of five
detectors, negative score-degree correlation. gamma at the BH threshold is
~7.0-7.4 there regardless of detector. Something dataset-level is breaking
exchangeability that has nothing to do with degree. This is a genuine open
question and probably a second finding rather than a nuisance. Weibo is also
the one dataset with a measurable contamination signal (exposure->score
r=0.111, p=1.4e-23, surviving control for degree), which may or may not be
connected.

### The empirical result that depends on none of this

Across the 13 cells that produced any discoveries, median realized FDR is 0.613
against nominal 0.10, with 7 of 13 above 0.50. Per dataset: amazon 0.900,
tolokers 0.635, weibo 0.634, reddit 0.167.

The clean condition -- the one condition Proposition 1 proves is exchangeable
-- fails by up to 9x on real graphs, across detectors spanning three
architectural families. That stands regardless of which mechanism explains it,
and it is currently the strongest result in the project.

### Why FDR is not a clean readout of the violation

FDR = V/(V+S). Anti-conservativeness raises V; weak detector signal lowers S.
Both inflate FDR, and the matrix confounded them. Synthetic runs at AUROC 1.0
where S is maximal, so a real violation appeared only as 0.132. Weibo runs at
power ~0.12, where a modest left-tail excess produces 0.65. Measure gamma
directly on the null p-values at the BH operating point; never infer the
violation from FDR alone.

### Next steps, in order

1. **Run the beta sweep.** Decisive, cheap, and replaces prediction 2.
2. Re-run the matrix with `left_tail_gamma` (added after this run) so the
   y-axis is measured where BH cuts, plus the block-permutation and
   within-detector analyses.
3. Investigate weibo on its own terms.
4. Only then decide what, if anything, becomes a theorem.


---

## Part 6: The project has never studied calibration contamination

**This section supersedes Part 4 as the theoretical core, and it reframes the
paper's premise. Read it before writing anything.**

### The misnomer

Every condition in `real_data_experiment.py` draws calibration from
`eligible_normal_idx`, which is a subset of `normal_idx = np.where(labels == 0)`:

    clean         calib_idx = clean_pool                       (labels == 0)
    contaminated  calib_idx = rng.choice(eligible_normal_idx)  (labels == 0)
    adversarial   calib_idx = top_exposed[:n_calib]            (labels == 0)

**No condition places a single anomaly in the calibration set.** In the
conformal outlier-detection literature -- Bates et al. 2023, AdaDetect --
"contaminated calibration" means the reference sample CONTAINS OUTLIERS. That
experiment has never been run in this project.

What the three conditions actually vary is which NORMAL nodes are selected, as
a function of their exposure (fraction of anomalous neighbours):

    clean         exposure == 0
    contaminated  exposure ~ population
    adversarial   exposure maximal

This is a **covariate selection** experiment wearing a contamination label. It
explains, cleanly and in retrospect, why the contamination story never worked
(handoff section 4): there was no contamination to detect. The exposure ->
score channel we spent so long measuring (|r| < 0.03 on four datasets, 0.111 on
weibo) is a second-order effect of message passing, not the first-order effect
the framing implied.

### The general statement

Let normals carry a covariate W (degree, exposure, clustering, anything). Let a
selection rule R produce calibration C, and let test-normals T be drawn from
the normal population. Conformal validity requires S|C ==d S|T.

**Theorem (selection-induced non-exchangeability).** If R shifts the
distribution of W -- that is, P_C^W != P^W -- and the conditional score
distribution S | W is stochastically monotone in W, then S|C and S|T differ in
the usual stochastic order. The conformal p-values of normal test points are
then stochastically dominated by (or dominate) Uniform, and BH's FDR guarantee
fails in the corresponding direction.

The proof is Part 4 steps 2-4 verbatim, with W in place of degree. Part 4 is
therefore not wrong -- it is the special case W = degree, and its error was
treating a corollary as the theorem.

**Corollary 1 (exposure channel).** The clean rule sets W = exposure and makes
P_C^W a point mass at 0 while P^W has mass above 0 -- the maximal possible
shift. Validity fails whenever S depends on exposure at all.

Contrapositive, and this is the point worth quoting: clean calibration is valid
only when the score is insensitive to exposure, which is exactly the regime
where contamination was never a threat. **The strategy is valid when it is
unnecessary and invalid when it is needed.**

**Corollary 2 (degree channel).** P(exposure == 0 | degree) is non-increasing
(measured: Kendall tau = -0.92 on amazon, reddit, tolokers), so the clean rule
shifts degree as a side effect. Validity fails whenever S depends on degree,
EVEN IF S is independent of exposure. This is the channel that dominates where
the filter is harsh -- amazon retains 2% of normals, tolokers 9%.

Weibo is the case that separates the two: the filter retains 70%, degree is not
shifted at all (tau = -0.02, 0/25 seeds monotone), yet gamma ~ 7. Corollary 2
predicts nothing there; Corollary 1 does, and weibo is the one dataset with a
measurable exposure -> score signal (Spearman 0.111, p = 1.4e-23, surviving
control for degree).

**Corollary 3 (the safe rule).** If R is a uniform random draw from the normal
population, then P_C^W = P^W for EVERY covariate W simultaneously. No shift
exists, so no channel can open. Random calibration is valid regardless of how
many calibration nodes happen to neighbour anomalies.

Corollary 3 is the practical payload and it is counterintuitive: **accepting
"contamination" into calibration is safer than filtering it out**, because the
filtering is what breaks the guarantee. `scripts/calibration_strategy_comparison.py`
tests exactly this at matched n_calib and a shared test set.

### The experiment that is still missing

True contamination -- calibration containing actual anomalies at rate epsilon
-- has never been run here, and it is cheap. The expected contrast completes
the story:

  - **True contamination** puts high-scoring anomalies into calibration. Test
    anomalies then face stiffer competition, p-values rise, power falls, and
    FDR stays controlled or becomes conservative. Safe but weak.
  - **Selection filtering** (our "clean") removes normals correlated with the
    anomalies, shifts the covariate distribution, and makes p-values
    anti-conservative. Dangerous.

If that holds, the paper's finding is sharp: the failure mode everyone guards
against is benign, and the standard guard is what actually breaks FDR control.
Add an `--epsilon` condition to the frame builder that injects anomalies into
calibration at a controlled rate.

### What this does to the paper

The title and premise change from contamination to selection. Concretely:

  - Sections built on "clustered, propagating calibration contamination"
    describe a mechanism that is real but second-order, and that no condition
    in the codebase actually isolates.
  - The empirical result stands unchanged and gets a correct explanation:
    median realized FDR 0.613 against nominal 0.10 across 13 discovering cells.
  - Proposition 1 is not contradicted. Its exchangeability precondition is
    simply violated by the selection rule, and the paper should say which rules
    satisfy it (random) and which do not (any covariate filter).
  - Part 4's degree machinery survives as Corollary 2 rather than as the
    headline.

### Status

The theorem is a restatement of Part 4 at the right level of generality and
inherits its proof, which rests on standard stochastic-order results. What is
NOT yet established is Corollary 3 on real data at matched n -- that is what is
running now. If random calibration is also broken, the account is wrong and
this section must be rewritten, not defended.

### Addendum: true contamination is safe because Part 1 blocks it

The true-contamination condition was built and smoke-tested (synthetic, frozen
detector, one seed -- wiring only, not a result). Injecting anomalies into
calibration at 5% produced ZERO discoveries, and the reason is Part 1's
discovery-threshold proposition, which closes the loop between the two halves
of this document.

With eps*n_calib anomalies in calibration occupying the top scores, no test
point can achieve a p-value below (eps*n + 1)/(n + 1). At n_calib=707, eps=0.05
that floor is 0.0508. BH rejects at sorted rank k only if p <= alpha*k/m, so
k >= 0.0508 * 723 / 0.10 = 368 simultaneous discoveries would be required to
justify even one. The procedure cannot reject anything.

So the two failure modes are opposite and have separate explanations:

    TRUE contamination   raises the resolution floor  -> no discoveries
                         -> FDR trivially 0.  SAFE BUT POWERLESS.  (Part 1)

    SELECTION filtering  shifts the null distribution -> anti-conservative
                         -> FDR up to 0.91.  DANGEROUS.             (Part 6)

That is the paper's spine. Part 1 and Part 6 are not two separate results
about calibration -- they are the two ways a calibration set can be wrong, and
the literature's stated worry is the benign one.

CAVEAT on the smoke test: it ran dominant_ours (dead final ReLU, sdeg ~ 0) on a
2500-node synthetic graph for a single seed, and its `random` arm looked WORSE
than `clean` (gamma 2.01 vs 0.59) on that seed while reversing on the next.
That is noise from a detector with no covariate sensitivity, on a graph too
small to resolve anything. It neither supports nor undermines Corollary 3. The
real-data runs decide.

### Part 6 results: the strategy comparison ran (amazon, weibo)

**Exchangeability: CONFIRMED, and starkly.** On amazon at matched n_calib,
matched test set, matched floor, mean over 5 seeds:

    strategy         calib_deg  test_deg   gap(d)   gamma   mean_p
    clean                105.6     737.2   -1.252   13.22   0.1438
    random               761.4     737.2   -0.005    0.82   0.5007
    exposed_only         728.1     737.2   +0.013    1.01   0.5067
    true_contam_05       727.2     737.2   -0.010    0.72   0.4976
    true_contam_10       732.2     737.2   +0.023    0.75   0.5077
    random_full (n=4000) 742.5     737.2   +0.002    1.16   0.5003

The clean filter selects calibration at 1/7th the test population's degree;
every other rule -- including calibration containing 10% ACTUAL ANOMALIES --
lands within 2% of the test set and is exchangeable (gamma 0.72-1.16,
mean_p ~ 0.50). Only the covariate-shifting rule breaks, exactly as Part 6
predicts, and gamma is computed from null p-values so this does not depend on
discovery counts.

**Prediction 4 was WRONG, informatively.** exposed_only is NOT broken
(gamma 0.96-1.01). Filtering per se does not hurt; filtering that SHIFTS a
score-relevant covariate does. On amazon nearly every high-degree node has an
anomalous neighbour, so "exposed" is nearly the whole population while
"unexposed" is the low-degree fringe. Corollary 1 must be restated in terms of
the induced covariate shift, not the act of filtering.

**The remedy claim is NOT established, and something bigger surfaced.**
random_full at n_calib=4000 has bh_min_rank ~ 7 -- only seven test points need
to beat the whole calibration set -- and still produced ZERO discoveries.
Fewer than 7 of amazon's 821 anomalies outrank 4000 random normals under
dominant_pygod: with score-degree Spearman +0.918, the detector's top ranks
are HIGH-DEGREE NORMALS, not anomalies. AUROC ~0.97 measures average ranking,
not top-of-list precision.

Which reframes clean's 1527 "discoveries" at FDR 0.787: ~325 true, ~1202
false. **Every discovery this pipeline ever made on amazon was manufactured by
the validity failure.** The degree gap let nearly all test nodes outrank the
low-degree calibration set, BH fired en masse, and the hits among them were
along for the ride. Under any VALID calibration, this detector detects nothing
on amazon. Power without validity was an illusion of the broken guarantee.

**Weibo: consistent once the dilution is accounted for.** Strategy comparison
shows clean gamma 1.42 vs random 0.82. Milder than the matrix's gamma ~ 7
because this design's test set is a random exposed/unexposed MIXTURE (the
documented conservative choice), while the matrix frame tests against the
fully-exposed complement -- on weibo the clean pool is 70% of normals, so the
dilution is large. Directionally identical, magnitude scaled by the frame.

**A correction to the weibo exposure-channel claim.** The r=0.111 repeatedly
cited for weibo's exposure->score signal was measured on DEGREE-NORMALIZED
scores (exposure_degree_confound_check, degree_norm=True; r_raw in the same
file is 0.045). On raw scores the strategy comparison measures +0.016. The
exposure channel on weibo is WEAK on raw scores; earlier statements that weibo
"fails through exposure" overstated this. What the weibo matrix failure (gamma
~7, FDR 0.634) actually runs through is not yet pinned down -- smaller degree
gap amplified by the full-separation frame is the leading candidate, but it is
OPEN, not explained.

### The difficulty sweep failed and was rebuilt

feature_shift down to 0.15 left AUROC at 1.0000 at every level: with p_aa=0.3
against p_nn=0.005, anomalies sit in 60x denser blocks and are structurally
obvious to a reconstruction detector regardless of features. p_aa is the real
difficulty knob and the sweep now varies it (0.005 -> 0.30). Note the frozen
detector cannot validate this locally -- its structure decoder is dead, so its
AUROC is feature-only and insensitive to p_aa by construction.

---

## Part 7: The score gap is the operative quantity, and it is deployable

### The law

Across every matched-frame cell run so far -- 3 datasets, 2 detectors, 6
selection rules, 17 cells -- the standardized score gap between calibration and
test-normals predicts the exchangeability violation:

    Spearman(gap_d, gamma) = -0.7285,  p = 9.11e-04

and it predicts SIGN, not just magnitude:

    amazon  clean    gap -1.252  (calib 7x LOWER degree)   gamma 13.22  broken
    weibo   clean    gap -0.078                            gamma  1.42  mild
    reddit  gae clean gap -0.065                           gamma  1.04  ~valid
    ...     every matched rule, |gap| < 0.05               gamma ~1     valid
    weibo   exposed  gap +0.145  (calib HIGHER degree)     gamma  0.23  conservative
    reddit  exposed  gap +0.435  (calib 5x HIGHER degree)  gamma  0.31  conservative

Calibration scoring LOWER than test makes the procedure anti-conservative;
scoring HIGHER makes it conservative; matched gives validity. Degree is one
route to a gap, exposure is another, and neither is privileged -- what matters
is the gap itself.

### What is and is not a contribution here

**Be honest about this.** gap_d and gamma are both functions of the same two
score distributions, so "a shift in scores produces a shift in p-values" is
close to definitional and a referee will say so. The gap -> gamma link is NOT
the contribution.

The contributions are the parts that are empirical and surprising:

  1. **The selection rule causes the gap, at a magnitude nobody would guess.**
     Filtering calibration for "cleanliness" produces a 7x degree gap on amazon
     (105.6 vs 737.2) and a 13x FDR violation. Nothing about the rule announces
     this.
  2. **True contamination does NOT cause a gap.** Calibration containing 5% and
     10% actual anomalies stays within 2% of the test degree and is
     exchangeable (gamma 0.72-0.83). The failure mode the literature guards
     against is benign.
  3. **The gap is computable WITHOUT LABELS.** It is a two-sample statistic on
     scores you already have at deployment time. That makes it an operational
     precondition check, not a post-hoc diagnosis: measure the standardized gap
     between calibration and test scores, and if it is large, the FDR guarantee
     is void regardless of what the method promises.

Point 3 is the practical payload and it is what the paper should lead with.

### AUROC on the synthetic generator is a degree measurement

The p_aa sweep produced an independent finding that is arguably more damaging
than anything above. Sweeping anomaly-anomaly density with features held fixed:

    p_aa     E[deg|anom]  E[deg|norm]  ratio   AUROC
    0.005       32.2         72.8      0.44    0.0793
    0.010       36.0         72.8      0.49    0.1038
    0.020       43.5         72.8      0.60    0.0661
    0.050       66.0         72.8      0.91    0.3337
    0.100      103.5         72.8      1.42    0.8597
    0.300      253.5         72.8      3.48    1.0000

AUROC crosses 0.5 exactly where the anomaly/normal degree ratio crosses 1.0.
The features are IDENTICAL at every level. dominant_pygod's AUROC on this
generator is a measurement of whether anomalies were planted as dense blocks,
nothing more -- consistent with its score-degree Spearman of +0.918 on amazon.

The AUROC 1.0000 that every synthetic result in this project was built on is
therefore not detection. It is degree ranking that happens to coincide with the
planted structure. Flip p_aa and the same detector on the same features ranks
anomalies at the BOTTOM (AUROC 0.066).

### A recurring error worth stating once

The difficulty sweep's printed verdict ("the effect SHRINKS on harder tasks")
is WRONG, and for the fifth time in this project the cause is zero-discovery
levels contaminating a summary statistic. Four of its six levels produced no
discoveries, so their FDR gap is trivially 0.000; averaging those into the
"harder" group manufactured the conclusion. The two levels where measurement
is possible give gaps of 0.017 (AUROC 0.86) and 0.016 (AUROC 1.00) -- flat, not
shrinking.

Every summary statistic in this codebase must exclude zero-discovery cells
before averaging. Prior instances: degree normalization (Part 3), the beta
sweep (Part 5), the strategy-comparison verdict, and now this.

---

## Part 8: The degree-baseline check ran. 16/20 cells cannot beat degree.

`scripts/degree_baseline_check.py`, 5 detectors x 4 datasets x 3 seeds,
commit 4937701.

    dataset    best free baseline         best detector          gap
    amazon     0.7446 (neg_degree)        0.8589 (gae)          +0.1143
    reddit     0.5561 (degree)            0.5769 (dominant_ours) +0.0207
    tolokers   0.5596 (degree)            0.5618 (ocgnn)        +0.0022
    weibo      0.7782 (neg_degree)        0.7733 (dominant_ours) -0.0049

16 of 20 detector-dataset cells fail to beat the training-free baseline by
more than 0.02 AUROC. On tolokers and weibo no detector clears the bar at all.
This supports the NeurIPS D&B framing (handoff section 8.1): reported AUROC on
these benchmarks does not, by itself, distinguish a trained detector from a
degree lookup table.

### The wrinkle: dominant_pygod, the detector behind every Part 6/7 result,
### is itself close to a degree proxy on 3 of 4 datasets

    dataset    dominant_pygod AUROC   best free baseline   sdeg
    amazon           0.3834               0.7446 (neg_deg) +0.902
    tolokers         0.5303               0.5596 (degree)  +0.909
    weibo            0.5352               0.7782 (neg_deg) +0.426
    reddit           0.5478               0.5561 (degree)  +0.579

On amazon it does not merely tie the baseline, it LOSES by 0.36 -- its raw
score ranks in the wrong direction relative to which nodes are actually
anomalous there. This is consistent with the score-gap law, not contradicted
by it: dominant_pygod's score being near-collinear with degree (sdeg 0.43-0.91)
is exactly why the clean-condition degree shift (2A.2, gap_d -1.25 on amazon)
produced such an extreme gamma (13.22) when measured through it.

But it raises the obvious referee question: is the selection-bias failure
GENERAL, or an artifact of choosing a near-degenerate detector? On amazon,
detectors with LOW degree sensitivity beat the baseline outright: gae +0.114
(sdeg=-0.091), anomalydae +0.107 (sdeg=-0.081), dominant_ours +0.081
(sdeg=-0.083).

### The decisive follow-up, not yet run

Rerun `calibration_strategy_comparison.py --dataset amazon --detector gae`.
gae has genuine signal (beats degree by 0.114) and near-zero, slightly negative
degree sensitivity -- the opposite profile from dominant_pygod.

  - If clean calibration STILL produces gamma >> 1 for gae, the selection-bias
    mechanism holds even for a detector doing real detection work, which is
    the strong and general version of the claim this paper needs.
  - If gae's clean gamma is ~1, the amazon gamma=13.22 result was substantially
    a degree-proxy artifact of dominant_pygod specifically, and the paper must
    say so rather than lead with that number.

Either outcome is reportable. Do not skip this to avoid the second one.

### Part 8 follow-up: gae on amazon settles it. The mechanism is REAL and CONDITIONAL.

The referee question raised above -- is the selection-bias failure general, or
an artifact of dominant_pygod being a degree proxy -- has a clean answer, and
it is the controlled experiment this project had been missing.

Same graph. Same clean filter. Same resulting degree gap. Only the detector's
degree sensitivity differs:

                        dominant_pygod        gae      ratio
    calib_deg                  105.6        106.3       1.0x
    test_deg                   737.2        771.3       1.0x
    degree ratio               0.143        0.138       1.0x
    sdeg (score~degree)       +0.902       -0.026      34.7x
    gap_d (score gap)         -1.252       -0.038      32.9x
    gamma                      13.22         0.76      17.4x

The clean filter shifts degree by 7x for BOTH detectors -- that part is a
property of the graph and the filter, not of the model. Whether that covariate
shift becomes a SCORE shift depends entirely on whether the score responds to
degree. It does for pygod (sdeg +0.902 -> gap -1.252 -> gamma 13.22, broken)
and does not for gae (sdeg -0.026 -> gap -0.038 -> gamma 0.76, valid).

**This completes the mechanism rather than undermining it.** The causal chain
is now demonstrated with the covariate shift HELD FIXED and only the score's
sensitivity varying:

    selection rule -> covariate shift -> (x score sensitivity) -> score gap -> gamma

and the middle multiplication is what the two detectors isolate. It also
restates the finding correctly: **calibration filtering is dangerous
CONDITIONAL on the detector being sensitive to whatever the filter selects on.**
Not universally dangerous, and not safe -- conditionally dangerous, with a
measurable condition.

That conditionality is a feature for the paper, not a hedge. It is exactly what
makes the label-free score-gap diagnostic (Part 7) the deliverable: you cannot
tell from the filter alone whether you are in trouble, you have to measure the
gap, and measuring it requires no labels.

### The first working configuration in this project

    gae + random_full on amazon:
        n_calib=4000   gamma=0.96   disc=101   FDR=0.059   power=0.116

Valid (gamma ~ 1, realized FDR 0.059 BELOW the nominal 0.10) and useful (101
discoveries, 11.6% of the 821 anomalies). Every configuration before this was
either broken (pygod/clean: FDR 0.787) or found nothing (pygod/random_full:
0 discoveries; reddit/gae/random_full: 3).

Note also that gae BEATS the degree baseline on amazon (+0.114 AUROC, Part 8),
so this is a detector doing real detection work, under a valid calibration
rule, achieving controlled FDR with non-trivial power. That is the existence
proof the paper needs: the pipeline CAN work, and the paper can say what it
takes -- a detector that is not a covariate proxy, plus an unfiltered
calibration set.

### Status of the claims after this run

  CONFIRMED  Calibration filtering induces a covariate shift (7x degree on
             amazon), independent of detector.
  CONFIRMED  That shift breaks exchangeability IF AND ONLY IF the score is
             sensitive to the shifted covariate. Demonstrated by holding the
             shift fixed across two detectors.
  CONFIRMED  The score gap predicts the violation and is label-free.
  CONFIRMED  True contamination remains harmless in both detectors.
  CONFIRMED  A working configuration exists (gae + unfiltered calibration).
  OPEN       Whether this replicates on tolokers and weibo with gae/anomalydae.
             That is now the highest-value remaining run.

### Part 8 replication: tolokers confirms the conditionality

                              amazon    tolokers
    calib_deg (clean)          106.3         5.7
    test_deg                   771.3        73.8
    degree gap                  7.3x       12.9x
    sdeg (gae)                -0.026      -0.172
    gap_d (score)             -0.038      +0.052
    gamma, clean                0.76        0.28
    gamma, random               0.81        0.99

tolokers' clean filter is HARSHER than amazon's (13x degree gap vs 7x), and
still produces no validity failure under a low-sdeg detector. On the same graph
dominant_pygod (sdeg +0.909) gives gamma@BH 9.61 and realized FDR 0.635.

Two graphs, same pattern, opposite detectors: **the covariate gap is a property
of the filter and the graph; the validity failure is a property of the
detector.** The sign prediction also holds again -- sdeg negative with
low-degree calibration puts calibration scores slightly higher, predicting
conservative, and gamma is 0.28.

**Honest limit.** gae on tolokers produces zero discoveries even unhandicapped
at n_calib=4000. The VALIDITY claim replicates across both graphs; the
USABILITY claim does not. amazon is so far the only graph where a configuration
is both valid and useful (gae + unfiltered, FDR 0.059 at power 0.116). The
paper should state this plainly: the diagnostic generalises, a working
configuration has been demonstrated once, and whether useful power is
achievable in general is open.
