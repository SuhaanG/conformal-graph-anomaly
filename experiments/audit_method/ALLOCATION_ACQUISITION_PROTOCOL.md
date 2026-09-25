# Error allocation and exposure-biased acquisition

Frozen 24 September 2026, before running these variants. This is a follow-up chosen after inspecting equal-alpha stratification results; it is not an untouched confirmatory study.

## Question and fixed scope

Separate the power cost of splitting alpha from the cost of separate reference ranks. Evaluate Amazon, T-Finance and GADBench Weibo; both existing HGB scorers; all ten cached training seeds and five test splits. No retraining, score-direction changes, detector tuning, result-based choice of graph, partition or allocation. Alpha is 0.10. Preserve all outcomes, including zero discoveries.

## A: Allocation ablation on the previous sampling design

Reuse STRATIFIED_PROTOCOL.md exactly: the same sampled test identities, revelation draws, score tie marks and two partitions (zero/positive exposure; lower80/upper20 exposure across the deployment universe). Revelation fractions are .25, .5 and 1. Compare full random reference, low-stratum reference pooled over all test points, equal-level separate BH (.05/.05), size-proportional separate BH (alpha*m_h/m), and one BH at .10 on concatenated stratum-specific conformal p-values. Require the first three existing arms to reproduce previous per-split outcomes before using the additions.

## B: Exposure-biased label acquisition

Freeze exposure using ONLY the already acquired training-panel anomaly labels and fixed graph degree. This avoids circularly defining selection from the labels being acquired. Freeze either zero/positive strata or 80/20 exposure-rank strata over the full deployment universe, using fixed independent exposure tie marks. Training nodes never re-enter calibration/test.

Use the same 25%-per-class test split generator as A. For each non-test deployment node acquire its label independently with probability rho_h according to its frozen stratum. Keep acquired normals as references; count every acquired label, including anomalies, in the cost. Do not update strata after acquisition. Probabilities (low,high) are (.5,.5), (.5,.1), (.5,.025), denoting neutral, moderate and severe selection. There is positive support everywhere. No test labels or unrevealed labels enter selection, weighting or p-values. Ten acquisition repetitions per split use shared uniform marks across probability settings and score families.

Compare pooled biased-reference BH; equal-alpha stratified BH; size-alpha stratified BH; pooled BH of stratum-wise p-values; known-propensity weighted ranks followed by BH (descriptive unless a joint guarantee is established); and the same weighted ranks followed by BY. Include uniform acquisition as a COUNTERFACTUAL design benchmark, not as an available reference in the biased-acquisition scenario. Its common probability is mean(rho_h) over the FULL deployment universe, fixed before test assignment and without deployment labels. This matches average acquisition propensity; class-specific test-count rounding can cause a small difference in expected total cost. Record actual label costs and reference sizes; do not claim exact cost matching. Computing that probability from the realized non-test set would introduce normal-role dependence, so do not do that.

Weighted ranks use w=1/rho_h and include the test point's weight in numerator and denominator. Empty references give p=1; all procedures retain all test hypotheses. Score ties use the previous independent lexicographic marks. Budget allocation and fixed probabilities must be stated explicitly.

## Validity checks before reporting guarantees

Condition on graph, fixed labels, training, role-invariant scores, anomaly-side assignments and all stratum role counts. For constant acquisition probability within each fixed stratum, feasible normal-role assignments have constant probability. Conditional on all counts, their law factors over strata. Apply the existing within-stratum distinct-score exchangeability theorem; prove independent PRDS blocks remain PRDS, then apply pooled BH. Size-proportional levels are fixed AFTER conditioning on counts, not before realized test identities; the existing sum-FDP argument yields alpha*m0/m.

For weighting, separately derive weighted rank super-uniformity by exchanging a designated null test position with reference positions: conditional target mass is proportional to 1/rho_h. BY needs only those marginal bounds. Do not borrow a BH guarantee from marginal validity. Verify the argument by exact enumeration of small finite-population role assignments with unequal probabilities, empty references and ties resolved before assignments. Finite enumerations are checks, not proofs.

## Reporting and stopping rule

Summaries average acquisition repetitions and splits within seed, then ten seeds. Pointwise paired intervals use the ten seed-average differences; they quantify repeated evaluation on a fixed graph. A primary FDP-versus-power figure uses rho=.5 and the 80/20 partition for A, and all three acquisition probability settings with the 80/20 partition for B. Appendices retain both partitions and every condition. Do not select an attractive partition post hoc. Report comparisons with equal-alpha and pooled full random, even if improvements are absent. Do not assert power ordering, novelty or acceptance probabilities. Pilot-panel optimization is parked.
