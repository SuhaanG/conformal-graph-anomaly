# Randomized WCS with estimated acquisition propensities

Frozen 25 September 2026 after inspecting covariate distributions only (share
of nodes with known anomalous neighbors, degree quantiles), before any outcome
under this scenario. Robustness check requested to answer "how would a
practitioner know rho?". No guarantee is claimed for estimated weights.

## Acquisition scenario (true propensity unknown to the analyst)

For deployment node v, k_v = number of training-labeled anomalous neighbors,
z_v = log(1 + degree) standardized over deployment nodes of that graph.
logit rho(v) = -1.2 sqrt(k_v) + 0.3 z_v, clipped to [0.01, 0.95].
Uniform 25%-per-class test assignment as before; each non-test deployment node
(normal or anomalous) is acquired independently with probability rho(v), using
the existing acquisition uniforms SeedSequence([2026092407, dsid, seed, split,
rep]). Acquired normals form the reference.

## Propensity estimates (use only acquisition indicators of non-test nodes)

- oracle: true rho (Theorem A applies).
- logit: logistic regression (sklearn, L2, C=1, standardized features
  sqrt(k), k, log(1+degree), training-label exposure fraction), 2-fold
  cross-fitting over non-test deployment nodes with folds from
  SeedSequence([2026092506, dsid, seed, split, rep]); each non-test node gets
  its out-of-fold prediction, test nodes the average of both fold models.
  Clipped to [0.005, 1].
- strata2 (deliberately misspecified): empirical acquisition rate within the
  fixed lower80/upper20 training-exposure strata.

## Arms

Pooled biased-reference BH; randomized WCS with homogeneous pruning using
oracle, logit and strata2 weights (same xi and U across the three); weighted BH
with oracle weights (descriptive). xi from SeedSequence([2026092507, ...]),
U from SeedSequence([2026092508, ...]) with (dsid, seed, split, rep, model).
Amazon, T-Finance, GADBench Weibo; both HGB scorers; ten seeds; five splits;
ten acquisition repetitions. alpha = 0.10. Report every arm and cell,
including failures of the estimated-weight arms.

## Reporting

Mean FDP and power, with test-set-level SEs (50 test sets per cell), and the
bound alpha m0/m. Estimated-weight results are described as empirical
robustness evidence under this generator only.
