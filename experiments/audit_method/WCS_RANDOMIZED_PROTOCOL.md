# Randomized weighted p-values and a weighted resolution condition

Frozen 25 September 2026, before any randomized-WCS outcome on graph scores.
Chosen after inspecting WCS_ACQUISITION_PROTOCOL.md results, where WCS lost
power in the severe scenario. Not an untouched confirmatory study.

## Hypothesis being tested

Severe-scenario power loss is caused by the test-point weight floor
f_j = w_j / (W + w_j) of the deterministic weighted p-value. Replacing the
test-point term w_j by U_j w_j, with independent U_j ~ Unif(0,1), removes the
floor while keeping exact conditional uniformity under the design.

## Arms (added to the unchanged WCS_ACQUISITION design)

Same graphs, scorers, seeds, splits, strata designs, scenarios, acquisition
repetitions and streams as wcs_acquisition.py. New arms:
- wcs_rand_homogeneous: randomized p_j, deterministic auxiliary values,
  homogeneous pruning with the SAME xi as wcs_homogeneous.
- wcs_rand_deterministic: randomized p_j, deterministic pruning.
- weighted_bh_rand: BH on randomized weighted p-values (descriptive only;
  no joint guarantee is claimed).
U_j from SeedSequence([2026092503, dsid, seed, split, design index, rep,
scenario index, model index]), one uniform per test node, shared by the three
randomized arms. Existing arms are recomputed and must reproduce
wcs_acquisition_trials exactly.

## Resolution diagnostics (recorded per run, all arms)

Kish effective calibration size (sum w)^2 / sum w^2; for anomalous test nodes
in each stratum, the median of m f_j / alpha (the smallest rejection count at
which that node could be discovered by deterministic weighted p-values); the
fraction of anomalous test nodes with f_j <= alpha. Check in every run that
each weighted-BH and deterministic-WCS (both prunings for deterministic
p-values: |R| >= m f_j / alpha for deterministic pruning, |R| >= xi m f_j /
alpha for homogeneous) rejection satisfies the stated necessary condition.

## Validation before reporting

1. Exact enumeration on the small finite designs of wcs_acquisition.validate:
   the design-averaged probability that a null test's randomized p-value is at
   most t equals t (to 1e-12) for a grid of t.
2. Monte Carlo over U (4,000 draws per case, homogeneous pruning integrated
   exactly over xi) on those designs: randomized-WCS FDR must not exceed
   alpha m0 / m by more than 4 Monte Carlo SEs.

## Reporting

Report every arm and scenario, including where randomization does not help or
reduces power. State that randomized p-values make individual decisions depend
on an auxiliary coin, which some practitioners may find unacceptable.
