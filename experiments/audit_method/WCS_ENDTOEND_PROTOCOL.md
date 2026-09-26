# End-to-end biased acquisition: scorer trained on the biased labels

Frozen 25 September 2026, before any outcome of this design. Follow-up chosen
after the WCS, randomized-WCS and estimated-propensity results; not an
untouched confirmatory study.

## Motivation

All earlier supervised results train the scorer on a uniform 10% panel while
only calibration labels are biased. Here every new label comes from one
exposure-biased acquisition and is randomly split between training and
calibration.

## Design (per dataset, seed, split, repetition, scenario)

- Prior known fraud: the cached seed panel (`train` in the score caches). Its
  labels define exposure (training-label exposure fraction and the fixed
  lower80/upper20 strata, as before) and it is included in scorer training.
  It never enters calibration or testing.
- Test: 25% of deployment normals and 25% of deployment anomalies, uniformly,
  with the existing SeedSequence([20260926, dsid, seed, split]) stream.
- Acquisition: each non-test deployment node is acquired independently with
  probability rho(v); existing uniforms SeedSequence([2026092407, dsid, seed,
  split, rep]).
- Split: each acquired node goes to training independently with probability
  f = 1/2 (SeedSequence([2026092509, dsid, seed, split, rep, scenario index]));
  acquired normals not sent to training form the calibration set.
- Scorers: the two fixed HGB configurations (attributes; attributes + neighbor
  mean + log degree), unchanged hyperparameters, random_state = seed, trained
  on prior panel + acquired training half, scoring all nodes.
- Scenarios: moderate (.5, .1) and severe (.5, .025) two-level rho on the
  lower80/upper20 strata; smooth rho from WCS_ESTIMATED_PROTOCOL.md.
- Repetitions: seeds 0-9, splits 0-4, reps 0-2 (reduced for retraining cost).

## Claim under test

Conditioning additionally on the training set, the probability that a given
member of calibration-plus-designated-test is the test node is proportional to
1/(rho (1 - f)), i.e. to 1/rho. Hence Theorem A applies with unchanged weights
whenever scores depend only on the training set and fixed graph information.

## Validation before reporting

Exact enumeration of small finite populations over test assignment,
acquisition and the training split, with scores that change arbitrarily with
the realized training set (pseudo-random function of it): exact FDR of
deterministic WCS (homogeneous integrated exactly over xi) must not exceed
alpha m0 / m.

## Arms

Pooled biased-reference BH; weighted BH (descriptive); randomized WCS with
homogeneous pruning (weights 1/rho, xi and U from new streams 2026092510 /
2026092511 keyed by dataset, seed, split, rep, scenario, model). Report every
cell, AUROC of the retrained scorers, and all failures.
