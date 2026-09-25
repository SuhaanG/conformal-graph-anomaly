# Weighted conformalized selection under exposure-biased acquisition

Frozen 25 September 2026, before any WCS outcome on the graph score caches.
This follow-up was chosen after inspecting ALLOCATION_ACQUISITION_PROTOCOL.md
results (weighted BH had high power without a joint guarantee; weighted BY had
a guarantee without power). It is not an untouched confirmatory study.

## Question

Under the frozen biased-acquisition design (uniform normal test assignment,
then independent acquisition of each remaining node with known probability
rho_h fixed by training-label exposure strata), does hypothesis-conditional
WCS (Jin and Candes, Algorithm 2, deterministic and homogeneous pruning) with
weights w = 1/rho_h retain power close to weighted BH while controlling FDR?

## Fixed scope

Exactly the acquisition phase of allocation_acquisition.py: Amazon, T-Finance,
GADBench Weibo; both HGB scorers; ten training seeds; five test splits; both
strata designs (zero/positive, lower80/upper20); scenarios neutral (.5,.5),
moderate (.5,.1), severe (.5,.025); ten acquisition repetitions; identical
SeedSequence streams, score tie marks, test identities and acquired sets.
alpha = 0.10. No retraining, tuning, or result-based choice of graph, design,
scenario, or pruning rule. Report both pruning rules and every outcome,
including zero discoveries.

Homogeneous pruning uses one uniform per (dataset, seed, split, design, rep,
scenario, model), from SeedSequence([2026092501, dsid, seed, split, design
index, rep, scenario index, model index]).

## Arms

Recomputed on the same inputs: pooled biased-reference BH, equal-alpha
stratified BH, pooled stratum BH, weighted BH, weighted BY. New: WCS
deterministic, WCS homogeneous. The recomputed arms must reproduce the stored
allocation_acquisition_trials outcomes exactly before any WCS result is used.

## Validation before reporting

1. The O(m) WCS implementation equals the existing validated m-by-m
   implementation (wcs_experiment.wcs) on random distinct-score cases with
   two-valued and continuous weights.
2. Exact enumeration of small finite populations under the stated design
   (all normal test sets, all acquisition subsets, unequal known rho, anomalies
   present or absent): exact FDR of WCS (both prunings, homogeneous averaged
   over its uniform exactly by enumeration of its breakpoints or a fine grid)
   must not exceed alpha * m0 / m. Weighted BH exact FDR is recorded
   descriptively, whatever it is.

## Decision rule (stated in advance)

WCS is carried into the main paper only if, at the moderate scenario and the
80/20 design, its mean power on T-Finance and GADBench Weibo is within 0.05 of
weighted BH for both scorers, or clearly exceeds equal-alpha stratified power.
Otherwise it is reported in the appendix as a guaranteed but less powerful
option, and the paper is submitted without the reframing.
