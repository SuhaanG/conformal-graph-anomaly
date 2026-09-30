# Protocol: does randomized weighted BH control FDR under the acquisition design?

Frozen 2026-09-30 before any outcome of this search was computed. Revision 2 (same day, still
before any outcome): question reframed as a validity investigation; reporting rules added. Script:
`wcs_counterexample.py`. Outputs: `wcs_counterexample_ledger.csv.gz` (every evaluated
design), `wcs_counterexample_confirm.csv`, `wcs_counterexample_scale.csv`,
`wcs_counterexample_manifest.json`. All results are reported whatever they show.

## Question

Under the finite-population design of Theorem 1 (condition on scores and labels; choose
the m0 normal test nodes uniformly among the n normals; acquire every other normal
independently with known probability rho(v); weights w = 1/rho), randomized WCS with
homogeneous pruning satisfies E[FDP] <= alpha m0/m. Does BH applied to the same randomized
weighted p-values, p_j = (sum_{i in C} w_i 1{s_i > s_j} + U_j w_j)/(W + w_j), also satisfy
it? An earlier exploratory search has no saved ledger and is not used.

## Exact evaluation

For fixed roles, p_j is uniform on [b_j/(W+w_j), (b_j+w_j)/(W+w_j)] and the p-values are
independent across test nodes. BH at level alpha depends on p only through each node's
band L_j = min{k : p_j <= alpha k/m} (m+1 if none), so
FDR = sum over roles of mass x sum over band vectors of prod_j P(L_j) x FDP(bands).
This is exact up to floating point. Deterministic weighted BH (U = 1) is evaluated exactly
as a secondary arm. The code is checked first: exact FDR must agree with a direct Monte
Carlo of the whole design (20,000 draws) within 4 SE on 20 random designs, and the
design-averaged null p-value must be exactly uniform on those designs.

## Search space (stage 1: 20,000 random designs, seed 2026093001)

- n normals in {2,...,6}; m0 = nt in {1,...,min(n-1,4)}; test anomalies na in {0,...,3};
  m = nt + na <= 4.
- rho(v) for every node drawn from {0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0}
  (anomalies' rho sets only their test weight).
- Scores: a uniformly random ordering of all n + na nodes (distinct).
- alpha in {0.05, 0.1, 0.2, 0.3, 0.5}.

## Stage 2: local search (seed 2026093002)

From the 20 designs with the largest excess FDR - alpha m0/m, run 300 steps of greedy hill
climbing; a step changes one rho to another grid value, swaps two scores, or changes
alpha, and is kept if the exact excess increases. n, nt and na stay fixed.

## Stage 3: confirmation

The five designs with the largest exact excess (distinct after stage 2) are re-evaluated by
an independent Monte Carlo of the full design (test set, acquisition, U, and xi for WCS),
400,000 draws each, seed 2026093003, for randomized weighted BH and randomized WCS.

**Primary criterion.** A counterexample is confirmed if exact FDR exceeds alpha m0/m by
more than 1e-9 AND the Monte Carlo mean exceeds it by more than 5 Monte Carlo SE.
**Secondary.** Whether the confirmed FDR also exceeds the nominal alpha. Randomized WCS on
the same design is expected at or below the bound (within 4 SE); anything else is a bug and
is reported as such.

## Stage 4: does it persist at scale? (seed 2026093004)

The confirmed design with the largest excess is replicated k-fold for k in {1, 2, 5, 10, 25}:
each node becomes k nodes with the same rho, each original score becomes a block of k
consecutive scores, nt and na scale by k. Monte Carlo, 20,000 draws per k, both procedures.
An excess at scale is claimed only if it exceeds 3 SE. No other scaling scheme is tried.

## Reporting rules

- Every design stays inside the theorem's sampling assumptions (known rho > 0, uniform test
  normals, independent acquisition, role-invariant distinct scores).
- Exceeding alpha m0/m and exceeding the nominal alpha are reported separately. Exceeding
  alpha m0/m does not by itself disprove nominal FDR control.
- Candidates found by the adaptive search (stage 2) are re-verified by exact calculation from
  a fresh re-implementation (explicit enumeration of roles and of the band vectors, written
  independently of the vectorized search code) and by Monte Carlo (stage 3).
- The search scope, the number of designs evaluated and the distribution of excess
  (including all designs that did not exceed the bound) are reported.
- Failure to find a violation is not evidence of validity beyond the searched space.
- A small constructed violation would establish a mathematical distinction only; it says
  nothing about practical relevance on the benchmark graphs.

## Use in the paper

If confirmed: report the design, exact FDR, bound and the WCS value, and state that it is a
constructed finite population, not evidence about the benchmark graphs (where the
end-to-end runs detected no clear violation). Possible outcomes are a confirmed counterexample,
no violation in the searched space (validity of randomized weighted BH under this design
remains open; the search scope is reported), or an unresolved result. All are reported.
