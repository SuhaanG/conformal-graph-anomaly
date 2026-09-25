# Role-randomized stratified calibration

2026-09-23, frozen before this remedy's outcomes. Existing detector/dose outcomes
have been inspected. This is an application of exchangeable conformal BH, not
a claim that a new conformal principle has been invented.

Reuse all available cached supervised scores, ten training seeds and five
randomized deployment splits for Amazon, Tolokers, PyGOD Weibo; add T-Finance
and GADBench Weibo if source acquisition succeeds. Reveal non-test labels at
.25,.50,1 with the existing streams. Condition in the proof on the entire graph,
labels, training panel, fitted score vector and all anomaly-side role/revelation
randomness. Exposure then depends only on frozen known-anomaly indicators and
the graph, not on which remaining normals were revealed or assigned to testing.

Two separate fixed stratum designs, both reported:
1. zero versus positive observed exposure;
2. lower 80% versus upper 20% of observed exposure ranks over ALL deployment
   nodes, with independent continuous tie marks before normal-role assignment.
The second cutoff must NOT be computed using just realized calibration normals:
that would let normal roles influence the strata and invalidate this proof.
Uniform marks also break score ties lexicographically, preserving strict order.

For each design, compare on exactly the same test set and revealed reference:
- pooled full random reference, BH at alpha=.10;
- biased low-stratum reference used for all test nodes, BH at .10;
- stratified calibration: each test node uses every reference normal in its
  own stratum; apply BH separately at .05 in each of the two strata; pool the
  discoveries. Empty reference strata abstain (p=1), empty test strata contribute
  zero. Do not reallocate alpha in response to outcomes or empty strata.

The full-reference and stratified arms have identical total reference/label
budgets. The filtered arm has fewer references and is a diagnostic baseline;
the existing graded matched-size experiment isolates composition separately.
No choice between stratum designs uses deployment FDP. Report each separately,
including any loss in power or failure to improve on pooled random calibration.
This is not post-hoc repair of scores/p-values computed on a selected reference.

Proof: uniform normal-role assignment yields conditional exchangeability of
reference and null-test scores within each fixed stratum, conditional also on
their role counts and anomalous scores. Apply the exchangeability/PRDS theorem
within each stratum. Pointwise FDP(union)<=sum_h FDP_h, so summed FDR <=sum_h
alpha_h<=alpha without any cross-stratum independence or PRDS argument. Sparse
strata reduce attainable rank resolution and power, not the claimed validity.

Validation: unknown-label perturbations must not alter exposure; holding anomaly
roles fixed and permuting normal roles must not alter strata; exhaustive small
finite-population FDR calculations with nonempty discoveries and randomized ties;
reference implementation checks, disjointness, per-stratum budget counts, and
complete seed coverage. Save per-split outcomes and paired seed summaries.
