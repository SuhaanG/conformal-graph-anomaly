# Protocol: uniform acquisition at the same expected label budget

Frozen 2026-09-30 before any outcome of this comparison was computed. Revision 2 (same day):
the first version set rho_u to the mean of rho over the realized non-test nodes of each split,
so a node's acquisition probability depended on which nodes were tested. That violates the
condition of Theorem 1 that propensities are fixed under changes of test roles (a two-node
example gives FDR 0.1225 at alpha 0.10). The run was stopped after four Amazon seeds with no
summaries written; its log and script are kept as `wcs_uniform_v0_exploratory.*` and are not
used. No outcome of that run was inspected beyond the progress log. Script:
`wcs_uniform.py`. Outputs: `wcs_uniform_trials.csv.gz`, `wcs_uniform_seeds.csv`,
`wcs_uniform_paired.csv`, `wcs_uniform_manifest.json`. All results are reported.

## Question

If the team controls the acquisition probabilities, is uneven acquisition worth using at
all? For each biased design of the end-to-end experiment (`WCS_ENDTOEND_PROTOCOL.md`:
moderate, severe, smooth), compare it with uniform acquisition at the same expected total
label budget.

## Design

Everything matches the end-to-end experiment (same graphs, prior panel, ten seeds, five test
splits, three acquisition repetitions, both HGB scorers, training fraction 1/2, alpha 0.10)
except the acquisition probability. For a biased design rho(v), the uniform design uses
rho_u = mean of rho(v) over the whole deployment population, fixed for each graph, seed
and bias setting before test nodes are assigned. Budgets are matched over the sampling
design, not per split: because test counts are class-stratified and rounded, the expected
per-split label counts rho_u |reference| and sum_{reference} rho(v) can differ, and both are
recorded and reported. The rate is never allowed to depend on which nodes are tested.
Acquisition reuses the same uniform draws u(v) (acquired if u(v) < rho_u), the same
train/calibration coins, the same scorer seeds and the same p-value auxiliaries as the
biased run, with the scorer retrained on each design's own acquired labels. The prior
panel is shared and counted identically in both arms.

## Arms under uniform acquisition

- Randomized WCS with homogeneous pruning and constant weights (covered by Theorem 1).
- Pooled BH with deterministic conformal p-values (valid under uniform acquisition).

The biased arms are read from `wcs_endtoend_trials.csv.gz` (randomized WCS, pooled BH,
randomized weighted BH), keyed by dataset, model, seed, split, rep and scenario.

## Primary comparison

Randomized WCS power under uniform minus randomized WCS power under the matched biased
design, per (graph, scorer, bias) cell, 18 cells. Seeds are the units: paired difference
of seed means, 95% t interval with 9 df. Also reported: FDP against alpha m0/m, realized
training and calibration label counts, training anomalies, AUROC, and pooled BH under
uniform.

## Interpretation, fixed in advance

- Interval above zero in a cell: uniform acquisition is better there.
- Interval below zero: biased acquisition with WCS is better there; the paper states where.
- Interval covering zero: no detectable difference.
- If uniform is better or similar across the grid, the paper recommends uniform acquisition
  when the team controls it and positions WCS as the correction for imbalance that is
  imposed or operationally required. No operational constraint (for example verification
  cost) is asserted without a measured source; such settings are named as future work.
- No further acquisition designs are added after seeing these results.
