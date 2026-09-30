# Fact sheet for the AISTATS rewrite

Regenerate with `python make_fact_sheet.py`; every number in section 3 is read from
the file named next to it. If the paper states a number, it must match this sheet.
Result files live in `audit_method/`; figures and tables in `aistats_revision/`.
Revised 2026-09-26 to incorporate the independent audit in `WCS_HANDOFF_REVIEW.md`.

## 1. The method (frozen)

Randomized weighted conformalized selection (WCS) with homogeneous pruning and
weights w(v) = 1/rho(v), where rho(v) is the known probability that node v's label
was acquired. It is the outlier-detection variant of Jin and Candes, Algorithm 2 of
arXiv v2 (`\citet{jin2023algorithm}`), with the randomized p-value (test-point term
U_j w_j). Code: `wcs_randomized.py` (`wcs_rand`); end-to-end version
`wcs_endtoend.py`. The figure and table label "WCS (deterministic p)" also uses
randomized homogeneous pruning; only its p-values are deterministic.

## 2. Theory (`wcs_design_proof.tex`; pre-review copy `wcs_design_proof_v1.tex`)

**Theorem A.** Condition on the graph, all labels, role-invariant scores (distinct via
tie marks), the anomalous test set, and propensities rho(v) in (0,1] fixed before
normal roles are assigned. Choose m0 normal test nodes uniformly; acquire each
remaining normal independently with probability rho(v). Then WCS with weights 1/rho,
deterministic or homogeneous pruning, and deterministic or randomized weighted
p-values satisfies E[FDP] <= alpha m0 / m <= alpha.

**Corollary (scorer trained on acquired labels).** Now a separate paragraph in the
proof file, proved directly by the swap argument, not by re-invoking the theorem.
With a common training fraction f in [0,1), each non-test normal is unused, training
or calibration with probabilities 1-rho, f rho, (1-f) rho. Conditioning on the
training set, the candidate test node's probability is proportional to
1/((1-f) rho), so the weights stay 1/rho. With node-dependent f(v) the weights become
1/(rho(v)(1-f(v))). Humans must still check this paragraph line by line.

**Assumptions to state in the paper:**
1. Propensities are true, known and strictly positive. Being fixed before roles are assigned does not make an estimated function correct.
2. Test normals are chosen uniformly given their count; acquisition is independent across nodes given rho.
3. Scores depend only on fixed information and (corollary) the randomly chosen training part of the acquired labels.
4. Ties are broken by independent continuous marks.
5. No i.i.d. assumption on nodes: arbitrary graph dependence is allowed because all randomness comes from the design. The argument is not specific to graphs.

**Not covered:** estimated propensities; acquisition that adapts to earlier labels;
filters applied after acquisition.

**Weighted resolution proposition.** With f_j = w_j / (w_j + calibration weight),
weighted BH and deterministic-pruning WCS rejections need |R| >= m f_j / alpha
(homogeneous: xi m f_j / alpha), and no node with f_j > alpha is discovered by any of
them. f_j is approximately 1/(1 + rho_j N_ref) only as a plug-in using the mean
Horvitz-Thompson calibration mass; it is not a concentration claim. For randomized
p-values, the first-stage rejection probability is ((t - b_j)/f_j) clipped to [0,1]
with t = alpha S_j / m; it is not the final selection probability.

**Framing:** WCS is existing work. The contribution is its finite-population
sampling-design justification (including the trained-on-acquired-labels corollary),
the resolution analysis, and the graph evidence. The 1/rho weights are
Horvitz-Thompson weights. Credit Hennhofer and Preisach (2026), which directly
addresses resolution and variance in weighted conformal anomaly detection. No
comprehensive novelty search has been done.

## 3. Numbers

### End-to-end: scorer retrained on the biased labels (`wcs_endtoend_summary.csv`)

FDP / power; z = (mean FDP - bound) / SE, with SE computed across the ten training-seed averages (seeds are the independent units; the five test splits within a seed share its prior panel). These z values are descriptive, unadjusted diagnostics, not tests of the theorem.

| Graph | Scorer | Bias | Bound | Pooled BH | Weighted BH (no guarantee) | Randomized WCS |
|---|---|---|---|---|---|---|
| Amazon | graph | moderate | 0.090 | 0.108 / 0.815 (z +4.3) | 0.080 / 0.700 (z -2.4) | 0.087 / 0.708 (z -1.0) |
| Amazon | graph | severe | 0.090 | 0.115 / 0.818 (z +6.0) | 0.075 / 0.295 (z -2.5) | 0.089 / 0.254 (z -0.1) |
| Amazon | graph | smooth | 0.090 | 0.088 / 0.789 (z -0.5) | 0.075 / 0.606 (z -3.2) | 0.093 / 0.626 (z +0.4) |
| Amazon | attr | moderate | 0.090 | 0.098 / 0.803 (z +2.6) | 0.077 / 0.661 (z -2.9) | 0.088 / 0.677 (z -0.5) |
| Amazon | attr | severe | 0.090 | 0.108 / 0.806 (z +4.4) | 0.073 / 0.292 (z -4.8) | 0.093 / 0.253 (z +0.3) |
| Amazon | attr | smooth | 0.090 | 0.090 / 0.775 (z -0.3) | 0.071 / 0.576 (z -7.8) | 0.091 / 0.616 (z +0.1) |
| T-Finance | graph | moderate | 0.095 | 0.159 / 0.828 (z +14.9) | 0.096 / 0.790 (z +0.2) | 0.102 / 0.796 (z +1.9) |
| T-Finance | graph | severe | 0.095 | 0.180 / 0.832 (z +17.8) | 0.066 / 0.542 (z -5.5) | 0.092 / 0.518 (z -0.6) |
| T-Finance | graph | smooth | 0.095 | 0.175 / 0.830 (z +15.8) | 0.007 / 0.001 (z -24.5) | 0.095 / 0.136 (z -0.1) |
| T-Finance | attr | moderate | 0.095 | 0.109 / 0.660 (z +3.8) | 0.095 / 0.645 (z -0.0) | 0.098 / 0.633 (z +0.6) |
| T-Finance | attr | severe | 0.095 | 0.113 / 0.659 (z +4.9) | 0.030 / 0.033 (z -5.7) | 0.092 / 0.138 (z -0.4) |
| T-Finance | attr | smooth | 0.095 | 0.101 / 0.653 (z +1.6) | 0.004 / 0.000 (z -36.1) | 0.102 / 0.108 (z +0.6) |
| Weibo (GADBench) | graph | moderate | 0.090 | 0.158 / 0.943 (z +6.3) | 0.081 / 0.783 (z -1.6) | 0.091 / 0.792 (z +0.4) |
| Weibo (GADBench) | graph | severe | 0.090 | 0.177 / 0.947 (z +6.0) | 0.025 / 0.010 (z -7.9) | 0.081 / 0.106 (z -0.8) |
| Weibo (GADBench) | graph | smooth | 0.090 | 0.176 / 0.950 (z +7.0) | 0.063 / 0.493 (z -5.2) | 0.089 / 0.660 (z -0.1) |
| Weibo (GADBench) | attr | moderate | 0.090 | 0.121 / 0.782 (z +11.1) | 0.082 / 0.671 (z -1.5) | 0.088 / 0.657 (z -0.2) |
| Weibo (GADBench) | attr | severe | 0.090 | 0.126 / 0.766 (z +7.3) | 0.020 / 0.007 (z -10.3) | 0.061 / 0.052 (z -3.2) |
| Weibo (GADBench) | attr | smooth | 0.090 | 0.139 / 0.795 (z +11.7) | 0.052 / 0.294 (z -5.1) | 0.096 / 0.527 (z +1.6) |

- Pooled BH exceeds the bound by more than 2 seed-clustered SEs in 15/18 cells.
- Randomized WCS: largest excess +1.87 SE; results are consistent with the design-based bound. Some sample means sit slightly above it (e.g. T-Finance graph, moderate: 0.1024 vs bound 0.0954); this neither proves nor refutes control. The proof supplies the claim.
- Randomized WCS mean power exceeds weighted BH in 13/18 cells and is at least that of deterministic-p WCS (same pruning coin) in 18/18. Weighted BY and stratified rules were not run end-to-end, so do not claim superiority over every valid method.
- Retrained-scorer AUROC: cell means 0.939-0.994; individual fits 0.920-0.998.
- Label budget: the design still includes the uniform 10% prior panel (exposure definition and training). Only newly acquired labels are biased. State this prominently and count the panel in label costs.

### Propensities: true vs estimated vs misspecified (`wcs_estimated_summary.csv`), FDP / power

| Graph | Scorer | Pooled BH | WCS, true rho | WCS, estimated rho | WCS, 2-stratum rho (misspecified) |
|---|---|---|---|---|---|
| Amazon | graph | 0.091 / 0.788 | 0.090 / 0.706 | 0.089 / 0.696 | 0.080 / 0.769 |
| Amazon | attr | 0.090 / 0.781 | 0.088 / 0.700 | 0.087 / 0.694 | 0.085 / 0.759 |
| T-Finance | graph | 0.169 / 0.823 | 0.093 / 0.252 | 0.092 / 0.242 | 0.157 / 0.817 |
| T-Finance | attr | 0.100 / 0.655 | 0.091 / 0.177 | 0.094 / 0.185 | 0.097 / 0.650 |
| Weibo (GADBench) | graph | 0.183 / 0.942 | 0.092 / 0.754 | 0.090 / 0.720 | 0.139 / 0.905 |
| Weibo (GADBench) | attr | 0.137 / 0.728 | 0.093 / 0.572 | 0.090 / 0.539 | 0.118 / 0.688 |

The estimated model is cross-fitted logistic regression that uses the same covariates that generate this synthetic acquisition mechanism. It is an empirical sensitivity analysis, not evidence for arbitrary operational selection, and the theorem does not cover it.

### Resolution diagnostic (`wcs_randomized_summary.csv`; known rho, 80/20 strata, graph scorer)

m f / alpha for high-exposure anomalies (deterministic benchmark; for homogeneous pruning the realized necessary bound is multiplied by xi), anomaly share in the sparse stratum, and power:

| Graph | Bias | median m f / alpha | anomalies in sparse stratum | WCS (det. p) power | Randomized WCS power |
|---|---|---|---|---|---|
| Amazon | none | 7 | 0.51 | 0.789 | 0.790 |
| Amazon | moderate | 37 | 0.51 | 0.758 | 0.773 |
| Amazon | severe | 146 | 0.51 | 0.160 | 0.331 |
| T-Finance | none | 7 | 0.88 | 0.791 | 0.791 |
| T-Finance | moderate | 35 | 0.88 | 0.773 | 0.780 |
| T-Finance | severe | 140 | 0.88 | 0.730 | 0.766 |
| Weibo (GADBench) | none | 7 | 0.91 | 0.848 | 0.850 |
| Weibo (GADBench) | moderate | 37 | 0.91 | 0.790 | 0.813 |
| Weibo (GADBench) | severe | 148 | 0.91 | 0.197 | 0.345 |

### Validation (json files in `audit_method/`)

- Fast WCS vs the repository's own m-by-m equation implementation: 400 random cases identical. Agreement with the authors' public code covers only six constant-weight cases; do not claim variable-weight agreement.
- Exact finite-design FDR (deterministic p, both prunings): 486 designs, 46224 assignments, none above the bound. Weighted BH: none above the bound in these designs (not a proof).
- Randomized p-values: exact uniformity in 243 designs; Monte Carlo FDR in 216 designs x 10000 draws, worst +2.2 SE (pre-set limit 4).
- Training-dependent scores: 96 designs, 40176 exact states, none above the bound.
- Resolution condition: 39207 nonempty rejection sets checked (each at its largest floor), 0 violations.
- An independent audit (`WCS_HANDOFF_REVIEW.md`) re-aggregated all 367,800 trial rows, matched every manifest hash, and added exact integration over the auxiliary coins (32 designs, max excess ~1e-16).

## 4. Figures and tables (regenerate with `make_wcs_figures.py`)

| File | Content | Caption facts |
|---|---|---|
| `wcs_endtoend_figure.pdf` | FDP (top) and power (bottom), graph-feature scorer, 3 graphs x 3 bias settings, 4 methods | Scorer retrained on the biased labels; means over 10 seeds x 5 test splits x 3 acquisition repetitions; FDP bars are pointwise 95% t intervals (9 df) across the ten seed averages; dashed line alpha m0/m; weighted BH has no guarantee here; bars spanning the bound do not prove control |
| `wcs_endtoend_figure_attributes.pdf` | Same, attribute scorer (appendix) | same |
| `wcs_endtoend_table.tex` | All 18 end-to-end cells, FDP / power | dagger marks no guarantee |
| `wcs_estimated_table.tex` | True vs estimated vs misspecified propensities | see section 3 caveat |
| `wcs_resolution_table.tex` | Kish ESS, m f / alpha, anomaly share, power | known rho, 80/20 strata |

## 5. Citations (new entries in `wcs_refs_additions.bib`, checked 2026-09-26)

Already in `aistats_refs.bib`: jin2026weighted (Biometrika 113(1), asaf066, 2026),
jin2023algorithm (arXiv v2, Algorithm 2), tibshirani2019conformal,
marandon2024adaptive, bates2023testing, benjamini2001control, jin2025focal.
New: Horvitz and Thompson (1952), JASA 47(260):663-685; Fithian and Lei (2022),
Annals of Statistics 50(6):3091-3118, doi 10.1214/21-AOS2137; Hennhofer and
Preisach (2026), arXiv:2603.23205.

## 6. Protocol disclosure (must appear in the paper or appendix)

The WCS, randomized-WCS, estimated-propensity and end-to-end experiments were
follow-ups chosen after inspecting earlier results. Each protocol was frozen before
its own outcomes: `WCS_ACQUISITION_PROTOCOL.md`, `WCS_RANDOMIZED_PROTOCOL.md`,
`WCS_ESTIMATED_PROTOCOL.md`, `WCS_ENDTOEND_PROTOCOL.md`. The pre-set decision rule
(WCS power within 0.05 of weighted BH at moderate bias on T-Finance and Weibo) was
met by homogeneous pruning and failed by deterministic pruning; both are reported.
The figure and summaries switched from test-set SEs to seed-clustered SEs after the
audit; the original `fdp_se_testset` column is kept for provenance.

## 6b. Adversarial review changes (2026-09-29)

- Power comparison must be like for like. End-to-end addendum (`WCS_ENDTOEND_ADDENDUM.md`, frozen before the rerun; all 10,800 earlier outcomes reproduced exactly): BH on the same randomized weighted p-values (`weighted_bh_rand`, no guarantee) has mean FDP <= 0.103 (max z +1.79). Randomized WCS power minus it: moderate 0.000 to 0.048 (WCS keeps >= 93%); severe 0.002 to 0.348 (keeps 28% to 98%; T-Finance attr 0.138 vs 0.486, Amazon gaps 0.195/0.216, Weibo both low); smooth 0.017 to 0.136. The gap is from WCS pruning, not the weights. Randomized WCS still beats *deterministic* weighted BH in 13/18 cells. Frozen-score numbers (moderate gap 0.004-0.013, severe up to 0.365) are appendix only. Do not write "matches or exceeds weighted BH" without saying which one.
- A deterministic filter (Table 1) sets rho = 0 on removed nodes; no reweighting repairs it. The remedy is procedural: down-sample exposed normals with a known probability and weight by 1/rho.
- Known non-uniform rho arises when the analyst sets it (limited capacity, uneven verification cost); unrecorded operational selection is an open case.

## 7. Claims not to make

- That WCS is a new method.
- That weighted BH fails. The finite-design checks found no violation. An earlier exploratory counterexample search has no saved ledger and must not be reported.
- That the theorem covers estimated propensities. A misspecified model loses control (see section 3).
- A universal "not useful under severe bias". Power under severe bias varies by cell (for example 0.518 on T-Finance with the graph scorer); report the full grid.
- That randomized decisions are deterministic. Individual decisions depend on the auxiliary coin U_j.
- That all scenarios run in linear time. Auxiliary counts are O(m) per distinct test weight after sorting; smooth propensities can give m distinct weights and quadratic total work.
- Anything about how common exposure-biased labeling is in deployment without a verified citation.

## 8. Submission facts

Abstract, title and final author list due Tuesday 29 September 2026, 23:59 AoE;
full paper and supplement Tuesday 6 October 2026. Major title or abstract changes
after 29 September can cause desk rejection, so the WCS framing must be in the
submitted abstract. Main text is limited to eight pages. AI assistance is permitted
with disclosure and author responsibility; update the AI Use Statement to cover
method design and proofs. A fresh clone does not yet reproduce the results: the
processed graphs, cached scores and raw trial ledgers are local only. Ship them in a
hash-checked supplement archive and test it from a clean checkout.
