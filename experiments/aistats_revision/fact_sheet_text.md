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

<<NUMBERS>>

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
