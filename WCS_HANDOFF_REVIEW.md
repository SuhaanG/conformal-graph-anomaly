# WCS handoff review, 26 September 2026

The new numerical evidence passed the checks below. The current Overleaf paper is
not yet the WCS paper described in the fact sheet. This review does not certify
submission readiness, novelty, or an acceptance probability.

## Verification performed

- Checked the live GitHub main ref: `cb29c6829d988987851d1adbb284688f7b90b808`.
  Both new experiment commits are on the remote.
- Reaggregated all four local raw WCS ledgers: 144,000 acquisition rows, 198,000
  randomized rows, 15,000 estimated-propensity rows, and 10,800 end-to-end rows.
  All 367,800 rows have unique design keys; FDP/power are in range. All 786 summary
  cells and their seed summaries agree with the tracked CSVs to numerical tolerance.
- Matched every recorded input, script, and protocol SHA256 in the four manifests.
  The end-to-end manifest records graphs but omits the ten cached training-panel
  files per graph read by that runner; those files are covered by the other manifests.
- Compared fast auxiliary BH counts, randomized p-values, and pruning with an
  independently implemented literal calculation in 1,000 new random cases.
- Reran the existing 400 implementation comparisons and 486 exact finite designs
  (46,224 assignments), and the 96 training-dependent designs (40,176 states).
  Their reported validation summaries reproduce.
- Added independent exact integration of the auxiliary p-value coins and homogeneous
  pruning coin, including training-dependent scores: 32 small designs, 824
  positive-probability states, largest FDR excess 1.11e-16 (rounding).
  These cover empty calibration and some acquisition probabilities equal to one.
- Retrained nine graph-feature models: seed 0, split 0, acquisition repetition 0,
  for each graph and all three scenarios. All 36 method outcomes reproduce the
  stored FDP, power, discoveries, and AUROC. This is a sample retraining check,
  not a rerun of every fitted model or a new confirmatory experiment.
- Regenerated the three new LaTeX result tables into the audit directory. All
  agree exactly with the supplied end-to-end, estimated-propensity, and resolution tables.
- Inspected the supplied figure, current proof source, local compiled manuscript,
  and live Overleaf file tree, abstract, and outline. The manuscript was not edited.

Reproduce with `python scripts/audit_wcs_handoff.py --retrain-smoke` from the
repository root. Requires local `paper/` inputs, numpy, pandas, scipy,
scikit-learn, and threadpoolctl. The script writes only its own audit outputs under
`notes/wcs_audit_20260926/`. No GPU is required. That directory contains `audit.json`,
`endtoend_uncertainty.csv`, and regenerated tables. The two existing deterministic
validation suites were also run separately with their output directory redirected
there, preserving the experiment artifacts.

## Must address before submission

1. **Integrate the work.** Live Overleaf is still the earlier selection/stratified
   calibration paper. Its file tree lacks the new proof, end-to-end figures, and
   tables. The local `aistats.tex` also does not input `wcs_design_proof.tex` and
   explicitly says the older WCS simulation does not give a graph correction
   theorem. The local PDF is 45 pages; the live preview is 44, so they are not the
   same revision either. Rewrite the abstract, contributions, design, and main
   results around the verified new evidence, then rebuild both copies.

2. **State the end-to-end corollary.** The base swap proof appears correct under
   its fixed-score assumptions, and the randomized-rank argument is sound. The
   current proof file does not give the acquired-training extension promised by
   the fact sheet. Conditioning on a training set does not by itself preserve
   uniform null-test assignment; prove the swap law directly as described below.
   Human authors still need to check the completed proof line by line.

3. **Use uncertainty appropriate to the design.** `fdp_se_testset` treats 50
   split means as independent, although five share each seed's initial panel.
   For inference averaging over random initial panels, use the ten seed averages
   as clusters (and, for pointwise 95% intervals, an appropriate small-sample
   method such as t with nine degrees of freedom). Retain the original summaries
   for provenance and label the changed uncertainty calculation. The SE changes
   by as much as a factor of 2.01 across cells. With seed-cluster SEs, pooled BH
   is still over the bound by more than 2 SE in 15/18 cells, and the maximum
   randomized-WCS excess is 1.87 SE. These are descriptive, unadjusted diagnostics,
   not simultaneous tests or evidence over newly sampled graphs. Pointwise bars
   spanning the bound do not prove control; the proof supplies that claim.

4. **Complete reproduction for a fresh clone.** The new source and summary CSVs
   are on GitHub, but the three processed graphs and 60 cached score/panel files
   referenced across the manifests are absent from `experiments/`. So are the
   four raw WCS trial ledgers. Local `paper/` caches do not satisfy paths resolved
   relative to scripts in `experiments/`. Provide a hash-checked artifact archive
   or a tested acquisition/bootstrap command, document one canonical directory
   layout, and test from a clean checkout. Root `requirements.txt` also omits
   pandas, scikit-learn, and other experiment dependencies. The new proof,
   figure generator, figure PDFs, result tables, and citation additions are
   currently only in ignored `paper/aistats_revision/`.

## Training-dependent extension to write explicitly

Fix the initial graph, labels, prior panel, propensities and anomaly-side roles.
Choose the normal test positions as in the base design. A non-test normal has
three statuses: unused with probability `1-rho(v)`, training with probability
`f*rho(v)`, or calibration with probability `(1-f)*rho(v)`, with a common fixed
`0 <= f < 1`. Condition on the acquired training set B, the model's independent
randomness, fixed tie marks, unused identities, other null-test positions, and
the unordered set S containing calibration plus the designated test node.

For the candidate j in S, the joint probability is proportional to

`product_{b in B} f*rho(b) * product_{u in unused}(1-rho(u)) * product_{i in S except j}(1-f)*rho(i)`.

All factors except `1/((1-f)*rho(j))` are constant as j varies. The common
`1/(1-f)` cancels, giving the same conditional probabilities proportional to
`1/rho(j)`. Scores fitted only from B, the prior panel, and fixed unlabeled graph
information stay unchanged under this swap. The invariant auxiliary multiset,
randomized-rank uniformity, and pruning bounds therefore apply. With m fixed,
averaging over the conditioning gives the same FDR bound. This needs a separate
corollary/proof paragraph; it is not an automatic invocation of the base theorem
with a newly conditioned uniform-test design. Node-dependent training fractions
would instead require weights proportional to `1/(rho(v)*(1-f(v)))`.

## Claim corrections for the fact sheet and rewrite

- **Novelty:** WCS is existing work. Describe the contribution as its finite-
  population sampling-design justification, the resolution analysis, and graph
  evidence. The same design argument is not specific to graphs. A comprehensive
  novelty search was not completed in this audit.
- **Power:** randomized WCS has higher observed mean power than weighted BH in
  13/18 end-to-end cells, and at least the mean power of deterministic-p-value
  WCS with homogeneous pruning in 18/18. The latter comparison uses a common
  pruning coin and smaller randomized p-values; do not call it superiority over
  every valid method. Weighted BY and stratified rules were not run end-to-end
  in this new experiment. The method labeled simply `WCS` in the figure also
  uses randomized homogeneous pruning; clarify that the p-values are deterministic.
- **Bound:** say the results are consistent with the design-based bound. Some
  sample means exceed it; T-Finance graph/moderate is about 0.1024 at a bound
  about 0.095. That is neither an empirical proof nor by itself a refutation.
- **AUROC:** 0.939-0.994 is the range of cell means. Individual stored fits span
  approximately 0.920-0.998.
- **Severe bias:** avoid a universal 'not useful under severe bias' claim. For
  example, T-Finance graph/severe randomized-WCS power is 0.518. The low-power
  examples are particular scorer/scenario cells; report the full grid.
- **Estimated propensities:** the theorem requires true known positive
  propensities. 'Fixed before roles' alone does not make an arbitrary estimated
  function correct. The plug-in models are empirical sensitivity analyses;
  misspecification yields the reported high FDPs. The cross-fitted logistic
  model uses the variables that generate this synthetic acquisition mechanism;
  avoid treating this as evidence for arbitrary operational selection.
- **Resolution:** the 39,207 checks count nonempty rejection sets, each checked
  via its maximum floor, not 39,207 individual discoveries. For homogeneous
  pruning, the realized necessary bound includes xi; the table's unscaled
  `m*f/alpha` is the deterministic benchmark, not a universal count for
  homogeneous WCS. The corrected first-stage probability uses `(t-b_j)/f_j`
  clipped to [0,1], with `t=alpha*S_j/m`; it is not the final selection probability.
- **Implementation reference:** the 400-case agreement is with the repository's
  own m-by-m equation implementation. The older comparison with the authors'
  public implementation covers only six constant-weight cases. Do not claim
  variable-weight agreement with the authors' code.
- **Training label budget:** the end-to-end experiment still includes a uniform
  prior 10% labeled panel for exposure and training. Only newly acquired labels
  are biased. Say this prominently and count the prior-panel labels in costs.
- **Unsupported scratchpad result:** the 11,314 weighted-BH searches and 45
  hill-climbing starts have no supplied ledger or implementation here. They are
  unverified and should not become a reported reproducible result. The rerun
  finite-design checks found no weighted-BH violation; this is not a proof.
- **Runtime:** the optimized auxiliary-count calculation is O(m) per distinct
  test weight after sorting. Smooth weights can give m distinct weights and
  quadratic total work; do not describe all scenarios as linear time.

## Citation and submission checks

Algorithm 2 is correct for the explicitly cited arXiv v2 outlier-detection
variant: https://arxiv.org/html/2307.09291v2#S6 . The published Biometrika paper is
https://doi.org/10.1093/biomet/asaf066 . The Hennhoefer--Preisach preprint exists
and directly addresses weighted-conformal resolution and variance, so credit
that overlap: https://arxiv.org/abs/2603.23205 . Horvitz--Thompson metadata agree
with the publisher (https://doi.org/10.1080/01621459.1952.10483446); the Fithian--Lei
title, authors and pages agree with the journal volume and author's manuscript
(https://www.stat.berkeley.edu/~wfithian/fdr-dependence.pdf). These checks do not
establish that every claim about related work in a future rewrite is supported.

The official AISTATS FAQ says abstract/authors are due Tuesday September 29,
2026 at 23:59 AoE; full paper and supplement are due Tuesday October 6. The
handoff's weekday labels for September 27-29 are incorrect. More importantly,
major title/abstract changes after September 29 can cause desk rejection, so
the WCS framing must already be in the submitted abstract. Main text is limited
to eight pages; references, AI statement, checklist and appendices are excluded.
AI-assisted authorship is permitted with disclosure and author responsibility;
the supplied instruction to write the abstract without assistance is not a
conference requirement. The present local PDF's body occupies six pages, but
the final rewritten and uploaded version must be checked again.

Official policy sources:
https://virtual.aistats.org/Conferences/2027/SubmissionFAQ
https://virtual.aistats.org/Conferences/2027/CallForPapers

## Limits and next action

This audit did not rerun the entire end-to-end training grid, rerun the original
2.16-million-draw randomized validation, or establish comprehensive novelty.
No Overleaf files, scientific results, or frozen protocols were changed. The
existing two deterministic suites were rerun, and new independent exact and
sample retraining checks were added. The checked evidence supports proceeding
with a focused rewrite; additional expensive experiments are not the immediate
priority. Complete the proof corollary, clustered uncertainty, clean-clone
reproduction, and actual manuscript integration first.
