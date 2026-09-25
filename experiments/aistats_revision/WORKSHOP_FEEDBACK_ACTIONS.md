# Workshop feedback applied to the AISTATS revision

The supplied record contains Accept and Borderline Accept reviews, not a final
workshop decision. These ratings cannot be converted into an AISTATS acceptance
probability. The reviewers assessed an older version, so numerical objections
are addressed with the corrected current experiments rather than by restoring
superseded claims.

- **Practical strategy (NJFa):** Section 8 now gives a three-step random-label
  batch certificate, its assumptions and exact error target, and the successful
  Amazon example. The abstract and introduction make this operational response
  visible. The exploratory pilot variant is excluded from the AISTATS manuscript; it is not claimed
  to be a novel confidence-bound principle or uniformly better method.
- **Concrete scenario (NJFa):** The introduction begins with an analyst acquiring
  verified normal accounts for a fraud-screening reference.
- **PRDS and pooled SD (NJFa):** The current text explains marginal rank validity
  versus joint p-value dependence without relying on the undefined PRDS acronym.
  Appendix B defines the weighted pooled sample variance used in the archived
  standardized gap and states what the gap cannot establish.
- **Other paradigms (5VXH):** The supervised controls already provide direct
  evidence beyond unsupervised reconstruction. Appendix E now distinguishes
  this evidence from the untested implications for semi-supervised, PU,
  generalist, and cross-domain scoring. Unlabeled does not mean verified normal.
- **Original citations (5VXH):** DOMINANT and Isolation Forest are cited at use;
  the GAE ancestry is cited with an explicit distinction between this attribute
  decoder and the original variational adjacency-reconstruction model.
- **Weibo mechanism (5VXH):** Section 5 explains why weak degree correlation or
  a small mean-degree shift does not exclude a null-score distribution shift.
  The corrected untrimmed and trimming-sensitive Weibo results remain separate
  from the historical .279 result. Degree is one sufficient route, not the
  asserted sole cause.
- **Dataset/scorer breadth (5VXH):** Sections 4 and Appendix B explain the
  mechanism-oriented subset and compute constraints. The manuscript retains
  the unfavorable Tolokers results and identifies untested generalization.

## New work completed

`paper/audit_method/PILOT_PROTOCOL.md` specifies a 20%-budget pilot and one fresh
hypergeometric certificate, with no retry after failure. The run includes all
120 cached graph/model/seed combinations and 72,000 audits. Exact enumeration
checks all 256 binary labelings of a small graph under an adaptive audit design;
2,000 synthetic full-procedure trials add nonvacuous controls. See the manifest,
validation JSON, raw trial CSV, and local exploratory table.

At a cap of 500 additional labels, Amazon's attribute scorer returns 134.1 true
automatic discoveries with 173.9 labels used on average. Candidate-uniform
auditing returns 211.5 true discoveries with 500 labels. Pilot nonempty frequency
is 55.6% versus 80.75%. Thus the prototype saves labels but reduces yield at the
same cap; it is not a matched-cost superiority result. All original-score and
Tolokers pilot cells abstain. No new H200 training was needed.

## Remaining scientific judgment

The new writing answers a concrete reviewer objection but does not transform
the study into a new general graph FDR method. The finite-batch guarantee assumes
correct randomly audited labels and does not certify future graphs. At FDP
target .10 and failure probability .05, the generic expected-FDP bound is .145.
Authors still need to review the proofs and claims; no statement that they have
already verified every AI-assisted proof was added. Workshop scores are useful
feedback, not a basis for a numerical acceptance promise.

User scope clarification: keep only changes relevant to AISTATS. The pilot method and its extra table are therefore excluded from the submission draft; their code and results remain a separate research record.
