# Exploratory pilot-selected batch certification

Specified 2026-09-22 before executing this extension. Prior score precision and
uniform-audit results have already been inspected; this is not a preregistered
independent confirmation. No new training, score tuning, or graph selection.

Evaluate all 120 cached graph/model/seed combinations, budgets 100/250/500,
200 repetitions, q=.10 and delta=.05. Use the existing score-prefix grid.
Random seed namespace: 20260929. Keep all abstentions and all graphs.

1. Draw floor(.2 B) pilot labels uniformly from the largest candidate prefix.
2. For each prefix with h pilot observations and x normal observations, use
   (x+1)/(h+2) only as a planning estimate. Remove all pilot nodes.
3. Consider certificate sample sizes in {25,50,100,200,400,B-floor(.2 B)},
   capped by the remaining budget and strictly smaller than candidate size.
   Predict the normal count by ceiling(sample size times the planning rate).
   Select the prefix/sample-size pair with the largest unreviewed size whose
   predicted exact delta-level bound is <=q. Ties follow increasing prefix
   then sample size. If no pair qualifies, abstain after the pilot.
4. Freeze that pair. Draw a fresh uniform sample without replacement from
   the selected candidate after removing pilot nodes. Calculate the exact
   hypergeometric bound at delta (no multiplicity penalty for this one test).
   Certify remaining nodes only if the bound is <=q; otherwise abstain. Do
   not retry another candidate after a failed certificate.

Validity conditions on the entire pilot: the selected finite population and
certificate sample size are then fixed, so the existing hypergeometric proof
applies. All reviewed nodes are excluded from automatic discoveries. Correct
audit labels are assumed. This is classical sample splitting/learn-then-test,
not a claim to a new confidence-bound principle or guaranteed efficiency gain.

Report actual label use as well as the budget cap. Compare against all existing
uniform-candidate results at the same caps; these are separately randomized
audits on the same frozen scores, not a paired-randomization comparison.
Do not choose a different heuristic after seeing these outcomes.
