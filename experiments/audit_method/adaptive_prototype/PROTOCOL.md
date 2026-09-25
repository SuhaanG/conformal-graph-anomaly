# Adaptive discovery certification: exploratory prototype

2026-09-23. Written before executing this prototype. Existing score and audit
results have already been inspected. This is method development, not independent
confirmation or a claim of established novelty. Do not insert into the manuscript
unless it improves a useful baseline and the novelty/assumptions survive review.

Question: can adaptive allocation of requested labels certify more unreviewed
anomalies than the existing uniform candidate audit, at the same label budget?

Freeze the existing trained HGB attribute and graph-feature scores on Amazon,
Tolokers, and Weibo, all ten training seeds. Training nodes are excluded. Candidate
prefix sizes remain 25,50,100,200,400,800,1600,floor(.2*N), sorted/deduplicated.
No hidden evaluation labels enter candidate construction or allocation.

Prototype arms:
1. Uniform candidate auditing with existing fixed-time hypergeometric bounds.
2. Adaptive score-band auditing.
3. Adaptive score-band plus degree-rank-half auditing.
4. Fixed equal allocation over score bands, using the same confidence sequences.
5. Fixed equal allocation over score-band/degree cells, with those same bounds.

Arms 4 and 5 were added during the first execution, before inspecting its numeric
outcomes, following independent methodological review. They isolate allocation
from changes to the confidence bound. The expanded run supersedes the initial
three-arm run. Existing three-arm seed streams are preserved.
The initial run halted on an incompatible historical confidence set (upper bound
below the subsequently observed normal count). The implementation now marks that
as a bound failure and uses a conservative feasible fallback; it never treats an
empty set as perfect certainty. All conditions are rerun after this repair.

Adaptive arms partition the fixed prefixes into disjoint cells. Each requested
block is sampled uniformly without replacement within its chosen cell. Maintain
a beta-binomial-mixture likelihood-ratio confidence sequence for each cell's
total normal count (Beta(1,1), error delta/H); intersect upper bounds over time.
The certificate sums remaining-normal upper bounds over the complete cells of
each candidate. Pick the largest unreviewed candidate with upper FDP <= .10.
Use delta=.05, so this is a 95% fixed-batch FDP certificate, not .10 FDR control.

Allocation is a heuristic, not part of coverage: after each block of at most 25
labels, estimate each cell's normal rate by (x+.5)/(h+10). For each candidate,
project the remaining budget proportionally to remaining cell sizes, predict
normal observations by rounding that estimate, and compute hypothetical bounds.
Target the largest candidate predicted to certify. If none, target the candidate
with greatest estimated true unreviewed count divided by one plus its positive
certificate deficit. Within that candidate query the cell with the greatest
predicted reduction in its remaining-normal upper bound per label. If gains are
nonpositive, select the cell with the largest remaining size. All ties resolve
by increasing cell/candidate index. Bound calculations always use actual queried
labels; projections never certify anything. Spend the budget (or exhaust union).

Budgets 100,250,500; 50 audit repetitions per graph/model/seed/budget/arm.
Seed namespace 2026092301. Save every outcome, including abstentions, actual
label count, true unreviewed discoveries, FDP, certificate failures and bound
failures. Average repetitions within training seed before summarizing across
seeds. These are reused fixed graphs, not independent graph replications.

Validation before benchmark: brute-force likelihood inversion versus optimized
bound; exhaustive small-population probability of any-time noncoverage; adaptive
allocation with no access to unqueried labels; audited nodes excluded; budgets
and disjointness checked. No GPU or retraining is needed.

Relevant established foundations (must not be presented as new):
- Waudby-Smith and Ramdas, NeurIPS 2020, Confidence sequences for sampling
  without replacement: https://arxiv.org/abs/2006.04347
- Shekhar et al., UAI 2023, Risk-limiting financial audits via weighted sampling
  without replacement: https://arxiv.org/abs/2305.06884
- Xu, Karampatziakis and Mineiro, NeurIPS 2024, Active, anytime-valid risk
  controlling prediction sets: https://arxiv.org/abs/2406.10490

The research opportunity, if any, is a useful allocation policy for certifying
graph anomaly discoveries despite selection bias. Graph stratification alone
does not establish novelty. The method certifies a frozen batch, not future data.
