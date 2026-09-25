# Audit-method feasibility protocol

Specified before inspecting these experiment outputs, 15 September 2026.
This is a research prototype. Existing manuscript results remain unchanged.

## Cached-score baselines

Use every saved H200 score: Amazon and Tolokers, DOMINANT/GAE/Isolation Forest,
10 training seeds, the same five primary 25% stratified test splits. Corrected
Amazon label mask; no trimming. Add full eligible normal reference calibration
(all non-test verified normals) to the existing matched-size comparison.
Evaluate a fixed DOMINANT transformation s/log(1+degree+1e-8), exactly the existing
repository rule, with raw-score comparisons on identical nodes. Keep all runs.
For normalized scores compare entire filtered pool, matched random (200 draws),
and full reference. No transform parameter or direction is selected using FDP.

## New finite-batch certification prototype

Use the entire verified labeled benchmark graph as a fixed deployment batch:
8,639 Amazon nodes and 11,758 Tolokers nodes. Scores are frozen, audit labels hidden.
This differs from the baseline's reference/test split and must be reported separately.
Scorers: the three cached models and fixed normalized DOMINANT, all 10 seeds.
No learning from certification labels. Stable node-ID tie breaking.

Candidate sets are top-score prefixes of sizes 25,50,100,200,400,800,1600, and
floor(0.20*N), deduplicated and sorted. Labels do not choose the candidate grid.
Audit budgets: 100,250,500 node labels, capped only by the candidate-union size.
200 independent audit repetitions per scorer/seed/budget/design.
Primary designs: uniform sampling from the candidate union; equal-budget
score-band stratification; equal-budget score-band-by-degree-half stratification.
Each score band is the difference between successive candidate prefixes. Split
each band by degree rank with node-ID tie breaking. Budget allocation is fixed
from stratum sizes only, via capped equal allocation, before seeing any labels.
Uniform sampling without replacement within each stratum. Also retain diagnostic
uniform auditing from the entire graph (same label budget).

Uniform designs invert hypergeometric tails for every prefix conditional on the
number audited in it, with Bonferroni delta/K. Stratified designs invert the null
count in each disjoint cell, with Bonferroni delta/H; sum cell bounds for prefixes.
Subtract observed normal counts, exclude ALL audited nodes from the automatic
discovery set, and certify its remaining FDP <= q=0.10 at confidence 1-delta=0.95.
Select the largest remaining certified set; abstain if no nonempty set qualifies.
Do not report reviewed anomalies as automated discoveries. The bounds cover all
candidates simultaneously, making this data-dependent selection valid. Report
FDP, automated true discoveries, fraction of all graph anomalies automatically
found, unreviewed-anomaly recall, label cost, abstention, and certificate failures.
This is high-probability fixed-batch FDP certification, not FDR control at 0.10.
Expected FDP is bounded by q+(1-q)*delta=0.145; stricter FDR needs adjusted q/delta.

The full-reference baseline assumes labels identifying every reference normal;
it is not an equal-label-budget competitor to the new audit. Report that cost.
The uniform certificate is the essential classical finite-population baseline.
An all-label oracle reports the achievable prefix precision/recall envelope only
as a diagnostic ceiling, never as a procedure's input or selected setting.

## Validation and continuation rules

Prove hypergeometric inversion coverage and simultaneous selection validity.
Exhaustively enumerate small populations and audit subsets, including ties,
empty strata, full audits, all-normal/all-anomaly populations, and adaptive
selection among prefixes. Check allocation, audit membership, output exclusion,
and invariance to unaudited labels. Retain all outcomes, including no discoveries.
No reruns selecting favorable seeds, budgets, score directions, or graphs.
If graph-informed auditing does not outperform simpler auditing, report it and
do not present it as a superior new method. The statistical ingredients are
established; novelty remains unproven.

## Closest prior work

- Learn then Test: https://arxiv.org/abs/2110.01052
- Active, anytime-valid risk controlling prediction sets: https://arxiv.org/abs/2406.10490
- Confidence sequences for sampling without replacement: https://arxiv.org/abs/2006.04347
- Finite-Sample Coverage Audits for High-Recall Candidate Generation:
  https://arxiv.org/abs/2607.21480 (2026 preprint; adjacent exact certification).

Any extension following inspection of the feasibility outputs is exploratory and
must be recorded explicitly, with fresh random streams for confirmation.
