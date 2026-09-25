# Audit-method feasibility and stronger-detector results

Completed September 15, 2026. All new work ran locally on CPU; no H200 needed.

## Decision

The strongest addition is evidence that calibration selection can hurt an accurate
detector. A finite-batch audit also produces useful certified discoveries on
Amazon, but the proposed degree-stratified audit fails its declared efficiency
criterion. Do not present it as a superior new graph-specific method. Include the
supervised controls in the paper's empirical argument; treat audit certification
as a practical extension with established statistical ingredients, subject to a
careful novelty comparison. These experiments cannot establish acceptance odds.

## Accurate-detector comparison

Two fixed histogram-gradient-boosting models use a uniform random 10% training
panel: attributes alone, or attributes plus neighbor-average attributes and degree.
Training nodes never enter calibration, testing, or certification. There are ten
training seeds per model per graph, five evaluation splits, no score trimming,
and 200 matched calibration draws. Amazon excludes unverified nodes below ID3305.
The extension was declared after the original cached-score experiment, so it is
exploratory. The controls are supervised and must not be described as unsupervised
detectors or as a new architecture.

On Amazon, the attribute model achieves mean AUROC 0.968569 and AP 0.895247.
Oracle neighborhood-filtered calibration gives mean FDP 0.192297 (SD across seed
means 0.087258) and power 0.809061. Matched random calibration gives FDP 0.066940
(SD 0.008867) and power 0.494959. Using the entire eligible normal reference gives
FDP 0.091741 and power 0.783129. These are empirical means, not a distribution-free
FDR theorem. Reference sizes average 143.22 matched and 5275.50 full.

The graph-feature model achieves AUROC 0.970793 and AP 0.898464. Its corresponding
FDP/power pairs are 0.154328/0.801470, 0.065775/0.500298, and 0.091484/0.792792.
The full reference requires labeling the reference panel; it is not a competitor
at equal label cost to the audit method below.

The extra check reveals only non-test deployment labels and uses already-known
training labels to identify anomalous neighbors. With 50% revelation on Amazon,
the attribute model has filtered FDP 0.139792 (seed SD 0.069438), matched FDP
0.069223, and full-reference FDP 0.088558. Powers are 0.769588, 0.622194, and
0.779359. For the graph-feature model, FDPs are 0.117608, 0.072115, and 0.090684.
With 25% revelation, both filtered means remain below 0.10 (0.092886 and
0.085041). Do not omit that condition or claim inflation always exceeds nominal.
Two hundred checks verify that changing every unknown label leaves the filter
unchanged. Benchmark labels construct stratified test splits only.

Tolokers does not replicate the strong-detector discovery regime. AUROC is
0.705775 for attributes and 0.789824 with graph features, but conformal power is
very low. All conditions and seed SDs are in the summary CSVs. The graph is not
dropped because its result is unfavorable.

## Finite-batch certification

Freeze the scorer and candidate prefixes before querying labels. Sample an audit,
invert exact hypergeometric tails to bound each candidate's normal count, apply
simultaneous coverage, subtract observed audited normals, and select the largest
remaining candidate whose upper FDP bound is at most 0.10. Exclude all audited
nodes from automatic discoveries. See THEORY.md and certify.py.

Under the specified sampling design and correct labels, the guarantee is
P_audit(FDP <= 0.10) >= 0.95 for this fixed batch, regardless of graph dependence.
This is not expected FDR <=0.10; the immediate expectation bound is 0.145.
Candidates, strata, and the scorer cannot be tuned using certification labels.
Different scorer results are separate evaluations, not jointly certified model
selection. It is not a guarantee for future graphs or future observations.

For Amazon's attribute model, candidate-uniform auditing with 500 additional
labels returns 216.0185 unreviewed automatic discoveries on average, of which
211.5200 are true anomalies. It makes discoveries in 80.75% of trials; mean FDP
including abstentions is 0.016490, and conditional mean FDP when nonempty is
0.020421. No FDP violations occurred in these 2000 trials (ten fitted scorers,
200 audits each); empirical zero violations is not the proof. Training separately
uses 864 labels. Thus total labeled nodes are 1364, not 500. The batch comprises
7775 deployment nodes. Reviewed anomalies are excluded from automatic yield.

At 250 audit labels it returns 88.9920 automatic discoveries on average; at 100 it
always abstains. The graph-feature scorer gives 207.3730 discoveries at 500 labels.
Score-only stratification returns only 4.8800 (attribute model) and 10.5530
(graph-feature model); degree stratification and whole-graph uniform auditing
return zero. Every Tolokers audit also abstains. The current capped equal allocation
and separate stratum bounds are inefficient here; this does not prove all
graph-stratified designs are inferior.

The classical audit is useful on one graph with sufficiently accurate scores.
The proposed graph-specific improvement is unsupported. Prior work includes
[Learn then Test](https://arxiv.org/abs/2110.01052),
[active risk control](https://arxiv.org/abs/2406.10490),
[finite-population confidence sequences](https://arxiv.org/abs/2006.04347), and
[candidate-generation coverage audits](https://arxiv.org/abs/2607.21480).
The last paper concerns an adjacent coverage/recall objective; it does not by
itself establish equivalence, but exact inversion plus a procedure name is not
enough to establish novelty.

## Negative results retained

All 400 full-reference configurations using the original three cached scorers
and degree-normalized DOMINANT make zero discoveries. The assumption that larger
reference size alone would fix their power was false on the corrected untrimmed
data. The historical trimmed result must not substitute for this experiment.

Degree normalization raises Amazon DOMINANT AUROC to 0.535716 and lowers filtered
mean FDP to 0.164527, but matched power remains approximately zero. On Tolokers,
normalization gives zero discoveries throughout. This is an intervention on the
score, not proof that degree is the unique causal mechanism.

All 192000 original-score certification trials abstain. Among the declared
prefixes, none has the population precision needed for a useful 90%-precision
certificate. This is a diagnostic over the declared grid, not a claim about
every possible threshold.

## Validation and reproducibility

- 40 new fitted models, all score caches saved; 96000 supervised-score audits.
- 192000 original-score audits; repetitions reuse graphs and are not independent
  graph replications.
- 64000 synthetic audit trials, including nonvacuous positive controls; largest
  cell failure rate 0.005, below delta=0.05.
- Exact enumeration: 18600 hypergeometric coverage cases, 2048 selection
  configurations; all checks passed.
- 40400 supervised baseline rows and 160400 partial-label comparison rows.
- Matrix completeness, calibration-size matching, train/deployment separation,
  label masks, and audit budgets checked in summarize.py. Checksums and package
  versions are recorded in RESULTS_VALIDATION.json.

Run from the repository root with the existing .venv-aistats Python:

```powershell
.venv-aistats/Scripts/python.exe paper/audit_method/validate_certificate.py
.venv-aistats/Scripts/python.exe paper/audit_method/cached_baselines.py
.venv-aistats/Scripts/python.exe paper/audit_method/run_certification.py
.venv-aistats/Scripts/python.exe paper/audit_method/strong_scorers.py
.venv-aistats/Scripts/python.exe paper/audit_method/partial_labels.py
.venv-aistats/Scripts/python.exe paper/audit_method/stress_certificate.py
.venv-aistats/Scripts/python.exe paper/audit_method/summarize.py
```

Inputs are the existing graph/score caches under
paper/aistats_followup/h200_results/paper/aistats_followup. Scripts use paths
relative to their own location. strong_scorers.py reuses saved supervised scores;
move those caches to a separate backup directory before intentionally retraining.
No remote jobs or manuscript submissions were launched. The current Overleaf and
submission PDF are unchanged; manuscript_extension.tex contains proposed text for
the next integrated revision, rather than silently replacing the paper's scope.
