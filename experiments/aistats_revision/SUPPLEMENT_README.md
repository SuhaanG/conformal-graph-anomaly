# Calibration selection: reproducibility materials

The current manuscript studies matched reference selection, a graded observed-exposure removal experiment, exact conditional rank diagnostics, and an established fixed-batch certificate. The main comparisons use no score trimming. Amazon calibration/evaluation excludes unlabeled node IDs 0--3304.

## Current sources and build

The root paper source is `paper/aistats_revision/aistats.tex`. Compile in that directory with pdflatex, bibtex, pdflatex, pdflatex. The source package uses the official unmodified 2027 style. Generated figures are ordinary vector PDFs; all source inputs are included. The final editorial display is regenerated with `python paper/aistats_revision/build_editorial_display.py`, after other numerical display generators if those are rerun.

Regenerate the new figures and result text with:

```
python paper/aistats_revision/build_focus_results.py
```

This reads stored results and does not retrain models or overwrite the manuscript narrative. Source editing utilities and older manuscript snapshots are not required for reproduction.

## Observed-exposure removal experiment

Read `paper/audit_method/DOSE_PROTOCOL.md` and `WEIBO_CONTROL_PROTOCOL.md`. The Weibo extension was specified after inspecting the Amazon/Tolokers sweep and before fitting Weibo models. All three graphs, both scorers, ten seeds, five splits, three label budgets, nine removal fractions, two rules, and fifty repetitions are retained (810,000 evaluations).

```
python paper/audit_method/validate_focus.py
python paper/audit_method/selection_dose.py
```

The first command independently recomputes split/seed summaries from the complete compressed trial ledger and exhaustively checks 11,340 finite-design assignments. The second reruns the sensitivity experiment on frozen scores; it requires NumPy, pandas, and SciPy, not a GPU. It checks train/test separation, normal references, equal reference sizes, zero-removal equality, unknown-label invariance, and BH implementation agreement. Protocols state the independent lexicographic score tie convention. Prior zero-exposure comparisons use conservative nonrandomized ties and are identified separately.

Outputs are `selection_dose_trials.csv.gz`, split/seed/aggregate CSVs, and a hash manifest under `paper/audit_method/`. Averages include zero-discovery runs. Seed SDs describe training variation on one graph and are not confidence intervals over new graphs. Weibo model scores, processed graph data, training-panel identities, and diagnostics are included. `weibo_control.py` reproduces training using the same fixed classifier settings; it downloads the official PyGOD input if uncached. The raw file has 347 anomalous labels, despite a conflicting README count of 868; `WEIBO_DATA_PROVENANCE.json` documents the inspected file.

## Original current graph comparisons

`paper/aistats_followup/h200_results/paper/aistats_followup/` contains the ten-seed untrimmed GPU scores and 2,325,000 calibration evaluations. `build_followup_tables.py` validates job manifests and regenerates summary tables. `audit_primary_ranks.py` reconstructs the exact primary-rank probabilities and their 300,000-draw Monte Carlo check. Raw public Amazon and Tolokers inputs and source hashes are included. GPU and CPU training outputs are not pooled.

`paper/audit_method/strong_scorers.py`, `partial_labels.py`, `summarize.py`, and `cached_baselines.py` (where present) support the original accurate-score, partial-label, and full-reference comparisons. Fixed protocols, cached predictions, trial CSVs, and input/output manifests are included. Training labels, reference-panel labels, and certification labels are distinct budgets.

## Supporting simulations and certificate

`paper/aistats_followup/independent_graphs.py` reproduces the 2,500 independent graphs. `paper/aistats_revision/validate_revision.py` reproduces the independent score simulations; `graph_rank_audit.py` reproduces the five-seed Weibo GAE conditional audit from saved scores. The exact audit is conditional on test identities; the random-partition BH guarantee averages over their assignment. They are different statements.

`paper/aistats_followup/wcs_experiment.py` runs the independent-observation WCS comparison. `fetch_wcs_reference.py` retrieves the versioned third-party source and checks its hash; that third-party code is not redistributed. The implementation follows the specified version's Algorithm 2.

`paper/audit_method/certify.py`, `validate_certificate.py`, and `stress_certificate.py` implement and validate the classical finite-population certificate. Fixed candidate families, correct labels, and random auditing are essential. The certificate controls a fixed batch's FDP with high probability, not future-batch FDR at the same nominal value. The pilot adaptation experiment is excluded from the submission.

## Archived inputs

Some retained `results/published/` CSVs and older audit inputs use superseded designs. They are supplied only where historical validation scripts refer to them. They do not replace the current untrimmed verified-label results. Invalid historical Amazon tables and the obsolete degree-ranking matrix are absent from the manuscript. The current source, explicit protocols, and result manifests identify the evidence used in the submission.

## Allocation and acquisition follow-up (24 September)

Read ALLOCATION_ACQUISITION_PROTOCOL.md under paper/audit_method before interpreting the additions. It was frozen after equal-alpha outcomes, before testing these variants. This is explicitly an exploratory follow-up. Pilot-panel adaptation remains excluded.

```
python paper/audit_method/allocation_acquisition.py
python paper/audit_method/validate_allocation_additional.py
python paper/aistats_revision/build_allocation_results.py
```

These use cached scores on Amazon, T-Finance and GADBench Weibo and require no GPU. The 135,000-row ledger includes all allocation/acquisition variants. Existing arms reproduce 5,400 stored per-split outcomes. Independent reaggregation verifies seed summaries. Conditional block enumeration includes nonzero false discoveries; unequal-propensity enumeration checks weighted marginal ranks. Pooled stratified BH requires the stated product role law. Weighted BH is descriptive; weighted BY uses marginal validity. Uniform acquisition in the biased scenario is a counterfactual, with actual label costs recorded. An editorial pass moved the complete nine-panel display to the appendix and displays the moderate-acquisition comparison for all three graphs in the main paper. This presentation choice was made after the results, changes no protocol or outcomes, and retains every setting in the appendix. The moderate-removal six-row table and its pointwise paired intervals are regenerated from stored seed outcomes by `build_editorial_display.py`.

## Additional graphs and source provenance

TFINANCE_PROTOCOL.md and STRATIFIED_PROTOCOL.md were fixed before their corresponding outcomes. GADBENCH_DATA_PROVENANCE.json records official archive members, field counts, hashes and conversion. T-Finance and GADBench Weibo add 270,000 removal evaluations each; they are separate from the original 810,000. PyGOD and GADBench Weibo are alternative releases of one graph, not independent graph replications. Raw features and symmetrized adjacency match, but distributed labels contain 347 versus 868 anomalies. Raw feature processing also differs between the older processed cache and the GADBench extension. Never pool the two releases.

```
python paper/audit_method/benchmark_extension.py tfinance
python paper/audit_method/benchmark_extension.py weibo_gadbench
python paper/audit_method/stratified_calibration.py --datasets amazon tolokers weibo
python paper/audit_method/stratified_calibration.py --datasets tfinance weibo_gadbench
python paper/aistats_revision/build_stratified_results.py
python paper/aistats_revision/build_extended_results.py
```

The processed sparse graph inputs are packaged, so routine reproduction does not require DGL or the >1GB raw T-Finance binary. acquire_gadbench.py provides optional official-archive acquisition and native-DGL conversion; provenance describes the original runtime. Complete dose curves and pointwise paired intervals are provided. Table/figure generators recreate numerical content; edited prose is maintained in the included source files. Author verification of proofs and scientific interpretations is required before submission.
