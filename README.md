# Calibration selection and conformal FDR control in graph anomaly detection

Research code for *Calibration Selection Can Inflate False Discoveries on Graphs*.
The study compares exposure-based selection among verified normal calibration
nodes with size-matched random references. The exposure rule is a controlled
diagnostic intervention, not an established industry practice. Selection can
distort reference ranks; useful discoveries also depend on detector quality,
reference size, and the label-acquisition design. Contamination is a separate
question and is not claimed to be harmless in general.

The current revision and its reproduction guide are under `paper/aistats_revision/`;
supervised controls and observed-exposure sensitivity experiments are under
`paper/audit_method/`. The `results/published/` directory retains historical
experiments. Some historical rows used an invalid Amazon label convention or
other superseded protocols and are not evidence for the current conclusions.
The script/table mapping below describes those historical experiments.

The newer experiment source, protocols, compact manifests, and summary diagnostics
are also collected under `experiments/` for collaborators. Large downloaded datasets,
model score arrays, and generated trial tables remain local and are described by the
provenance files in that bundle.

## Layout

```
src/            conformal p-values and BH (conformal_fdr.py), detectors (detectors.py),
                anti-conservativeness estimator and simulated null (selection_bias.py),
                inverse-propensity weighted conformal p-values (weighted_conformal.py)
scripts/        runnable experiments; the ones behind the paper are listed below
results/
  published/    committed CSVs cited in the paper, plus an index (README.md)
  logs/         scratch output of reruns (gitignored)
theory/         derivations behind Sections 3 and 4
tests/          unit tests for the estimators
```

## Scripts behind each part of the paper

| Paper | Script | Output in `results/published/` |
|---|---|---|
| Tables 2, 3, 5, 6, 8 (clean, weighted, random, contaminated calibration) | `scripts/calibration_strategy_comparison.py --dataset <amazon\|tolokers\|weibo\|reddit> --detector <dominant_pygod\|gae>` | `calibration_strategy_<dataset>_<detector>.csv` |
| Table 4 (eligibility–degree Kendall correlation) and the 20-cell detector matrix | `scripts/selection_bias_matrix.py` | `selection_bias_matrix_v2.csv` |
| Table 7 and the degree-baseline heatmap | `scripts/degree_baseline_check.py` | `degree_baseline_check.csv` |
| Worked example in Appendix A (clearance counts) | `scripts/calibration_distribution_check.py --detector dominant_pygod` | `calibration_distribution_check_dominant_pygod.csv` |
| Recompute every per-cell mean reported in the paper | `scripts/verify_tables.py` | prints to stdout |

Each cell uses five seeds (0–4). Within a seed the test set is identical across
calibration strategies, and the clean and weighted strategies use the identical
calibration draw, so differences between strategies are not confounded by
sampling variation.

## Setup

```
pip install -r requirements.txt
pip install torch torch_geometric pygod dgl
```

Datasets (Amazon, Tolokers, Weibo, Reddit) are downloaded at first use by the
PyTorch Geometric and DGL loaders in the scripts; see `RUNBOOK_COLAB.md` for a
GPU walkthrough. All detectors use hidden dimension 64 and no dropout; encoder
depth and the scoring convention for each detector are set in
`src/detectors.py`.

## Definitions used throughout

- Conformal p-value: `(#{calibration scores ≥ test score} + 1) / (n_cal + 1)`.
- Anti-conservativeness factor: the fraction of normal test p-values at or
  below `t`, divided by `t`, reported at `t_lo = 10 / (n_cal + 1)`.
- Selection propensity for the weighted arm: the empirical probability of zero
  anomalous exposure within quantile bins of degree, estimated over the normal
  calibration pool; calibration weights are its reciprocal, test weight is one.
