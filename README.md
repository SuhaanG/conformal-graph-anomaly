# Calibration selection and conformal FDR control in graph anomaly detection

Code and data for the paper *Calibration Selection, Not Contamination, Breaks
Conformal FDR Control in Graph Anomaly Detection* (under review). The paper asks
when conformal p-values combined with Benjamini–Hochberg keep their false
discovery rate guarantee in unsupervised graph anomaly detection, and shows that
the standard precaution of restricting calibration to normal nodes with no
anomalous neighbors is itself a covariate selection filter that can break the
guarantee, while true contamination of the calibration set does not.

Every number in the paper traces to a CSV in `results/published/`; the index in
`results/published/README.md` maps each table to the file that produced it.

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
