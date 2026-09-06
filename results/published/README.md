# Published results

Every number that appears in the paper must trace to a CSV in this directory.

`results/logs/` is scratch — gitignored, rerun and overwritten freely. This
directory is the opposite: committed, stable, and the thing a reviewer or a
future co-author can check a claim against.

## How to add a result

1. Run the experiment; it writes to `results/logs/`.
2. Copy the CSV here, keeping the generated filename.
3. Add a row to the index below naming the script, the flags, and the claim it
   backs.
4. Commit the CSV and the index row together.

If a rerun changes a number, do **not** quietly overwrite — the paper may
already cite the old value. Add the new file with a `_v2` suffix and record
both, with a line explaining what changed.

## Index (TMLR draft, verified 2026-09-01)

Per-cell means over five seeds were recomputed from these files with a
stdlib-only script; Tolokers, Weibo and Reddit entries match the draft to the
reported precision.

| CSV | Produced by | Backs |
|---|---|---|
| `calibration_strategy_tolokers_dominant_pygod.csv` | `scripts/calibration_strategy_comparison.py --dataset tolokers --detector dominant_pygod` (weighted arm, 5 seeds) | Tables 3, 5, 6, 8 (Tolokers/pygod rows); §6.2 (ρ 0.899, γ̂ 15.15, FDR 0.382); §6.3 (5 % contamination: discoveries 0/148/93/173/0, FDR 0.018, power 0.031); the random / random_full rows (140 and 162 discoveries at FDR 0.016 / 0.033) not yet in the paper |
| `calibration_strategy_tolokers_gae.csv` | same, `--detector gae` | Tables 3, 5, 6, 7, 8 (Tolokers/gae rows) |
| `calibration_strategy_weibo_dominant_pygod.csv` | same, `--dataset weibo --detector dominant_pygod` | Tables 3, 5, 6, 8 (Weibo/pygod rows) |
| `calibration_strategy_weibo_gae.csv` | same, `--detector gae` | Tables 3, 5, 6, 7, 8 (Weibo/gae rows); §6.5 null p-values 0.205/0.49/0.615/0.325/0.045 |
| `calibration_strategy_reddit_dominant_pygod.csv` | same, `--dataset reddit --detector dominant_pygod` | Tables 3, 5, 6, 8 (Reddit/pygod rows) |
| `calibration_strategy_reddit_gae.csv` | same, `--detector gae` | Tables 3, 5, 6, 7, 8 (Reddit/gae rows); §6.5 null p-values 0.295/0.61/0.99/0.065/0.9 |
| `calibration_strategy_amazon_dominant_pygod.csv` | same, `--dataset amazon` — **pre-weighted-arm run** | **Stale relative to the draft.** Gives γ̂ 13.22, FDR 0.787, power 0.396, n_cal 224, no `weighted` rows. The draft's Amazon numbers (Table 2: n_cal 222, γ̂ 13.15, FDR 0.785, power 0.404; Table 8 weighted 9.55 / 0.844 / 0.196) come from the 2026-08-23 GPU rerun recorded in `theory/joint_discovery_threshold_proposition.md` Part 9, whose CSV was never committed. Commit it as `_v2` before submission. |
| `calibration_strategy_amazon_gae.csv` | same, `--detector gae` — **pre-weighted-arm run** | Stale for the same reason (FDR 0.033 / power 0.030 vs the draft's 0.075 / 0.060; `random_full` 0.059 / 0.116 vs 0.037 / 0.113). |
| `degree_baseline_check.csv` | `scripts/degree_baseline_check.py` | Table 9 and the AUROC heatmap: 20 detector–graph cells, five detectors = `dominant_ours`, `dominant_pygod`, `gae`, `anomalydae`, `ocgnn`; 16/20 within 0.02 AUROC of the best degree baseline |
| `selection_bias_matrix_v2.csv` | `scripts/selection_bias_matrix.py` (5 detectors × 4 graphs × 5 seeds) | Table 4 (Kendall τ between zero-exposure eligibility and degree: Amazon −0.921, Tolokers −0.921, Reddit −0.78 to −1.00 across detectors/seeds, Weibo −0.018) |
| `selection_bias_matrix_weibo_cola.csv` | same, CoLA on Weibo only | not used in the paper (CoLA is not one of the five detectors) |
| `calibration_distribution_check_dominant_pygod.csv` | `scripts/calibration_distribution_check.py --detector dominant_pygod` | Remark 5 illustration (Amazon, n_cal 267, m 5,821, required clearance 218, observed 130/141/130, discoveries 3,420/3,488/3,347) |
| `exposure_degree_confound_check.csv` | `scripts/exposure_degree_confound_check.py` | Weibo exposure→score r = 0.111 with partial r = 0.118 controlling for degree (theory notes; not cited in the draft) |
| `condition_comparison_pygod*.csv`, `severity_sweep.csv`, `multi_seed_sweep.csv`, `synthetic_difficulty_dominant_pygod.csv`, `adadetect_comparison_synthetic.csv`, `degree_sensitivity_sweep_*.csv`, `real_data_experiment_*.csv`, `baseline_comparison*.csv`, `real_data_exposure_diagnostic.csv` | earlier phases (synthetic contamination study, AdaDetect comparison, degree sweeps) | historical; not cited in the TMLR draft |

## Not yet committed

| Expected file | Produced by | Numbers it backs |
|---|---|---|
| `calibration_strategy_amazon_dominant_pygod_v2.csv`, `calibration_strategy_amazon_gae_v2.csv` | `calibration_strategy_comparison.py --dataset amazon` with the weighted arm (GPU rerun of 2026-08-23) | every Amazon entry in Tables 2, 3, 5, 6, 7, 8 |
| figure generators for `degree_distribution_amazon_figure`, `gamma_comparison_figure`, `score_gap_figure`, `degree_baseline_heatmap` | — | Figures 2–5 (currently Overleaf-only) |
