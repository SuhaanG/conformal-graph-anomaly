# Published results

Every number that appears in the paper must trace to a CSV in this directory.

`results/logs/` is scratch — gitignored, rerun and overwritten freely. This
directory is the opposite: committed, stable, and the thing a reviewer or a
future co-author can check a claim against.

**Why this exists.** For a stretch, the entire degree-confound investigation —
the clean-condition FDR inflation, the degree-matched partial fix, the
clustering null — lived only inside git commit messages, because
`results/logs/*` is gitignored and nothing copied the CSVs out. Those numbers
are the core of the paper. Commit messages are not a results store.

## How to add a result

1. Run the experiment; it writes to `results/logs/`.
2. Copy the CSV here, keeping the generated filename.
3. Add a row to the index below naming the script, the flags, and the claim it
   backs.
4. Commit the CSV and the index row together.

If a rerun changes a number, do **not** quietly overwrite — the paper may
already cite the old value. Add the new file with a `_v2` suffix and record
both, with a line explaining what changed.

## Index

| CSV | Produced by | Backs |
|---|---|---|
| _(empty — populate from Colab)_ | | |

## Still to be copied from Colab

These are the runs whose numbers are currently cited in
`theory/joint_discovery_threshold_proposition.md` Part 3 and in commit messages,
with no committed CSV behind them. Highest priority first.

| Expected file | Produced by | Numbers it backs |
|---|---|---|
| `condition_comparison_pygod.csv` | `condition_comparison_pygod.py --n_seeds 20` | **The central finding.** Clean-condition FDR 0.132 ± 0.037, d=0.837, p=0.0007; contaminated and adversarial at/below nominal (d=−0.349, −0.530) |
| `condition_comparison_pygod_degreematched.csv` | same, `--degree_matched_calib` | The partial fix: FDR 0.116 ± 0.034, d=0.449, p=0.0295; matching imperfect in 18/20 seeds |
| `condition_comparison_pygod_degreenorm.csv` | same, `--use_degree_norm` | The failed fix: AUROC 1.0→0.90, power 1.0→0.002, conditional FDR 40–44% |
| `clean_selection_degree_diagnostic.csv` | `clean_selection_degree_diagnostic.py --n_seeds 10` | Degree confound t=−24.959, p<0.0001, 10/10 seeds; score–degree Spearman 0.56; clustering null (0/10 seeds, r=0.004) |
| `severity_sweep_pygod.csv` + `_ranks.csv` | `severity_sweep_pygod_instrumented.py` | "Fails into silence" is dead (AUROC 1.0, power 1.0 at every severity); pooled FDR 0.109 vs 0.10, t=3.142, p=0.0011 |
| `adadetect_comparison_synthetic.csv` | `adadetect_comparison.py --dataset synthetic --n_seeds 20` | **Must be re-run.** The existing file predates commit `9b1eff5` and used the broken detector — its `embed` variant shows the dead-encoder signature (AUROC 0.5, zero discoveries every seed) |
| `selection_bias_matrix.csv` | `scripts/selection_bias_matrix.py` | Part 4 prediction 2 — the falsification test. Not yet run. |
| `exposure_degree_confound_check.csv` | `exposure_degree_confound_check.py` | Weibo exposure→score r=0.111, p=1.4e-23, partial r=0.118 controlling for degree |

`clean_selection_degree_diagnostic.py` currently prints to stdout and writes no
CSV. Either add CSV output or save the console log here as
`clean_selection_degree_diagnostic.log` — its numbers are cited in Part 3 and
need a source.
