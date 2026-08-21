# Colab runbook

Ordered by value. **Stop anywhere** — each cell produces an independently
reportable result, and cell 3 is the one that decides what the paper is.

Everything writes to `results/logs/`. Anything you want to keep must be copied
into `results/published/` and committed — see that directory's README for why.

---

## Setup

```python
import os
if os.path.exists('conformal-graph-anomaly'):
    os.chdir('conformal-graph-anomaly'); !git pull origin main
else:
    !git clone https://github.com/SuhaanG/conformal-graph-anomaly.git
    os.chdir('conformal-graph-anomaly')
```

```python
!pip install -q torch_geometric pygod networkx scipy
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Confirm the new files pulled in:

```python
!ls src/selection_bias.py scripts/selection_bias_matrix.py tests/test_selection_bias.py results/published/README.md
```

---

## Cell 1 — Gate: the estimator must pass its own tests

Run this first. If it fails, nothing downstream means anything.

```python
!python -m pytest tests/ -q
```

Expect **17 passed** — 4 from `test_normalize_equivalence.py`, 13 from the new
`test_selection_bias.py`.

The tests that matter are `test_valid_procedure_is_not_flagged_as_broken` and
`test_planted_calibration_shift_is_detected` — a genuinely exchangeable trial
must not be flagged, and a real tilt must be caught. An earlier version of the
estimator failed the first one, reporting p = 3e-07 on data that was
exchangeable by construction.

---

## Cell 2 — Hard gate: the refactor must not have changed behavior

`--detector` was threaded through `condition_comparison_pygod.py`. The default
path must still reproduce the clean-condition FDR of **0.132**.

```python
!python scripts/condition_comparison_pygod.py --n_seeds 20 --alpha 0.10 --device cuda
```

Check the printed clean-condition summary against `0.132 +/- 0.037`. **If it
moved, stop and say so** — it means the refactor changed the computation, and
every number built on top of it is suspect.

Also copy the CSV out, since this is the run the whole paper rests on:

```python
!cp results/logs/condition_comparison_pygod.csv results/published/
```

---

## Cell 3 — THE DECISION POINT

This is Part 4's falsification test. It answers whether the paper has a theorem.

```python
!python scripts/selection_bias_matrix.py --n_seeds 5 --device cuda
```

5 detectors × 4 real datasets × 5 seeds = 100 trials. Budget a few hours.

It prints a verdict at the end, one of three:

- **consistent with Part 4** — γ tracks score–degree dependence across cells.
  The mechanism holds, Part 4 becomes Theorem 2, and IEEE TNSE is a real target.
- **MIXED** — some statistics track, others don't. Report honestly; which ones
  disagree is itself informative.
- **NOT SUPPORTED** — γ does not track. **Part 4's mechanism as stated is
  wrong.** The selection effect may still be real, but the *degree* explanation
  for it failed. Revise the theory doc; do not defend it.

All three are publishable outcomes. Only pretending is not.

```python
!cp results/logs/selection_bias_matrix.csv results/published/
```

If it's too slow, cut scope in this order — seeds first, then datasets:
`--n_seeds 3`, then `--datasets amazon reddit weibo`. Do **not** cut detectors
below 4; the correlation needs cells, and the script refuses below n=5.

### The control worth running if cell 3 confirms

If the mechanism is real, turning degree normalization ON should lower Spearman
*and* γ together. That's a cheap causal check:

```python
!python scripts/selection_bias_matrix.py --n_seeds 3 --device cuda --degree_norm on --out results/logs/selection_bias_matrix_degreenorm.csv
```

---

## Cell 4 — Re-run AdaDetect (the existing CSV is invalid)

The synthetic AdaDetect results predate commit `9b1eff5` and used
`train_dominant`, the broken detector. Its `embed` variant shows the
dead-encoder signature — AUROC 0.5, `clf_partition_auc` 0.5, zero discoveries
in every seed. Those numbers cannot be reported.

```python
!python scripts/adadetect_comparison.py --dataset synthetic --n_seeds 20 --alpha 0.10 --device cuda
!cp results/logs/adadetect_comparison_synthetic.csv results/published/
```

Sanity check: `adadetect_embed` should now produce a `clf_partition_auc`
meaningfully above 0.5 and a nonzero discovery count. If it still reads exactly
0.5 with zero discoveries, the detector fix did not take effect — flag it.

---

## Cell 5 — Backfill the missing CSVs

These runs are cited in `theory/joint_discovery_threshold_proposition.md` Part 3
but have no committed file behind them.

```python
!python scripts/condition_comparison_pygod.py --n_seeds 20 --device cuda --degree_matched_calib
!python scripts/condition_comparison_pygod.py --n_seeds 20 --device cuda --use_degree_norm
!python scripts/severity_sweep_pygod_instrumented.py --n_seeds 20 --device cuda
!cp results/logs/condition_comparison_pygod_degree*.csv results/logs/severity_sweep_pygod*.csv results/published/
```

`clean_selection_degree_diagnostic.py` prints to stdout and writes no CSV, so
capture the log:

```python
!python scripts/clean_selection_degree_diagnostic.py --n_seeds 10 --device cuda 2>&1 | tee results/published/clean_selection_degree_diagnostic.log
```

---

## Cell 6 — Lower priority

Rank-logged real-data runs, closing Part 2's outstanding numerical check:

```python
for ds in ["amazon", "reddit", "tolokers", "weibo"]:
    !python scripts/real_data_experiment.py --dataset {ds} --n_seeds 15 --alpha 0.10 --device cuda --detector dominant_pygod --log_ranks
!python scripts/verify_extended_proposition.py --all
```

Note `--detector` now defaults to `dominant_pygod` rather than the broken
`dominant_ours`. Passing it explicitly above is belt-and-braces.

Yelp needs `--use_sparse_prop` and runs ~65 min per call — only if there's time.

---

## When you're done

```python
!git add results/published/ && git commit -m "Add published result CSVs" && git push origin main
```

Then update the index table in `results/published/README.md` so each CSV names
the claim it backs, and say which of the three cell-3 verdicts came back. That
verdict determines whether the next step is writing Theorem 2 or revising
Part 4.
