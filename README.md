# conformal-graph-anomaly

Testing whether finite-sample false discovery rate (FDR) control for conformal
anomaly detection survives clustered, propagating calibration contamination on
attributed graphs -- and if not, what calibration design restores it.

Target venue: IEEE Transactions on Cybernetics.

## Status

**Steps 1-6 complete: core empirical question answered.**

Result (50-150 seeds per condition, n_nodes=15000, DOMINANT detector,
conformal p-values + BH at alpha=0.10):

- **H4 supported, not H1.** Realized FDR stays below nominal across clean,
  contaminated (random), and adversarial (worst-case) calibration
  conditions, and across a 25x escalation in contamination severity
  (p_an from 0.002 to 0.05). No condition or severity level showed
  statistically significant FDR violation (all one-sided p > 0.88).
- **Important nuance:** at the highest severity tested (p_an=0.05, mean
  calibration exposure=0.42), the system produced ZERO discoveries across
  all 20 seeds. This is not confirmation of validity under stress -- it's
  a power collapse. The honest, defensible claim is that validity holds
  WITH meaningful detection power through p_an=0.02 (12.5x baseline), and
  beyond that the system fails safe into silence rather than into false
  discoveries, which is itself a genuine, positive, publishable property.
- Raw results: `results/logs/multi_seed_sweep.csv`, `results/logs/severity_sweep.csv`.

**Framing for the paper going forward:** this is a certification result
("Contamination-Robust Conformal Anomaly Discovery on Attributed Graphs"),
not a break-and-repair result. The contribution is proving the fail-safe
property survives graph-structured contamination (random and adversarial)
up to a characterized severity boundary, plus the graceful-degradation-to-
silence property beyond that boundary.

**Next milestone: move from synthetic SBM graphs to real organic-anomaly
datasets** (YelpChi, Elliptic, T-Finance from GADBench) to confirm the
certification result isn't an artifact of the synthetic contamination
mechanism. This is the step that makes the paper's empirical section
credible to reviewers, since synthetic-only validation invites the
"unrealistic anomaly injection" critique documented in the literature audit.

## Repo structure

```
src/            core modules (graph generator, detector, conformal pipeline)
scripts/        standalone runnable checks and experiments
notebooks/       exploratory analysis
data/           raw/processed synthetic and real datasets (gitignored)
results/        figures and logs (gitignored except structure)
tests/          unit tests
```

## Key files

- `src/graph_gen.py` -- synthetic contaminated attributed graph generator
- `src/detector.py` -- DOMINANT-style unsupervised GNN anomaly detector
- `src/conformal_fdr.py` -- conformal p-values, BH procedure, clean/contaminated/adversarial trial logic
- `scripts/check_graph_gen.py` -- sanity checks for the graph generator
- `scripts/sweep_contamination_contrast.py` -- Step 2 parameter sweep (locked config: n_nodes=3000 for detector calibration work, n_nodes=15000 for FDR experiments)
- `scripts/multi_seed_sweep.py` -- main 3-condition (clean/contaminated/adversarial) FDR experiment with statistical tests
- `scripts/severity_sweep.py` -- contamination severity escalation under the adversarial condition

## Setup

```bash
pip install -r requirements.txt
pip install torch scikit-learn
python scripts/check_graph_gen.py
```