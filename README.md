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

## Real-data validation attempt (FraudAmazonDataset) — documented limitation

Applying the validated synthetic pipeline to DGL's FraudAmazonDataset
(11,944 nodes, real organic fraud labels, avg degree 740) surfaced a
DIFFERENT structural failure mode than the one this project set out to
test, worth documenting as a distinct finding rather than a bug:

- Normal-node scores: mean=1502, 99th percentile=1534, but MAX=7596 -- a
  small number of legitimate high-degree "hub" accounts (heavy
  reviewers/sellers) get extreme reconstruction scores purely from unusual
  connectivity, unrelated to fraud.
- Anomaly scores are tightly clustered (1445-1642) with real but modest
  separation from typical normal nodes.
- 74% of true anomalies exceed the top-2000 normal scores (real signal is
  detectable), but 0% exceed the single highest-scoring normal node --
  meaning if even one extreme hub lands in the calibration set, it single-
  handedly blocks any discovery, since conformal p-values require beating
  nearly the entire calibration set.

**Conclusion:** the graph-structured contamination mechanism validated on
synthetic data (H4: fail-safe property survives clustered/adversarial
anomaly contamination) is a DIFFERENT phenomenon from hub-node score
inflation. Real-world deployment on dense, hub-heavy graphs would need a
degree-robust scoring scheme (e.g. degree-normalized reconstruction error,
or explicit hub exclusion/down-weighting in calibration) before the
certification claim can be extended to this class of graph. This is
documented here as a clearly scoped limitation and future-work direction,
not resolved within this project's compute budget.

## UPDATE: Real-data H4 confirmation (after fixing trimming/exchangeability bug)

The hub-node limitation above was resolved via degree-normalized scoring
plus symmetric trimming of extreme-score outlier nodes from BOTH
calibration eligibility and the test set (an earlier attempt that trimmed
calibration only broke exchangeability and manufactured a spurious
FDR~0.51 result -- documented as a cautionary methodological note).

After the fix, on FraudAmazonDataset (15 seeds/condition):
- contaminated: realized_fdr=0.048+/-0.044 (nominal=0.10), power=0.076,
  not significant (p=0.9997)
- adversarial: realized_fdr=0.019+/-0.012 (nominal=0.10), power=0.073,
  not significant (p=1.0000)
- clean: 0 discoveries throughout (known data limitation -- only ~267
  true zero-exposure nodes exist on this dense graph, low power by
  construction, not a validity concern)

**H4 REPLICATES ON REAL DATA.** The fail-safe property survives both
average-case and worst-case adversarial contamination on a real, organic
fraud-detection graph, with genuine detection power (not a power collapse
like the earlier buggy attempts). This directly answers the "unrealistic
anomaly injection" critique from the literature audit -- the certification
claim now holds on synthetic AND real data.

Pilot status: core empirical claim validated on both synthetic (3
conditions + severity sweep) and real (FraudAmazonDataset) data. Ready to
move toward: (1) additional real datasets for robustness (YelpChi,
Elliptic, Tolokers), (2) formal theoretical characterization (the
theorems this was meant to motivate), (3) baseline comparisons against
AdaDetect/CRC-SGAD, (4) writeup.