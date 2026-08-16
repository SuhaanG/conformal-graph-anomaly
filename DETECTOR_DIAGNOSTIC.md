# Diagnostic: the contamination mechanism is not reaching the anomaly scores

**Status:** confirmed empirically on synthetic data, 2026-08-16. Not yet
checked on the real datasets (dgl / torch-geometric / pygod are not installed
on the dev box -- that check must run on the GPU machine).

**Why this matters:** the paper's central causal claim is that GNN message
passing propagates anomalous influence into the scores of neighboring normal
nodes, contaminating the calibration set. That claim is what makes the
clean / contaminated / adversarial comparison meaningful. On synthetic data,
that pathway is severed, and the three conditions are measuring essentially
the same thing.

This was found while smoke-testing an AdaDetect baseline: its `embed` variant
returned power = 1.0000 with FDR = 0.0000, which was too clean to accept.

---

## 1. The finding

Correlation between a normal node's exposure (fraction of its neighbors that
are anomalous) and its anomaly score, n=3000, 3 seeds, 100 epochs:

| p_an  | mean exposure | Pearson r | p-value | Spearman |
|-------|---------------|-----------|---------|----------|
| 0.002 | 0.021         | +0.0154   | 0.434   | +0.0146  |
| 0.010 | 0.096         | +0.0027   | 0.354   | +0.0004  |
| 0.050 | **0.342**     | **-0.0064** | 0.574 | -0.0029  |

Every correlation is |r| < 0.016 and none is significant. At the highest
severity -- where a typical calibration node has 34% anomalous neighbors --
the correlation is slightly *negative*, the opposite of the predicted
direction.

**Consequence.** The "adversarial" condition selects the most anomaly-exposed
normal nodes, but their scores are statistically indistinguishable from
randomly selected normal nodes. The reported result (clean ~ contaminated ~
adversarial, no significant difference) follows trivially: there is no score
contamination for FDR control to be robust to.

---

## 2. Mechanism, traced end to end

### 2a. The generator never propagates features

`ContaminatedGraphGenerator.generate()` calls `_generate_features(labels)`,
which draws each node's features from a class-conditional Gaussian using that
node's **own** label. Neighbors are not consulted.

`propagate_contamination(hops, mix_weight)` implements the intended mechanism
and its docstring says so explicitly -- *"This is the literal mechanism H1
predicts corrupts normal nodes' calibration scores when anomalies cluster."*
**No experiment script calls it.** It is dead code.

### 2b. The GCN encoder's final layer is dead

Stepping through `DOMINANT.encode` (n=3000, seed 0, 100 epochs):

```
layer0 PRE-relu  : min=-1.7255 max= 2.1986  frac_pos=0.4760   <- healthy
layer1 PRE-relu  : min=-2.5554 max=-0.0080  frac_pos=0.0000   <- all negative
layer1 POST-relu : frac_zero=1.0000
final Z          : all_zero=True
```

Systematic, not seed- or size-specific -- 6/6 configurations, including
n=15000, the size used in every synthetic experiment in the paper:

| n      | seed | Z all-zero | Z std    | AUROC  |
|--------|------|------------|----------|--------|
| 3000   | 0    | True       | 0.000000 | 0.9100 |
| 3000   | 1    | True       | 0.000000 | 0.9053 |
| 3000   | 2    | True       | 0.000000 | 0.9168 |
| 15000  | 0    | True       | 0.000000 | 0.9113 |
| 15000  | 1    | True       | 0.000000 | 0.9089 |
| 15000  | 2    | True       | 0.000000 | 0.9094 |

Also confirmed at n=5000 at both 10 and 100 epochs, so it is not a
matter of undertraining. Note that AUROC is flat at ~0.91 across a 5x
change in graph size -- consistent with a feature-magnitude detector,
which is indifferent to graph size, and hard to reconcile with a
functioning GNN.

**Root cause.** ReLU on the *final* encoder layer forces `Z >= 0` elementwise,
so `z_i . z_j >= 0` and `sigmoid(z_i . z_j) >= 0.5` for every pair. The graph
is ~99.7% non-edges, all pulling that value toward 0. The nearest reachable
point is exactly 0.5, attained at `Z = 0`. The structure loss therefore
actively drives the embedding to zero. Standard DOMINANT does not put an
activation on the final embedding layer.

Confirmed by ablation -- removing the final ReLU revives it (`Z std` 0.000000
-> 0.080975) and, critically, **restores the exposure correlation to +0.346**.
So the mechanism the paper describes is real when the encoder functions; it
simply is not functioning.

### 2c. The structure decoder contributes exactly nothing

With `Z = 0`, `A_hat = sigmoid(0) = 0.5` for every pair (verified:
`n_unique(A_hat) == 1`). Then

```
struct_err_i = sum_j (A_ij - 0.5)^2
```

and since `(1-0.5)^2 == (0-0.5)^2 == 0.25`, this equals `0.25 * n` for **every**
node regardless of its edges. It is a constant offset and cannot affect
ranking.

### 2d. What the score actually is

With `Z = 0`, the attribute decoder emits `X_hat_i = rowsum_i * b` (a learned
constant vector scaled by the node's normalized-adjacency row sum), so

```
score_i ~ || X_i - rowsum_i * b ||^2  +  constant
```

i.e. essentially **feature magnitude**. Anomalies carry `feature_shift = 1.0`
in all 16 dimensions, so this separates them well -- AUROC 0.91. The detector
works, but by a route that ignores the graph.

---

## 3. Why the naive fix does not work

Removing the final ReLU revives the encoder and the exposure correlation, but
collapses detection:

| variant | p_an | Z std | AUROC | exposure r |
|---|---|---|---|---|
| current (ReLU) | 0.002 | 0.000000 | 0.9107 | +0.0154 |
| no final ReLU  | 0.002 | 0.080975 | **0.1214** | **+0.3460** |
| current (ReLU) | 0.050 | 0.000000 | 0.7619 | -0.0064 |
| no final ReLU  | 0.050 | 0.008192 | 0.7571 | -0.0015 |

AUROC 0.12 is far *below* chance -- the signal is inverted. This is already
documented in this codebase, in `train_dominant_scalable`'s docstring: with
`p_aa = 0.3` vs `p_nn = 0.005`, anomalies form dense, highly predictable
clusters that a reconstruction decoder rebuilds *easily*, giving them **lower**
error. Structure-based detection is inverted by construction on this generator.

So the current AUROC of 0.91 is obtained *because* the bug disables the graph
pathway, leaving a feature-only detector on data where features are trivially
separable.

---

## 4. What survives

- **The conformal + BH machinery is sound.** FDR control, the marginal /
  conditional / zero-discovery reporting convention, and the symmetric-trimming
  fix are all unaffected.
- **The severity-sweep power collapse is real**, but its cause is not
  calibration contamination. Raising `p_an` to 0.05 adds ~500K anomaly-normal
  edges, degrading overall detector separation.
- **`theory/joint_discovery_threshold_proposition.md` does not depend on the
  contamination mechanism.** It concerns base rate, calibration size and
  clearance rate, all of which stand.
- **`scripts/signal_quality_boundary_analysis.py`'s framing may be the correct
  one** -- power collapses as detector signal quality degrades regardless of
  cause. That is consistent with everything measured here.

---

## 5. Options

1. **Reframe to a signal-quality boundary result.** Drop the contamination
   claim, keep the discovery-threshold proposition and the boundary analysis.
   Most of the empirical work survives; title, abstract, intro and the
   three-condition design all change. Lowest risk, smallest new compute.
2. ~~**Wire up `propagate_contamination()`.**~~ **RULED OUT -- tested, does not
   work.** See section 5a.
3. **Fix the detector properly.** Remove the final ReLU *and* address the
   homophily inversion (dense anomaly clusters being easy to reconstruct).
   This is a research problem, not a bug fix, and would invalidate every
   existing number.

### 5a. Why wiring up `propagate_contamination()` does not work

Tested at n=3000, 3 seeds, hops=1, using the current unmodified detector:

| mix_weight | p_an  | AUROC  | exposure r | p-value |
|------------|-------|--------|------------|---------|
| 0.00 (current) | 0.002 | 0.9107 | +0.0154 | 0.434 |
| 0.30       | 0.002 | 0.9868 | +0.0086 | 0.625 |
| 0.50       | 0.002 | **0.9959** | +0.0023 | 0.810 |
| 0.00 (current) | 0.020 | 0.8720 | +0.0123 | 0.553 |
| 0.30       | 0.020 | 0.8847 | -0.0079 | 0.553 |
| 0.50       | 0.020 | 0.8864 | -0.0260 | 0.250 |

Exposure correlation stays at zero and drifts *negative* as mixing increases,
while AUROC *improves* to 0.996.

**Why.** Under mean aggregation, an anomaly's neighbors are mostly other
anomalies (`p_aa = 0.3`), so mixing tightens anomalies around the shifted mean.
A normal node has roughly 14 normal neighbors (`p_nn = 0.005 x 2850`) against a
fraction of an anomalous one, so its neighbor average is overwhelmingly normal
and mixing tightens it around the *normal* mean. Both classes get tighter,
separation improves, and the small anomalous pull on exposed normals is
swamped by variance reduction.

Contamination-by-propagation is therefore second-order **in this generator by
construction**: normal nodes have too many normal neighbors for anomalous
influence to survive averaging. Producing a real effect would require anomalies
to be high-degree hubs attached to sparse normals, or a detector genuinely
sensitive to neighborhood composition -- i.e. a generator redesign or a working
GNN, not a wiring change.

Not tested: mix_weight > 0.5, hops > 1, or p_an = 0.05 with propagation. The
trend across the tested grid runs the wrong way, so these are unlikely to
rescue it, but they are cheap to check if you want the negative result nailed
down before reframing.

---

## 6. Open items

- [x] Does `propagate_contamination()` restore the exposure-score correlation?
      **No -- see 5a.**
- [ ] **Do the real datasets show the same thing?** Cannot be checked locally.
      Run the exposure-vs-score correlation on Amazon / Reddit on the GPU box.
      Note the real case may differ: fraudulent accounts and their neighbors
      could genuinely share features, producing real exposure-score
      correlation through the *data* rather than through message passing.
      If so, the real-data results may hold for a different reason than stated.
- [ ] Re-examine whether `n_calib` and the three-condition design still make
      sense under whichever reframing is chosen.

## Reproducing

```bash
python tests/test_normalize_equivalence.py      # unrelated, should pass
```

The diagnostics above were run as one-off scripts. The decisive one is the
exposure-vs-score correlation: generate a graph, train the detector, compute
per-normal-node exposure, and correlate against score. If |r| ~ 0, the
mechanism is not operating.
