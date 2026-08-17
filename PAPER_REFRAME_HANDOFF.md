# Project handoff: conformal graph anomaly detection

**For Gopal, or anyone (human or assistant) picking this up cold.
Last updated 2026-08-17.**

This document is self-contained. You should not need any prior conversation to
continue the work. Read sections 1-3 for orientation, 4-6 for evidence, 7-9 for
what to do next.

---

## 1. What this project is

**Repo:** `conformal-graph-anomaly` (github.com/SuhaanG/conformal-graph-anomaly)
**Authors:** Suhaan Khan (UIUC), Suhaan Gopal (McNeil HS) -- equal contribution
**Paper draft:** lives in Overleaf, mirrored in the untracked `paper/` directory

**The domain.** Graph anomaly detection (GAD): find fraudulent/anomalous nodes
in an attributed graph. A detector assigns each node an anomaly score, but a
score is not a decision -- you need a threshold, and on a large graph any
threshold produces false alarms. Conformal prediction plus the
Benjamini-Hochberg (BH) procedure converts scores into a discovery set with a
finite-sample bound on the false discovery rate (FDR).

**The original question.** GNNs use message passing, so a normal node adjacent
to anomalies may receive an inflated score. If your "known normal" calibration
set contains such nodes, it is contaminated. Does FDR control survive that?

**The answer, after extensive testing: the question is not testable with this
setup, because the contamination never reaches the scores.** See section 4.

**What the paper should be instead:** a discovery-threshold result (section 5)
plus two methodological pitfalls (section 6). See section 7 for the rewrite.

---

## 2. Status in one page

| | Status |
|---|---|
| Contamination-robustness claim | **DEAD.** Disproven on 5 real datasets, under a correct detector at AUROC 1.00, and across 5 detectors spanning 3 families (section 4.10) |
| "Fails into silence" claim | **SUSPECT.** Probably an artifact of a broken detector. Re-test or drop |
| Discovery-threshold proposition | **NEEDS RESTATING.** Valid as a *sufficient* condition; the bidirectional reading behind the old "15/15" is falsified (section 5.5) |
| Methodological pitfalls | **REAL.** Two of them, both reproducible |
| All real-data FDR numbers | **VALID** as measured, but produced by a detector whose graph pathway is inert |
| Synthetic study | **COMPROMISED.** Detector broken; generator trivially easy for a working one |
| Current paper draft | **NOT SUBMITTABLE.** Contains factual errors, see section 6.4 |
| Venue | arXiv now; then a workshop or COPA. Not IEEE at current scope (section 8) |

---

## 3. The story, chronologically

1. The paper claimed FDR control survives calibration contamination from message
   passing, tested via three conditions: **clean** (calibration drawn only from
   normal nodes with zero anomalous neighbours), **contaminated** (drawn at
   random), **adversarial** (drawn from the most anomaly-exposed normals).
2. While smoke-testing an AdaDetect baseline, one variant returned power=1.0000
   and FDR=0.0000 -- implausibly perfect. Chasing that anomaly is what started
   everything below.
3. Found: our DOMINANT encoder's final layer is **dead**, output identically
   zero. The detector's 0.91 AUROC comes entirely from raw feature magnitude;
   the graph contributes nothing.
4. Found: exposure to anomalous neighbours has **~zero correlation** with a
   node's score. The contamination mechanism was never operating.
5. Confirmed on real data (Amazon, Reddit, Tolokers) by measuring the
   calibration score *distribution* directly rather than a correlation proxy.
6. Confirmed the bug is ours: **PyGOD's DOMINANT is correct**; ours is not.
7. Confirmed the null is **not** an artifact of our bug: PyGOD's correct
   implementation reaches AUROC 0.99-1.00 and exposure *still* does not
   correlate.
8. Added Weibo and Yelp. Fixed a wrong preprocessing setting for each.
   Proposition now 15/15.

---

## 4. Why the contamination claim is dead

### 4.1 The measurement that matters

A conformal p-value is `(#{calib >= test} + 1) / (n_cal + 1)`, so the only
property of the calibration set that reaches the p-values is its **score
distribution** -- especially the upper tail and above all the maximum, since a
test point hits the p-value floor exactly when it beats every calibration score.

Earlier diagnostics used exposure-score *correlation* as a proxy. That proxy
failed audit three ways (ran on the untrimmed set while the pipeline trims 1%;
used linear partial correlation against a multiplicative confound; and
correlation is not what p-values depend on). `calibration_distribution_check.py`
measures the distribution directly, replicating the pipeline's frame verbatim.

### 4.2 The full result (5 datasets x 3 seeds x 3 conditions)

| dataset | condition | n_cal | calib_exp | q95 | max | clear_anom | power |
|---|---|---|---|---|---|---|---|
| amazon | clean | 236 | 0.0000 | 432.07 | 454.3855 | **0.0731** | 0.0000 |
| amazon | contaminated | 4000 | 0.0309 | 349.16 | 455.0225 | **0.0731** | 0.0775 |
| amazon | adversarial | 4000 | 0.0583 | 373.44 | 456.8574 | **0.0719** | 0.0731 |
| reddit | clean | 9755 | 0.0000 | 1417.23 | **1454.8347** | **0.0155** | 0.0155 |
| reddit | contaminated | 4000 | 0.0056 | 1416.88 | **1454.8347** | **0.0155** | 0.0000 |
| reddit | adversarial | 4000 | 0.0143 | 1417.40 | **1454.8347** | **0.0155** | 0.0000 |
| tolokers | clean | 786 | 0.0000 | 2125.96 | 2131.8108 | **0.0078** | 0.0000 |
| tolokers | contaminated | 4000 | 0.3192 | 1339.26 | 2131.8069 | **0.0078** | 0.0078 |
| tolokers | adversarial | 4000 | **0.5330** | 1340.49 | 2131.8050 | **0.0078** | 0.0078 |
| weibo | clean | 5620 | 0.0000 | 342.72 | 1358.81 | 0.0173 | 0.0365 |
| weibo | contaminated | 4000 | 0.0318 | 363.71 | 1383.68 | 0.0163 | 0.0000 |
| weibo | adversarial | 4000 | 0.0642 | 399.15 | 1431.07 | 0.0144 | 0.0000 |
| **yelp** | clean | 926 | 0.0000 | 5774.33 | 5782.5965 | 0.0384 | 0.0401 |
| **yelp** | contaminated | 4000 | 0.1327 | 5769.50 | 5782.5521 | 0.0385 | 0.0434 |
| **yelp** | adversarial | 4000 | **0.2736** | 5769.49 | 5782.5566 | **0.0388** | 0.0434 |

**Sanity gate passes everywhere:** `calib_exp` separates correctly
(0 < base rate < adversarial), so the conditions genuinely select different
nodes. They just do not produce different score distributions.

### 4.3 Yelp is the decisive dataset

It is the **only** dataset where all three conditions produce substantial
discovery activity (~300 discoveries each, power ~0.04). Everywhere else at
least one condition sat at zero, and a procedure that discovers nothing cannot
demonstrate anything about calibration.

On Yelp, with adversarial calibration at **27.4% exposure**:

- **KS contaminated vs adversarial: indistinguishable in all three seeds**
  (D = 0.0210 / 0.0203 / 0.0115, p = 0.341 / 0.385 / **0.954**)
- power identical to four decimals (0.0434 both)
- `clear_anom` moves the **wrong way** -- adversarial is *highest* (0.0388 vs
  0.0385). Contamination should reduce clearance, not raise it.

So on the one dataset where composition could plausibly have mattered, it
demonstrably does not.

### 4.4 Tolokers is the severity check

Adversarial reaches **0.5330 mean exposure** -- over half of each calibration
node's neighbours anomalous, more than synthetic's worst case (0.34). Result:
q95 shifts 0.07%, identical max, identical clearance, **identical discoveries
(21 vs 21)**, KS p=0.141. If contamination drove anything, Tolokers would show
the largest effect. It shows none.

### 4.5 The one apparent effect is a degree-normalization artifact

Amazon is the only dataset with a real contaminated-vs-adversarial difference
(KS D~0.20, discoveries 79 -> 60). It does **not** track contamination severity;
it tracks the degree confound times whether degree normalization is on:

| dataset | degree_norm | exposure~degree | adversarial exposure | KS D |
|---|---|---|---|---|
| amazon | **True** | **-0.188** | 0.058 | **0.20** |
| tolokers | True | -0.040 | **0.533** | 0.03 |
| reddit | **False** | +0.228 | 0.014 | 0.03 |

Mechanism: adversarial picks high-exposure nodes; on Amazon exposure correlates
-0.188 with degree, so it picks **low-degree** nodes; `degree_normalize_scores`
divides by `log1p(degree)`, **inflating** them; higher calibration scores mean
higher p-values and fewer discoveries. No contamination required.

Confirmed by the Reddit control (`degree_norm=False`): raw and as-used
correlations are **identical** there (+0.0220 both), while on Amazon
normalization turns **-0.0621 into +0.2697** -- a sign flip and 4x magnitude
change.

### 4.10 The decisive test: five detectors, three families

Goals 1 and 2 from the old section 9.2 are complete. `src/detectors.py` runs the
study under five scorers; all five work (AUROC on a 1000-node synthetic graph):

| detector | AUROC | family |
|---|---|---|
| `dominant_pygod` | 0.9707 | reconstruction, attribute + structure |
| `dominant_ours` | 0.9230 | reconstruction (broken encoder) |
| `ocgnn` | 0.8973 | **one-class** |
| `anomalydae` | 0.7238 | dual autoencoder |
| `gae` | 0.6212 | reconstruction, attribute only |

Ran the full matrix on amazon/reddit/weibo/tolokers, 3 seeds. The clean
comparison is **contaminated vs adversarial** -- identical `n_cal` by
construction, differing only in which nodes were selected. Direction of change
in `clear_anom`, which is what determines discovery:

| dataset | dominant_ours | dominant_pygod | ocgnn | anomalydae | gae |
|---|---|---|---|---|---|
| amazon | DOWN | same | same | **UP** | **UP** |
| reddit | same | same | same | **UP** | same |
| weibo | DOWN | same | same | same | same |
| tolokers | same | same | **UP** | same | DOWN |

**Across 20 detector-dataset cells: 13 identical, 4 UP, 3 DOWN.**

Contamination predicts DOWN everywhere -- adversarial calibration should raise
calibration scores and make anomalies harder to clear. It occurs in 3 of 20,
while the *opposite* occurs in 4 of 20. **The direction is a coin flip.**

This is a stronger argument than "the effect is small". A real mechanism would
have a consistent sign across scorers, because selecting the most anomaly-exposed
nodes should degrade calibration the same way regardless of what computes the
score. Instead the sign depends on the detector, which is the signature of a
score-distribution artifact rather than contamination.

Note also Tolokers, where adversarial calibration reaches **54% anomalous
neighbours** -- the most extreme contamination anywhere in this project.
`clear_anom` there is identical across all four PyGOD detectors
(0.0500 / 0.0126 vs 0.0133 / 0.0035 / 0.0025 vs 0.0023).

### 4.11 Better AUROC does not mean better discovery

On Amazon, `dominant_pygod` (AUROC 0.9707) produces **zero** discoveries under
contaminated and adversarial calibration -- `clear_anom = 0.0000` exactly, not a
single anomaly beats the calibration maximum. Our broken detector (AUROC 0.9230)
produced ~64.

The working detector has a heavier normal-score tail, so with 4,000 calibration
nodes you capture normals that no anomaly exceeds. Ranking quality improved;
conformal discovery collapsed.

This is a clean, quotable result and it *supports* the paper's thesis: discovery
is governed by clearance behaviour in the extreme tail, not by ranking quality.
It also means AUROC is the wrong summary statistic for this pipeline, which is a
practical point worth making explicitly.

### 4.6 Why the mechanism is severed

Three independent breaks:

1. **The generator never propagates features.** `generate()` calls
   `_generate_features(labels)`, using each node's own label only.
   `propagate_contamination()` implements the intended mechanism, its docstring
   says so explicitly, and **nothing calls it**. Wiring it up was tested: it does
   *not* restore the effect (mean aggregation over mostly-normal neighbours
   swamps it) and it raises AUROC to 0.996.
2. **Our GCN encoder's final layer is dead.** `F.relu` on every layer including
   the last forces `Z >= 0`, so `sigmoid(z_i . z_j) >= 0.5`, while ~99.7% of
   node pairs are non-edges pulling toward 0. The loss minimizes at `Z = 0`
   exactly. Confirmed 6/6 across sizes and seeds including n=15,000. With `Z=0`,
   `A_hat = 0.5` everywhere and `struct_err = 0.25n` for every node -- a constant
   that cannot affect ranking.
3. **The effect is absent even with a working encoder** (4.7).

### 4.7 PyGOD is correct; the null survives anyway

**PyGOD's DOMINANT does not have our bug.** `torch_geometric`'s
`BasicGNN.forward` gates the activation behind
`if i < self.num_layers - 1 or self.jk_mode is not None`, so the final conv
output stays linear. Measured on the identical graph:

| | frac_zero | std | min | negatives? |
|---|---|---|---|---|
| ours | **1.0000** | 0.00000000 | 0 | no |
| PyGOD | **0.0000** | 0.05182356 | **-0.219214** | **yes** |

Negatives are impossible after a ReLU, so this is proof, not inference.

**And the null holds under it.** Running PyGOD's correct implementation on the
same synthetic graphs:

| p_an | mean exposure | PyGOD AUROC | exposure r | p |
|---|---|---|---|---|
| 0.002 | 0.021 | 0.9865 | +0.0320 | 0.0895 |
| 0.010 | 0.096 | 0.9998 | +0.0137 | 0.5291 |
| 0.050 | **0.342** | **1.0000** | **-0.0234** | 0.3315 |

Encoder alive, detection near-perfect, exposure still uncorrelated -- max
|r| = 0.032, nothing significant, negative at the highest severity. **The
confound was removed and the effect is still absent.**

### 4.8 A second headline claim is now suspect

PyGOD's AUROC *rises* with contamination severity (0.9865 -> 1.0000). Ours
*fell* (0.9107 -> 0.7619). So the severity sweep's power collapse -- the "fails
into silence rather than into false discovery" finding -- is most likely also an
artifact of the dead encoder degrading as the graph densifies. With AUROC 1.0,
clearance would be near-total and discoveries would occur.

**Do not carry this claim into the rewrite without re-testing it.**

### 4.9 Consequence for the synthetic study

AUROC **1.0000** means the generator is trivially easy for a working detector
(`feature_shift=1.0` across 16 dims with functioning message passing). Re-running
synthetic with a correct detector produces no interesting regime. The generator
would need to be made materially harder first.

**Therefore: the paper should lead with real data**, where results were measured
on the actual pipeline, and treat synthetic as illustrative with caveats stated.

---

## 5. The positive result: the discovery-threshold proposition

Source: `theory/joint_discovery_threshold_proposition.md` (Gopal's).

**Setup.** `n` = calibration size, `m` = test size, `m_1` = anomalies in test,
`pi_1 = m_1/m`, `alpha` = nominal FDR.

- **Fact 1.** Minimum achievable conformal p-value is `1/(n+1)`, attained when a
  test point beats every calibration score.
- **Fact 2.** BH rejects `k` points tied at that floor only if
  `1/(n+1) <= alpha*k/m`, i.e. `k >= m/(alpha*(n+1))`.

Define the **clearance rate** `c` = fraction of true anomalies beating the entire
calibration set. Then discovery is possible iff

    pi_1 * c >= 1 / (alpha * (n_cal + 1))

equivalently `clears >= bh_min_rank = ceil(m / (alpha*(n_cal+1)))`.

### 5.1 Validation: 15 of 15 on real data

| dataset | condition | n_cal | m | needs | clears | predicts | observed | ok |
|---|---|---|---|---|---|---|---|---|
| amazon | clean | 236 | 5821 | 246 | 60 | none | 0 | Y |
| amazon | contaminated | 4000 | 5821 | 15 | 60 | discovery | ~64 | Y |
| amazon | adversarial | 4000 | 5821 | 15 | 59 | discovery | 60 | Y |
| reddit | clean | 9755 | 1123 | 2 | 5.7 | discovery | 5.7 | Y |
| reddit | contaminated | 4000 | 5366 | 14 | 5.7 | none | 0 | Y |
| reddit | adversarial | 4000 | 5366 | 14 | 5.7 | none | 0 | Y |
| tolokers | clean | 786 | 7566 | 97 | 20 | none | 0 | Y |
| tolokers | contaminated | 4000 | 7566 | **19** | **20** | discovery | 20.7 | Y |
| tolokers | adversarial | 4000 | 7566 | **19** | **20** | discovery | 21 | Y |
| weibo | clean | 5620 | 2704 | **5** | **6.0** | discovery | 33-35 | Y |
| weibo | contaminated | 4000 | 4324 | 11 | 5.7 | none | 0 | Y |
| weibo | adversarial | 4000 | 4324 | 11 | 5.0 | none | 0 | Y |
| yelp | clean | 926 | 11677 | 126 | 256 | discovery | 279 | Y |
| yelp | contaminated | 4000 | 11677 | 30 | 257 | discovery | 318 | Y |
| yelp | adversarial | 4000 | 11677 | 30 | 259 | discovery | 309 | Y |

**The informative cells are Tolokers and Weibo**, where the prediction is tight
at the boundary (19 vs 20; 5 vs 6.0) *and* the neighbouring condition on the
same dataset falls the other side and correctly produces zero. Yelp's cells are
comfortable (2x-8.6x margin) -- they add count, not sharpness. Say so honestly.

**`m` values are reconstructed; confirm against the `m_test` column in
`results/logs/calibration_distribution_check.csv` before publishing.**

### 5.2 Why it is interesting

It overturns the obvious ordering: **Reddit has a better detector than Tolokers
(AUROC 0.577 vs 0.409, the latter below chance) yet discovers nothing, while
Tolokers discovers.** Detector quality alone predicts the reverse. Base rate and
calibration size resolve it.

### 5.3 It reinterprets our own "condition effects"

Every clean-vs-other difference in the real-data results is a **calibration
size** effect the proposition already explains -- not contamination. Reddit and
Weibo are clearest: clean discovers while the others do not, purely because
clean has a larger pool (9755 and 5620 vs 4000). Yelp shows it as a graded
effect: clean has the *smallest* pool (926), hence the highest p-value floor,
hence fewest discoveries and the most conservative FDR (0.018 vs 0.045).

### 5.5 IMPORTANT: the bidirectional reading is falsified

**Read this before using the 15/15 table.**

The proposition as written in `theory/` is a **sufficient** condition: *if*
`pi_1 * c >= 1/(alpha(n+1))`, at least one discovery occurs. It never claimed
the converse.

The 15/15 table above used it **bidirectionally**, predicting "none" whenever
`clears < needs`. That direction is now falsified. Running the same check under
`dominant_pygod`:

| dataset | condition | n_cal | m | needs | clears | predicted | observed |
|---|---|---|---|---|---|---|---|
| amazon | clean | 267 | 5821 | 218 | 134 | none | **3,420** |

Off by a factor of 25. The mechanism: BH rejects at rank *k* where
`p_(k) <= alpha*k/m`. At m=5821 and alpha=0.1, rank 3420 admits p-values up to
**0.0588**, while the floor is `1/268 = 0.0037`. Thousands of points sitting
*between* the floor and that threshold get rejected. The floor-based condition
only ever governed points AT the floor.

**Worse, the failures are seed-unstable.** Amazon clean under `ocgnn`, identical
configuration, three seeds:

| seed | clear_anom | n_disc |
|---|---|---|
| 0 | 0.0877 | **424** |
| 1 | 0.0889 | **0** |
| 2 | 0.0828 | **0** |

Clearance essentially constant (72-73 anomalies against a threshold of 218), yet
discoveries swing from 424 to zero. Near the boundary, whether discovery happens
is governed by the *bulk* of the p-value distribution, which varies seed to seed.

Why the old 15/15 held: under `dominant_ours` the score distributions happened
to make both readings coincide. **Adding a second detector broke it immediately**
-- which is exactly what multi-detector validation is for, and is itself an
argument for having done it.

**What this means for the paper.** Either:
  (a) restate the result honestly as "predicts discovery correctly in N cases",
      dropping the negative direction, which is weaker but true; or
  (b) strengthen the proposition to account for non-floor rejections, which
      would be a real theoretical contribution and is the thing most likely to
      lift this paper a venue tier (old section 9.2 item 3).

Do not ship the bidirectional framing.

### 5.4 Correction needed in the theory doc

It computes `c*` using the **graph-wide** anomaly rate, but its own definition
says `pi_1 = m_1/m` (test-set prevalence). Calibration removes *normal* nodes and
the pipeline caps test normals at 5000 while keeping every anomaly, so the test
set is enriched ~2x:

| dataset | pi_1 in doc | pi_1 actual | c* in doc | c* corrected |
|---|---|---|---|---|
| Amazon | 0.0687 | 0.1410 | 0.0364 | 0.0177 |
| Reddit | 0.0333 | 0.0682 | 0.0750 | 0.0366 |
| Tolokers | 0.2182 | 0.3391 | 0.0115 | 0.0074 |

Synthetic: doc says 0.0716 using the 5% graph rate; test-set prevalence is 6.14%,
giving **0.0582**. `clearance_rate_verification.py` had the same bug hardcoded
(`pi1 = 0.05`) and is now fixed to compute per trial.

**Ordering and argument survive unchanged** (Tolokers < Amazon < Reddit either
way). Only numbers need updating. Also: the doc quotes Weibo's anomaly rate from
a PyGOD docstring as ~10.3%; the loaded graph is **4.13%** (347/8405). And its
item 4 references `theory/theoretical_characterization_draft.md`, which does not
exist.

---

## 6. The pitfalls (the second contribution)

### 6.1 An encoder can silently die while AUROC looks fine

Our detector reported **AUROC 0.91 with a completely inert graph pathway**:
`Z = 0`, `A_hat = 0.5` uniformly, `struct_err` constant across all nodes. Nothing
in a standard evaluation catches this. The signal came entirely from feature
magnitude.

**Root cause:** ReLU on the final embedding layer. **Fix:**

```python
for i, layer in enumerate(self.encoder_layers):
    H = layer(A_norm, H)
    if i < len(self.encoder_layers) - 1:   # no activation on the final embedding
        H = F.relu(H)
```

PyGOD/PyG get this right; we did not. **The headline for practitioners: good
AUROC does not verify that your graph pathway is doing anything.** The diagnostic
is one line -- check whether the embedding is identically zero.

**Do not fix and re-run naively:** removing the ReLU revives the mechanism
(exposure r -> +0.346) but *inverts* detection (AUROC -> 0.12), because dense
anomaly clusters (`p_aa=0.3`) are easier to reconstruct. That failure is already
documented in `train_dominant_scalable`'s own docstring.

### 6.2 A zero-variance signature worth reporting

On Amazon, **all 20 adversarial seeds returned bit-identical results** --
`n_discoveries=60`, `realized_fdr=0.000`, `power=0.073`, `std = 0.000` exactly.
Twenty independently trained models, one outcome.

Explanation: with `Z=0` the score is feature magnitude plus a rowsum term,
neither seed-dependent, and adversarial selection is deterministic (top-4000 by
exposure, no RNG). Contaminated *does* vary (FDR 0.000-0.146) because its draw is
random.

Two reasons to report it: a reviewer will ask why a standard deviation is exactly
zero across 20 seeds, and it is a cheap diagnostic anyone can run. Note also that
two contaminated seeds individually exceed nominal 0.10 (0.139, 0.146) --
legitimate since FDR control bounds the average, but better disclosed than
discovered.

Yelp is the counter-example: its adversarial condition **does** vary across
seeds (FDR 0.000-0.056), making it the strongest test.

### 6.3 Degree normalization must be measured per dataset, not inherited

Four datasets, four independent measurements, no clean rule:

| dataset | median degree | max degree | ratio | degree_norm | evidence |
|---|---|---|---|---|---|
| Reddit | 8 | 2,557 | 320x | **False** | norm pushed AUROC 0.577 -> 0.452, below chance |
| Weibo | 59 | 4,359 | 74x | **True** | raw 0.773 -> norm 0.843 |
| Tolokers | 29 | 1,804 | 62x | True | (inherited, not independently measured) |
| Amazon | 425 | 6,981 | 16x | **True** | corrects documented hub inflation |
| **Yelp** | **168** | **501** | **3x** | **False** | measured; see below |

Yelp is the clearest mechanism: mean degree 167.4 ~ median 168.0 and max only
501, i.e. **essentially no hubs**. Degree normalization exists to correct hub
inflation; with nothing to correct it only injects noise. But Reddit breaks any
simple "heavy tail -> normalize" rule, so report this as an observation with the
counterexample, not a law.

**This was not academic.** Yelp's setting was inherited by analogy to Amazon back
when Yelp could not be run at all. Correcting it changed the result
qualitatively:

| | clean | contaminated FDR | contaminated power |
|---|---|---|---|
| degree_norm=True (wrong) | 0 discoveries | 0.036 | 0.019 |
| degree_norm=False (correct) | **~271** | 0.045 | **0.044** |

Power more than doubled and clean went from silent to productive. An assumed
preprocessing choice was suppressing the result.

### 6.4 Factual errors in the current paper draft -- fix regardless of framing

1. **§Datasets describes an experiment that was not run.** It says anomalous
   influence "is propagated to neighbouring nodes through a message-passing step
   that mixes each node's feature vector with a weighted average of its
   neighbours'... This propagation mechanism instantiates the calibration
   contamination process examined throughout the paper." **Nothing calls
   `propagate_contamination()`.** Highest-priority fix -- a methods section
   describing a procedure not in the code.
2. **The abstract asserts the mechanism operates** ("message passing propagates
   their influence... shorting those nodes' scores"). It does not. Also
   "shorting" should read "shifting".
3. **§Detector protocol claims the degree correction was applied to "Amazon and
   Yelp"** as a per-dataset property that was checked. For Yelp it was never
   checked, and the measurement contradicts it (6.3).
4. **Amazon AUROC stated as 0.893**; measured **0.7542**. Tolokers stated 0.4093;
   measured 0.4390. `signal_quality_boundary_analysis.py` hardcodes the old
   values and the theory doc's verification table depends on them.
5. **Practical Guidance and Appendix A cite Tolokers results never reported.**
   They now exist -- report them.
6. **Appendix A calls the closed-form threshold a *failed* approach.** Gopal's
   rank-k version fixes exactly the flaw described there. It is now the paper's
   central positive result, not a failure.
7. The intro says the graph setting introduces contamination "considered in many
   non-graph settings", which concedes the novelty; should read *distinct from*.
8. Citation keys: the draft uses `bates2023testing`, `huang2023uncertainty`,
   `ding2019deep`, `dou2020camouflaged`. `references.bib` has been renamed to
   match these, so they are consistent -- do not "fix" them back.

---

## 7. The rewrite

**Old question:** does FDR control survive calibration contamination?
**New question:** when does conformal FDR control discover anything at all on
graphs?

**Working title:** *When Does Conformal Anomaly Detection Discover Anything on
Graphs? A Joint Condition on Base Rate, Calibration Size, and Detector Quality*

**Contributions:**
1. The joint discovery-threshold condition, derived from conformal p-value
   mechanics and the BH rejection rule, validated **15/15** across five real
   datasets, tight at the boundary on two.
2. A negative result, honestly reported and hard to obtain: calibration
   *composition* -- including worst-case selection up to 53% anomalous
   neighbours -- does not measurably change the calibration score distribution.
   Calibration *size* does, and dominates. Established across five datasets and
   under a correct detector implementation at AUROC 1.00.
3. Two pitfalls that make GAD pipelines look like they work when they do not:
   silent encoder collapse with AUROC intact, and degree normalization
   manufacturing an apparent exposure effect.

### Section-by-section

| Section | Action |
|---|---|
| Title / Abstract / Keywords | **Rewrite** |
| Introduction | **Rewrite** around the discovery question |
| Related work | Trim contamination emphasis; keep conformal + GAD |
| Preliminaries | **Add the proposition as the theoretical core** |
| Experimental setup | Mostly unchanged, but **delete the propagation sentences** (6.4.1) |
| Results: real-world | **Lead with this.** Same numbers + the 15/15 table |
| Results: synthetic | Demote; caveat per 4.8-4.9 |
| Results: baseline comparison | Unchanged |
| Ablation: calibration size | **Promote to a main result** -- it demonstrates the `n_cal` term directly |
| Detection power mechanism | Keep; supports the proposition |
| Discussion | Add the negative result, both pitfalls, the zero-variance signature |
| Conclusion | **Rewrite** |
| Appendix A | **Rewrite** from failure to central result (6.4.6) |

---

## 8. Venue

**Not IEEE at current scope.** The objection that caps us is not evidence, it is
depth: the proposition is a few lines of algebra combining the p-value floor with
the BH rule. Correct, useful, well-validated -- and a reviewer will still call it
an observation rather than a theorem. More validation cells improve evidence
without making the result deeper.

Both routes to TKDE are closed:
- *"A widely-used detector is silently broken"* -- closed, PyGOD is correct (4.7)
- *"Contamination is real and we characterize it"* -- closed, null holds under a
  correct detector (4.7)

**Ranked recommendation:**

1. **arXiv, immediately.** Free, citable, blocks nothing.
2. **NeurIPS Datasets & Benchmarks Track.** Best fit for the pitfalls framing;
   ~25-30% acceptance; CV line reads "NeurIPS". Precedent: `tang2023gadbench`
   in our own bibliography was published there.
3. **AISTATS / UAI.** Right audience for the statistical core -- they will
   actually engage with the p-value floor argument, PRDS testing, and the e-BH
   appendix. ~10-20% for this paper.
4. **COPA** (Symposium on Conformal and Probabilistic Prediction). Best topical
   fit; `clarkson2024contamination` in our bib was published there. PMLR
   proceedings, no fee. **COPA 2026 closed (papers May 20, posters Aug 1);
   COPA 2027 ~May 2027.**
5. **SDM** -- where DOMINANT itself was published. ~20-25%.
6. IEEE TNSE / TAI / TETCI / TCSS -- ~20-30% each after the rewrite. TNSE has the
   best scope fit of the IEEE options.

Note that in CS, conferences generally outrank mid-tier journals, so options 2-5
are *both* more prestigious and better-fitting than the IEEE journals.

### What would actually reach IEEE Transactions level

See section 9.2. Short version: multiple detectors, a formalized theorem, and
re-running everything under a correct detector. Months, not weeks.

---

## 9. What to do next

### 9.1 Immediate (this week)

- [ ] Do the rewrite (section 7). Experiments are done; numbers do not change.
- [ ] Fix all factual errors in 6.4 -- these are not optional.
- [ ] Post to arXiv.
- [ ] Paste Yelp's measured raw vs degree_norm AUROCs into the TODO comment in
      `DEGREE_NORM_BY_DATASET` (`scripts/real_data_experiment.py`), matching the
      style of the weibo entry.
- [ ] Confirm `m` values in the 15/15 table against `m_test` in the CSV.
- [ ] Update `theory/joint_discovery_threshold_proposition.md` per 5.4.
- [ ] Measure `exposure~degree` on Weibo -- the last loose end. Weibo is the only
      dataset where the calibration distribution moves monotonically with
      exposure (q95 342->364->399, clear_anom 0.0173->0.0163->0.0144), and the
      degree explanation does not obviously fit its direction.
      `exposure_degree_confound_check.py --datasets weibo`
- [ ] **Re-test or drop "fails into silence"** (4.8).

### 9.2 To reach IEEE Transactions level (months)

Ranked by impact:

1. ~~**Re-run everything with PyGOD's DOMINANT.**~~ **DONE** -- see 4.10/4.11.
2. ~~**Add 2-3 more detectors.**~~ **DONE** -- five detectors across three
   families now run via `src/detectors.py`. Remaining: Yelp under each detector
   (needs `--use_sparse_prop`), which would take the matrix to 25 cells per
   condition.
3. **Formalize the proposition into a theorem** -- now the single highest-value
   item, and more urgent than before: 5.5 shows the current statement cannot
   predict absence of discovery, and fixing that IS the theoretical contribution.
   Specifically, extend beyond "can any floor-tied point be rejected" to a
   condition covering rejections from the bulk of the p-value distribution. with stated conditions and a
   proof, ideally extended beyond "can any discovery occur" to a bound on
   expected discovery count. This is the only thing that addresses the depth
   objection.
4. **Run AdaDetect at scale** (built and gate-verified, never run). Caveat when
   reporting: it consumes our DOMINANT scores, so it compares calibration
   *procedures* on a degraded score.
5. **Harden the synthetic generator** so a working detector does not hit
   AUROC 1.0, making synthetic informative again.

Items 1+2 together are the highest value per unit effort and would plausibly move
this to TAI/TNSE range; adding 3 is what a TKDE reviewer would want.

---

## 10. Repo guide

### Core library (`src/`)
| File | Purpose |
|---|---|
| `graph_gen.py` | Synthetic SBM generator. **Note `propagate_contamination()` is never called** |
| `detector.py` | DOMINANT. `normalize_adj` is FROZEN (validated results depend on it). `normalize_adj_fast` = O(n^2) equivalent. `train_dominant(use_sparse_prop=True)` = large-graph path. `train_dominant_scalable` is BROKEN, do not use |
| `conformal_fdr.py` | `conformal_p_values`, `benjamini_hochberg`. FROZEN -- imported by 7+ scripts with committed results |
| `adadetect.py` | AdaDetect mechanism, resolution-floor-matched frame |

### Diagnostics (the ones that produced section 4)
| Script | What it answers |
|---|---|
| `calibration_distribution_check.py` | **The decisive one.** Do the three conditions produce different calibration distributions? |
| `exposure_degree_confound_check.py` | Is the exposure-score correlation real or a degree artifact? |
| `real_data_exposure_diagnostic.py` | Does exposure correlate with score on real data? |
| `pygod_architecture_check.py` | Does PyGOD have our dead-ReLU bug? (sampler-free) |
| `pygod_exposure_check.py` | Does contamination operate under a *correct* detector? |
| `pygod_dominant_collapse_check.py` | Same as architecture_check but via PyGOD's wrapper -- **needs pyg-lib/torch-sparse, usually fails** |
| `degree_norm_diagnostic.py` | Should degree normalization be on for this dataset? |

### Experiments
| Script | Purpose |
|---|---|
| `real_data_experiment.py` | Main 3-condition FDR experiment. Holds `DEGREE_NORM_BY_DATASET` |
| `multi_seed_sweep.py`, `severity_sweep.py` | Synthetic experiments (see 4.9 caveats) |
| `baseline_comparison*.py` | Ensemble vs single-seed |
| `adadetect_comparison.py` | AdaDetect vs ours |
| `clearance_rate_verification.py` | Tests the proposition on the severity sweep |
| `signal_quality_boundary_analysis.py` | Gopal's signal-quality framing |

### Theory & docs
- `theory/joint_discovery_threshold_proposition.md` -- the proposition (needs 5.4 fixes)
- `DETECTOR_DIAGNOSTIC.md` -- full technical writeup of the encoder collapse
- `PAPER_REFRAME_HANDOFF.md` -- this file

---

## 11. Environment

Runs on a Jupyter box with an **NVIDIA H200 NVL**. Use `--device cuda`.

```bash
~/envs/dgl311/bin/python <script>
```

`setup_dgl_env.sh` requests CPU torch but the env actually has **torch
2.4.0+cu121** with CUDA available -- do not assume CPU-only.

Installed: numpy 2.4.6, scipy 1.17.1, networkx 3.6.1, torch 2.4.0+cu121,
dgl 2.4.0, torch_geometric 2.8.0.post1, pygod 1.1.0.
`torch_sparse` fails to import (missing `torch_scatter`) -- only affects
`pygod_dominant_collapse_check.py`; use `pygod_architecture_check.py` instead.

**Yelp requires `--use_sparse_prop`** in both `real_data_experiment.py` and
`calibration_distribution_check.py`. Without it you get the frozen dense path:
two dense 45,954 x 45,954 numpy matmuls per call, ~65 min each, and the GPU does
not help because that work is numpy on CPU. Verified bit-identical to the dense
path on Amazon (all 60 trials).

### Reproducing everything

```bash
~/envs/dgl311/bin/python scripts/calibration_distribution_check.py --datasets amazon reddit tolokers weibo --n_seeds 3 --device cuda
```

```bash
~/envs/dgl311/bin/python scripts/calibration_distribution_check.py --datasets yelp --n_seeds 3 --device cuda --use_sparse_prop
```

```bash
~/envs/dgl311/bin/python scripts/pygod_architecture_check.py --n_nodes 3000 --epochs 100
```

```bash
~/envs/dgl311/bin/python scripts/pygod_exposure_check.py --n_nodes 3000 --epochs 100 --seeds 3 --p_an 0.002 0.01 0.05
```

Minutes each on GPU. Outputs land in `results/logs/`.

---

## 12. Notes for whoever continues this

- **The frozen-code norm is real.** `conformal_fdr.py` and `normalize_adj` are
  imported by scripts that produced committed results. Add alongside; do not
  edit. Every performance path added so far was verified bit-identical first.
- **This project has a history of subtle bugs found by chasing implausible
  results.** Asymmetric trimming once manufactured FDR ~0.51; the dead encoder
  was found because an AdaDetect variant returned power=1.0. When a number looks
  too good, chase it.
- **Measure, do not inherit.** Two preprocessing settings (Weibo, Yelp) were
  inherited by analogy and one was wrong in a way that changed conclusions.
- **The negative results here were expensive to establish and are the paper's
  most defensible content.** Do not quietly drop them for being unexciting.
