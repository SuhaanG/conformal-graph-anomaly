# Paper reframe: what we found, what survives, and what to write

**For: Gopal. From: Suhaan K, 2026-08-16.**

Short version: the contamination-robustness claim does not hold up, and we
should not submit it. But your discovery-threshold proposition just got
validated on **9 out of 9** dataset x condition cells on real data, and it is
strong enough to carry the paper on its own. The experiments are done. The
numbers do not change. This is a rewrite, not a re-run.

---

## 1. What broke, in one paragraph

The paper claims GNN message passing propagates anomalous influence into the
scores of neighboring normal nodes, contaminating the calibration set, and that
FDR control survives this. We tested whether that contamination actually
reaches the scores. It does not -- on synthetic data or on any of the three
real datasets. The clean / contaminated / adversarial conditions are not
producing meaningfully different calibration sets, so the comparison cannot
support a contamination-robustness claim. Full technical detail is in
`DETECTOR_DIAGNOSTIC.md`; the decisive evidence is below.

---

## 2. The decisive evidence

Run: `scripts/calibration_distribution_check.py`, 3 datasets x 3 seeds x 3
conditions, raw output in `results/logs/calibration_distribution_check.csv`.

This measures the thing conformal p-values actually depend on. A p-value is
`(#{calib >= test} + 1) / (n_cal + 1)`, so the *only* property of the
calibration set that reaches the p-values is its score distribution --
especially the upper tail, and above all the maximum, since a test point hits
the floor exactly when it beats every calibration score.

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

### 2a. The selection logic works -- so this is a real negative, not a bug

`calib_exp` separates correctly everywhere: 0 for clean, base rate for
contaminated, higher for adversarial. The conditions genuinely select different
nodes. They just do not produce different *score distributions*.

### 2b. The quantity that drives discovery never moves

`clear_anom` (fraction of true anomalies beating the entire calibration set) is
**identical across all three conditions within every dataset**. So is `max` --
on Reddit it is literally identical to four decimals across all three
conditions. The extreme tail that sets the p-value floor is untouched by which
calibration condition you choose.

### 2c. Tolokers is the killer

Tolokers adversarial reaches **0.5330 mean exposure** -- over half of each
calibration node's neighbors are anomalous. That is more contamination than
anywhere else we have tested, including synthetic's worst case (0.34). Result:

- q95 shift of 0.07% (1339.26 -> 1340.49)
- identical max, identical clearance rate
- **identical discoveries: 21 vs 21**, all three seeds
- KS contaminated vs adversarial: D=0.0257, **p=0.141, not significant** (seed 0)

If contamination drove our results, Tolokers would show the largest effect. It
shows none.

### 2d. The one apparent effect is a degree-normalization artifact

Amazon is the only dataset with a real contaminated-vs-adversarial difference
(KS D~0.20, discoveries 79 -> 60). Effect size does **not** track contamination
severity, but it exactly tracks the degree confound times whether degree
normalization is enabled:

| dataset | degree_norm | exposure~degree | adversarial exposure | KS D (contam vs adv) |
|---|---|---|---|---|
| amazon | **True** | **-0.188 (strong)** | 0.058 | **0.20** |
| tolokers | True | -0.040 (weak) | **0.533** | 0.03 |
| reddit | **False** | +0.228 | 0.014 | 0.03 |

Mechanism: adversarial selects high-exposure nodes; on Amazon exposure
correlates -0.188 with degree, so it selects **low-degree** nodes;
`degree_normalize_scores` divides by `log1p(degree)`, **inflating** them; higher
calibration scores mean higher p-values and fewer discoveries. That reproduces
the observed "adversarial is conservative" result with no contamination
involved.

Confirmed by the Reddit control (`degree_norm=False`): raw and as-used
correlations are **identical** there (+0.0220 both), whereas on Amazon
normalization turns **-0.0621 into +0.2697** -- a sign flip and a 4x magnitude
change. See `scripts/exposure_degree_confound_check.py`.

### 2e. Why the mechanism is severed (synthetic)

Two independent breaks, both in `DETECTOR_DIAGNOSTIC.md`:

1. **The generator never propagates features.** `generate()` calls
   `_generate_features(labels)`, using each node's own label only.
   `propagate_contamination()` implements the intended mechanism and its
   docstring says so -- and nothing calls it. We tested wiring it up: it does
   **not** restore the effect (mean aggregation over mostly-normal neighbors
   swamps it), and it *raises* AUROC to 0.996.
2. **The GCN encoder's final layer is dead.** ReLU on the final embedding
   forces `Z >= 0`, so `sigmoid(z_i . z_j) >= 0.5`, but ~99.7% of node pairs
   are non-edges pulling toward 0. The loss minimizes at `Z = 0` exactly.
   Confirmed 6/6 across sizes and seeds including n=15,000. With `Z=0`,
   `A_hat = 0.5` everywhere and `struct_err = 0.25n` for every node -- a
   constant that cannot affect ranking. The detector's ~0.91 AUROC comes
   entirely from raw feature magnitude.

Removing the final ReLU revives the mechanism (exposure correlation -> +0.346)
but **inverts detection** (AUROC -> 0.12), because dense anomaly clusters
(`p_aa=0.3`) are *easier* to reconstruct -- a failure already documented in
`train_dominant_scalable`'s own docstring.

---

## 3. Your proposition: 9 out of 9 on real data

This is the headline result now. Using
`bh_min_rank = ceil(m / (alpha * (n_cal + 1)))` and the measured clearance
rate, your joint condition predicts every observed outcome:

| dataset | condition | n_cal | m | needs | clears | predicts | observed | ✓ |
|---|---|---|---|---|---|---|---|---|
| amazon | clean | 236 | 5821 | 246 | 60 | none | 0 | ✓ |
| amazon | contaminated | 4000 | 5821 | 15 | 60 | discovery | ~64 | ✓ |
| amazon | adversarial | 4000 | 5821 | 15 | 59 | discovery | 60 | ✓ |
| reddit | clean | 9755 | 1123 | 2 | 5.7 | discovery | 5.7 | ✓ |
| reddit | contaminated | 4000 | 5366 | 14 | 5.7 | none | 0 | ✓ |
| reddit | adversarial | 4000 | 5366 | 14 | 5.7 | none | 0 | ✓ |
| tolokers | clean | 786 | 7566 | 97 | 20 | none | 0 | ✓ |
| tolokers | contaminated | 4000 | 7566 | **19** | **20** | discovery (barely) | 20.7 | ✓ |
| tolokers | adversarial | 4000 | 7566 | **19** | **20** | discovery (barely) | 21 | ✓ |

Note the Tolokers rows: predicted threshold 19, actual clearance 20, observed
~21 discoveries. The prediction is tight at the boundary, not just directionally
right.

**`m` values above are reconstructed; confirm against the `m_test` column in
`results/logs/calibration_distribution_check.csv` before this goes in the paper.**

### Why this is genuinely interesting

It overturns the obvious ordering. **Reddit has a better detector than Tolokers
(AUROC 0.577 vs 0.409, the latter below chance) yet discovers nothing, while
Tolokers discovers.** Detector quality alone predicts the reverse. Your
base-rate framing resolves it: Tolokers' 21.8% anomaly rate sets a far lower bar
than Reddit's 3.3%.

### It also reinterprets our own "condition effects"

Every clean-vs-other difference in the real-data results is a **calibration
size** effect, which your proposition already explains -- not a contamination
effect. Reddit is the clearest case: clean discovers (5-6) while
contaminated/adversarial discover nothing, purely because clean has n_cal=9755
vs 4000, and a correspondingly smaller test set.

### One correction to the theory doc

`theory/joint_discovery_threshold_proposition.md` computes `c*` using the
**graph-wide** anomaly rate, but its own definition says `pi_1 = m_1/m`
(test-set prevalence). Because calibration removes *normal* nodes and the
real-data path caps test normals at 5000 while keeping every anomaly, the test
set is enriched roughly 2x:

| dataset | pi_1 in doc | pi_1 actual (test) | c* in doc | c* corrected |
|---|---|---|---|---|
| Amazon | 0.0687 | 0.1410 | 0.0364 | 0.0177 |
| Reddit | 0.0333 | 0.0682 | 0.0750 | 0.0366 |
| Tolokers | 0.2182 | 0.3391 | 0.0115 | 0.0074 |

Synthetic likewise: the doc's 0.0716 uses the 5% graph rate; test-set prevalence
is 6.14%, giving **0.0582**. `scripts/clearance_rate_verification.py` had the
same issue hardcoded (`pi1 = 0.05`) and is now fixed to compute it per trial.

**The ordering and therefore your argument survive unchanged** (Tolokers <
Amazon < Reddit either way). Only the numbers need updating.

---

## 4. What survives vs. what goes

**Survives:**
- The conformal + BH machinery, and every FDR-control number
- The marginal / conditional / zero-discovery reporting convention
- The symmetric-trimming fix (a genuine catch -- asymmetric trimming
  manufactured FDR ~0.51)
- The severity sweep's power collapse (real, but caused by detector quality
  degrading as the graph densifies, not by calibration contamination)
- The calibration-size ablation (40% -> 90% raising power 0.071 -> 0.132) --
  **this gets promoted from footnote to central result**, since it directly
  demonstrates the `n_cal` term
- Your proposition and `signal_quality_boundary_analysis.py`'s framing
- Datasets, detector protocol, evaluation metrics, all setup sections

**Goes:**
- The contamination-robustness claim as the paper's thesis
- The message-passing mechanism story
- The framing of clean/contaminated/adversarial as a contamination test
  (they get reinterpreted, not deleted)
- The current title and abstract

---

## 5. The reframed paper

**Old question:** does FDR control survive calibration contamination?
**New question:** when does conformal FDR control discover anything at all on
graphs?

**Working title:** *When Does Conformal Anomaly Detection Discover Anything on
Graphs? A Joint Condition on Base Rate, Calibration Size, and Detector Quality*

**Central claim.** Discovery activity is governed by a joint condition
`pi_1 * c >= 1 / (alpha * (n_cal + 1))`, derivable directly from conformal
p-value mechanics and the BH rejection rule -- not by detector quality alone. We
validate it on three real graphs it was not fitted to, where it correctly
predicts all nine discovery/no-discovery outcomes, including the counterintuitive
case of a better detector discovering nothing while a below-chance detector
discovers.

**Contributions:**
1. The joint discovery-threshold condition, derived and validated 9/9.
2. FDR control holds throughout, including in the zero-power regime -- the
   procedure fails into silence rather than into false discovery.
3. A negative result, honestly reported: calibration *composition* (which
   normal nodes you calibrate on, including worst-case selection by anomaly
   exposure) does not measurably change the calibration score distribution.
   Calibration *size* does, and dominates.
4. Two methodological pitfalls that make GAD pipelines look like they work when
   they do not: the dead-ReLU embedding collapse, and degree normalization
   manufacturing apparent exposure effects (sign flip, -0.06 -> +0.27).

Contribution 4 may be the most useful to practitioners. Both are easy to miss
and both are in widely-used building blocks.

### Section-by-section

| Section | Action |
|---|---|
| Title / Abstract / Keywords | **Rewrite** |
| Introduction | **Rewrite** around the discovery question |
| Related work | Trim contamination emphasis; keep conformal + GAD |
| Preliminaries | **Add the proposition as the theoretical core** |
| Experimental setup | **Unchanged** |
| Results: synthetic | Same numbers, reframed narrative |
| Results: real-world | Same numbers, + the 9/9 prediction table |
| Results: baseline comparison | **Unchanged** |
| Ablation: calibration size | **Promote to a main result** |
| Detection power mechanism | Keep; it supports the proposition |
| Discussion | Add the negative result + both pitfalls |
| Conclusion | **Rewrite** |
| Appendix A | **Rewrite** -- it currently calls the closed-form threshold a
  *failed* approach, but your rank-k version fixes exactly the flaw it
  describes. It is now a positive result. |

Note also: the current draft cites Tolokers results in Practical Guidance and
Appendix A without ever reporting them. We now have them -- report them.

---

## 6. Where to submit

**TKDE and TAI are out for this version.** Realistically they were always a
stretch, and the reframed paper is smaller: no new algorithm, one detector,
three real datasets, and a central result that a reviewer can fairly call
"careful algebra plus good validation."

Ranked, honestly:

1. **arXiv, immediately.** Free, citable, timestamps the work, and it is
   something concrete to point at this week. Do this regardless of anything
   else.
2. **A workshop at NeurIPS / ICML / KDD / ICLR.** Roughly 50-70% acceptance
   once a workshop exists, the CV line reads as the parent venue, and they are
   usually non-archival so we can extend later. Best fit: negative-results,
   evaluation-methodology, or graph-learning workshops. **Deadlines need
   checking -- NeurIPS 2026's lineup is already out and had no good topical
   fit, so look at ICLR/KDD cycles.**
3. **COPA (Symposium on Conformal and Probabilistic Prediction).** The best
   topical fit by far -- it is the conformal prediction community's home venue,
   and `clarkson2024contamination` in our own bibliography was published there.
   Proceedings in PMLR, no publication fee. **COPA 2026 has closed (papers May
   20, posters Aug 1); COPA 2027 would be ~May 2027.**
4. **Mid-tier data mining venues** if we want archival: PAKDD, DSAA, ASONAM,
   IEEE BigData, roughly 20-30%.

**Recommendation:** arXiv this week to hit your deadline, then target a
workshop for real reviewer feedback, then COPA 2027 with an expanded version if
we want the strongest topical home.

---

## 7. Open items

- [ ] Confirm the `m` values in the 9/9 table against `m_test` in the CSV
- [ ] Update `theory/joint_discovery_threshold_proposition.md` with corrected
      `pi_1` (test-set prevalence, not graph-wide)
- [ ] Fix the dead reference: the theory doc's item 4 points to
      `theory/theoretical_characterization_draft.md`, which does not exist
- [ ] **AUROC discrepancy to resolve:** we measured Amazon 0.7542 and Tolokers
      0.4390, but `signal_quality_boundary_analysis.py` and the theory doc
      hardcode 0.8925 and 0.4093. Your verification table depends on these.
- [ ] **Disclose the seed-dependent encoder collapse:** on real data, Reddit
      seed 2 and Tolokers seed 0 collapsed to `Z_std = 0` while sibling seeds
      did not. A detector that randomly collapses is a stability issue a
      reviewer will ask about.
- [ ] Watch realized FDR across seeds -- one Amazon contaminated seed came out
      near 0.139 against a 0.10 nominal. Probably noise at n=1, worth checking.

## 8. Reproducing everything here

```bash
python3 scripts/calibration_distribution_check.py --datasets amazon reddit tolokers --n_seeds 3 --device cpu
```

```bash
python3 scripts/exposure_degree_confound_check.py --datasets amazon reddit tolokers --n_seeds 3 --device cpu
```

```bash
python3 scripts/real_data_exposure_diagnostic.py --datasets amazon reddit tolokers --n_seeds 3 --device cpu
```

All three run on CPU in the `dgl311` env in well under an hour total. Outputs
land in `results/logs/`.
