# Paper reframe: what we found, what survives, and what to write

**For: Gopal. From: Suhaan K. Updated 2026-08-17.**

Short version: the contamination-robustness claim does not hold up, and we
should not submit it. But your discovery-threshold proposition is now validated
on **12 out of 12** dataset x condition cells across four real datasets it was
never fitted to, and it is strong enough to carry the paper on its own. The
experiments are done. The numbers do not change. This is a rewrite, not a
re-run.

**What changed since the 08-16 version** (details in sections 2e, 3, 9):
- **The contamination question is now closed definitively.** PyGOD's DOMINANT
  implementation is correct where ours is broken, and even with its working
  encoder achieving AUROC 0.99-1.00, exposure still does not correlate with
  score (max |r| = 0.032, nothing significant). So the null is not an artifact
  of our bug -- the mechanism genuinely does not operate.
- **A second headline claim is now suspect.** PyGOD's AUROC *rises* with
  contamination severity (0.9865 -> 1.0000) where ours *fell* (0.9107 ->
  0.7619), so the severity sweep's "fails into silence" power collapse is
  most likely also an artifact of the dead encoder.
- **Weibo added**: proposition now 12/12, and degree normalization measured
  rather than inherited.
- **The Yelp blocker is fixed and verified**, so Yelp is finally runnable.

---

## 1. What broke, in one paragraph

The paper claims GNN message passing propagates anomalous influence into the
scores of neighboring normal nodes, contaminating the calibration set, and that
FDR control survives this. We tested whether that contamination actually
reaches the scores. It does not -- on synthetic data, on any of the four real
datasets, and (critically) not even under a correct detector implementation
that achieves near-perfect AUROC. The clean / contaminated / adversarial
conditions are not producing meaningfully different calibration sets, so the
comparison cannot support a contamination-robustness claim. Full technical
detail is in `DETECTOR_DIAGNOSTIC.md`; the decisive evidence is below.

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
| weibo | clean | 5620 | 0.0000 | 342.72 | 1358.81 | 0.0173 | 0.0365 |
| weibo | contaminated | 4000 | 0.0318 | 363.71 | 1383.68 | 0.0163 | 0.0000 |
| weibo | adversarial | 4000 | 0.0642 | 399.15 | 1431.07 | 0.0144 | 0.0000 |

### 2a. The selection logic works -- so this is a real negative, not a bug

`calib_exp` separates correctly everywhere: 0 for clean, base rate for
contaminated, higher for adversarial. The conditions genuinely select different
nodes. They just do not produce different *score distributions*.

### 2b. The quantity that drives discovery barely moves

`clear_anom` (fraction of true anomalies beating the entire calibration set) is
**identical across all three conditions** on Amazon, Reddit and Tolokers. So is
`max` -- on Reddit it is literally identical to four decimals across all three
conditions. The extreme tail that sets the p-value floor is untouched by which
calibration condition you choose.

**Weibo is the one partial exception, and it is worth stating honestly.** There,
`q95` (342.7 -> 363.7 -> 399.1), `max` (1358.8 -> 1383.7 -> 1431.1) and
`clear_anom` (0.0173 -> 0.0163 -> 0.0144) all move *monotonically* with
exposure, in the direction the contamination hypothesis predicts. The magnitude
is small (clearance falls 17% relative) and it changes no discovery outcome --
all three conditions are still fully explained by the proposition, see section 3
-- but it is the single place in five datasets where the effect appears at all.

Note the direction argument does not obviously reduce to the degree confound
here: Weibo is sparse in the Reddit sense (70% of normals have zero exposure),
which would predict *positive* exposure~degree, hence adversarial selecting
high-degree nodes, hence *lower* normalized scores. We observe higher.
**`exposure~degree` on Weibo has not been measured yet** -- one run of
`scripts/exposure_degree_confound_check.py --datasets weibo` closes this, and it
is the last loose end in the contamination analysis. Report it either way.

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

### 2f. The bug is ours; PyGOD's implementation is correct

We checked whether the reference implementation shares the flaw. It does not.
`torch_geometric`'s `BasicGNN.forward` gates the activation behind

    if i < self.num_layers - 1 or self.jk_mode is not None:

so on the final conv no activation is applied and the embedding stays linear.
PyGOD's `DOMINANTBase` delegates its encoder to that backbone, and is therefore
structurally immune. Measured on the identical graph:

| | frac_zero | std | min | negatives? |
|---|---|---|---|---|
| ours | **1.0000** | 0.00000000 | 0 | no |
| PyGOD | **0.0000** | 0.05182356 | **-0.219214** | **yes** |

Negative values are impossible after a ReLU, so this is proof rather than
inference. Source reading and measurement agree.
(`scripts/pygod_architecture_check.py`)

### 2g. The contamination null is NOT an artifact of our bug

This is the result that closes the question. We ran PyGOD's correct
implementation on the same synthetic graphs, sweeping severity
(`scripts/pygod_exposure_check.py`):

| p_an | mean exposure | PyGOD AUROC | exposure r | p |
|------|---------------|-------------|------------|---|
| 0.002 | 0.021 | 0.9865 | +0.0320 | 0.0895 |
| 0.010 | 0.096 | 0.9998 | +0.0137 | 0.5291 |
| 0.050 | **0.342** | **1.0000** | **-0.0234** | 0.3315 |

The encoder is alive (Z std ~0.05) and detection is near-perfect, and exposure
*still* does not correlate -- max |r| = 0.032, nothing significant, and at the
highest contamination it turns negative, exactly as our broken detector did.
**We removed the confound and the effect is still absent.** The contamination
framing cannot be rescued by fixing or swapping the detector.

### 2h. A SECOND headline claim is now suspect: "fails into silence"

Note the direction in the table above. PyGOD's AUROC *rises* with contamination
severity (0.9865 -> 0.9998 -> 1.0000). Ours *fell* (0.9107 -> 0.7619).

The severity sweep's power collapse -- the "fails into silence rather than into
false discovery" finding, one of the paper's headline claims -- is therefore
most likely **also an artifact of the dead encoder degrading detection as the
graph densifies**. With AUROC 1.0 the clearance rate would be near-total,
comfortably clearing `bh_min_rank ~ 44`, so discoveries would occur rather than
collapse to zero.

Treat that claim as unsupported until re-tested. It is the second headline
finding to fall, and it must not survive into the rewrite unexamined.

### 2i. A complication for any synthetic follow-up

AUROC **1.0000** means the synthetic generator is trivially easy for a working
detector: `feature_shift=1.0` across 16 dimensions with functioning message
passing gives perfect separation. Re-running the synthetic study with a correct
detector would therefore produce no interesting regime -- everything gets
discovered. The generator would need to be made materially harder first.

Practical consequence: **the paper should lead with real data**, where results
were measured on the actual pipeline and where the proposition's validation
lives, and treat synthetic as illustrative with the caveats in 2e-2h stated
plainly.

---

## 3. Your proposition: 12 out of 12 on real data

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
| weibo | clean | 5620 | 2704 | **5** | **6.0** | discovery (barely) | 33-35 | ✓ |
| weibo | contaminated | 4000 | 4324 | 11 | 5.7 | none | 0 | ✓ |
| weibo | adversarial | 4000 | 4324 | 11 | 5.0 | none | 0 | ✓ |

Note the Tolokers and Weibo rows: Tolokers predicts 19 needed against 20
clearing and observes ~21 discoveries; Weibo predicts 5 needed against 6.0
clearing. Both are tight at the boundary, not merely directionally right --
and in both cases the *neighbouring* condition on the same dataset falls the
other side of the threshold and correctly produces zero.

Weibo also adds a case the earlier three did not: a dataset where clean has a
LARGER calibration pool than the 4000 cap (5620), so it discovers while
contaminated and adversarial do not. Same mechanism, opposite direction from
Amazon and Tolokers, still predicted correctly.

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
vs 4000, and a correspondingly smaller test set. Weibo reproduces the same
pattern independently (5620 vs 4000).

### A diagnostic signature worth reporting

On Amazon, **all 20 adversarial seeds return bit-identical results** --
`n_discoveries=60`, `realized_fdr=0.000`, `power=0.073`, `std = 0.000` exactly.
Twenty independently trained models, one outcome.

That is explained by the dead encoder: with `Z=0` the score reduces to feature
magnitude plus a rowsum term, neither of which depends meaningfully on the
training seed, and adversarial selection is deterministic (top-4000 by exposure,
no RNG). Contaminated *does* vary (FDR 0.000 to 0.146) because its calibration
draw is random -- the RNG is the only source of variation left.

Two reasons this belongs in the paper: a reviewer will ask why a standard
deviation is exactly zero across 20 seeds, and it is a cheap diagnostic anyone
can run on their own pipeline. Also worth stating plainly that two contaminated
seeds individually exceed the nominal 0.10 (0.139, 0.146) -- legitimate, since
FDR control bounds the average, but better disclosed than discovered.

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

One more input correction while updating that doc: it lists Weibo's anomaly
rate from the PyGOD docstring, but the loaded graph is **8,405 nodes / 347
anomalies = 4.13%**, not the ~10.3% quoted. Use the measured value.

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
- The calibration-size ablation (40% -> 90% raising power 0.071 -> 0.132) --
  **this gets promoted from footnote to central result**, since it directly
  demonstrates the `n_cal` term
- Your proposition (now 12/12) and `signal_quality_boundary_analysis.py`'s framing
- Datasets, detector protocol, evaluation metrics, all setup sections
- All real-data results: they were measured on the actual pipeline and
  reproduce exactly (Amazon re-ran bit-identically, twice)

**Goes:**
- The contamination-robustness claim as the paper's thesis
- The message-passing mechanism story
- The framing of clean/contaminated/adversarial as a contamination test
  (they get reinterpreted, not deleted)
- The current title and abstract

**Now in doubt -- do NOT carry forward unexamined:**
- **"Fails into silence" / the severity-sweep power collapse.** See 2h. PyGOD's
  AUROC *rises* with severity where ours fell, so the collapse is most likely an
  artifact of the dead encoder. This was contribution #2 in the 08-16 version of
  this document and has to be either re-tested or dropped.
- Any synthetic-data claim that depends on detector behaviour rather than on
  the conformal procedure itself.

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
validate it on four real graphs it was not fitted to, where it correctly
predicts all twelve discovery/no-discovery outcomes, including the
counterintuitive case of a better detector discovering nothing while a
below-chance detector discovers.

**Contributions:**
1. The joint discovery-threshold condition, derived and validated **12/12**
   across four real datasets, tight at the boundary in two of them.
2. A negative result, honestly reported and hard to obtain: calibration
   *composition* -- which normal nodes you calibrate on, including worst-case
   selection by anomaly exposure up to 53% of neighbours -- does not measurably
   change the calibration score distribution. Calibration *size* does, and
   dominates. Established across five datasets **and** under a correct detector
   implementation reaching AUROC 1.00, so it is not an artifact of a weak
   scorer.
3. Two methodological pitfalls that make GAD pipelines look like they work when
   they do not: an encoder that silently collapses to zero while AUROC stays
   at 0.91, and degree normalization manufacturing an apparent exposure effect
   (sign flip, -0.06 -> +0.27).

Contribution 3 may be the most useful to practitioners: **AUROC does not verify
that your graph pathway is doing anything.** Ours reported 0.91 with every
structural signal inert. Nothing in a standard evaluation would catch it, and
the diagnostic is one line once you know to look.

Note contribution 2 in the 08-16 version of this document was "fails into
silence"; it has been removed pending re-test (see section 4).

### Section-by-section

| Section | Action |
|---|---|
| Title / Abstract / Keywords | **Rewrite** |
| Introduction | **Rewrite** around the discovery question |
| Related work | Trim contamination emphasis; keep conformal + GAD |
| Preliminaries | **Add the proposition as the theoretical core** |
| Experimental setup | **Unchanged**, except: delete the propagation sentences in
  §Datasets -- they describe `propagate_contamination()`, which no experiment
  calls. That is a methods section describing a procedure not in the code. |
| Results: synthetic | Reframe, and caveat per 2h/2i |
| Results: real-world | **Lead with this.** Same numbers, + the 12/12 table |
| Results: baseline comparison | **Unchanged** |
| Ablation: calibration size | **Promote to a main result** |
| Detection power mechanism | Keep; it supports the proposition |
| Discussion | Add the negative result + both pitfalls + the zero-variance
  diagnostic |
| Conclusion | **Rewrite** |
| Appendix A | **Rewrite** -- it currently calls the closed-form threshold a
  *failed* approach, but your rank-k version fixes exactly the flaw it
  describes. It is now a positive result. |

Other factual corrections the current draft needs regardless of framing:
- **§Datasets describes an experiment that was not run** (the propagation
  sentences above). Highest-priority fix -- this is the category of error that
  causes real problems if found post-publication.
- **The abstract asserts the mechanism operates** ("message passing propagates
  their influence... shorting those nodes' scores"). It does not. Also
  "shorting" should read "shifting".
- **Amazon AUROC is stated as 0.893**; we measure **0.7542**. Tolokers is
  stated 0.4093; we measure 0.4390. `signal_quality_boundary_analysis.py`
  hardcodes the old values and your verification table depends on them.
- **Practical Guidance and Appendix A cite Tolokers results never reported.**
  We now have them -- report them.
- The intro says the graph setting introduces contamination "considered in many
  non-graph settings", which concedes the novelty; it should say *distinct
  from*.

---

## 6. Where to submit

**TKDE and TAI are both still out of reach for this version, and the new
results did not change that** -- though they did change *why*.

The 12/12 validation and the PyGOD-confirmed null are genuinely stronger
evidence than we had on 08-16. But neither addresses the objection that caps
us: the proposition is a few lines of algebra combining the conformal p-value
floor with the BH rejection rule. It is correct, useful, and well-validated, and
a reviewer will still call it an observation rather than a theorem. More
validation cells make the evidence better; they do not make the result deeper.

The 08-17 results also cost us something. "Fails into silence" was contribution
#2 and is now suspect (2h), and the synthetic study is compromised by a detector
whose graph pathway was inert. So the paper gained rigour and lost a claim --
roughly a wash on venue, with the honest version being smaller and more solid.

There were two paths to TKDE and both are now closed:
- *"A widely-used detector is silently broken"* -- closed. PyGOD is correct;
  the bug is ours (2f).
- *"Contamination is real and we characterize it"* -- closed. The null holds
  even under a correct detector at AUROC 1.00 (2g).

Ranked, honestly:

1. **arXiv, immediately.** Free, citable, timestamps the work, and it is
   something concrete to point at this week. Do this regardless of anything
   else, and note it does **not** block a journal submission later.
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
   20, posters Aug 1); COPA 2027 would be ~May 2027.** A COPA audience is the
   one that will actually engage with the p-value floor argument, the PRDS
   testing, and the e-BH appendix -- at a general venue those go unread.
4. **Mid-tier data mining venues** if we want archival: PAKDD, DSAA, ASONAM,
   IEEE BigData, roughly 20-30%.
5. **IEEE TAI**, only if we later add 2-3 more detectors and a formalized
   version of the proposition. That is months, not weeks, and it is the only
   route back to an IEEE Transactions.

**Recommendation:** arXiv this week to hit the deadline, then a workshop for
real reviewer feedback, then COPA 2027 as the strongest topical home if we want
to expand it.

---

## 7. Open items

- [ ] Confirm the `m` values in the 12/12 table against `m_test` in the CSV
- [ ] **Measure `exposure~degree` on Weibo** -- the last loose end (2b). Weibo
      is the only dataset where the calibration distribution moves with
      exposure, and the degree explanation does not obviously fit its direction.
      One run: `exposure_degree_confound_check.py --datasets weibo`
- [ ] **Re-test or drop "fails into silence"** (2h). It was a headline claim
      and is now suspect.
- [ ] Update `theory/joint_discovery_threshold_proposition.md`: corrected
      `pi_1` (test-set prevalence, not graph-wide), Weibo's real anomaly rate
      (4.13%, not 10.3%), and add the Weibo rows to the verification table
- [ ] Fix the dead reference: the theory doc's item 4 points to
      `theory/theoretical_characterization_draft.md`, which does not exist
- [ ] **AUROC discrepancy to resolve:** we measured Amazon 0.7542 and Tolokers
      0.4390, but `signal_quality_boundary_analysis.py` and the theory doc
      hardcode 0.8925 and 0.4093. Your verification table depends on these.
- [ ] **Disclose the encoder collapse and its zero-variance signature.** On real
      data, Reddit seed 2 and Tolokers seed 0 collapsed to `Z_std = 0` while
      sibling seeds did not, and all 20 Amazon adversarial seeds returned
      bit-identical results.
- [ ] Yelp: `real_data_experiment.py` and `calibration_distribution_check.py`
      both need `--use_sparse_prop` (verified bit-identical on Amazon). Running
      it takes the proposition to 15/15. Measure `degree_norm` for Yelp first --
      it currently inherits Amazon's `True` without ever having been checked.
- [ ] AdaDetect on Amazon -- built and gate-verified, not yet run at scale.
      Caveat when reporting: it consumes our DOMINANT scores, whose graph
      pathway is inert, so it compares calibration *procedures* on a degraded
      score.

## 8. Reproducing everything here

The environment has a **CUDA torch build and an H200** despite
`setup_dgl_env.sh` requesting the CPU index -- use `--device cuda`.

```bash
~/envs/dgl311/bin/python scripts/calibration_distribution_check.py --datasets amazon reddit tolokers weibo --n_seeds 3 --device cuda
```

```bash
~/envs/dgl311/bin/python scripts/exposure_degree_confound_check.py --datasets amazon reddit tolokers weibo --n_seeds 3 --device cuda
```

```bash
~/envs/dgl311/bin/python scripts/pygod_architecture_check.py --n_nodes 3000 --epochs 100
```

```bash
~/envs/dgl311/bin/python scripts/pygod_exposure_check.py --n_nodes 3000 --epochs 100 --seeds 3 --p_an 0.002 0.01 0.05
```

Minutes each on GPU. Outputs land in `results/logs/`.

Note `pygod_dominant_collapse_check.py` (the detector-wrapper version) needs
`pyg-lib` or `torch-sparse`, and `torch-sparse` will not import without
`torch-scatter`. Use `pygod_architecture_check.py` instead -- it bypasses the
NeighborLoader entirely and answers the same question.
