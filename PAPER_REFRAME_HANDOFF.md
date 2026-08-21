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

**The project changed direction on 2026-08-20. Read section 2A before anything
else -- the premise the paper is named after turned out not to be what any
experiment in this repo was measuring.**

| | Status |
|---|---|
| Contamination-robustness claim | **DEAD, and it was never tested.** No condition in the codebase ever put an anomaly in calibration (2A.1) |
| **Calibration SELECTION breaks FDR control** | **CONFIRMED on real data, causally, at matched frames.** gamma 13.22 vs 0.82 on amazon (2A.2). This is the paper |
| **The score-gap law** | **CONFIRMED.** Spearman(gap, gamma) = -0.73, p=9.1e-04 over 17 cells, predicts sign and magnitude (2A.3) |
| **True contamination is HARMLESS** | **CONFIRMED.** 5% and 10% real anomalies in calibration stay exchangeable (2A.2) |
| **Benchmark AUROC largely measures degree** | **CONFIRMED. 16 of 20 cells cannot beat a degree lookup** (2A.4) |
| **The mechanism is CONDITIONAL on the detector** | **CONFIRMED by controlled experiment** (2A.4b). Same filter, same 7x degree gap, gamma 13.22 for pygod vs 0.76 for gae |
| **A working configuration exists** | **YES.** gae + unfiltered calibration on amazon: FDR 0.059, 101 discoveries, power 0.116 (2A.4b) |
| Discovery-threshold proposition (Part 1) | **VALID** as a sufficient condition, and it explains why true contamination is safe |
| Part 4 degree-tilt theorem | **SUPERSEDED** by Part 7. Degree was one channel, not the mechanism |
| Detector power under valid calibration | **DEPENDS ON THE DETECTOR.** 0 for pygod on amazon; 101 (power 0.116, FDR 0.059) for gae (2A.4b) |
| Synthetic study | **VACUOUS.** AUROC 1.0 is a degree artifact of p_aa=0.3 (2A.4) |
| Current paper draft | **NOT SUBMITTABLE.** Wrong premise plus the factual errors in 6.4 |
| Venue | AISTATS (Oct deadline) as the real attempt, TNSE as the fallback. NeurIPS D&B deadline is ~May 2027, too far. Section 8 |

---

## 2A. What happened on 2026-08-20 (READ FIRST)

Four findings, in dependency order. Everything is in
`theory/joint_discovery_threshold_proposition.md` Parts 6 and 7 with full
numbers; this is the orientation.

### 2A.1 The project never studied calibration contamination

Every condition in `real_data_experiment.py` draws calibration from
`eligible_normal_idx`, a subset of `normal_idx = np.where(labels == 0)`:

    clean         calib_idx = clean_pool                       (labels == 0)
    contaminated  calib_idx = rng.choice(eligible_normal_idx)  (labels == 0)
    adversarial   calib_idx = top_exposed[:n_calib]            (labels == 0)

**No anomaly has ever entered a calibration set in this repo.** In Bates et al.
2023 and AdaDetect, "contaminated calibration" means the reference sample
CONTAINS OUTLIERS. That experiment had never been run.

What the three conditions actually vary is which NORMALS are selected, by
exposure (fraction of anomalous neighbours): clean is exposure==0,
contaminated is population-rate, adversarial is maximal. It is a
covariate-selection experiment wearing a contamination label.

This retroactively explains section 4: the exposure->score channel we spent
weeks measuring (|r| < 0.03 on four datasets) is a second-order message-passing
effect. There was no contamination to detect because none was ever injected.

### 2A.2 The real finding: the selection RULE breaks exchangeability

Under the clean condition, calibration = {normal : exposure == 0} and
test-normals = its complement. **They are disjoint in exposure by
construction.** Conformal validity requires S|calib ==d S|test, so if the score
responds to exposure or to anything correlated with it, validity fails by
design.

`scripts/calibration_strategy_comparison.py` tests this by varying ONLY the
selection rule -- same test set, same n_calib, same p-value floor, invariants
asserted at runtime. Amazon, dominant_pygod, 5 seeds:

    strategy         calib_deg  test_deg   gap(d)   gamma   mean_p   disc    fdr
    clean                105.6     737.2   -1.252   13.22   0.1438   1527  0.787
    random               761.4     737.2   -0.005    0.82   0.5007      0  0.000
    exposed_only         728.1     737.2   +0.013    1.01   0.5067      0  0.000
    true_contam_05       727.2     737.2   -0.010    0.72   0.4976      0  0.000
    true_contam_10       732.2     737.2   +0.023    0.75   0.5077      0  0.000
    random_full (n=4000) 742.5     737.2   +0.002    1.16   0.5003      0  0.000

**The clean filter selects calibration at 1/7th the test population's degree.**
Every other rule -- including calibration containing 10% ACTUAL ANOMALIES --
lands within 2% of the test set and is exchangeable. gamma is computed from
null p-values, so this does not depend on discovery counts.

The quotable form: *clean calibration is valid only if the score is insensitive
to exposure, which is exactly when contamination was never a threat. The
strategy is valid when it is unnecessary and invalid when it is needed.*

**A prediction we got wrong, informatively.** `exposed_only` is NOT broken
(gamma 1.01). Filtering per se does not hurt; filtering that SHIFTS a
score-relevant covariate does. On amazon nearly every high-degree node has an
anomalous neighbour, so "exposed" is almost the whole population while
"unexposed" is the low-degree fringe.

### 2A.3 The score-gap law, and why it is deployable

Across every matched-frame cell so far -- 3 datasets, 2 detectors, 6 selection
rules, 17 cells -- the standardized score gap between calibration and
test-normals predicts the violation, including its SIGN:

    Spearman(gap_d, gamma) = -0.7285,  p = 9.11e-04

    amazon  clean         gap -1.252  (calib 7x LOWER degree)   gamma 13.22
    weibo   clean         gap -0.078                            gamma  1.42
    reddit  gae clean     gap -0.065                            gamma  1.04
    ...     matched rules |gap| < 0.05                          gamma ~1
    weibo   exposed_only  gap +0.145  (calib HIGHER degree)     gamma  0.23
    reddit  exposed_only  gap +0.435  (calib 5x HIGHER degree)  gamma  0.31

Calibration scoring lower than test -> anti-conservative. Higher ->
conservative. Matched -> valid. Degree and exposure are both merely routes to a
gap; neither is privileged. **This supersedes Part 4's degree-specific
theorem.**

**Be honest about what is and is not a contribution.** gap_d and gamma are both
functions of the same two score distributions, so "shifted scores produce
shifted p-values" is near definitional and a referee will say so. The
contributions are:

  1. the selection rule causes a gap of unguessable magnitude (7x degree gap
     from a rule that sounds prudent);
  2. true contamination does NOT cause one;
  3. **the gap is computable WITHOUT LABELS** -- a two-sample statistic on
     scores you already have at deployment. That makes it an operational
     precondition check, not a post-hoc diagnosis, and it is the practical
     payload the paper should lead with.

### 2A.4 Benchmark AUROC may be measuring node degree

The strongest and least expected finding, and the one that could reach Tier 1.

Sweeping anomaly-anomaly edge density with FEATURES HELD IDENTICAL
(`scripts/synthetic_difficulty_sweep.py`):

    p_aa     E[deg|anom]  E[deg|norm]  ratio   detector AUROC   degree-only AUROC
    0.005       32.2         72.8      0.44       0.0793            0.0330
    0.010       36.0         72.8      0.49       0.1038              --
    0.020       43.5         72.8      0.60       0.0661              --
    0.050       66.0         72.8      0.91       0.3337            0.3690
    0.100      103.5         72.8      1.42       0.8597            0.8192
    0.300      253.5         72.8      3.48       1.0000            1.0000

AUROC crosses 0.5 exactly where the anomaly/normal degree ratio crosses 1.0,
and **an untrained degree lookup reproduces the trained detector's curve.**
dominant_pygod's score-degree Spearman is +0.918 on amazon. The AUROC 1.0000
that every synthetic result in this project rests on is not detection -- it is
degree ranking coinciding with the planted block structure. Flip p_aa and the
same detector on the same features scores 0.066.

**RESOLVED. `degree_baseline_check.py` ran: 16 of 20 detector-dataset cells
fail to beat an untrained degree lookup by more than 0.02 AUROC.**

    dataset    best free baseline         best detector           gap
    amazon     0.7446 (neg_degree)        0.8589 (gae)          +0.1143
    reddit     0.5561 (degree)            0.5769 (dominant_ours) +0.0207
    tolokers   0.5596 (degree)            0.5618 (ocgnn)        +0.0022
    weibo      0.7782 (neg_degree)        0.7733 (dominant_ours) -0.0049

On tolokers and weibo NO detector clears the bar. The claim "benchmark AUROC in
graph anomaly detection substantially measures node degree" is now supported by
measurement, not conjecture. Full table in theory doc Part 8.

Note what it also revealed: dominant_pygod -- the detector behind every Part
6/7 result -- itself LOSES to the baseline on 3 of 4 datasets (amazon 0.383 vs
0.745). That raised an obvious referee question, which 2A.4b answers.

### 2A.4b The controlled experiment: the mechanism is REAL and CONDITIONAL

Running the strategy comparison on amazon with `gae` instead of
`dominant_pygod` isolates the mechanism, because the two detectors have
opposite degree profiles on an otherwise identical setup:

                            dominant_pygod        gae      ratio
        calib_deg                  105.6        106.3       1.0x
        test_deg                   737.2        771.3       1.0x
        degree ratio               0.143        0.138       1.0x
        sdeg (score~degree)       +0.902       -0.026      34.7x
        gap_d (score gap)         -1.252       -0.038      32.9x
        gamma                      13.22         0.76      17.4x

**The clean filter produces the SAME 7x degree gap for both.** That part is a
property of the graph and the rule. Whether it becomes a validity failure
depends entirely on whether the score responds to degree -- it does for pygod
(broken), it does not for gae (valid).

This is the causal chain demonstrated with the covariate shift HELD FIXED and
only the score's sensitivity varying. It does not weaken the selection-bias
finding; it completes it, and it pre-empts the referee objection rather than
leaving it open.

The correct statement of the claim is therefore: **calibration filtering is
CONDITIONALLY dangerous, and the condition is measurable.** That conditionality
is why the label-free score-gap diagnostic (2A.3) is the paper's deliverable --
you cannot tell from the filter alone whether you are in trouble.

**And it gives the project its first working configuration:**

    gae + random_full on amazon:
        n_calib=4000   gamma=0.96   disc=101   FDR=0.059   power=0.116

Valid (FDR below the nominal 0.10), useful (11.6% of the 821 anomalies), with a
detector that beats the degree baseline by +0.114 AUROC. Every earlier
configuration was either broken (pygod/clean, FDR 0.787) or found nothing. This
is the existence proof that keeps the paper from being purely cautionary.

### 2A.5 The uncomfortable result -- NOW PARTLY RESOLVED, see 2A.4b

> This section was written before the gae run. Its conclusion holds for
> dominant_pygod but NOT in general: gae under valid calibration achieves 101
> discoveries at FDR 0.059 on amazon. Read 2A.4b. What survives is that a
> degree-proxy detector has no real power once its validity failure is removed.

Under a VALID calibration rule, dominant_pygod finds almost nothing:

  - amazon, `random_full` at n_calib=4000 (bh_min_rank ~7, so only 7 test
    points need to beat the whole calibration set): **0 discoveries**
  - reddit, gae, `random_full`: **3 discoveries**, FDR 0.132, power 0.005

So clean's 1527 amazon "discoveries" at FDR 0.787 (~325 true, ~1202 false) were
manufactured by the validity failure. The degree gap let nearly every test node
outrank the low-degree calibration set, BH fired en masse, and the true hits
were along for the ride.

**Power without validity was an illusion of the broken guarantee.** The paper's
claim must be about VALIDITY, not detection performance. That is defensible for
a conformal paper -- validity is the product -- but do not oversell power.

### 2A.6 New scripts, and what each is for

    scripts/calibration_strategy_comparison.py
        THE central experiment. Varies only the calibration selection rule at
        matched n_calib / test set / floor. Six strategies including
        true_contam_05/10 (real anomalies injected) and random_full (unmatched
        n, the deployment-realistic arm). Runtime frame-invariant assertions.

    scripts/degree_baseline_check.py
        Trained detectors vs an untrained degree lookup. The Tier-1 decider.

    scripts/synthetic_difficulty_sweep.py
        Sweeps p_aa (NOT feature_shift -- see below) to find a non-trivial
        detection regime, with a matched clean-vs-random comparison per level.

    scripts/degree_sensitivity_sweep.py
        Manipulates score-degree dependence directly via score/log1p(deg)**beta
        on a fixed frame. Superseded as a mechanism test by the strategy
        comparison, but its beta curve is still the cleanest demonstration that
        no score-level degree correction can fix the problem.

    src/selection_bias.py
        gamma estimator, simulated exchangeable null, adaptive t grid,
        score-degree dependence, empirical q(d). 13 tests in
        tests/test_selection_bias.py.

### 2A.7 Traps in this codebase, learned the hard way

**Six measurement errors produced confident wrong verdicts today.** Every one
was caught by testing the statistic against a case with a known answer. Do this
before believing anything this codebase prints, including code written after
these fixes.

  1. **Zero-discovery cells contaminating averages.** BH scores a
     zero-discovery trial as FDR=0. This has now faked a positive result FIVE
     separate times: degree normalization (6.3), the beta sweep, the strategy
     comparison verdict, the difficulty sweep, and the original Method B/C.
     **Never average an FDR over cells with no discoveries.**
  2. **gamma measured where BH does not operate.** `min_rank = n_calib//4` put
     the measurement 112x further into the bulk than BH's actual threshold on
     weibo. Use `left_tail_gamma` with `adaptive_t_grid`.
  3. **A fixed t that the calibration size cannot resolve.** At n_calib=267,
     t=0.01 covers calibration ranks 1-2 and collapses to 0.00.
  4. **Correlating two monotone functions of a swept parameter.**
     Spearman(beta, sdeg) = -1.0000 by construction, so anything monotone in
     beta correlates with sdeg. That produced a bogus "SUPPORTS" verdict.
  5. **argsort-based AUROC with tied scores.** Returns 1.0 on all-tied input.
     Degree is heavily tied, so this would have corrupted 2A.4 specifically.
     Fixed in `degree_baseline_check.py` via scipy `rankdata`;
     `synthetic_difficulty_sweep.py` and `severity_sweep_pygod_instrumented.py`
     still carry the old version and should be updated.
  6. **Citing a number without checking its provenance.** The weibo
     exposure->score r=0.111 quoted repeatedly was measured on
     DEGREE-NORMALIZED scores; raw is 0.045, and the strategy run measures
     +0.016. A story was built on it before anyone checked.

### 2A.8 What to run next, in priority order

    DONE  degree_baseline_check.py           -> 16/20 cells lose to degree (2A.4)
    DONE  strategy comparison, amazon + gae  -> mechanism is conditional (2A.4b)

    DONE  strategy comparison, tolokers + gae -> conditionality REPLICATES

          tolokers' clean filter is even harsher than amazon's -- a 13x degree
          gap (calib 5.7 vs test 73.8) against amazon's 7x -- yet with gae
          (sdeg -0.172) the score gap stays at +0.052 and gamma stays valid at
          0.28, with random at 0.99. On the SAME graph dominant_pygod
          (sdeg +0.909) gives gamma@BH 9.61 and FDR 0.635. Two graphs, same
          pattern: the covariate gap is a property of the filter, the validity
          failure is a property of the detector.

          Sign check passes again: sdeg NEGATIVE with low-degree calibration
          means calibration scores slightly HIGHER, predicting conservative --
          and gamma is 0.28.

          HONEST LIMIT: gae on tolokers finds NOTHING even at n_calib=4000. The
          validity result replicates; the usability result does not. amazon
          remains the only graph with a configuration that is both valid and
          useful.

    1. Same run on weibo (--dataset weibo --detector gae). Third graph.

    2. Widen the score-gap law past 17 cells -- anomalydae and ocgnn on the
       datasets already run. It is the paper's central quantitative claim.

    3. Fix the argsort AUROC in synthetic_difficulty_sweep.py and
       severity_sweep_pygod_instrumented.py (2A.7 item 5). Ties are mishandled.

    4. Copy every CSV into results/published/ with an index row.

    5. WRITE. The draft is the bottleneck. Section 7's rewrite plan predates
       2A entirely and needs redoing.

**Do not** add more detectors or datasets hoping for a better headline. The
empirical breadth already exceeds most GAD papers. What is missing is writing,
and a faculty collaborator (section 8.8).

---

## 2B. Writing the TMLR paper (Gopal — this is your section)

**Venue decided: TMLR.** Not TNSE, not AISTATS, not IEEE. Rationale in 8.1.
Short version: TMLR's acceptance criteria are (1) are the claims supported by
evidence and (2) would some subset of the ML audience care. **Novelty is
explicitly not required.** This paper's weakness is novelty and its strength is
rigor, so TMLR is the venue whose bar it clears most comfortably (~40-50% vs
~25% at TNSE).

**DO NOT port the IEEE draft.** Its premise is falsified (2A.1) and every
results table in it came from the broken detector. Start from the TMLR
template and pull pieces across deliberately, per 2B.3.

### 2B.1 Title and abstract

    Calibration Selection, Not Contamination, Breaks Conformal
    FDR Control in Graph Anomaly Detection

The abstract must state the claims TMLR will check evidence against. Draft:

> Conformal prediction converts graph anomaly scores into p-values with a
> finite-sample FDR guarantee, provided the calibration set is exchangeable
> with normal test data. A natural precaution on graphs is to build the
> calibration set from nodes with no anomalous neighbours, avoiding
> contamination propagated by message passing. We show this precaution is the
> failure mode. Filtering on neighbourhood exposure is a covariate filter: on
> Amazon it draws calibration at one seventh the test population's mean degree,
> and a detector whose score correlates with degree inherits that shift,
> driving realized FDR to 0.787 against a nominal 0.10. Holding the graph, the
> filter, and the resulting degree shift fixed while changing only the detector
> removes the failure entirely, isolating detector covariate-sensitivity as the
> operative condition; the pattern replicates on a second graph. Injecting
> ACTUAL anomalies into calibration at 5% and 10% leaves exchangeability
> intact, so the failure the literature guards against is benign while the
> standard guard is not. We give a label-free diagnostic: the standardized
> score gap between calibration and test predicts the violation and requires no
> ground truth. Finally, on four standard benchmarks, 16 of 20
> detector-dataset pairs fail to beat an untrained node-degree lookup by more
> than 0.02 AUROC. We identify one configuration that is both valid and useful,
> and report plainly that its usefulness does not generalize.

### 2B.2 Section skeleton

    1  Introduction          the instinct to filter; why it is a covariate
                             filter; contributions as checkable claims
    2  Related Work          PORT from IEEE draft, reframe positioning
    3  Preliminaries         PORT. Keep Prop 1 but state that the selection
                             rule VIOLATES its precondition -- Prop 1 is not
                             contradicted, it is inapplicable under filtering
    4  Selection-Induced Non-Exchangeability
                             NEW. The chain: selection -> covariate shift ->
                             (x score sensitivity) -> score gap -> gamma
    5  An Extended Discovery Condition
                             PORT Lemma 1 / Prop 2 / Cor 1 / Remarks 1-2
                             intact. Add: this explains why true contamination
                             is safe (injected anomalies raise the floor)
    6  Experimental Setup    matched-frame protocol, gamma estimator,
                             simulated exchangeable null, frame invariants
    7  Results               6 subsections, see 2B.4
    8  Discussion            what we claim vs do not; power is weak; the claim
                             is about VALIDITY
    9  Conclusion

### 2B.3 What to port from the IEEE draft, and what to burn

**PORT MOSTLY INTACT**
  - Section 2 Related Work (reframe the positioning paragraph only)
  - Section 3 Preliminaries through the p-value construction
  - Proposition 1 and its discussion
  - Section 3.1 "An Extended Discovery Condition" ENTIRE -- Lemma 1,
    Proposition 2, Corollary 1, Remarks 1 and 2. This is correct work.
  - Algorithm 1
  - Appendix A entirely (closed-form threshold, PRDS testing, e-BH scale
    mismatch). Real contributions, honestly reported.
  - Multiple Testing Correction subsection
  - Reproducibility statement (now points at a real repo)

**BURN**
  - Title, abstract, introduction -- premise falsified
  - EVERY results table in Section 5 -- broken-detector numbers
  - "Fails into silence" everywhere (abstract, results, conclusion, Figure 1)
  - The Yelp and Tolokers exclusion paragraphs -- both since resolved
  - Table `related-comparison` "Contam. -> Yes" for Ours -- never tested
  - Conclusion -- rests on the false claim

**FIX IF PORTED**
  - Amazon AUROC stated as 0.893; measured 0.7542
  - `\markboth{...TKDE...}` -- irrelevant in TMLR, delete
  - Author block and ORCID -- TMLR is ANONYMOUS at submission.
    Non-anonymous submissions are rejected without review.
  - `\bibliography{tmlr}` in the template -> `\bibliography{references}`
  - Re-add the macros (\ncal, \pval, \calibset, ...) -- Section 5 needs them

### 2B.4 Results section, subsection by subsection

    7.1  Selection filtering breaks FDR control
         Amazon matched-frame table (2A.2). Lead with calib_deg 105.6 vs
         test_deg 737.2 -- the 7x gap is the most legible single fact.
    7.2  The failure is conditional on the detector
         pygod vs gae on the SAME graph and filter (2A.4b), plus the tolokers
         replication. This is the strongest evidence in the paper; give it room.
    7.3  True contamination is harmless
         true_contam_05/10 stay exchangeable. Connect to Section 5: the
         resolution floor is why.
    7.4  The score gap as a label-free diagnostic
         The law + the argument that it is computable without labels.
         STATE HONESTLY that gap -> gamma is near definitional.
    7.5  Benchmark AUROC largely tracks node degree
         16/20 table + the p_aa sweep showing AUROC crossing 0.5 exactly where
         the degree ratio crosses 1.0, with features held identical.
    7.6  A working configuration, and its limits
         gae + unfiltered on amazon works; the same on tolokers finds nothing.
         Do not bury the second half.

### 2B.5 Every number you may cite, with its source

Do not cite a number that is not in this table or in a committed CSV.

| Quantity | Value | Source |
|---|---|---|
| Synthetic clean FDR | 0.132 +/- 0.037, d=0.837, p=0.0007 | `condition_comparison_pygod.csv` |
| Synthetic contaminated | 0.091 +/- 0.026 | same |
| Synthetic adversarial | 0.086 +/- 0.026 | same |
| Amazon clean, pygod: calib_deg | 105.6 | `calibration_strategy_amazon_dominant_pygod.csv` |
| Amazon test_deg | 737.2 | same |
| Amazon clean gap_d | -1.252 | same |
| Amazon clean gamma | 13.22 | same |
| Amazon clean discoveries / FDR | 1527 / 0.787 | same |
| Amazon random gamma | 0.82 | same |
| Amazon true_contam_05 gamma | 0.72 | same |
| Amazon true_contam_10 gamma | 0.75 | same |
| Amazon pygod sdeg | +0.902 | same |
| Amazon gae: calib_deg / test_deg | 106.3 / 771.3 | `calibration_strategy_amazon_gae.csv` |
| Amazon gae gap_d / gamma | -0.038 / 0.76 | same |
| Amazon gae sdeg | -0.026 | same |
| Amazon gae random_full | n=4000, gamma 0.96, 101 disc, FDR 0.059, power 0.116 | same |
| Tolokers gae: calib_deg / test_deg | 5.7 / 73.8 | `calibration_strategy_tolokers_gae.csv` |
| Tolokers gae gap_d / gamma | +0.052 / 0.28 | same |
| Tolokers gae random gamma | 0.99 | same |
| Tolokers gae sdeg | -0.172 | same |
| Degree baseline: cells failing to beat | 16 of 20 | `degree_baseline_check.csv` |
| amazon best free baseline | 0.7446 (neg_degree) | same |
| reddit / tolokers / weibo baselines | 0.5561 / 0.5596 / 0.7782 | same |
| best detector per dataset | 0.8589 gae / 0.5769 / 0.5618 / 0.7733 | same |
| (A1) monotone, amazon/reddit/tolokers | 25/25 seeds, tau ~ -0.92 | `selection_bias_matrix.csv` |
| (A1) monotone, weibo | 0/25 seeds, tau -0.02 | same |
| p_aa sweep AUROC | 0.0793 -> 1.0000 across p_aa 0.005 -> 0.3 | `synthetic_difficulty_dominant_pygod.csv` |
| degree-only AUROC, same sweep | 0.0330 -> 1.0000 | theory doc Part 7 |

**BLOCKER: five of these CSVs are NOT in the repo yet.** They exist only in
`results/logs/` on the H200. Until they are copied to `results/published/` and
committed, Gopal cannot verify any number sourced from them, and the paper
would be citing figures with no checkable provenance -- the exact problem
`results/published/README.md` exists to prevent.

    PRESENT   condition_comparison_pygod.csv
    PRESENT   selection_bias_matrix.csv          (regenerate first, see 2A.7)
    TO COPY   calibration_strategy_amazon_dominant_pygod.csv
    TO COPY   calibration_strategy_amazon_gae.csv
    TO COPY   calibration_strategy_tolokers_gae.csv
    TO COPY   degree_baseline_check.csv
    TO COPY   synthetic_difficulty_dominant_pygod.csv

On the H200, after the runs finish:

    cp results/logs/*.csv results/published/
    git add results/published/ && git commit -m "Add result CSVs" && git push

Then add an index row for each in `results/published/README.md` naming the
claim it backs. **Do this before writing Section 7.**

**NEEDS RECOMPUTING BEFORE USE:** the score-gap law (Spearman -0.7285,
p=9.11e-04, 17 cells). It used `gamma_hat` from the pre-endpoint-fix matrix,
which had 18 rows floored at exactly 1.000000. Recompute from the regenerated
CSV. A `mean_p`-based version is unaffected and can be used immediately.

### 2B.6 Claims you must NOT make

Each of these was in the IEEE draft and each is false:

  - "Realized FDR stays at or below nominal in every condition." Clean is
    0.132, significantly ABOVE (p=0.0007).
  - "The procedure fails into silence." Dead -- under the correct detector the
    severity sweep gives AUROC 1.0 and power 1.0 at every level.
  - "We study calibration contamination." No anomaly ever entered a
    calibration set in the original conditions (2A.1).
  - "The clean condition results are consistent with Proposition 1 in every
    trial." Inverted -- clean is the condition that breaks.
  - Amazon AUROC 0.893. Measured 0.7542.
  - Anything asserting the detectors usefully detect anomalies. Under valid
    calibration, power is 0.116 at best and zero on most graphs.

And two framing rules:

  - **Claim VALIDITY, not performance.** The paper is about when the guarantee
    holds. It is not a detection-performance paper and cannot survive being
    read as one.
  - **State the near-definitional gap yourself**, in Section 4, before a
    reviewer finds it. TMLR rewards that; hiding it is what gets papers
    rejected there.

### 2B.7 TMLR mechanics

  - **Anonymous.** No names, no ORCID, no acknowledgements, no "our previous
    work". Non-anonymous submissions are rejected without review.
  - Rolling submission, no deadline. Do not rush.
  - Reviews are public on OpenReview -- assume the submission is visible.
  - Action-editor model: one AE plus reviewers, decisions on the two criteria
    above rather than on a score threshold.
  - The repo is a genuine asset. Full history, tests, and a record of
    falsifying our own hypothesis twice. Point at it in Reproducibility.

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

> **SUPERSEDED BY 2A.1, and read that first.** This section is still accurate
> as a record of what was measured, but the conclusion is now understood
> differently. We spent weeks measuring an exposure->score channel and finding
> it near zero. The reason is not that contamination is harmless in general --
> it is that **no anomaly was ever placed in a calibration set**, so there was
> no contamination present to have an effect. The conditions differ in which
> NORMALS they select, not in contamination. Keep the measurements; discard the
> framing.

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

> **STILL VALID, and it gained a second use.** Beyond predicting when discovery
> is possible, the floor condition explains why TRUE contamination is safe
> (2A.2): injecting anomalies into calibration puts them at the top of the
> calibration scores, which raises the minimum achievable p-value to
> (eps*n+1)/(n+1) and pushes bh_min_rank beyond what any test set can supply.
> True contamination is safe *because Part 1 blocks it*. That ties the two
> halves of the theory document together.

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

**Rewritten 2026-08-20. Earlier versions of this section said IEEE was out,
then said TAI, then TNSE. Those were all written before section 2A. The honest
current read is below, and it is CONDITIONAL on one experiment.**

### 8.1 DECIDED: TMLR

Both gating experiments ran (2A.4, 2A.4b) and the venue question is settled.
**Target TMLR.** The writing guide is section 2B.

| venue | fit | estimate | deadline |
|---|---|---|---|
| **TMLR** | **Best.** Judges correctness and interest, NOT novelty -- which is exactly this paper's weak/strong split | **~40-50%** | rolling |
| IEEE TNSE | Good scope, but competing on novelty and prestige where we are weakest | ~25% | rolling |
| AISTATS | Wants theory; ours is the near-definitional part | ~10-15% | ~Oct |
| NeurIPS D&B | 2A.4 would suit it, but next deadline is ~May 2027 | n/a | dead cycle |
| IEEE TAI / TETCI | Fallbacks if TMLR rejects | ~35-40% | rolling |

**Why TMLR over TNSE, concretely.** This paper's weakness is novelty -- the
gap-to-gamma link in 2A.3 is near definitional and any reviewer who knows
conformal prediction will see it. Its strengths are rigor, reproducibility,
honest self-correction, and empirical breadth. TMLR's stated criteria are
exactly (1) claims supported by evidence and (2) some subset of the audience
would care. It is the one venue where the paper is judged on its strengths
rather than its weakness.

Costs, stated plainly: no impact factor, weaker recognition outside ML, and
submissions are public on OpenReview. For an ML-for-Science MS the reviewers
are ML people who know TMLR, so the recognition cost is small in that context.

**Fallback order if TMLR rejects:** TNSE, then TAI. TMLR reviews are public and
substantive, so a rejection still improves whatever goes out next.

### 8.2 What changed the estimate upward

Before 2A.4b this was a cautionary null: "this fails, and fixing it reveals the
method never worked." After 2A.4b it is a conditional result with an existence
proof -- the mechanism is real, the condition is measurable without labels, and
one configuration is both valid and useful (gae + unfiltered calibration, FDR
0.059 at power 0.116). "Here is when it breaks, here is how to check, here is a
setting that works" is a materially stronger paper.

### 8.3 What would move it further, in order of value

  1. **A faculty co-author.** Fixes arXiv endorsement, tightens the theory
     section, and changes how referees read a submission from unaffiliated
     students. Worth more than any remaining experiment.
  2. **Replication of 2A.4b on weibo and reddit.** One run each.
  3. **Writing.** The bottleneck, and it has been for weeks.

### 8.4 What is off the table, and why

  - **NeurIPS / ICML / ICLR main track**: no novel method, no proven theorem.
    Under 5%. The Part 4 theorem was superseded by Part 7, and Part 7's core
    link is near definitional (2A.3).
  - **TPAMI**: same reason, higher bar.
  - **TKDE**: ~15-20%. It wants a method. Ruled out earlier in this project for
    exactly this reason and nothing since has changed it.
  - **IEEE Access / MDPI**: would accept, but committees know the bar is low.
    Not worth it given the work is genuinely better than that tier.

### 8.5 Framing for an ML-for-Science application

Frame this as **reliability of AI-driven discovery**, not as graph anomaly
detection. FDR control IS the false-discovery problem in scientific screening --
the same statistical machinery used in genomics and drug screening. "Can we
trust the discoveries an AI pipeline reports?" is a first-class SciML question,
and the answer here -- *not unless you check the calibration gap, and here is
the label-free check* -- is directly usable by other people.

That framing makes both NeurIPS D&B and TNSE read as on-topic for a SciML MS,
rather than as a methods detour.

### 8.6 COPA, still the best topical fit

The Symposium on Conformal and Probabilistic Prediction remains where the
conformal-prediction community actually reads.
`clarkson2024contamination`, in our bibliography, was published there.
Proceedings to PMLR, no publication fee. These are the only reviewers who will
engage seriously with the exchangeability argument rather than skim it.
**COPA 2026 has closed; 2027 is around May.**

### 8.7 The arXiv blocker, unchanged

We do not have arXiv endorsement, and cs.LG requires it for authors without
prior submissions in the category. Two routes, neither instant: get endorsed
(most naturally via a UIUC faculty member, who would also improve the paper and
change how reviewers read a submission from unaffiliated students), or get
accepted somewhere first. Until then the GitHub repo is the citable artifact --
it is public, has full history, and everything in this document reproduces from
it.

### 8.8 Honest overall assessment

The work is genuinely good and it is not the work we set out to do. What exists
now is: a documented failure mode in a procedure people rely on, a quantitative
law with sign prediction, a label-free diagnostic anyone can run, a
counterintuitive negative control, and a serious question about whether a whole
benchmark measures what it claims.

What does NOT exist: a method, a proven theorem, or a demonstration that any of
these detectors usefully work. Power under valid calibration is near zero
(2A.5), and no amount of writing will hide that from a referee. The paper must
claim validity, not performance.

Two structural handicaps worth stating plainly: no faculty co-author, and no
arXiv presence. Both are fixable and both matter more than one more experiment.
Getting a UIUC faculty member involved would do more for the outcome than
anything else on the next-steps list.

---

## 9. What to do next

### 9.1 Immediate -- for Gopal

**Read section 2A first. The premise changed.** The paper is no longer about
contamination; it is about calibration SELECTION. Sections 4 and 5 are still
accurate as records but carry superseded framing, and both now have pointers
saying so.

In priority order:

- [ ] **Run `degree_baseline_check.py`** (~15 min). This single run decides the
      venue, because it decides whether the Tier-1 framing in 8.2 exists. Do it
      before anything else.

      python scripts/degree_baseline_check.py --n_seeds 3 --device cuda

- [ ] **Run the strategy comparison with `ocgnn`** on reddit and weibo. It is
      the only detector that both discovers and controls FDR on two datasets
      (weibo 15.6 discoveries at FDR 0.041; reddit 2.6 at 0.108). If random
      calibration improves on clean for it, that is the demonstrated remedy the
      paper currently lacks.

- [ ] **Widen the score-gap law past 17 cells** -- tolokers, plus anomalydae
      and gae on the datasets already run. The law is the paper's central
      quantitative claim and 17 cells is thin.

- [ ] **Fix the argsort AUROC** in `synthetic_difficulty_sweep.py` and
      `severity_sweep_pygod_instrumented.py` (2A.7 item 5). It mishandles ties.

- [ ] **Copy every CSV into `results/published/`** with an index row. Several
      numbers cited in this document still have no committed file behind them.

- [ ] **Start writing.** The draft is the bottleneck, not the experiments. It
      still contains the factual errors in 6.4 AND now a premise we know is
      wrong. Section 7's rewrite plan needs redoing against 2A.

**Do not** run more detectors or datasets hoping for a better result. The
empirical breadth already exceeds most GAD papers. What is missing is writing
and, more than any experiment, a faculty collaborator (8.8).

### 9.2 To reach IEEE Transactions level (months)

> **PARTLY OBSOLETE.** Items 1 and 2 are done. Item 3 (formalise the
> proposition into a theorem) was attempted as Part 4 and superseded by Part 7,
> whose central link is near definitional -- so "write the theorem" is no
> longer the path to a better venue. See section 8, which is rewritten.


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
