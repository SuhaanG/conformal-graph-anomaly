# Status — 2026-08-19

Short, current, and honest. `PAPER_REFRAME_HANDOFF.md` has the full history;
`theory/joint_discovery_threshold_proposition.md` has the maths and every
result in detail. This file is the "where are we actually" page.

---

## The one-line version

We have a strong, reproducible empirical finding and no confirmed theorem. The
finding alone is publishable. The theorem is still open and one experiment away
from resolution either way.

---

## What is solid

**The headline result.** Constructing a calibration set by topological
filtering breaks conformal FDR control on real graphs, badly:

    median realized FDR = 0.613   against nominal 0.10
    7 of 13 discovering cells above 0.50
    per dataset: amazon 0.900, tolokers 0.635, weibo 0.634, reddit 0.167

This is the *clean* condition — the one Proposition 1 proves is exchangeable.
The two conditions without proofs (contaminated, adversarial) sit at or below
nominal. Measured across 5 detectors spanning three architectural families and
4 real graphs, 100 trials.

**Supporting evidence.**
- Synthetic: clean FDR 0.132 vs nominal 0.10 (d=0.837, p=0.0007, 20 seeds),
  while contaminated 0.091 and adversarial 0.086 are both *below* nominal.
- Degree-matched calibration sampling closes about half the synthetic gap
  (d 0.837 -> 0.449) without touching scores.
- Degree normalization makes it worse, not better (conditional FDR 40-44%).
- Clustering coefficient: clean null, and we know why (it is flat on that
  generator, 0.0049-0.0051).
- AdaDetect baseline runs and is now valid (the pre-`9b1eff5` numbers were
  produced by the broken detector and are discarded).

**Infrastructure that would be a contribution in its own right.**
`src/selection_bias.py` (gamma estimator + simulated exchangeable null),
`scripts/selection_bias_matrix.py`, `scripts/degree_sensitivity_sweep.py`,
`scripts/pygod_architecture_check.py`, `scripts/calibration_distribution_check.py`.

---

## What is NOT solid

**Part 4's mechanism (degree-tilt) is unresolved.** Not refuted, not confirmed.

- Restricted to datasets where its own precondition (A1) holds, measured at the
  BH operating point: rho=+0.81, p=0.0149, n=8 cells.
- But drops to rho=+0.50, p=0.3910 without `dominant_pygod`, because across all
  100 trials *every* cell with score-degree correlation > 0.5 is that one
  detector. Coverage problem, not a subtle statistical one.
- The corrected matrix rerun gives the strongest within-detector result to the
  statistic measured where BH actually cuts: `gamma_t0.01` 5/5 right-signed,
  sign test p=0.0312; block permutation p=0.0547. Suggestive, straddling 0.05.
- **A warning sign:** in the first beta sweep, the one usable column
  (`gamma_t0.05`) crossed gamma=1 at sdeg=+0.65, not at sdeg=0. If that holds
  under the extended sweep, the mechanism is wrong — the procedure would be
  valid while the score is still strongly degree-dependent.

**Weibo is unexplained and is probably a second finding.** FDR 0.634 with no
degree tilt at all ((A1) fails there: 0/25 seeds monotone, tau=-0.02) and
negative score-degree correlation on three of five detectors. Something
dataset-level breaks exchangeability that has nothing to do with degree.

**Synthetic is near-vacuous.** A working detector hits AUROC 1.000 on the
current generator, so FDR claims there are made in a perfect-separation regime.
Harden the generator or stop leaning on synthetic.

**No demonstrated fix.** Every remedy tried either fails (degree
normalization), only half-works (degree matching), or "works" by making zero
discoveries. Weighted conformal (Tibshirani et al. 2019) with weights ~ 1/q(d)
is the principled candidate and has not been implemented.

---

## Running right now

`degree_sensitivity_sweep.py --dataset amazon` (job 3, `beta_amazon2.log`).

This is the decisive test. It manipulates degree sensitivity directly via
`score / log1p(degree)**beta` on one detector, one graph, one fixed calibration
frame — causal, not observational. Read the last ~25 lines when it finishes.

**The number that matters is `gamma at sdeg = 0`.** Part 4 predicts ~1.0.

| verdict printed | meaning |
|---|---|
| SUPPORTS | gamma ~ 1 at sdeg=0. Mechanism holds; Part 4 can become Theorem 2. |
| NOT SUPPORTED | still anti-conservative with a degree-neutral score. Degree is not the channel. Revise Part 4. |
| OVERSHOOT | conservative at sdeg=0; beta distorts more than degree dependence. |
| INCONCLUSIVE | sweep never reached sdeg=0. Extend `--betas` and rerun. |

Then the same on `tolokers` — one graph is not enough either way.

---

## Next, in priority order

1. **Read `beta_amazon2.log`.** Everything below branches on it.
2. **Run tolokers.** `--dataset tolokers --n_seeds 5 --device cuda`.
3. **If SUPPORTS:** write Theorem 2, then implement weighted conformal as the
   fix. Paper becomes mechanism + theorem + remedy.
4. **If NOT SUPPORTED:** drop the degree explanation, keep the empirical
   finding, and pivot the theory section to "we proposed a mechanism and
   falsified it." Still publishable, one tier lower.
5. **Either way:** investigate weibo, and copy every CSV into
   `results/published/` with an index row.
6. **Start writing.** The Overleaf draft still contains claims we know are
   false — see handoff §6.4. That work is independent of 1-4 and is currently
   the bottleneck.

---

## Venue, honestly

**TKDE is out.** It wants a novel method or a theorem, and we have neither
confirmed. Even with the beta sweep landing, ~15-20%. That is the lottery we
agreed to avoid.

**Realistic targets, given the >50% bar:**

| outcome of the sweep | target | estimate |
|---|---|---|
| SUPPORTS | **TNSE** | 45-55% |
| SUPPORTS | TAI | ~60% |
| NOT SUPPORTED | **TAI** | ~50% |
| NOT SUPPORTED | TETCI | ~60% |

**TAI clears the bar in both branches.** It is the safe answer, it is Q1, and
its scope (uncertainty quantification, trustworthy AI) fits a conformal-
prediction-reliability paper directly. TNSE is the upgrade, available only if
the mechanism confirms.

What carries the paper at TAI even without a theorem: a documented,
reproducible failure of a guarantee people rely on, measured across 5 detectors
and 5 datasets, with a mechanism proposed and honestly tested. "We falsified
our own hypothesis" reads well to referees — it is rarer than it should be.

The thing that would move this up a tier is not another dataset. It is the
weighted-conformal fix working. Mechanism + theorem + remedy is a TNSE paper;
mechanism + theorem alone is borderline; empirical finding alone is TAI.

---

## Working notes for whoever picks this up

Three times this session a printed verdict was wrong and the error was in the
measurement, not the data:

1. `gamma_hat` and `ks_uniform` were floored by their own grid endpoint, so
   every conservative detector returned identical boundary values.
2. `min_rank = n_calib//4` moved the measurement 112x further into the bulk
   than BH actually cuts, which inverted the conclusion.
3. `Spearman(sdeg, gamma)` along a beta sweep is vacuous — `Spearman(beta,
   sdeg) = -1.0000` by construction, so anything monotone in beta correlates.

Each was caught by checking the statistic against a case with a known answer
(an exchangeable simulation, an analytic gamma, a planted effect). **Do that
before believing any verdict this codebase prints**, including the ones it
prints after these fixes.
