# Machine-checked combinatorial core (Lean 4 + Mathlib)

This directory contains a Lean 4 formalization of the deterministic, finite
statements that the paper's theory rests on. It is meant as a companion to the
paper, not a replacement for its proofs: the probabilistic layer (exchangeability
making every position of a pooled score vector equally likely, Strassen's
coupling, and the Benjamini–Yekutieli PRDS theorem) is **not** formalized. What
is formalized is every counting and arithmetic step, so the only things a reader
has to take on trust are those three standard results.

All statements live in `ConformalFDR/Basic.lean`. The build has no `sorry`, and
`#print axioms` on every theorem reports only the three standard axioms
(`propext`, `Classical.choice`, `Quot.sound`); see `AxiomCheck.lean`.

## Map from the paper

| Paper | Lean |
|---|---|
| calibration rank r(v), p-value p̂(v) = r(v)/(n_cal+1) | `rank`, `pval` |
| Fact 1 (resolution floor 1/(n_cal+1), attained iff the score beats every calibration score) | `one_le_rank`, `rank_le_succ`, `pval_ge_floor`, `rank_eq_one_iff`, `pval_le_one` |
| rank is nonincreasing in the test score and nondecreasing in the calibration scores (the step inside the proof of the selection theorem) | `rank_antitone_score`, `rank_mono_calib`, `pval_mono_calib` |
| selection theorem, deterministic core: under the coupling, {p̂_cf ≤ t} ⊆ {p̂ ≤ t} | `pval_le_of_coupling` |
| Lemma (null-rank): P(r(v) ≤ r) ≤ r/(n_cal+1), with equality absent ties — counting form over the pooled set of n_cal+1 scores | `card_looRank_le` (≤, tie-robust), `card_looRank_eq` (=, distinct scores), `card_pval_le_eq` (exact grid law ⌊t(n_cal+1)⌋/(n_cal+1)) |
| Fact 2 (BH crossing characterization) | `bh_nonempty_iff` |
| Corollary (floor-only discovery condition, r = 1) | `bh_nonempty_of_floor`, `bh_nonempty_of_floor_split` |
| "BH at level α controls only up to γα": rescaling identity | `bhK_smul`, `countLe_smul` |
| contamination remark: anomalies at the top of calibration raise every attainable rank; adding a calibration point above the test score raises its p-value | `rank_ge_of_dominated`, `rank_snoc`, `pval_snoc_ge` |

`bhK p α` is the Benjamini–Hochberg rejection count, defined without sorting as
the largest k ≤ m with at least k p-values below αk/m (equivalent to
max{k : p_(k) ≤ αk/m}); `bhK_spec` shows the count satisfies the rejection
condition and `bhK_pos_iff` characterizes a nonempty discovery set.

## What the formalization forced us to make precise

Writing the statements down at this level of detail surfaced four places where
the paper's wording is looser than the mathematics:

1. **Ties go the wrong way for the lower bound.** `card_looRank_le` is the
   tie-robust direction (P(r(v) ≤ r) ≤ r/(n_cal+1)). The selection theorem's
   lower bound needs the *equality* `card_looRank_eq`, which requires distinct
   scores. So the theorem should assume almost-surely distinct scores (true for
   the continuous detector scores used), rather than saying the tie case is
   "conservative".
2. **Strict inequality cannot hold for every t ∈ (0,1].** `pval_le_one` gives
   P(p̂ ≤ 1) = 1 for any calibration set, and `pval_ge_floor` gives
   P(p̂ ≤ t) = 0 for t < 1/(n_cal+1). Strictness can only be claimed on the grid
   points 1/(n_cal+1), …, n_cal/(n_cal+1).
3. **The γα step is a rescaling, not a Benjamini–Yekutieli correction.**
   `bhK_smul` is the whole deterministic content: BH at level α on p equals BH
   at level γα on γ·p. The remaining step is the PRDS guarantee for BH applied to
   the super-uniform p-values γ·p.
4. **Fact 2's converse needs N(r) ≥ 1.** `bh_nonempty_iff` derives it from the
   right-hand side being strictly positive; the paper's proof uses it silently.

## Building

```
cd formal
lake exe cache get   # downloads Mathlib's compiled oleans (~several GB)
lake build
lake env lean AxiomCheck.lean
```

Toolchain: `leanprover/lean4:v4.33.1`, Mathlib tag `v4.33.1` (pinned in
`lakefile.toml` and `lake-manifest.json`).
