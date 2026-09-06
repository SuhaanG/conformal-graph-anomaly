import Mathlib

/-!
# Machine-checked combinatorial core of
"Calibration Selection, Not Contamination, Breaks Conformal FDR Control
 in Graph Anomaly Detection"

Everything here is deterministic (finite counting and arithmetic); no measure
theory is used. The probabilistic statements in the paper reduce to these facts
plus two standard ingredients that are NOT formalized: (i) exchangeability makes
every position of a pooled score vector equally likely, and (ii) Strassen's
theorem supplies the coupling used in the proof of the selection theorem.

Map from the paper to this file:

* `rank`, `pval`                  the calibration rank r(v) and p̂(v) = r(v)/(n_cal+1)
* Fact 1 (resolution floor)       `one_le_rank`, `rank_le_succ`, `pval_ge_floor`,
                                  `rank_eq_one_iff`, `pval_le_one`
* rank monotonicity               `rank_antitone_score`, `rank_mono_calib`, `pval_mono_calib`
  (the deterministic step inside the proof of the selection theorem and inside
   the contamination remark)
* Lemma (null-rank), counting form
                                  `card_looRank_le`  (≤ r positions have rank ≤ r; tie-robust)
                                  `card_looRank_eq`  (= r when scores are distinct)
                                  `card_pval_le_eq`  (the exact grid law ⌊t(n+1)⌋/(n+1))
* Fact 2 (BH crossing)            `bh_nonempty_iff`
* Corollary (floor-only case)     `bh_nonempty_of_floor`, `bh_nonempty_of_floor_split`
* BH rescaling (γα consequence)   `bhK_smul`
* Contamination remark            `rank_ge_of_dominated`, `pval_snoc_ge`
* Selection theorem, deterministic core
                                  `pval_le_of_coupling`
-/

open Finset

namespace ConformalFDR

noncomputable section

/-! ## Calibration rank and conformal p-value -/

section Rank

variable {n : ℕ}

/-- Calibration rank of a test score `s` against calibration scores `c`:
one plus the number of calibration scores that are at least `s`
(ties count against the test point, as in the paper). -/
def rank (c : Fin n → ℝ) (s : ℝ) : ℕ :=
  (univ.filter fun i => s ≤ c i).card + 1

/-- Conformal p-value `rank / (n + 1)`. -/
def pval (c : Fin n → ℝ) (s : ℝ) : ℝ :=
  (rank c s : ℝ) / (n + 1)

lemma one_le_rank (c : Fin n → ℝ) (s : ℝ) : 1 ≤ rank c s := by
  unfold rank; omega

lemma rank_le_succ (c : Fin n → ℝ) (s : ℝ) : rank c s ≤ n + 1 := by
  unfold rank
  have h := card_filter_le (univ : Finset (Fin n)) (fun i => s ≤ c i)
  simp only [card_univ, Fintype.card_fin] at h
  omega

/-- Fact 1: the p-value is never below `1/(n+1)`. -/
lemma pval_ge_floor (c : Fin n → ℝ) (s : ℝ) : 1 / (n + 1 : ℝ) ≤ pval c s := by
  unfold pval
  gcongr
  exact_mod_cast one_le_rank c s

lemma pval_le_one (c : Fin n → ℝ) (s : ℝ) : pval c s ≤ 1 := by
  unfold pval
  rw [div_le_one (by positivity)]
  exact_mod_cast rank_le_succ c s

/-- Fact 1, equality case: the floor `1/(n+1)` is attained exactly when the test
score exceeds every calibration score. -/
lemma rank_eq_one_iff (c : Fin n → ℝ) (s : ℝ) : rank c s = 1 ↔ ∀ i, c i < s := by
  unfold rank
  constructor
  · intro h i
    have h0 : (univ.filter fun i => s ≤ c i).card = 0 := by omega
    rw [card_eq_zero, filter_eq_empty_iff] at h0
    exact not_le.mp (h0 (mem_univ i))
  · intro h
    have h0 : (univ.filter fun i => s ≤ c i) = ∅ := by
      rw [filter_eq_empty_iff]
      intro i _
      exact not_le.mpr (h i)
    rw [h0, card_empty]

/-- The rank is nonincreasing in the test score. -/
lemma rank_antitone_score (c : Fin n → ℝ) {s s' : ℝ} (h : s ≤ s') :
    rank c s' ≤ rank c s := by
  unfold rank
  have hsub : (univ.filter fun i => s' ≤ c i) ⊆ (univ.filter fun i => s ≤ c i) := by
    intro i hi
    simp only [mem_filter, mem_univ, true_and] at hi ⊢
    exact h.trans hi
  have := card_le_card hsub
  omega

/-- The rank is nondecreasing in the calibration scores (pointwise). -/
lemma rank_mono_calib {c c' : Fin n → ℝ} (h : ∀ i, c i ≤ c' i) (s : ℝ) :
    rank c s ≤ rank c' s := by
  unfold rank
  have hsub : (univ.filter fun i => s ≤ c i) ⊆ (univ.filter fun i => s ≤ c' i) := by
    intro i hi
    simp only [mem_filter, mem_univ, true_and] at hi ⊢
    exact hi.trans (h i)
  have := card_le_card hsub
  omega

lemma pval_mono_calib {c c' : Fin n → ℝ} (h : ∀ i, c i ≤ c' i) (s : ℝ) :
    pval c s ≤ pval c' s := by
  unfold pval
  gcongr
  exact_mod_cast rank_mono_calib h s

/-- Deterministic core of the selection theorem: under a coupling in which every
shifted calibration score lies below the corresponding counterfactual
(exchangeable) score, the event `{counterfactual p-value ≤ t}` is contained in
`{shifted p-value ≤ t}`. Combined with the exact grid law `card_pval_le_eq` for
the counterfactual p-value and Strassen's coupling, this gives
`P(p̂(v) ≤ t) ≥ ⌊t(n+1)⌋/(n+1)`. -/
theorem pval_le_of_coupling {c c' : Fin n → ℝ} (h : ∀ i, c i ≤ c' i) (s t : ℝ)
    (ht : pval c' s ≤ t) : pval c s ≤ t :=
  (pval_mono_calib h s).trans ht

/-- Contamination, deterministic core: if a set `A` of calibration points all score at
least `s`, the rank of `s` is at least `|A| + 1`. Anomalies sitting at the top of the
calibration set therefore raise the smallest rank a normal test point can attain. -/
lemma rank_ge_of_dominated (c : Fin n → ℝ) (s : ℝ) (A : Finset (Fin n))
    (hA : ∀ i ∈ A, s ≤ c i) : A.card + 1 ≤ rank c s := by
  unfold rank
  have hsub : A ⊆ univ.filter fun i => s ≤ c i := by
    intro i hi
    simp only [mem_filter, mem_univ, true_and]
    exact hA i hi
  have := card_le_card hsub
  omega

/-- Appending one calibration score changes the rank by one exactly when the new
score is at least the test score. -/
lemma rank_snoc (c : Fin n → ℝ) (a s : ℝ) :
    rank (Fin.snoc c a : Fin (n + 1) → ℝ) s = rank c s + if s ≤ a then 1 else 0 := by
  unfold rank
  rw [card_filter, card_filter, Fin.sum_univ_castSucc]
  simp only [Fin.snoc_castSucc, Fin.snoc_last]
  split_ifs <;> omega

/-- Contamination, deterministic core: adding a calibration point that scores at least
the test score can only raise the test point's p-value, even though the denominator
grows. (Adding a point below the test score lowers it: `r/(n+1) → r/(n+2)`.) -/
lemma pval_snoc_ge (c : Fin n → ℝ) {a s : ℝ} (h : s ≤ a) :
    pval c s ≤ pval (Fin.snoc c a : Fin (n + 1) → ℝ) s := by
  unfold pval
  rw [rank_snoc, if_pos h]
  have hr : (rank c s : ℝ) ≤ n + 1 := by exact_mod_cast rank_le_succ c s
  push_cast
  have hd1 : (0 : ℝ) < n + 1 := by positivity
  have hd2 : (0 : ℝ) < n + 1 + 1 := by positivity
  rw [div_le_iff₀ hd1, div_mul_eq_mul_div, le_div_iff₀ hd2]
  nlinarith

end Rank

/-! ## Lemma (null-rank): ranks inside a pooled set

Pool the `n` calibration scores with one test score into `x : Fin (n+1) → ℝ`.
The leave-one-out rank of position `i` is the calibration rank the score `x i`
would receive against the other `n` scores. Under exchangeability every position
is equally likely to be the test point, so `P(r(v) ≤ r)` equals
`#{i : looRank x i ≤ r} / (n+1)`. The two counting lemmas below give
`P(r(v) ≤ r) ≤ r/(n+1)` in general and `= r/(n+1)` when scores are distinct. -/

section LooRank

variable {N : ℕ}

/-- Leave-one-out rank of position `i` in the pooled scores `x`. -/
def looRank (x : Fin N → ℝ) (i : Fin N) : ℕ :=
  (univ.filter fun j => j ≠ i ∧ x i ≤ x j).card + 1

/-- Tie-robust direction: at most `r` positions have leave-one-out rank `≤ r`.
Proof: among any collection of positions with rank `≤ r`, the one with the smallest
score is outranked by all the others, so the collection has at most `r` members. -/
theorem card_looRank_le (x : Fin N → ℝ) (r : ℕ) :
    (univ.filter fun i => looRank x i ≤ r).card ≤ r := by
  by_contra hcon
  rw [not_le] at hcon
  have hne : (univ.filter fun i => looRank x i ≤ r).Nonempty := by
    rw [← card_pos]; omega
  obtain ⟨i₀, hi₀, hmin⟩ := (univ.filter fun i => looRank x i ≤ r).exists_min_image x hne
  have hsub : (univ.filter fun i => looRank x i ≤ r).erase i₀ ⊆
      univ.filter fun j => j ≠ i₀ ∧ x i₀ ≤ x j := by
    intro j hj
    rw [mem_erase] at hj
    simp only [mem_filter, mem_univ, true_and]
    exact ⟨hj.1, hmin j hj.2⟩
  have h1 := card_le_card hsub
  rw [card_erase_of_mem hi₀] at h1
  have h2 : looRank x i₀ ≤ r := (mem_filter.mp hi₀).2
  unfold looRank at h2
  omega

/-- Number of positions with a strictly larger score. -/
def above (x : Fin N → ℝ) (i : Fin N) : ℕ := (univ.filter fun j => x i < x j).card

lemma looRank_eq_above_succ (x : Fin N → ℝ) (hx : Function.Injective x) (i : Fin N) :
    looRank x i = above x i + 1 := by
  unfold looRank above
  congr 1
  congr 1
  ext j
  simp only [mem_filter, mem_univ, true_and]
  constructor
  · rintro ⟨hji, hle⟩
    exact lt_of_le_of_ne hle (fun h => hji (hx h).symm)
  · intro hlt
    refine ⟨?_, hlt.le⟩
    rintro rfl
    exact lt_irrefl _ hlt

lemma above_lt_of_lt (x : Fin N → ℝ) {i i' : Fin N} (h : x i < x i') :
    above x i' < above x i := by
  unfold above
  have hsub : (univ.filter fun j => x i' < x j) ⊆ univ.filter fun j => x i < x j := by
    intro j hj
    simp only [mem_filter, mem_univ, true_and] at hj ⊢
    exact h.trans hj
  apply card_lt_card
  rw [ssubset_iff_of_subset hsub]
  exact ⟨i', by simp [h], by simp⟩

lemma above_injective (x : Fin N → ℝ) (hx : Function.Injective x) :
    Function.Injective (above x) := by
  intro i i' h
  by_contra hne
  rcases lt_or_gt_of_ne (hx.ne hne) with hlt | hlt
  · exact absurd h (ne_of_gt (above_lt_of_lt x hlt))
  · exact absurd h (ne_of_lt (above_lt_of_lt x hlt))

lemma above_lt (x : Fin N → ℝ) (i : Fin N) : above x i < N := by
  unfold above
  have hss : (univ.filter fun j => x i < x j) ⊂ univ := by
    rw [ssubset_iff_of_subset (subset_univ _)]
    exact ⟨i, mem_univ _, by simp⟩
  have := card_lt_card hss
  simpa using this

lemma image_above (x : Fin N → ℝ) (hx : Function.Injective x) :
    univ.image (above x) = range N := by
  apply eq_of_subset_of_card_le
  · intro k hk
    rw [mem_image] at hk
    obtain ⟨i, _, rfl⟩ := hk
    exact mem_range.mpr (above_lt x i)
  · rw [card_range, card_image_of_injective _ (above_injective x hx)]
    simp

/-- Distinct scores: exactly `r` positions have leave-one-out rank `≤ r` (for `r ≤ N`).
This is the "exactly uniform on the grid" case of the null-rank lemma. -/
theorem card_looRank_eq (x : Fin N → ℝ) (hx : Function.Injective x) {r : ℕ} (hr : r ≤ N) :
    (univ.filter fun i => looRank x i ≤ r).card = r := by
  have hfilt : (univ.filter fun i => looRank x i ≤ r) = univ.filter fun i => above x i < r := by
    ext i
    simp only [mem_filter, mem_univ, true_and, looRank_eq_above_succ x hx]
    omega
  rw [hfilt]
  have hinj := above_injective x hx
  calc (univ.filter fun i => above x i < r).card
      = ((univ.filter fun i => above x i < r).image (above x)).card :=
        (card_image_of_injective _ hinj).symm
    _ = ((univ.image (above x)).filter fun k => k < r).card := by rw [filter_image]
    _ = ((range N).filter fun k => k < r).card := by rw [image_above x hx]
    _ = (range r).card := by
        congr 1
        ext k
        simp only [mem_filter, mem_range]
        omega
    _ = r := card_range r

/-- Grid arithmetic: `r/(n+1) ≤ t` exactly when `r ≤ ⌊t(n+1)⌋`. -/
lemma rank_div_le_iff {n r : ℕ} {t : ℝ} (ht : 0 ≤ t) :
    (r : ℝ) / (n + 1) ≤ t ↔ r ≤ ⌊t * (n + 1)⌋₊ := by
  rw [Nat.le_floor_iff (mul_nonneg ht (by positivity)), div_le_iff₀ (by positivity)]

/-- Exact grid law for an exchangeable (distinct-score) pool of `n+1` scores: the
number of positions whose p-value is at most `t` is `⌊t(n+1)⌋`, for `0 ≤ t ≤ 1`.
Dividing by `n+1` gives `P(p̂ ≤ t) = ⌊t(n+1)⌋/(n+1)` under exchangeability. -/
theorem card_pval_le_eq {n : ℕ} (x : Fin (n + 1) → ℝ) (hx : Function.Injective x)
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    (univ.filter fun i => (looRank x i : ℝ) / (n + 1) ≤ t).card = ⌊t * (n + 1)⌋₊ := by
  have hfl : ⌊t * (n + 1)⌋₊ ≤ n + 1 := by
    have h1 : t * (n + 1) ≤ ((n + 1 : ℕ) : ℝ) := by
      push_cast
      have := mul_le_mul_of_nonneg_right ht1 (by positivity : (0 : ℝ) ≤ n + 1)
      simpa using this
    have := Nat.floor_mono h1
    rwa [Nat.floor_natCast] at this
  have hfilt : (univ.filter fun i => (looRank x i : ℝ) / (n + 1) ≤ t) =
      univ.filter fun i => looRank x i ≤ ⌊t * (n + 1)⌋₊ := by
    ext i
    simp only [mem_filter, mem_univ, true_and]
    exact rank_div_le_iff ht0
  rw [hfilt]
  exact card_looRank_eq x hx hfl

end LooRank

/-! ## Benjamini–Hochberg on a finite test set -/

section BH

variable {ι : Type*} [Fintype ι]

/-- Number of test points whose p-value is at most `c`. -/
def countLe (p : ι → ℝ) (c : ℝ) : ℕ := (univ.filter fun i => p i ≤ c).card

/-- Benjamini–Hochberg rejection count at level `α`: the largest `k ≤ m` such that at
least `k` p-values are `≤ αk/m`. Since the `k`-th smallest p-value is `≤ c` exactly when
at least `k` p-values are `≤ c`, this is the usual `max {k : p_(k) ≤ αk/m}` (with `0`
when no such `k` exists). -/
def bhK (p : ι → ℝ) (α : ℝ) : ℕ :=
  ((range (Fintype.card ι + 1)).filter fun k : ℕ =>
    k ≤ countLe p (α * (k : ℝ) / Fintype.card ι)).sup id

lemma bhK_pos_iff (p : ι → ℝ) (α : ℝ) :
    0 < bhK p α ↔ ∃ k : ℕ, 1 ≤ k ∧ k ≤ Fintype.card ι ∧
      k ≤ countLe p (α * (k : ℝ) / Fintype.card ι) := by
  unfold bhK
  rw [Finset.lt_sup_iff]
  constructor
  · rintro ⟨k, hk, hpos⟩
    simp only [mem_filter, mem_range] at hk
    simp only [id] at hpos
    exact ⟨k, by omega, by omega, hk.2⟩
  · rintro ⟨k, hk1, hkm, hkc⟩
    refine ⟨k, ?_, ?_⟩
    · simp only [mem_filter, mem_range]
      exact ⟨by omega, hkc⟩
    · simp only [id]
      omega

/-- The BH count itself satisfies the rejection condition. -/
lemma bhK_spec (p : ι → ℝ) (α : ℝ) :
    bhK p α ≤ countLe p (α * (bhK p α : ℝ) / Fintype.card ι) := by
  unfold bhK
  set S := (range (Fintype.card ι + 1)).filter fun k : ℕ =>
    k ≤ countLe p (α * (k : ℝ) / Fintype.card ι) with hS
  have hne : S.Nonempty := ⟨0, by simp [hS]⟩
  obtain ⟨k, hk, hsup⟩ := S.exists_mem_eq_sup hne id
  rw [hsup]
  rw [hS] at hk
  simpa using (mem_filter.mp hk).2

lemma countLe_smul (p : ι → ℝ) {γ : ℝ} (hγ : 0 < γ) (c : ℝ) :
    countLe (fun i => γ * p i) (γ * c) = countLe p c := by
  unfold countLe
  congr 1
  ext i
  simp only [mem_filter, mem_univ, true_and]
  exact ⟨fun h => le_of_mul_le_mul_left h hγ, fun h => mul_le_mul_of_nonneg_left h hγ.le⟩

/-- BH rescaling: running BH at level `α` on p-values `p` rejects exactly as many
hypotheses as running BH at level `γα` on the rescaled p-values `γ·p`. This is the
deterministic content of "if `P(p ≤ t) ≤ γt` for all `t`, then BH at level `α` behaves
like BH at level `γα` on super-uniform p-values". -/
theorem bhK_smul (p : ι → ℝ) {γ : ℝ} (hγ : 0 < γ) (α : ℝ) :
    bhK (fun i => γ * p i) (γ * α) = bhK p α := by
  unfold bhK
  have hset : ((range (Fintype.card ι + 1)).filter fun k : ℕ =>
      k ≤ countLe (fun i => γ * p i) (γ * α * (k : ℝ) / Fintype.card ι)) =
      (range (Fintype.card ι + 1)).filter fun k : ℕ =>
      k ≤ countLe p (α * (k : ℝ) / Fintype.card ι) := by
    apply filter_congr
    intro k _
    rw [show γ * α * (k : ℝ) / Fintype.card ι = γ * (α * (k : ℝ) / Fintype.card ι) by ring,
      countLe_smul p hγ]
  rw [hset]

/-- `N(r)`: number of test points whose calibration rank is at most `r`. -/
def countRank (rk : ι → ℕ) (r : ℕ) : ℕ := (univ.filter fun i => rk i ≤ r).card

/-- Fact 2 (BH crossing characterization). With p-values on the grid `rk/(n+1)`,
BH rejects at least one hypothesis iff some rank threshold `r` satisfies
`N(r) ≥ (m/α) · r/(n+1)`. -/
theorem bh_nonempty_iff [Nonempty ι] {n : ℕ} (rk : ι → ℕ)
    (hrk : ∀ i, 1 ≤ rk i ∧ rk i ≤ n + 1) {α : ℝ} (hα : 0 < α) :
    0 < bhK (fun i => (rk i : ℝ) / (n + 1)) α ↔
      ∃ r, 1 ≤ r ∧ r ≤ n + 1 ∧
        (Fintype.card ι : ℝ) / α * (r / (n + 1)) ≤ countRank rk r := by
  rw [bhK_pos_iff]
  have hm0 : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hmne : (Fintype.card ι : ℝ) ≠ 0 := hm0.ne'
  have hαne : α ≠ 0 := hα.ne'
  constructor
  · rintro ⟨k, hk1, hkm, hkc⟩
    have hTne : (univ.filter fun i =>
        (rk i : ℝ) / (n + 1) ≤ α * k / Fintype.card ι).Nonempty := by
      rw [← card_pos]
      exact lt_of_lt_of_le hk1 hkc
    obtain ⟨i₀, hi₀, hmax⟩ := (univ.filter fun i =>
        (rk i : ℝ) / (n + 1) ≤ α * k / Fintype.card ι).exists_max_image rk hTne
    refine ⟨rk i₀, (hrk i₀).1, (hrk i₀).2, ?_⟩
    have hsub : (univ.filter fun i => (rk i : ℝ) / (n + 1) ≤ α * k / Fintype.card ι) ⊆
        univ.filter fun i => rk i ≤ rk i₀ := by
      intro j hj
      simp only [mem_filter, mem_univ, true_and]
      exact hmax j hj
    have hcard : k ≤ countRank rk (rk i₀) := le_trans hkc (card_le_card hsub)
    have hi₀' : (rk i₀ : ℝ) / (n + 1) ≤ α * k / Fintype.card ι := (mem_filter.mp hi₀).2
    calc (Fintype.card ι : ℝ) / α * ((rk i₀ : ℝ) / (n + 1))
        ≤ (Fintype.card ι : ℝ) / α * (α * k / Fintype.card ι) := by gcongr
      _ = k := by field_simp
      _ ≤ countRank rk (rk i₀) := by exact_mod_cast hcard
  · rintro ⟨r, hr1, hrn, hcond⟩
    have hr0 : (0 : ℝ) < r := by exact_mod_cast hr1
    have hpos : (0 : ℝ) < (Fintype.card ι : ℝ) / α * (r / (n + 1)) :=
      mul_pos (div_pos hm0 hα) (div_pos hr0 (by positivity))
    have hN : 0 < countRank rk r := by
      have : (0 : ℝ) < countRank rk r := lt_of_lt_of_le hpos hcond
      exact_mod_cast this
    refine ⟨countRank rk r, hN, ?_, ?_⟩
    · exact (card_filter_le _ _).trans (le_of_eq card_univ)
    · apply card_le_card
      intro i hi
      simp only [mem_filter, mem_univ, true_and] at hi ⊢
      have hi' : (rk i : ℝ) ≤ r := by exact_mod_cast hi
      have h1 : (rk i : ℝ) / (n + 1) ≤ r / (n + 1) := by gcongr
      have h2 : (r : ℝ) / (n + 1) ≤ α * countRank rk r / Fintype.card ι := by
        rw [le_div_iff₀ hm0]
        have := mul_le_mul_of_nonneg_left hcond hα.le
        calc (r : ℝ) / (n + 1) * Fintype.card ι
            = α * ((Fintype.card ι : ℝ) / α * (r / (n + 1))) := by field_simp
          _ ≤ α * countRank rk r := this
      exact h1.trans h2

/-- Corollary (floor-only case), `r = 1`: if `N(1) ≥ (m/α)/(n+1)` then BH rejects. -/
theorem bh_nonempty_of_floor [Nonempty ι] {n : ℕ} (rk : ι → ℕ)
    (hrk : ∀ i, 1 ≤ rk i ∧ rk i ≤ n + 1) {α : ℝ} (hα : 0 < α)
    (h : (Fintype.card ι : ℝ) / α * (1 / (n + 1)) ≤ countRank rk 1) :
    0 < bhK (fun i => (rk i : ℝ) / (n + 1)) α :=
  (bh_nonempty_iff rk hrk hα).mpr ⟨1, le_refl _, by omega, by simpa using h⟩

/-- `N(r)` splits into its anomalous and normal parts. -/
lemma countRank_split (rk : ι → ℕ) (anom : ι → Prop) [DecidablePred anom] (r : ℕ) :
    countRank rk r = (univ.filter fun i => anom i ∧ rk i ≤ r).card +
      (univ.filter fun i => ¬ anom i ∧ rk i ≤ r).card := by
  unfold countRank
  rw [← card_filter_add_card_filter_not (s := univ.filter fun i => rk i ≤ r) anom]
  simp only [filter_filter]
  congr 1 <;> (congr 1; ext i; simp only [mem_filter, mem_univ, true_and]; tauto)

/-- The paper's form of the floor-only corollary: if no normal test point sits at the
resolution floor (`N₀(1) = 0`) and the anomalies alone satisfy
`N₁(1) ≥ (m/α)/(n+1)`, i.e. `π₁ c(1) ≥ 1/(α(n+1))` after dividing by `m`, then BH rejects. -/
theorem bh_nonempty_of_floor_split [Nonempty ι] {n : ℕ} (rk : ι → ℕ)
    (hrk : ∀ i, 1 ≤ rk i ∧ rk i ≤ n + 1) {α : ℝ} (hα : 0 < α)
    (anom : ι → Prop) [DecidablePred anom]
    (h0 : (univ.filter fun i => ¬ anom i ∧ rk i ≤ 1).card = 0)
    (h1 : (Fintype.card ι : ℝ) / α * (1 / (n + 1)) ≤
      (univ.filter fun i => anom i ∧ rk i ≤ 1).card) :
    0 < bhK (fun i => (rk i : ℝ) / (n + 1)) α := by
  apply bh_nonempty_of_floor rk hrk hα
  rw [countRank_split rk anom 1, h0]
  simpa using h1

end BH

end

end ConformalFDR
