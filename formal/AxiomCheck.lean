-- Run with `lake env lean AxiomCheck.lean` from the `formal/` directory.
-- Every theorem should report only [propext, Classical.choice, Quot.sound].
import ConformalFDR.Basic
open ConformalFDR
#print axioms ConformalFDR.pval_ge_floor
#print axioms ConformalFDR.rank_eq_one_iff
#print axioms ConformalFDR.pval_le_of_coupling
#print axioms ConformalFDR.pval_snoc_ge
#print axioms ConformalFDR.rank_ge_of_dominated
#print axioms ConformalFDR.card_looRank_le
#print axioms ConformalFDR.card_looRank_eq
#print axioms ConformalFDR.card_pval_le_eq
#print axioms ConformalFDR.bh_nonempty_iff
#print axioms ConformalFDR.bh_nonempty_of_floor_split
#print axioms ConformalFDR.bhK_smul
#print axioms ConformalFDR.bhK_spec
