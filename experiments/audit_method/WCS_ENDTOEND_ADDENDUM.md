# Addendum to WCS_ENDTOEND_PROTOCOL.md: randomized weighted BH arm

Frozen 29 September 2026, before any outcome of this arm. Requested by an
internal review: the main end-to-end comparison contrasted randomized WCS with
weighted BH on deterministic p-values, mixing two differences (randomization and
the selection procedure).

Change: add one arm, `weighted_bh_rand`, equal to BH at alpha = 0.10 on the
randomized weighted p-values of the same run, using the SAME auxiliary uniforms
U_j, fitted scores, acquired reference, test nodes and weights as
`wcs_rand_homogeneous`. No other change. The rerun must reproduce every existing
arm's FDP, power and discovery count exactly (compared with the archived
`wcs_endtoend_v1_trials.csv.gz`) before the new arm is reported. The arm has no
FDR guarantee under this design and is reported as an empirical comparator,
whatever its results.
