# Park this prototype; do not add it to the AISTATS manuscript

2026-09-23. Completed 45,000 audit evaluations: three fixed graphs, two cached
HGB scorers, ten training seeds, three budgets, five arms, fifty repetitions.
The adaptive score and adaptive degree arms returned no certified discoveries
in every benchmark condition. Same-confidence-sequence fixed-allocation arms
also abstained everywhere. At 500 labels on Amazon, existing uniform fixed-time
auditing returned mean 209.902 true unreviewed discoveries for attributes and
200.194 for graph features (nonempty fractions .802 and .764). This baseline
comparison reproduces the earlier useful audit regime with fresh audit draws.

Zero certificate failures for an always-abstaining method demonstrate no useful
empirical success. There were 160 bound failures across all five arms and 45,000
evaluations, but none led to an FDP exceedance for the selected output. These
are separate diagnostics; do not silently report all bounds as always correct.

Independent recomputation from the trial file matches the seed-averaged summary.
The confidence-bound code passed 282 direct inversion checks and 2,044 ordered
binary-path enumeration cases. A two-cell adaptive enumeration checked another
256 paths. Nonvacuous stress output occurred for the score-only adaptive arm in
the 1,600-node all-anomaly candidate-union control, but not when only the first
400 nodes were highly precise. The prototype is not hard-coded to abstain;
it is too conservative/inefficient in the relevant tested regimes.

No allocation heuristic was retuned after observing benchmark outcomes. One
initial implementation failure (empty historical confidence sets) was fixed
conservatively and the whole expanded comparison was rerun; see PROTOCOL.md.

The underlying confidence-sequence and risk-limiting-audit ideas are established
prior work. There is no demonstrated new useful method here, and no basis to
claim improved acceptance chances. Preserve this negative research result
outside the manuscript and current submission ZIPs. Prioritize the moderate
selection effect, transparent uncertainty, an additional independently chosen
benchmark, and a sampling-design-valid remedy directly tied to calibration.

The independent AI proof critique is a debugging aid, not human author
verification. The authors must understand and verify any proof they submit.
