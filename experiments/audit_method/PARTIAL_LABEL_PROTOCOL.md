# Exploratory check with non-test labels only

Declared after the supervised oracle-filter comparison completed. Keep both graphs,
both scorers, all ten seeds, and the same five deployment test splits. Reveal 25%
and 50% of the non-test deployment panel using a single nested uniform draw.
The training panel is already labeled; its labels may identify anomalous neighbors,
but training nodes never enter calibration or testing. Use only revealed normal
deployment nodes as the reference pool. A node passes the filter if it has no
known anomalous neighbor among training and revealed reference nodes.

Use min(1000, filtered pool size) nodes for both filtered and random calibration,
200 draws of each. Add the entire revealed normal reference as a separate arm.
No score trimming, test-label filtering, model changes, or score tuning. Record
all outcomes, including zero discoveries. Master seed 20260928 determines label
revelation and calibration draws; the existing test-split seed remains 20260926.
Check filter invariance to changing every label outside the known set.

This is an exploratory robustness check prompted by the oracle results, not a
prespecified confirmatory test. Test stratification uses benchmark labels to
construct evaluation splits; these labels are not available to either detector,
calibration selection, or rejection rule.
