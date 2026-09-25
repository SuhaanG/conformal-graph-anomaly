# T-Finance benchmark protocol, frozen before loading outcomes

2026-09-23. Additional benchmark selected by the authors before any T-Finance
model/evaluation outcomes. Use the publicly distributed GADBench T-Finance file
(39,357 nodes according to its documentation). Record the raw source URL, file
hash, actual label counts, feature dimensions and edge counts before training.
Do not choose a substitute dataset based on performance. Do not silently change
binary label meanings. Use sparse adjacency, never dense reconstruction.

Fit the same two histogram-gradient-boosting classifiers already used in the
Amazon/Tolokers/Weibo controls: attributes only; attributes plus mean one-hop
neighbor attributes and log(1+degree). No label propagation covariates. Uniform
10% training panel, 10 training seeds, independent of labels. Hyperparameters:
200 iterations, learning rate .1, 31 leaves, minimum leaf size 20, L2=1,
no class weighting, no early stopping. Frozen transductive feature processing:
finite numeric attributes; use distributed features, no outcome-driven feature
selection. Symmetrize and binarize adjacency, remove self loops; report this
choice and raw/processed counts. Training labels excluded from every test and
calibration population. CPU training; H200 is not required.

Use dataset seed index 3 with existing master namespaces 20260924 training,
20260926 splits and 20260928 revelation. Five independently randomized 25%
label-stratified deployment test splits per training panel. Reveal non-test
labels at .25,.50,1.00, nested by the same uniform stream. Score and exposure
tie handling, graded removal levels (0,.05,.10,.20,.40,.60,.80,.90,.97), fifty
paired calibration repetitions, no reference-size cap, BH .10, and retained
negative/abstention outcomes follow DOSE_PROTOCOL.md exactly. Also report the
full random reference at zero removal. Save every trial and seed-level summary.

Report AUROC/AP and the entire FDP/power curve. The .20-removal, .50-revelation
contrast is designated for comparison with the earlier Amazon result; no
claim of independent confirmation across all previously inspected datasets.
Uncertainty uses paired training-seed averages conditional on this fixed graph,
not 50 draws or five splits as independent graph observations. No retuning
after results. Run stratified remedy according to STRATIFIED_PROTOCOL.md.

# Separate Weibo source sensitivity

If the GADBench Weibo file can be obtained and read cheaply, compare its labels,
attributes and adjacency with the PyGOD file. Record hashes and counts. If labels
differ, never overwrite the PyGOD dataset or pool their results. Fit the same
two classifiers and repeat the same dose/remedy grids, ten seeds and five splits,
as a separately named weibo_gadbench sensitivity with dataset seed index 4.
Use matched index-based training/split RNG namespace 2 where compatible to help
source comparisons; stratified split identities can change when labels change.
Document any representation differences instead of attributing all differences
to labels. A second source of the same graph is not another graph replication.
