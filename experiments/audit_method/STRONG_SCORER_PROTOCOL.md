# Exploratory extension: stronger frozen scorers

Specified after the cached-score feasibility experiment returned no certificates.
The original 192,000 trials remain unchanged. All 400 full-reference comparisons
(raw three models plus normalized DOMINANT) made no discoveries. The smallest
oracle FDP among the prespecified raw Amazon GAE prefixes was 0.485, despite high
AUROC. This motivates testing score quality rather than increasing audit budget
until a favorable result appears.

Train two fixed supervised histogram gradient-boosting controls on a uniform
random 10% sample of verified nodes. Training IDs are sampled without looking at
their labels, with seeds 0--9 and master seed 20260924. Remaining nodes form the
deployment batch. Training labels are never reused as certification labels.
Model A uses original attributes. Model B adds one-hop average attributes and
log(1+degree), computed with no labels on the full graph. These are simple
supervised controls, not new graph anomaly architectures or GADBench replicas.

Fixed sklearn HistGradientBoostingClassifier settings: 200 iterations, learning
rate .1, max_leaf_nodes 31, max_depth None, min_samples_leaf 20, l2_regularization 1,
early_stopping False, no class weighting. Use one CPU thread. No hyperparameter
or score-direction tuning on deployment labels. Cache all 40 trained score vectors.

Repeat the same certificate grid, budgets, designs, and 200 audit repetitions on
the untrained 90% of verified nodes. Use fresh master seed 20260925. Report training
labels and audit labels separately. Compute AUROC, AP, exact prefix precision, and
automated discoveries. The hidden labels score the procedures only.

Additionally repeat the filtered/matched/full-reference conformal comparison on
five 25% stratified splits of the remaining deployment nodes, never using training
nodes as calibration or test. Budget1000 and 200 matched-random calibration draws.
These additional normal reference labels are an oracle diagnostic resource and
must not be hidden in label-cost comparisons. Oracle filtered pool counts all true
anomalous neighbors, as in the original diagnostic.

Success criterion: nontrivial certified automated discoveries at q=.10/delta=.05,
and a reproducible advantage of score-degree stratification over BOTH score-only
stratification and candidate-uniform auditing at equal training and audit budgets.
If a simpler method matches or wins, report that. Do not add a superiority claim.

Runtime note: the initial four-thread attempt failed before fitting the first model
because Windows sandboxing denied creation of a joblib pipe. Use one CPU thread;
no outcomes had been inspected and no model parameters changed.
