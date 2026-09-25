# What this prototype does and does not establish

Freeze the graph, its attributes and labels, the trained scores, the candidate
prefixes and a partition whose cells exactly align with every prefix boundary.
Training labels are outside the discovery/audit population. Labels need not be
independent across graph nodes. Correct requested labels are assumed.

For a cell of size N with unknown fixed total K of normal nodes, draw uniformly
without replacement within that cell. After n local draws let x be the observed
normal count. The ordered binary path has likelihood

    L_n(k) = (k)_x (N-k)_(n-x) / (N)_n,

where (a)_b is a falling factorial. A uniform prior on the cell's total count,
equivalently a Beta(1,1) Bernoulli mixture integrated over the finite population,
has ordered-path predictive mass

    Q_n = B(1+x, 1+n-x) / B(1,1).

Under the true K, Q_n/L_n(K) is a nonnegative supermartingale starting at one.
Its conditional expected multiplicative increment sums the alternative
predictive probabilities over the outcomes possible under the true remaining
population. That sum is at most one; when a type is exhausted it need not equal
one. Thus this is not asserted to be an everywhere mean-one martingale.

Select the next cell using only already observed information, then sample
uniformly from its unqueried members. Each cell process remains a supermartingale
in the global audit filtration. Ville's inequality at delta/H and a union bound
give simultaneous coverage of every cell's true total at every sampling time.
No cell independence or optional-stopping penalty is required. Beta parameters
are frozen, not retrospectively fit to the labels being certified.

Take the upper endpoint of the feasible counts for which Q_n/L_n(k) < H/delta,
and intersect upper endpoints over time. Using just the intersection of upper
endpoints is a conservative relaxation of intersecting the full confidence
sets. If historical bounds become inconsistent with newly observed counts,
report a bound failure and fall back conservatively, never to an empty-set
interpretation of zero uncertainty.

For candidate prefix P, write h_c,x_c,U_c for audited count, observed normals and
total-normal upper bound in its complete cells. The unreviewed FDP is bounded by

    sum_(c in P) (U_c-x_c) / sum_(c in P) (N_c-h_c).

Release the largest positive-size remainder whose bound is <=q, or abstain.
The simultaneous event covers every fixed candidate and every stopping time,
so selecting among them requires no additional multiplicity correction.

Guarantee: with probability at least 1-delta over the audit design, every
released remainder has FDP <=q, conditional on the frozen population. This is
an FDP exceedance guarantee. It gives E[FDP] <= q+(1-q)*delta, not E[FDP]<=q.
At q=.10 and delta=.05 the derived expectation bound is .145. The guarantee
does not transfer to new graphs or future nodes.

The allocation policy aims to maximize remaining certified discoveries, not
merely minimize confidence-interval width. Auditing an anomaly consumes a true
automatic discovery, so spending more labels can reduce the output size. The
planner's predictions are not confidence bounds and are used only to allocate
labels; a poor planner can waste budget while the certificate remains valid.

Novelty assessment: the finite-population CS and adaptive sampling guarantee
are established statistical machinery (Waudby-Smith & Ramdas 2020; Shekhar et
al. 2023). Active anytime-valid risk control already exists (Xu et al. 2024).
Neither a new name nor degree stratification establishes a new method worthy
of a methodological novelty claim. A plausible contribution would require a
distinctive, well-motivated allocation objective/policy with a substantial
efficiency advantage, backed by comparisons and preferably an allocation result.
This prototype is a feasibility test of that opportunity, not evidence that
such a contribution has already been achieved.
