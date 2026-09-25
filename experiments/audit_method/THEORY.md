# Prototype guarantee and boundaries

Condition on a finite graph, its binary labels, trained scores, and a family of
candidate sets fixed before querying certification labels. Graph nodes and scores
may be arbitrarily dependent. The randomness in this guarantee is audit sampling.
It is not a claim of novel hypergeometric statistics.

For a set of N nodes with unknown V normal nodes, a uniform n-node audit observes
X ~ Hypergeom(N,V,n). Let U(x) be the largest feasible v for which
P(Hypergeom(N,v,n) <= x) >= a. The CDF decreases in v, so U(X)<V implies
F_V(X)<a. A discrete CDF evaluated at its own observation is super-uniform:
P(F_V(X)<a) <= a. Thus P(V<=U(X)) >= 1-a. Ties in the anomaly score do not enter
this sampling statement; candidate membership uses a fixed node-ID tiebreaker.

For uniform auditing of a superset of K candidates, condition on the number of
audited nodes inside each candidate. Its audited subset is uniform within that
candidate, so use a=delta/K for each upper bound. A union bound makes all K bounds
simultaneously valid with probability at least 1-delta; neither independent
candidates nor independent graph nodes are needed.

For H disjoint prespecified strata, sample a fixed number uniformly from each.
Use a=delta/H for the stratum null-count bounds. With probability at least
1-delta they all hold. Every candidate in this implementation is a union of
complete strata, so its null upper bound is the sum of the corresponding bounds.
All sample sizes are determined without certification labels.

For candidate k of size R_k with h_k audited nodes and x_k audited normals,
the unreviewed candidate has exactly V_k-x_k normals among R_k-h_k nodes.
Therefore its FDP is bounded by min(R_k-h_k,U_k-x_k)/(R_k-h_k) whenever R_k>h_k.
Because the bounds hold simultaneously, selecting the largest unreviewed set
whose bound is <=q retains P(FDP<=q)>=1-delta. If no nonempty candidate qualifies,
return no automatic discoveries and define FDP=0. Reviewed anomalies are not
counted as automatic discoveries. This avoids inflating apparent usefulness by
reporting human-reviewed positives as model discoveries.

This implies E[FDP] <= q+(1-q)*delta, not necessarily <=q. For an expected FDR
target alpha and delta<alpha, q=(alpha-delta)/(1-delta) is sufficient. The primary
prototype instead explicitly evaluates a q=.10, delta=.05 FDP certificate.

Audit labels must be correct, audit selection must follow the declared random
design, and candidates/strata/scores must be frozen before those labels are seen.
The procedure certifies this batch, not a new graph or future population. Model
choice after certification would require joint bounds over the model family;
this prototype evaluates each frozen scorer separately and does not choose one
by its certification outcomes.

The method uses classical finite-population bounds and a simultaneous testing
argument closely related to Learn then Test. Graph-informed stratification is a
candidate efficiency improvement, not a new validity principle. Novelty requires
further comparison with existing audit, risk-control, and sequential-sampling work.
