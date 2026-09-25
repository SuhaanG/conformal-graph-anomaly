"""Check exact primary null tails against fresh Monte Carlo calibration draws.

Conditions on each saved graph, trained score vector and primary test split.
Each draw's null fraction is one observation for MCSE; nodes are not independent
replicates. Fresh random streams do not reuse the GPU calibration draws.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom

D = Path(__file__).resolve().parent
G = D.parent / 'aistats_followup/h200_results/paper/aistats_followup'

def main(repetitions=500):
    rows = []
    for dsid, ds in enumerate(('amazon', 'tolokers')):
        g = np.load(G / f'{ds}_graph.npz')
        y = g['labels']; N = len(y)
        a = sparse.csr_matrix((np.ones(len(g['indices'])), g['indices'], g['indptr']), shape=(N, N))
        verified = np.arange(3305, N) if ds == 'amazon' else np.arange(N)
        normals = verified[y[verified] == 0]; anomalies = verified[y[verified] == 1]
        mask = np.asarray(a @ y).ravel() == 0
        for mid, model in enumerate(('dominant_pygod', 'gae', 'isolation_forest')):
            for seed in range(10):
                scores = np.load(G / f'{ds}_{model}_scores_{seed}.npz')['scores']
                stored = pd.read_csv(G / f'evaluation_{ds}_{model}_{seed}_summary.csv')
                for split in range(5):
                    rng = np.random.default_rng(np.random.SeedSequence([20260917, dsid, split]))
                    test = np.r_[rng.choice(normals, int(round(.25 * len(normals))), False),
                                 rng.choice(anomalies, int(round(.25 * len(anomalies))), False)]
                    pool = np.setdiff1d(normals, test)
                    filtered = pool[mask[pool]]; n = min(1000, len(filtered))
                    null_scores = scores[test[y[test] == 0]]
                    cutoff = int(np.floor(.01 * (n + 1) + 1e-10)) - 1
                    for methodid, (method, eligible) in enumerate((('filtered', filtered), ('random', pool))):
                        M = len(eligible)
                        K = M - np.searchsorted(np.sort(scores[eligible]), null_scores, side='left')
                        exact = float(hypergeom.cdf(cutoff, M, K, n).mean())
                        q = stored[(stored.design == 'primary') & (stored.label_fraction == 1) &
                                   (stored.budget == 1000) & (stored.split == split) & (stored.method == method)]
                        assert len(q) == 1
                        assert abs(exact - q.iloc[0].null_tail001_exact) < 1e-12
                        rng = np.random.default_rng(np.random.SeedSequence([20260921, dsid, mid, seed, split, methodid]))
                        values = np.empty(repetitions)
                        for rep in range(repetitions):
                            c = np.sort(scores[rng.choice(eligible, n, False)])
                            exceedances = n - np.searchsorted(c, null_scores, side='left')
                            values[rep] = np.mean(exceedances <= cutoff)
                        mc = float(values.mean()); se = float(values.std(ddof=1) / np.sqrt(repetitions))
                        rows.append(dict(dataset=ds, detector=model, seed=seed, split=split, method=method,
                                         n_calib=n, repetitions=repetitions, exact=exact, mc=mc, mcse=se,
                                         error=mc-exact, z=(mc-exact)/se if se > 1e-14 else np.nan))
            print('AUDITED', ds, model, flush=True)
    df = pd.DataFrame(rows)
    target = D / 'primary_rank_validation.csv'; df.to_csv(target, index=False)
    nonconstant = df.mcse > 1e-14
    report = dict(conditions=len(df), fresh_calibration_draws=len(df)*repetitions,
                  maximum_absolute_error=float(df.error.abs().max()),
                  max_absolute_z=float(df.loc[nonconstant, 'z'].abs().max()),
                  nonconstant_conditions=int(nonconstant.sum()),
                  number_above_3_mcse=int((df.loc[nonconstant, 'z'].abs() > 3).sum()),
                  constant_conditions_exact=bool((df.loc[~nonconstant, 'error'].abs() < 1e-12).all()),
                  csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  master_seed=20260921, selection='all primary untrimmed oracle/random learned-detector conditions',
                  interpretation='MCSE is across calibration draws conditional on fixed scores and test identities; no independence of graph nodes is assumed.')
    (D / 'primary_rank_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
