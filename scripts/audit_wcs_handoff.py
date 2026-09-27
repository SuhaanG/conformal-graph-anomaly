"""Independent checks of the September 26 WCS handoff; does not alter results.

Run from the repository: python scripts/audit_wcs_handoff.py --data-root paper
Requires numpy/pandas/scipy/scikit-learn. Raw trials and caches are local inputs.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_sizes(rank, cal, test, weights, alpha):
    m = len(test)
    mass = weights[cal].sum()
    tail = np.array([weights[cal][rank[cal] > rank[j]].sum() for j in test])
    sizes = []
    for j in test:
        aux = (tail + weights[j] * (rank[j] > rank[test])) / (mass + weights[j])
        aux[test == j] = 0
        crossing = np.flatnonzero(np.sort(aux) <= alpha * np.arange(1, m + 1) / m)
        sizes.append(int(crossing[-1] + 1) if len(crossing) else 0)
    return tail, np.array(sizes)


def literal_prune(first, sizes, xi):
    m = len(first)
    qualifying = [r for r in range(m + 1) if np.count_nonzero(first & (xi * sizes <= r)) >= r]
    return first & (xi * sizes <= max(qualifying))


def expected_pruning(first, sizes, null, randomized):
    if not first.any():
        return 0.
    if not randomized:
        r = literal_prune(first, sizes, 1.)
        return np.count_nonzero(r & null) / max(1, r.sum())
    cuts = sorted({0., 1.} | {r / int(s) for s in sizes[first] for r in range(1, int(s))})
    ans = 0.
    for lo, hi in zip(cuts, cuts[1:]):
        r = literal_prune(first, sizes, (lo + hi) / 2)
        ans += (hi - lo) * np.count_nonzero(r & null) / max(1, r.sum())
    return ans


def exact_randomized_training():
    """Integrate roles, training, all U_j and xi; no Monte Carlo p-value draws."""
    cases = states = 0
    maximum = -1.
    for n, nt, na, alpha, f in itertools.product((3, 4), (1, 2), (0, 1), (.1, .3), (0., .5)):
        nall = n + na
        rho = np.resize(np.array([.8, .2, 1.]), nall)
        weights = 1 / rho
        base = np.random.default_rng([1201, n, nt, na]).permutation(nall)
        sums = np.zeros(2)
        total = 0.
        for tt in itertools.combinations(range(n), nt):
            remaining = np.setdiff1d(np.arange(n), tt)
            test = np.r_[tt, np.arange(n, nall)].astype(int)
            for st in itertools.product((0, 1, 2), repeat=len(remaining)):
                st = np.array(st)
                mass = np.prod(np.where(st == 0, 1 - rho[remaining], np.where(st == 1, f * rho[remaining], (1 - f) * rho[remaining]))) / math.comb(n, nt)
                if mass == 0:
                    continue
                total += mass
                train = remaining[st == 1]
                rank = base + np.random.default_rng([44, *train.tolist()]).normal(size=nall)
                cal = remaining[st == 2]
                tail, sizes = literal_sizes(rank, cal, test, weights, alpha)
                probs = np.clip((alpha * sizes / len(test) * (weights[cal].sum() + weights[test]) - tail) / weights[test], 0, 1)
                for indicators in itertools.product((False, True), repeat=len(test)):
                    first = np.array(indicators)
                    probability = np.prod(np.where(first, probs, 1 - probs))
                    for k, randomized in enumerate((False, True)):
                        sums[k] += mass * probability * expected_pruning(first, sizes, test < n, randomized)
                states += 1
        assert abs(total - 1) < 1e-10
        excess = float(max(sums) - alpha * nt / (nt + na))
        assert excess < 1e-10, (n, nt, na, alpha, f, sums)
        maximum = max(maximum, excess)
        cases += 1
    return dict(cases=cases,positive_mass_states=states,max_excess=maximum,
                scope='Independent exact integration including training-dependent scores, empty calibration, and rho=1; small designs only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'paper')
    parser.add_argument('--output', type=Path, default=ROOT / 'notes/wcs_audit_20260926')
    parser.add_argument('--retrain-smoke', action='store_true', help='Retrain one graph-feature run per graph and scenario; CPU only')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.data_root / 'audit_method'
    source = ROOT / 'experiments/audit_method'
    report = dict(result_checks={}, hash_checks={}, missing_clone_inputs=[])
    for name in ('wcs_acquisition', 'wcs_randomized', 'wcs_estimated', 'wcs_endtoend'):
        trials = pd.read_csv(data / (name + '_trials.csv.gz'))
        saved = pd.read_csv(source / (name + '_summary.csv'))
        keys = [x for x in ('dataset', 'model', 'design', 'scenario', 'method') if x in saved]
        trialkeys = keys + ['seed', 'split', 'rep']
        assert not trials.duplicated(trialkeys).any()
        for metric in ('fdp', 'power'):
            assert trials[metric].between(0, 1).all()
        seed = trials.groupby(keys + ['seed']).mean(numeric_only=True)
        expected = seed.groupby(keys).agg(['mean','std'])
        expected.columns = ['_'.join(c) for c in expected.columns]
        indexed = saved.set_index(keys).sort_index()
        expected = expected.sort_index()
        common = [c for c in expected if c in indexed]
        assert expected.index.equals(indexed.index)
        assert np.allclose(expected[common], indexed[common], rtol=1e-10, atol=1e-12, equal_nan=True), name
        savedseed = pd.read_csv(source / (name + '_seeds.csv')).set_index(keys + ['seed']).sort_index()
        commonseed = [c for c in seed if c in savedseed]
        assert np.allclose(seed.sort_index()[commonseed],savedseed[commonseed],rtol=1e-10,atol=1e-12,equal_nan=True)
        manifest = json.loads((source / (name + '_manifest.json')).read_text())
        assert len(trials) == manifest['rows']
        hashes = {}
        for key, path in [('script_sha256', source / (name + '.py')), ('protocol_sha256', source / (name.upper() + '_PROTOCOL.md'))]:
            # The acquisition protocol is named WCS_ACQUISITION_PROTOCOL.md, etc.
            hashes[key] = dict(matches=digest(path)==manifest[key], file=str(path.relative_to(ROOT)))
        for filename, expectedhash in manifest['inputs'].items():
            path = (args.data_root / 'aistats_followup/h200_results/paper/aistats_followup' if filename=='amazon_graph.npz' else data) / filename
            hashes[filename] = dict(matches=path.exists() and digest(path)==expectedhash)
            corresponding = ROOT / 'experiments' / path.relative_to(args.data_root)
            if not corresponding.exists():
                report['missing_clone_inputs'].append(str(corresponding.relative_to(ROOT)))
        report['hash_checks'][name] = hashes
        report['result_checks'][name] = dict(rows=len(trials),summary_cells=len(saved),metrics_checked=len(common))
        if name == 'wcs_endtoend':
            splitmean = trials.groupby(keys+['seed','split']).mean(numeric_only=True)
            naive_se = splitmean.groupby(keys).fdp.std() / np.sqrt(50)
            assert np.allclose(naive_se.sort_index(),indexed.fdp_se_testset)
            cluster_se = seed.groupby(keys).fdp.std() / np.sqrt(10)
            comp = indexed[['fdp_mean','power_mean','bound_mean','fdp_se_testset']].copy()
            comp['fdp_se_seed_cluster'] = cluster_se
            comp['z_50_testsets'] = (comp.fdp_mean-comp.bound_mean)/comp.fdp_se_testset
            comp['z_10_seed_clusters'] = (comp.fdp_mean-comp.bound_mean)/comp.fdp_se_seed_cluster
            comp.to_csv(args.output/'endtoend_uncertainty.csv')
            pivot = indexed.power_mean.unstack('method')
            z = comp.xs('wcs_rand_homogeneous',level='method')
            pooled = comp.xs('pooled_reference',level='method')
            report['headlines'] = dict(randomized_power_beats_weighted_bh=int((pivot.wcs_rand_homogeneous>pivot.weighted_bh).sum()),
                randomized_power_at_least_deterministic_p_wcs=int((pivot.wcs_rand_homogeneous>=pivot.wcs_homogeneous-1e-12).sum()),
                pooled_above_bound_2_naive_se=int((pooled.z_50_testsets>2).sum()),
                pooled_above_bound_2_cluster_se=int((pooled.z_10_seed_clusters>2).sum()),
                randomized_max_naive_z=float(z.z_50_testsets.max()),randomized_max_cluster_z=float(z.z_10_seed_clusters.max()),
                auroc_trial_range=[float(trials.auroc.min()),float(trials.auroc.max())],
                auroc_cell_mean_range=[float(saved.auroc_mean.min()),float(saved.auroc_mean.max())])
    report['missing_clone_inputs'] = sorted(set(report['missing_clone_inputs']))
    sys.path.insert(0,str(source))
    from wcs_acquisition import wcs_sizes
    from wcs_randomized import wcs_rand
    rng = np.random.default_rng(90262026)
    for case in range(1000):
        n, m = int(rng.integers(0,21)),int(rng.integers(1,15))
        ranks = rng.permutation(n+m).astype(float)
        weights = np.exp(rng.normal(0,2,n+m))
        cal,test = np.arange(n),np.arange(n,n+m)
        alpha = float(rng.choice([.01,.1,.3,.8]))
        tail,sizes = literal_sizes(ranks,cal,test,weights,alpha)
        p,fastsizes = wcs_sizes(ranks[cal],weights[cal],ranks[test],weights[test],alpha)
        assert np.array_equal(sizes,fastsizes)
        assert np.allclose(p,(tail+weights[test])/(weights[cal].sum()+weights[test]))
        U,xi = rng.random(m),float(rng.random())
        r,pr = wcs_rand(ranks,cal,test,weights,U,xi,alpha)
        literal_p = (tail+U*weights[test])/(weights[cal].sum()+weights[test])
        assert np.allclose(pr,literal_p)
        assert np.array_equal(r,literal_prune(literal_p<=alpha*sizes/m,sizes,xi))
    report['independent_equation_cases'] = 1000
    report['exact_randomized_training'] = exact_randomized_training()
    if args.retrain_smoke:
        report['retraining_smoke'] = retrain_smoke(args.data_root)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('hash_checks','missing_clone_inputs')},indent=2))
    print('Manifest mismatches:',[(run,k) for run,hs in report['hash_checks'].items() for k,v in hs.items() if not v['matches']])
    print('Missing clone inputs:',len(report['missing_clone_inputs']))


def retrain_smoke(data_root):
    from scipy import sparse
    from scipy.special import expit
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits
    from selection_dose import exposure, score_ranks
    from stratified_calibration import make_strata
    d = data_root / 'audit_method'
    stored = pd.read_csv(d/'wcs_endtoend_trials.csv.gz')
    checked = []
    for ds, dsid in [('amazon',0),('tfinance',3),('weibo_gadbench',2)]:
        path = (data_root/'aistats_followup/h200_results/paper/aistats_followup' if ds=='amazon' else d)/f'{ds}_graph.npz'
        g = np.load(path)
        y,x = g['labels'].astype(int),g['features'].astype(np.float32)
        a = sparse.csr_matrix((np.ones(len(g['indices']),dtype=np.float32),g['indices'],g['indptr']),shape=(len(y),len(y)))
        degree = np.asarray(a.sum(axis=1)).ravel()
        features = np.c_[x,(a@x)/np.maximum(1,degree[:,None]),np.log1p(degree)].astype(np.float32)
        c = np.load(d/f'{ds}_hgb_attributes_scores_0.npz')
        panel,deploy = c['train'],c['deploy']
        assert np.array_equal(c['labels'],y)
        assert not np.intersect1d(panel,deploy).size
        if ds=='amazon':
            assert np.all(np.r_[panel,deploy]>=3305)
        normals,anoms = deploy[y[deploy]==0],deploy[y[deploy]==1]
        rng = np.random.default_rng(np.random.SeedSequence([20260926,dsid,0,0]))
        test = np.r_[rng.choice(normals,round(.25*len(normals)),False),rng.choice(anoms,round(.25*len(anoms)),False)]
        reference = np.setdiff1d(deploy,test)
        obs = np.zeros(len(y)); obs[panel] = y[panel]
        counts = np.asarray(a@obs).ravel()
        ld = np.log1p(degree); z = (ld-ld[deploy].mean())/ld[deploy].std()
        smooth = np.clip(expit(-1.2*np.sqrt(counts)+.3*z),.01,.95)
        marks = np.random.default_rng(np.random.SeedSequence([2026092303,dsid,0])).random(len(y))
        strata = make_strata(exposure(a,y,panel),deploy,marks,'exposure_80')
        for si,(scenario,probabilities) in enumerate([('moderate',(.5,.1)),('severe',(.5,.025)),('smooth',None)]):
            rho = smooth if probabilities is None else np.array(probabilities)[strata]
            u = np.random.default_rng(np.random.SeedSequence([2026092407,dsid,0,0,0])).random(len(reference))
            acquired = reference[u<rho[reference]]
            assign = np.random.default_rng(np.random.SeedSequence([2026092509,dsid,0,0,0,si])).random(len(acquired))<.5
            train = np.r_[panel,acquired[assign]]
            cal = acquired[~assign]; cal = cal[y[cal]==0]
            assert not np.intersect1d(train,test).size and not np.intersect1d(train,cal).size
            model = HistGradientBoostingClassifier(max_iter=200,learning_rate=.1,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=0)
            with threadpool_limits(limits=1):
                model.fit(features[train],y[train]); scores=model.predict_proba(features)[:,1]
            ranks = score_ranks(scores,np.random.SeedSequence([2026092512,dsid,0,0,0,si,1]))
            xi = np.random.default_rng(np.random.SeedSequence([2026092510,dsid,0,0,0,si,1])).random()
            U = np.random.default_rng(np.random.SeedSequence([2026092511,dsid,0,0,0,si,1])).random(len(test))
            weights=1/rho
            tail,sizes = literal_sizes(ranks,cal,test,weights,.1)
            detp=(tail+weights[test])/(weights[cal].sum()+weights[test])
            randp=(tail+U*weights[test])/(weights[cal].sum()+weights[test])
            ordinary=(1+np.array([np.count_nonzero(ranks[cal]>ranks[j]) for j in test]))/(len(cal)+1)
            def bh(p):
                crossed=np.flatnonzero(np.sort(p)<=.1*np.arange(1,len(p)+1)/len(p))
                return p<=np.sort(p)[crossed[-1]] if len(crossed) else np.zeros(len(p),bool)
            outcomes={'pooled_reference':bh(ordinary),'weighted_bh':bh(detp),'wcs_homogeneous':literal_prune(detp<=.1*sizes/len(test),sizes,xi),'wcs_rand_homogeneous':literal_prune(randp<=.1*sizes/len(test),sizes,xi)}
            actual=stored[(stored.dataset==ds)&(stored.model=='hgb_graph_features')&(stored.seed==0)&(stored.split==0)&(stored.rep==0)&(stored.scenario==scenario)].set_index('method')
            for method,r in outcomes.items():
                tp=int(y[test[r]].sum()); k=int(r.sum())
                metrics=[(k-tp)/max(k,1),tp/y[test].sum(),k,roc_auc_score(y[test],scores[test])]
                assert np.allclose(metrics,actual.loc[method,['fdp','power','discoveries','auroc']].to_numpy(float),rtol=1e-10,atol=1e-12),(ds,scenario,method,metrics)
            checked.append(dict(dataset=ds,scenario=scenario,methods=4,matched=True))
            print('RETRAINED',ds,scenario,flush=True)
    return checked


if __name__ == '__main__':
    main()
