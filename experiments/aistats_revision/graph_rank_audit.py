"""Exact finite-population rank audit and repeated calibration on fixed test sets.
Uses the fresh Weibo GAE score caches; no detector is retrained here.
"""
from pathlib import Path
import sys, csv, json, hashlib, platform
import numpy as np
from scipy.stats import hypergeom
from itertools import combinations
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from real_data_experiment import load_pygod_graph
from validate_revision import pvals,bh,write_csv

def exact_rank_check():
    pool=np.array([0.,1.,1.,2.,4.]); targets=np.array([0.,1.,3.,5.]); n=3
    draws=list(combinations(range(len(pool)),n))
    observed=np.array([pvals(pool[list(c)],targets) for c in draws])
    K=np.sum(pool[:,None]>=targets[None,:],axis=0)
    for t in (.25,.5,.75,1.):
        exact=hypergeom.cdf(int(np.floor(t*(n+1)))-1,len(pool),K,n)
        assert np.allclose(np.mean(observed<=t,axis=0),exact,atol=1e-12)
    print('Exact rank check passed: exhaustive calibration subsets with ties',flush=True)

def main(repetitions=200):
    exact_rank_check()
    graph,x,y=load_pygod_graph('weibo')
    normals=np.flatnonzero(y==0); anomalies=np.flatnonzero(y==1)
    exposure=np.array([sum(y[j]==1 for j in graph.neighbors(int(i)))/max(1,graph.degree(int(i))) for i in normals])
    rows=[]; summary=[]
    for seed in range(5):
        cache=OUT/f'weibo_gae_scores_{seed}.npz'
        if not cache.exists(): continue
        data=np.load(cache); scores=data['scores']; assert np.array_equal(y,data['labels'])
        for trimi,trim in enumerate((0.,.01,.05)):
            mask=scores[normals]<=np.percentile(scores[normals],100*(1-trim))
            eligible=normals[mask]; rng=np.random.default_rng(seed)
            test_normal=rng.choice(eligible,min(2000,len(eligible)//2),replace=False)
            test=np.r_[test_normal,anomalies]; n0=len(test_normal)
            pools={'clean':np.setdiff1d(normals[mask & (exposure==0)],test_normal),
                   'random':np.setdiff1d(eligible,test_normal),
                   'exposed_only':np.setdiff1d(normals[mask & (exposure>0)],test_normal)}
            n=min(map(len,pools.values())); assert n>=50
            for method in ('clean','random','exposed_only','random_full'):
                pool=pools['random'] if method=='random_full' else pools[method]
                nh=min(4000,len(pool)) if method=='random_full' else n
                assert not np.intersect1d(pool,test).size
                M=len(pool); K=M-np.searchsorted(np.sort(scores[pool]),scores[test_normal],side='left')
                exact={key:float(hypergeom.cdf(int(np.floor(t*(nh+1)+1e-10))-1,M,K,nh).mean()) for t,key in ((.01,'tail001'),(.05,'tail005'))}
                sub=[]
                for rep in range(repetitions):
                    rng=np.random.default_rng(np.random.SeedSequence([20260914,seed,trimi,rep]))
                    cal=rng.choice(pool,nh,replace=False)
                    p=pvals(scores[cal],scores[test]); reject=bh(p); k=int(reject.sum())
                    row=dict(seed=seed,trim=trim,method=method,rep=rep,n_calib=nh,pool_size=M,
                             fdp=float(reject[:n0].sum()/max(k,1)),power=float(reject[n0:].mean()),discoveries=k,
                             tail001=float(np.mean(p[:n0]<=.01)),tail005=float(np.mean(p[:n0]<=.05)))
                    rows.append(row);sub.append(row)
                out=dict(seed=seed,trim=trim,method=method,repetitions=repetitions,n_calib=nh,pool_size=M)
                for key in ('fdp','power','discoveries','tail001','tail005'):
                    a=np.array([r[key] for r in sub]);out[key+'_mean']=float(a.mean());out[key+'_mcse']=float(a.std(ddof=1)/np.sqrt(len(a)))
                for key,value in exact.items():
                    out[key+'_exact']=value
                    out[key+'_error']=out[key+'_mean']-value
                    out[key+'_zerror']=out[key+'_error']/max(1e-12,out[key+'_mcse'])
                summary.append(out)
        print('CALIBRATION AUDIT completed seed',seed,flush=True)
    write_csv(OUT/'graph_rank_trials.csv',rows);write_csv(OUT/'graph_rank_summary.csv',summary)
    aggregate=[]
    for trim in (0.,.01,.05):
        for method in ('clean','random','exposed_only','random_full'):
            sub=[r for r in summary if r['trim']==trim and r['method']==method]
            a=dict(trim=trim,method=method,training_seeds=len(sub),calibration_repetitions=repetitions)
            for key in ('fdp','power','discoveries','tail001','tail005'):
                z=np.array([r[key+'_mean'] for r in sub]);a[key+'_mean']=float(z.mean());a[key+'_training_sd']=float(z.std(ddof=1))
            for key in ('tail001','tail005'): a[key+'_exact']=float(np.mean([r[key+'_exact'] for r in sub]))
            aggregate.append(a)
    write_csv(OUT/'graph_rank_aggregate.csv',aggregate)
    edge_array=np.array(sorted(graph.edges()),dtype=np.int64)
    manifest=dict(python=platform.python_version(),numpy=np.__version__,dataset='weibo',detector='gae',
        nodes=len(y),undirected_edges=graph.number_of_edges(),edge_array_sha256=hashlib.sha256(edge_array.tobytes()).hexdigest(),
        completed_seeds=sorted({r['seed'] for r in rows}),calibration_repetitions=repetitions,
        fixed_thresholds=[.01,.05],master_seed=20260914,
        maximum_absolute_mcse_units=max(abs(r[k+'_zerror']) for r in summary for k in ('tail001','tail005')),
        exhaustive_tied_score_check='passed')
    (OUT/'graph_rank_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Completed',len({r['seed'] for r in rows}),'training seeds;',len(rows),'calibration evaluations',flush=True)
    print('Maximum standardized tail discrepancy',max(abs(r[k+'_zerror']) for r in summary for k in ('tail001','tail005')),flush=True)

if __name__=='__main__': main()
