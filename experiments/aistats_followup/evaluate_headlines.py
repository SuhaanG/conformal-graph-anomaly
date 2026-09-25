"""Matched calibration design comparisons on frozen score caches.
Run per detector/seed after training; retain every calibration draw.
"""
from pathlib import Path
import sys,argparse,csv,json,time
import numpy as np
from scipy import sparse
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score,average_precision_score
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'aistats_revision'))
from validate_revision import pvals,bh,write_csv

def observed_filter(a,y,test,revealed):
    assert not np.intersect1d(test,revealed).size
    obs=np.zeros(len(y));obs[revealed]=y[revealed]
    return np.asarray(a@obs).ravel()==0

def checks():
    a=sparse.csr_matrix([[0,1,1],[1,0,0],[1,0,0]]);y=np.array([0,1,0]);test=np.array([1]);known=np.array([0,2])
    f=observed_filter(a,y,test,known);y2=y.copy();y2[test]=1-y2[test]
    assert np.array_equal(f,observed_filter(a,y2,test,known))

def main(dataset,detector,seed,reps=200):
    checks();d=np.load(OUT/f'{dataset}_graph.npz');y=d['labels'];N=len(y)
    a=sparse.csr_matrix((np.ones(len(d['indices'])),d['indices'],d['indptr']),shape=(N,N))
    verified=np.arange(3305,N) if dataset=='amazon' else np.arange(N)
    normals=verified[y[verified]==0];anoms=verified[y[verified]==1]
    if detector=='degree':s=d['degree'].astype(float)
    else:
        c=np.load(OUT/f'{dataset}_{detector}_scores_{seed}.npz');assert np.array_equal(c['labels'],y);s=c['scores']
    meta=dict(dataset=dataset,detector=detector,seed=seed,verified_normals=len(normals),verified_anomalies=len(anoms),
              auroc=float(roc_auc_score(y[verified],s[verified])),auprc=float(average_precision_score(y[verified],s[verified])))
    rows=[];summary=[];dsid=0 if dataset=='amazon' else 1
    for split in range(5):
        rng=np.random.default_rng(np.random.SeedSequence([20260917,dsid,split]));n0=int(round(.25*len(normals)));n1=int(round(.25*len(anoms)))
        test=np.r_[rng.choice(normals,n0,False),rng.choice(anoms,n1,False)]
        reference=np.setdiff1d(verified,test);reveal_u=rng.random(len(reference))
        oracle=np.asarray(a@y).ravel()==0
        scenarios=[]
        for fraction in (1.,.25,.5):
            revealed=reference if fraction==1. else reference[reveal_u<fraction]
            pool=revealed[y[revealed]==0]
            filtermask=oracle if fraction==1. else observed_filter(a,y,test,revealed)
            for budget in (200,1000):scenarios.append(('primary',fraction,0.,budget,test,pool,filtermask))
        # Keep every untrimmed test node; remove high scores from references only.
        pool=reference[y[reference]==0]
        for trim in (.01,.05):
            trimmed=pool[s[pool]<=np.quantile(s[pool],1-trim)]
            scenarios.append(('calibration_only_trim',1.,trim,1000,test,trimmed,oracle))
        # Historical protocol sensitivity: all anomalies, 2000 normals,
        # trimming BOTH normal pools. Corrected-label and historical-label arms.
        for domain,normalset in [('verified',normals)]+([('historical_unlabeled_as_normal',np.flatnonzero(y==0))] if dataset=='amazon' else []):
            for trim in (0.,.01,.05):
                elig=normalset[s[normalset]<=np.quantile(s[normalset],1-trim)]
                rr=np.random.default_rng(np.random.SeedSequence([20260918,dsid,split]))
                tn=rr.choice(elig,min(2000,len(elig)//2),False);tt=np.r_[tn,anoms];pp=np.setdiff1d(elig,tn)
                scenarios.append(('enriched_both_trim_'+domain,1.,trim,1000,tt,pp,oracle))
        for scenid,(design,fraction,trim,budget,tt,pool,fmask) in enumerate(scenarios):
            filtered=pool[fmask[pool]];n=min(budget,len(filtered));yn=y[tt];nnull=int((yn==0).sum())
            base=dict(**meta,split=split,design=design,label_fraction=fraction,trim=trim,budget=budget,n_calib=n,
                      n_test=len(tt),n_null=nnull,n_anomaly=int(yn.sum()),random_pool=len(pool),filtered_pool=len(filtered))
            if n<1:
                summary.append(dict(**base,method='unavailable',repetitions=0,fdp_mean=np.nan,power_mean=np.nan,discoveries_mean=np.nan,
                                    fdp_mcse=np.nan,power_mcse=np.nan,discoveries_mcse=np.nan,null_tail001_exact=np.nan));continue
            assert not np.intersect1d(pool,tt).size
            if design!='enriched_both_trim_historical_unlabeled_as_normal':assert np.all(np.isin(pool,verified))
            samples={k:[] for k in ('filtered','random','mixture_half')}
            for rep in range(reps):
                rng=np.random.default_rng(np.random.SeedSequence([20260919,dsid,seed,split,scenid,rep]))
                cf=rng.choice(filtered,n,False);cr=rng.choice(pool,n,False)
                half=cf[:n//2];remaining=np.setdiff1d(pool,half);cm=np.r_[half,rng.choice(remaining,n-len(half),False)]
                for method,cal in [('filtered',cf),('random',cr),('mixture_half',cm)]:
                    assert len(np.unique(cal))==n
                    p=pvals(s[cal],s[tt]);reject=bh(p);nr=int(reject.sum())
                    r=dict(**base,rep=rep,method=method,fdp=float(np.sum(reject&(yn==0))/max(nr,1)),
                           power=float(np.sum(reject&(yn==1))/max(int(yn.sum()),1)),discoveries=nr)
                    rows.append(r);samples[method].append(r)
            for method,g in samples.items():
                z=dict(**base,method=method,repetitions=reps)
                for key in ('fdp','power','discoveries'):
                    v=np.array([r[key] for r in g]);z[key+'_mean']=float(v.mean());z[key+'_mcse']=float(v.std(ddof=1)/np.sqrt(reps))
                if method=='mixture_half':z['null_tail001_exact']=np.nan
                else:
                    ppool=filtered if method=='filtered' else pool;M=len(ppool)
                    K=M-np.searchsorted(np.sort(s[ppool]),s[tt[yn==0]],side='left')
                    z['null_tail001_exact']=float(hypergeom.cdf(int(np.floor(.01*(n+1)+1e-10))-1,M,K,n).mean())
                summary.append(z)
        print('EVAL',dataset,detector,seed,'split',split,flush=True)
    stem=f'{dataset}_{detector}_{seed}'
    write_csv(OUT/f'evaluation_{stem}_trials.csv',rows);write_csv(OUT/f'evaluation_{stem}_summary.csv',summary)
    print('EVAL COMPLETE',stem,len(rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset');ap.add_argument('detector');ap.add_argument('seed',type=int);ap.add_argument('--reps',type=int,default=200)
    args=ap.parse_args();main(args.dataset,args.detector,args.seed,args.reps)
