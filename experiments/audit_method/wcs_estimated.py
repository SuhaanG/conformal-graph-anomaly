"""Randomized WCS with estimated acquisition propensities (WCS_ESTIMATED_PROTOCOL.md)."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from stratified_calibration import D,G,MODELS,DATASETS,make_strata
from selection_dose import exposure,score_ranks
from allocation_acquisition import pvalues,bh
from wcs_randomized import wcs_rand
from wcs_acquisition import ALPHA

def fit_predict(X,z,fit_idx,pred_idx):
    sc=StandardScaler().fit(X[fit_idx]);lr=LogisticRegression(C=1.,max_iter=2000).fit(sc.transform(X[fit_idx]),z[fit_idx])
    return lr.predict_proba(sc.transform(X[pred_idx]))[:,1]

def run(datasets=('amazon','tfinance','weibo_gadbench')):
    rows=[];inputs={};started=time.time()
    for ds in datasets:
        dsid=DATASETS[ds];path=(G if ds=='amazon' else D)/f'{ds}_graph.npz';g=np.load(path)
        y=g['labels'].astype(int);a=sparse.csr_matrix((np.ones(len(g['indices'])),g['indices'],g['indptr']),shape=(len(y),len(y)))
        inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();deg=np.asarray(a.sum(axis=1)).ravel()
        for seed in range(10):
            ranks={};train=None
            for mid,model in enumerate(MODELS):
                p=D/f'{ds}_{model}_scores_{seed}.npz';c=np.load(p);inputs[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
                if train is None:train=c['train'];deploy=c['deploy']
                ranks[model]=score_ranks(c['scores'],np.random.SeedSequence([20260923,dsid,seed,mid,7]))
            normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
            obs=np.zeros(len(y));obs[train]=y[train];k=np.asarray(a@obs).ravel();e=exposure(a,y,train)
            ld=np.log1p(deg);z=(ld-ld[deploy].mean())/ld[deploy].std()
            rho=np.clip(expit(-1.2*np.sqrt(k)+.3*z),.01,.95)
            X=np.c_[np.sqrt(k),k,ld,e]
            marks=np.random.default_rng(np.random.SeedSequence([2026092303,dsid,seed])).random(len(y))
            strata=make_strata(e,deploy,marks,'exposure_80')
            for split in range(5):
                rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                reference=np.setdiff1d(deploy,test);m=len(test);npos=max(int(y[test].sum()),1)
                for rep in range(10):
                    u=np.random.default_rng(np.random.SeedSequence([2026092407,dsid,seed,split,rep])).random(len(reference))
                    acq=u<rho[reference];revealed=reference[acq];cal=revealed[y[revealed]==0]
                    zind=np.zeros(len(y),int);zind[revealed]=1
                    # Cross-fitted logistic propensity on non-test nodes; test nodes average both folds.
                    fold=np.random.default_rng(np.random.SeedSequence([2026092506,dsid,seed,split,rep])).random(len(reference))<.5
                    rho_hat=np.ones(len(y))
                    A,B=reference[fold],reference[~fold]
                    rho_hat[B]=fit_predict(X,zind,A,B);rho_hat[A]=fit_predict(X,zind,B,A)
                    rho_hat[test]=(fit_predict(X,zind,A,test)+fit_predict(X,zind,B,test))/2
                    rho_hat=np.clip(rho_hat,.005,1.)
                    rho_s=np.ones(len(y))
                    for h in (0,1):
                        members=reference[strata[reference]==h];rate=max(zind[members].mean(),1e-3)
                        rho_s[deploy[strata[deploy]==h]]=rate
                    for mi,(model,rank) in enumerate(ranks.items()):
                        xi=np.random.default_rng(np.random.SeedSequence([2026092507,dsid,seed,split,rep,mi])).random()
                        U=np.random.default_rng(np.random.SeedSequence([2026092508,dsid,seed,split,rep,mi])).random(m)
                        out={'pooled_reference':bh(pvalues(rank,cal,test)),
                             'weighted_bh_oracle':bh(pvalues(rank,cal,test,1/rho)),
                             'wcs_rand_oracle':wcs_rand(rank,cal,test,1/rho,U,xi)[0],
                             'wcs_rand_logit':wcs_rand(rank,cal,test,1/rho_hat,U,xi)[0],
                             'wcs_rand_strata2':wcs_rand(rank,cal,test,1/rho_s,U,xi)[0]}
                        for method,rr in out.items():
                            ids=test[rr];tp=int(y[ids].sum());kk=len(ids)
                            rows.append(dict(dataset=ds,model=model,seed=seed,split=split,rep=rep,method=method,
                                fdp=(kk-tp)/max(kk,1),power=tp/npos,discoveries=kk,n_reference=len(cal),
                                bound=ALPHA*(m-int(y[test].sum()))/m,
                                rho_mae=float(np.abs(rho_hat[deploy]-rho[deploy]).mean()),
                                rho_min=float(rho[deploy].min()),acq_rate=float(acq.mean())))
            print('DONE',ds,seed,round(time.time()-started,1),flush=True)
    df=pd.DataFrame(rows);df.to_csv(D/'wcs_estimated_trials.csv.gz',index=False,compression='gzip')
    groups=['dataset','model','method'];metrics=['fdp','power','discoveries','n_reference','bound','rho_mae','acq_rate']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/'wcs_estimated_seeds.csv',index=False)
    sp=df.groupby(groups+['seed','split']).fdp.mean().reset_index()
    se=sp.groupby(groups).fdp.agg(lambda x:x.std(ddof=1)/np.sqrt(len(x))).rename('fdp_se_testset')
    out=seeds.groupby(groups)[metrics].agg(['mean','std']);out.columns=['_'.join(c) for c in out.columns]
    out.join(se).reset_index().to_csv(D/'wcs_estimated_summary.csv',index=False)
    report=dict(rows=len(df),seconds=time.time()-started,
                protocol_sha256=hashlib.sha256((D/'WCS_ESTIMATED_PROTOCOL.md').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs)
    (D/'wcs_estimated_manifest.json').write_text(json.dumps(report,indent=2));print(report['rows'],'rows',flush=True)

if __name__=='__main__':run()
