"""Exploratory, fixed supervised controls; no audit-label reuse or tuning."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,hashlib,sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from run_certification import trials,BUDGETS
D=Path(__file__).resolve().parent
G=D.parent/'aistats_followup/h200_results/paper/aistats_followup'
sys.path.insert(0,str(D.parent/'aistats_revision'))
from validate_revision import pvals,bh

def main():
    started=time.time();evals=[];diagnostics=[];base=[]
    for dsid,ds in enumerate(('amazon','tolokers')):
        g=np.load(G/f'{ds}_graph.npz');y=g['labels'].astype(int);x=g['features'];degree=g['degree']
        a=sparse.csr_matrix((np.ones(len(g['indices'])),g['indices'],g['indptr']),shape=(len(y),len(y)))
        avg=(a@x)/np.maximum(np.asarray(a.sum(axis=1)),1)
        xg=np.c_[x,avg,np.log1p(degree)].astype(np.float32)
        verified=np.arange(3305,len(y)) if ds=='amazon' else np.arange(len(y))
        oracle=np.asarray(a@y).ravel()==0
        for seed in range(10):
            rng=np.random.default_rng(np.random.SeedSequence([20260924,dsid,seed]))
            train=np.sort(rng.choice(verified,int(round(.1*len(verified))),False));deploy=np.setdiff1d(verified,train)
            assert not np.intersect1d(train,deploy).size
            for mid,(model,features) in enumerate((('hgb_attributes',x),('hgb_graph_features',xg))):
                dest=D/f'{ds}_{model}_scores_{seed}.npz'
                t=time.time()
                if dest.exists():
                    c=np.load(dest);assert np.array_equal(c['train'],train);s=c['scores']
                else:
                    clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=.1,max_leaf_nodes=31,
                        min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=seed)
                    with threadpool_limits(limits=1):
                        clf.fit(features[train],y[train]);s=clf.predict_proba(features)[:,1]
                    np.savez_compressed(dest,scores=s,train=train,deploy=deploy,labels=y)
                assert np.isfinite(s).all()
                yy=y[deploy];dd=degree[deploy];N=len(deploy);ss=s[deploy]
                ends=np.array(sorted(set([25,50,100,200,400,800,1600,int(.2*N)])))
                # For smaller N, the prespecified 1600 endpoint may exceed .2N;
                # retain this fixed endpoint and define the union by max(ends).
                order=np.lexsort((deploy,-ss));tp=np.cumsum(yy[order])[ends-1]
                for n,ntp in zip(ends,tp):
                    diagnostics.append(dict(dataset=ds,model=model,seed=seed,training_labels=len(train),
                        training_anomalies=int(y[train].sum()),n_deploy=N,auroc=roc_auc_score(yy,ss),
                        average_precision=average_precision_score(yy,ss),n=int(n),fdp=(n-ntp)/n,
                        power=ntp/yy.sum(),true_discoveries=int(ntp)))
                for b in BUDGETS:
                    for methodid,method in enumerate(('uniform_candidates','score_strata','score_degree_strata','uniform_graph')):
                        rng=np.random.default_rng(np.random.SeedSequence([20260925,dsid,mid,seed,b,methodid]))
                        z=trials(yy,dd,order,ends,method,b,rng)
                        for key,value in dict(dataset=ds,model=model,seed=seed,method=method,budget=b,training_labels=len(train),n_deploy=N).items():z[key]=value
                        evals.append(z)
                normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
                for split in range(5):
                    rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                    test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                    pool=np.setdiff1d(normals,test);filtered=pool[oracle[pool]];n=min(1000,len(filtered))
                    for method in ('filtered','matched_random','full_reference'):
                        count=200 if method=='matched_random' else 1
                        for rep in range(count):
                            if method=='filtered':cal=filtered
                            elif method=='full_reference':cal=pool
                            else:cal=rng.choice(pool,n,False)
                            r=bh(pvals(s[cal],s[test]));nr=int(r.sum());ntp=int(y[test[r]].sum())
                            base.append(dict(dataset=ds,model=model,seed=seed,split=split,rep=rep,method=method,
                                n_calib=len(cal),training_labels=len(train),reference_label_panel=len(deploy)-len(test),
                                fdp=(nr-ntp)/max(nr,1),power=ntp/max(int(y[test].sum()),1),discoveries=nr,true_discoveries=ntp))
                print('STRONG',ds,model,seed,'seconds',round(time.time()-t),flush=True)
            pd.concat(evals,ignore_index=True).to_csv(D/'strong_certificate_partial.csv',index=False)
            pd.DataFrame(diagnostics).to_csv(D/'strong_diagnostics_partial.csv',index=False)
    pd.concat(evals,ignore_index=True).to_csv(D/'strong_certificate_trials.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(D/'strong_diagnostics.csv',index=False)
    pd.DataFrame(base).to_csv(D/'strong_baseline_trials.csv',index=False)
    (D/'strong_manifest.json').write_text(json.dumps(dict(completed_models=40,training_seeds=10,
        seconds=time.time()-started,training_master_seed=20260924,audit_master_seed=20260925,
        outputs={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in
                 ['strong_certificate_trials.csv','strong_diagnostics.csv','strong_baseline_trials.csv']}),indent=2))
    print('STRONG COMPLETE',len(evals)*200,'audit trials',len(base),'baseline rows',flush=True)
if __name__=='__main__':main()
