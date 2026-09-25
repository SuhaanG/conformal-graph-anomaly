"""Frozen T-Finance / GADBench Weibo supervised dose extension, sparse CPU."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, csv, gzip, hashlib, json, time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from selection_dose import exposure,score_ranks,evaluate,FRACTIONS,REVEAL,REPS

D=Path(__file__).resolve().parent


def main(ds):
    dsid={'tfinance':3,'weibo_gadbench':2}[ds]
    path=D/f'{ds}_graph.npz';g=np.load(path);y=g['labels'].astype(int);x=g['features']
    assert set(np.unique(y))=={0,1} and np.isfinite(x).all()
    a=sparse.csr_matrix((np.ones(len(g['indices']),dtype=np.float32),g['indices'],g['indptr']),shape=(len(y),len(y)))
    assert (a!=a.T).nnz==0 and not a.diagonal().any()
    degree=np.asarray(a.sum(axis=1)).ravel()
    assert np.allclose(degree,g['degree'])
    avg=(a@x)/np.maximum(1,degree[:,None]);xg=np.c_[x,avg,np.log1p(degree)].astype(np.float32)
    started=time.time();diagnostics=[];summaries=[];checks=0;nrows=0
    dest=D/f'{ds}_dose_trials.csv.gz'
    fields=['dataset','model','seed','split','label_fraction','removed','method','rep','n_calib','pool_size','fdp','power','discoveries','nonempty']
    with gzip.open(dest,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for seed in range(10):
            rng=np.random.default_rng(np.random.SeedSequence([20260924,dsid,seed]))
            train=np.sort(rng.choice(len(y),int(round(.1*len(y))),False));deploy=np.setdiff1d(np.arange(len(y)),train)
            models={}
            for mid,(model,xx) in enumerate((('hgb_attributes',x),('hgb_graph_features',xg))):
                out=D/f'{ds}_{model}_scores_{seed}.npz'
                if out.exists():
                    c=np.load(out);assert np.array_equal(c['train'],train) and np.array_equal(c['labels'],y);s=c['scores']
                else:
                    clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=.1,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=seed)
                    with threadpool_limits(limits=1):
                        clf.fit(xx[train],y[train]);s=clf.predict_proba(xx)[:,1]
                    np.savez_compressed(out,scores=s,train=train,deploy=deploy,labels=y)
                diagnostics.append(dict(dataset=ds,model=model,seed=seed,training_labels=len(train),training_anomalies=int(y[train].sum()),n_deploy=len(deploy),auroc=roc_auc_score(y[deploy],s[deploy]),average_precision=average_precision_score(y[deploy],s[deploy])))
                models[model]=score_ranks(s,np.random.SeedSequence([20260923,dsid,seed,mid,7]))
            normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
            for split in range(5):
                rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                reference=np.setdiff1d(deploy,test)
                rng=np.random.default_rng(np.random.SeedSequence([20260928,dsid,seed,split]));u=rng.random(len(reference))
                prepared={}
                for model,rank in models.items():
                    order=test[np.argsort(-rank[test])];prepared[model]=(order,np.cumsum(y[order]),.1*np.arange(1,len(test)+1)/len(test))
                for rho in REVEAL:
                    revealed=reference[u<rho];known=np.r_[train,revealed];pool=revealed[y[revealed]==0]
                    e=exposure(a,y,known)
                    assert np.array_equal(e,exposure(a,y,known[y[known]==1]));checks+=1
                    assert not np.intersect1d(known,test).size and not np.intersect1d(pool,train).size
                    acc={(m,d,k):[] for m in models for d in FRACTIONS for k in ('exposure','random')}
                    for rep in range(REPS):
                        rng=np.random.default_rng(np.random.SeedSequence([20260923,dsid,seed,split,int(rho*100),rep]))
                        filtered_order=pool[np.lexsort((rng.random(len(pool)),e[pool]))];random_order=rng.permutation(pool)
                        for drop in FRACTIONS:
                            n=int(np.ceil((1-drop)*len(pool)));assert n>0
                            for model,rank in models.items():
                                zero=[]
                                for method,order in (('exposure',filtered_order),('random',random_order)):
                                    val=evaluate(rank,order[:n],*prepared[model]);zero.append(val);acc[model,drop,method].append(val)
                                    writer.writerow(dict(zip(fields,[ds,model,seed,split,rho,drop,method,rep,n,len(pool),*val])));nrows+=1
                                if drop==0:assert zero[0]==zero[1]
                    for (model,drop,method),values in acc.items():
                        vals=np.asarray(values);row=dict(dataset=ds,model=model,seed=seed,split=split,label_fraction=rho,removed=drop,method=method,n_calib=int(np.ceil((1-drop)*len(pool))),pool_size=len(pool))
                        for j,key in enumerate(('fdp','power','discoveries','nonempty')):
                            row[key]=vals[:,j].mean();row[key+'_mcse']=vals[:,j].std(ddof=1)/np.sqrt(REPS)
                        summaries.append(row)
            print('EXTENSION',ds,seed,'rows',nrows,'seconds',round(time.time()-started),flush=True)
            pd.DataFrame(diagnostics).to_csv(D/f'{ds}_diagnostics.csv',index=False)
    df=pd.DataFrame(summaries);df.to_csv(D/f'{ds}_dose_splits.csv',index=False)
    groups=['dataset','model','label_fraction','removed','method'];metrics=['fdp','power','discoveries','nonempty','n_calib','pool_size']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/f'{ds}_dose_seeds.csv',index=False)
    agg=seeds.groupby(groups)[metrics].agg(['mean','std']);agg.columns=['_'.join(c) for c in agg.columns];agg.reset_index().to_csv(D/f'{ds}_dose_summary.csv',index=False)
    files=[path,dest,D/f'{ds}_dose_splits.csv',D/f'{ds}_dose_seeds.csv',D/f'{ds}_dose_summary.csv',D/f'{ds}_diagnostics.csv',*sorted(D.glob(f'{ds}_hgb_*_scores_*.npz'))]
    (D/f'{ds}_extension_manifest.json').write_text(json.dumps(dict(rows=nrows,anomaly_only_exposure_checks=checks,seconds=time.time()-started,protocol_sha256=hashlib.sha256((D/'TFINANCE_PROTOCOL.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['tfinance','weibo_gadbench']);args=parser.parse_args();main(args.dataset)
