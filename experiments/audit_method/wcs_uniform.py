"""Uniform acquisition at the same expected label budget as each biased design (WCS_UNIFORM_PROTOCOL.md)."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from stratified_calibration import D,G,DATASETS,make_strata
from selection_dose import exposure,score_ranks
from allocation_acquisition import pvalues,bh
from wcs_acquisition import ALPHA
from wcs_randomized import wcs_rand
from wcs_endtoend import F_TRAIN,SCEN

def run(datasets=('amazon','tfinance','weibo_gadbench')):
    rows=[];inputs={};started=time.time()
    for ds in datasets:
        dsid=DATASETS[ds];path=(G if ds=='amazon' else D)/f'{ds}_graph.npz';g=np.load(path)
        y=g['labels'].astype(int);x=g['features'].astype(np.float32)
        a=sparse.csr_matrix((np.ones(len(g['indices']),dtype=np.float32),g['indices'],g['indptr']),shape=(len(y),len(y)))
        deg=np.asarray(a.sum(axis=1)).ravel();avg=(a@x)/np.maximum(1,deg[:,None]);xg=np.c_[x,avg,np.log1p(deg)].astype(np.float32)
        inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for seed in range(10):
            c=np.load(D/f'{ds}_hgb_attributes_scores_{seed}.npz');panel=c['train'];deploy=c['deploy']
            normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
            obs=np.zeros(len(y));obs[panel]=y[panel];k=np.asarray(a@obs).ravel();e=exposure(a,y,panel)
            ld=np.log1p(deg);z=(ld-ld[deploy].mean())/ld[deploy].std()
            smooth=np.clip(expit(-1.2*np.sqrt(k)+.3*z),.01,.95)
            marks=np.random.default_rng(np.random.SeedSequence([2026092303,dsid,seed])).random(len(y))
            strata=make_strata(e,deploy,marks,'exposure_80')
            for split in range(5):
                rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                reference=np.setdiff1d(deploy,test);m=len(test);npos=max(int(y[test].sum()),1)
                for rep in range(3):
                    u=np.random.default_rng(np.random.SeedSequence([2026092407,dsid,seed,split,rep])).random(len(reference))
                    for si,(scen,kind,pp) in enumerate(SCEN):
                        rho_b=np.array(pp)[strata] if kind=='two' else smooth
                        # Fixed before test assignment (protocol revision 2): mean prescribed probability over the deployment population.
                        rho_u=float(rho_b[deploy].mean());rho=np.full(len(y),rho_u)
                        budget_u=rho_u*len(reference);budget_b=float(rho_b[reference].sum())
                        acquired=reference[u<rho_u]
                        to_train=np.random.default_rng(np.random.SeedSequence([2026092509,dsid,seed,split,rep,si])).random(len(acquired))<F_TRAIN
                        tr=acquired[to_train];rest=acquired[~to_train];cal=rest[y[rest]==0]
                        training=np.r_[panel,tr];w=1/rho
                        for mi,(model,feats) in enumerate((('hgb_attributes',x),('hgb_graph_features',xg))):
                            clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=.1,max_leaf_nodes=31,min_samples_leaf=20,
                                l2_regularization=1,early_stopping=False,random_state=seed)
                            clf.fit(feats[training],y[training]);s=clf.predict_proba(feats)[:,1]
                            rank=score_ranks(s,np.random.SeedSequence([2026092512,dsid,seed,split,rep,si,mi]))
                            xi=np.random.default_rng(np.random.SeedSequence([2026092510,dsid,seed,split,rep,si,mi])).random()
                            U=np.random.default_rng(np.random.SeedSequence([2026092511,dsid,seed,split,rep,si,mi])).random(m)
                            out={'pooled_reference':bh(pvalues(rank,cal,test))}
                            out['wcs_rand_homogeneous'],_=wcs_rand(rank,cal,test,w,U,xi)
                            auc=roc_auc_score(y[test],s[test])
                            for method,rr in out.items():
                                ids=test[rr];tp=int(y[ids].sum());kk=len(ids)
                                rows.append(dict(dataset=ds,model=model,seed=seed,split=split,rep=rep,scenario=scen,method=method,
                                    fdp=(kk-tp)/max(kk,1),power=tp/npos,discoveries=kk,n_acquired=len(acquired),n_reference=len(cal),
                                    n_train_new=len(tr),train_anomalies=int(y[training].sum()),auroc=auc,rho_u=rho_u,expected_budget_uniform=budget_u,expected_budget_biased=budget_b,
                                    bound=ALPHA*(m-int(y[test].sum()))/m))
            print('DONE',ds,seed,round(time.time()-started,1),flush=True)
    df=pd.DataFrame(rows);df.to_csv(D/'wcs_uniform_trials.csv.gz',index=False,compression='gzip')
    groups=['dataset','model','scenario','method'];metrics=['fdp','power','discoveries','n_acquired','n_reference','n_train_new','train_anomalies','auroc','bound','rho_u','expected_budget_uniform','expected_budget_biased']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/'wcs_uniform_seeds.csv',index=False)
    # biased arms from the end-to-end run, same keys
    b=pd.read_csv(D/'wcs_endtoend_trials.csv.gz');b=b[b.dataset.isin(datasets)]
    bs=b.groupby(groups+['seed'])[['fdp','power','n_reference','n_train_new','train_anomalies','auroc']].mean().reset_index()
    uw=seeds[seeds.method=='wcs_rand_homogeneous'];bw=bs[bs.method=='wcs_rand_homogeneous']
    pr=uw.merge(bw,on=['dataset','model','scenario','seed'],suffixes=('_uniform','_biased'),validate='one_to_one')
    out=[]
    for key,gdf in pr.groupby(['dataset','model','scenario']):
        dlt=gdf.power_uniform-gdf.power_biased;se=dlt.std(ddof=1)/np.sqrt(len(dlt));T9=2.262157
        rec=dict(zip(['dataset','model','scenario'],key),n_seeds=len(dlt),power_uniform=gdf.power_uniform.mean(),power_biased=gdf.power_biased.mean(),
                 diff=dlt.mean(),lo=dlt.mean()-T9*se,hi=dlt.mean()+T9*se,fdp_uniform=gdf.fdp_uniform.mean(),fdp_biased=gdf.fdp_biased.mean(),
                 bound=gdf.bound.mean(),n_reference_uniform=gdf.n_reference_uniform.mean(),n_reference_biased=gdf.n_reference_biased.mean(),
                 n_train_new_uniform=gdf.n_train_new_uniform.mean(),n_train_new_biased=gdf.n_train_new_biased.mean(),
                 train_anomalies_uniform=gdf.train_anomalies_uniform.mean(),train_anomalies_biased=gdf.train_anomalies_biased.mean(),
                 auroc_uniform=gdf.auroc_uniform.mean(),auroc_biased=gdf.auroc_biased.mean(),
                 expected_budget_uniform=gdf.expected_budget_uniform.mean(),expected_budget_biased=gdf.expected_budget_biased.mean())
        rec['verdict']='uniform better' if rec['lo']>0 else ('biased better' if rec['hi']<0 else 'no detectable difference')
        out.append(rec)
    pd.DataFrame(out).to_csv(D/'wcs_uniform_paired.csv',index=False)
    report=dict(rows=len(df),seconds=time.time()-started,
                protocol_sha256=hashlib.sha256((D/'WCS_UNIFORM_PROTOCOL.md').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs)
    (D/'wcs_uniform_manifest.json').write_text(json.dumps(report,indent=2));print(report['rows'],'rows',flush=True)

if __name__=='__main__':run()
