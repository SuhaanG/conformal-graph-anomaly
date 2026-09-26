"""End-to-end biased acquisition with scorers retrained on the biased labels (WCS_ENDTOEND_PROTOCOL.md)."""
from pathlib import Path
import itertools,json,hashlib,time,math,argparse
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from stratified_calibration import D,G,DATASETS,make_strata
from selection_dose import exposure,score_ranks
from allocation_acquisition import pvalues,bh
from wcs_acquisition import wcs_select,homogeneous_exact_fdp,ALPHA
from wcs_randomized import wcs_rand

F_TRAIN=.5
SCEN=[('moderate','two',(.5,.1)),('severe','two',(.5,.025)),('smooth','smooth',None)]

def validate():
    """Exact FDR when scores depend arbitrarily on the realized training set."""
    worst={'wcs_det_homog':-1.,'wcs_det_determ':-1.};wbh=[];cases=0;states_total=0
    for n in (4,5):
        for nt in (1,2):
            for na_t in (1,2):
                for na_x in (0,1):
                    for probs in ((.8,.2),(.5,.1),(.9,.3)):
                        for alpha in (.3,.5):
                            N=n+na_t+na_x;y=np.r_[np.zeros(n),np.ones(na_t+na_x)].astype(int)
                            prob=np.array(probs)[np.arange(N)%2];w=1/prob
                            base=np.random.default_rng([n,nt,na_t,na_x]).permutation(N).astype(float)
                            anom_test=np.arange(n,n+na_t);extra=np.arange(n+na_t,N)
                            sums={'wcs_det_homog':0.,'wcs_det_determ':0.,'weighted_bh':0.};total=0.
                            for tt in itertools.combinations(range(n),nt):
                                nontest=np.r_[np.setdiff1d(np.arange(n),tt),extra].astype(int);test=np.r_[tt,anom_test].astype(int)
                                for st in itertools.product((0,1,2),repeat=len(nontest)):  # 0 unused, 1 train, 2 calibration
                                    st=np.array(st);pr=np.where(st==0,1-prob[nontest],np.where(st==1,prob[nontest]*F_TRAIN,prob[nontest]*(1-F_TRAIN)))
                                    mass=pr.prod()/math.comb(n,nt);total+=mass;states_total+=1
                                    tr=tuple(sorted(nontest[st==1].tolist()))
                                    rank=base+np.random.default_rng([97,len(tr),*tr]).normal(0,3,N)  # scores depend on training set only
                                    cal=nontest[(st==2)&(y[nontest]==0)]
                                    rr,_,_=wcs_select(rank,cal,test,w,1.,alpha);sums['wcs_det_determ']+=mass*(1-y[test[rr]]).sum()/max(rr.sum(),1)
                                    sums['wcs_det_homog']+=mass*homogeneous_exact_fdp(rank,cal,test,w,y,alpha)
                                    rr=bh(pvalues(rank,cal,test,w),alpha);sums['weighted_bh']+=mass*(1-y[test[rr]]).sum()/max(rr.sum(),1)
                            assert abs(total-1)<1e-12
                            bound=alpha*nt/(nt+na_t)
                            for k in worst:
                                assert sums[k]<=bound+1e-12,(k,n,nt,na_t,na_x,probs,alpha,sums,bound)
                                worst[k]=max(worst[k],sums[k]-bound)
                            wbh.append(sums['weighted_bh']-bound);cases+=1
    report=dict(exact_cases=cases,exact_states=states_total,max_fdr_minus_bound=worst,
                weighted_bh_max_fdr_minus_bound=max(wbh),weighted_bh_cases_above=int(sum(x>1e-12 for x in wbh)))
    (D/'wcs_endtoend_validation.json').write_text(json.dumps(report,indent=2));print('VALIDATION',report,flush=True)

def run(datasets):
    validate();rows=[];inputs={};started=time.time()
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
                        rho=np.array(pp)[strata] if kind=='two' else smooth
                        acquired=reference[u<rho[reference]]
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
                            out={'pooled_reference':bh(pvalues(rank,cal,test)),'weighted_bh':bh(pvalues(rank,cal,test,w)),
                                 'wcs_homogeneous':wcs_select(rank,cal,test,w,xi)[0],'wcs_rand_homogeneous':wcs_rand(rank,cal,test,w,U,xi)[0]}
                            auc=roc_auc_score(y[test],s[test])
                            for method,rr in out.items():
                                ids=test[rr];tp=int(y[ids].sum());kk=len(ids)
                                rows.append(dict(dataset=ds,model=model,seed=seed,split=split,rep=rep,scenario=scen,method=method,
                                    fdp=(kk-tp)/max(kk,1),power=tp/npos,discoveries=kk,n_reference=len(cal),n_train_new=len(tr),
                                    train_anomalies=int(y[training].sum()),auroc=auc,bound=ALPHA*(m-int(y[test].sum()))/m))
            print('DONE',ds,seed,round(time.time()-started,1),flush=True)
    df=pd.DataFrame(rows);df.to_csv(D/'wcs_endtoend_trials.csv.gz',index=False,compression='gzip')
    groups=['dataset','model','scenario','method'];metrics=['fdp','power','discoveries','n_reference','n_train_new','train_anomalies','auroc','bound']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/'wcs_endtoend_seeds.csv',index=False)
    sp=df.groupby(groups+['seed','split']).fdp.mean().reset_index()
    se=sp.groupby(groups).fdp.agg(lambda v:v.std(ddof=1)/np.sqrt(len(v))).rename('fdp_se_testset')
    out=seeds.groupby(groups)[metrics].agg(['mean','std']);out.columns=['_'.join(c) for c in out.columns]
    out.join(se).reset_index().to_csv(D/'wcs_endtoend_summary.csv',index=False)
    report=dict(rows=len(df),seconds=time.time()-started,
                protocol_sha256=hashlib.sha256((D/'WCS_ENDTOEND_PROTOCOL.md').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs)
    (D/'wcs_endtoend_manifest.json').write_text(json.dumps(report,indent=2));print(report['rows'],'rows',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--datasets',nargs='*',default=['amazon','tfinance','weibo_gadbench'])
    args=ap.parse_args()
    if args.validate_only:validate()
    else:run(args.datasets)
