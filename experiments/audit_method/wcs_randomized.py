"""Randomized weighted p-values for WCS and the weighted resolution condition (WCS_RANDOMIZED_PROTOCOL.md)."""
from pathlib import Path
import itertools,json,hashlib,time,math,argparse
import numpy as np
import pandas as pd
from scipy import sparse
from stratified_calibration import D,G,MODELS,DATASETS,make_strata
from selection_dose import exposure,score_ranks
from allocation_acquisition import pvalues,bh,methods
from wcs_acquisition import wcs_sizes,wcs_select,ALPHA,SCENARIOS,DESIGNS
import wcs_experiment

def weighted_base(rank,cal,test,w):
    W=float(w[cal].sum())
    if not len(cal):return np.zeros(len(test)),W
    c=rank[cal];order=np.argsort(c);tail=np.r_[np.cumsum(w[cal][order][::-1])[::-1],0.]
    return tail[np.searchsorted(c[order],rank[test],side='right')],W

def wcs_rand(rank,cal,test,w,U,xi=1.,alpha=ALPHA):
    """Randomized p_j = (sum_C w 1{s>s_j} + U_j w_j)/(W + w_j); auxiliary values unchanged."""
    _,sizes=wcs_sizes(rank[cal].astype(float),w[cal],rank[test].astype(float),w[test],alpha)
    base,W=weighted_base(rank,cal,test,w);wt=w[test]
    p=(base+U*wt)/(W+wt)
    first=p<=alpha*sizes/len(test)
    return wcs_experiment.prune(first,sizes,xi),p

def floors(cal,test,w):
    W=float(w[cal].sum());wt=w[test];return wt/(W+wt)

def validate():
    rng=np.random.default_rng(2026092504)
    thresholds=np.linspace(.02,.98,49);exact_cases=0;mc_cases=0;worst=-1.;assignments=0
    for n in (4,5,6):
        for nt in (1,2,3):
            if nt>=n:continue
            for na in (0,1,2):
                for probs in ((.8,.2),(.5,.1),(.9,.3)):
                    for ordering in range(3):
                        N=n+na;strata=np.arange(N)%2;prob=np.array(probs)[strata];w=1/prob
                        y=np.r_[np.zeros(n),np.ones(na)].astype(int)
                        rank=np.random.default_rng([n,nt,na,ordering]).permutation(N)
                        if ordering==2:rank[n:]=np.arange(N-na,N);rank[:n]=np.random.default_rng(ordering).permutation(n)
                        # 1. Exact design-averaged uniformity of a null randomized p-value.
                        cdf=np.zeros(len(thresholds));total=0.;support=[]
                        for tt in itertools.combinations(range(n),nt):
                            rem=np.setdiff1d(np.arange(n),tt);test=np.r_[tt,np.arange(n,N)].astype(int)
                            for flags in itertools.product((False,True),repeat=len(rem)):
                                flags=np.array(flags,bool);cal=rem[flags]
                                mass=np.prod(np.where(flags,prob[rem],1-prob[rem]))/math.comb(n,nt)
                                total+=mass;assignments+=1;support.append((mass,test,cal))
                                base,W=weighted_base(rank,cal,test,w);wt=w[test]
                                pr=np.clip((thresholds[None,:]*(W+wt[:nt,None])-base[:nt,None])/wt[:nt,None],0,1)
                                cdf+=mass*pr.mean(axis=0)
                        assert abs(total-1)<1e-12 and np.allclose(cdf,thresholds,atol=1e-12),(n,nt,na,probs,ordering)
                        exact_cases+=1
                        # 2. Monte Carlo FDR of randomized WCS (homogeneous xi also random) over design and U.
                        if n<=5 and ordering<2:
                            for alpha in (.3,.5):
                                masses=np.array([s[0] for s in support]);draws=10000
                                idx=rng.choice(len(support),draws,p=masses/masses.sum());f=np.empty(draws)
                                for d,i in enumerate(idx):
                                    _,test,cal=support[i];U=rng.random(len(test));xi=rng.random()
                                    rr,_=wcs_rand(rank,cal,test,w,U,xi,alpha)
                                    f[d]=(1-y[test[rr]]).sum()/max(rr.sum(),1)
                                bound=alpha*nt/(nt+na);se=f.std(ddof=1)/np.sqrt(draws) if f.std()>0 else 0.
                                assert f.mean()<=bound+4*se+1e-12,(n,nt,na,probs,ordering,alpha,f.mean(),bound,se)
                                worst=max(worst,(f.mean()-bound)/max(se,1e-12));mc_cases+=1
    report=dict(exact_uniformity_cases=exact_cases,exact_assignments=assignments,thresholds=len(thresholds),
                mc_fdr_cases=mc_cases,mc_draws_per_case=10000,max_fdr_minus_bound_in_se=worst)
    (D/'wcs_randomized_validation.json').write_text(json.dumps(report,indent=2));print('VALIDATION',report,flush=True)

def run(datasets):
    validate();rows=[];inputs={};started=time.time();violations=0;checked=0
    for ds in datasets:
        dsid=DATASETS[ds];path=(G if ds=='amazon' else D)/f'{ds}_graph.npz';g=np.load(path)
        y=g['labels'].astype(int);a=sparse.csr_matrix((np.ones(len(g['indices'])),g['indices'],g['indptr']),shape=(len(y),len(y)))
        inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for seed in range(10):
            ranks={};train=None
            for mid,model in enumerate(MODELS):
                path=D/f'{ds}_{model}_scores_{seed}.npz';c=np.load(path)
                inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
                if train is None:train=c['train'];deploy=c['deploy']
                ranks[model]=score_ranks(c['scores'],np.random.SeedSequence([20260923,dsid,seed,mid,7]))
            normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
            marks=np.random.default_rng(np.random.SeedSequence([2026092303,dsid,seed])).random(len(y))
            fixed={d:make_strata(exposure(a,y,train),deploy,marks,d) for d in DESIGNS}
            for split in range(5):
                rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                reference=np.setdiff1d(deploy,test);m=len(test);anom_test=y[test]==1
                for di,(design,strata) in enumerate(fixed.items()):
                    for rep in range(10):
                        u=np.random.default_rng(np.random.SeedSequence([2026092407,dsid,seed,split,rep])).random(len(reference))
                        for si,(scenario,pp) in enumerate(SCENARIOS):
                            prob=np.array(pp)[strata];revealed=reference[u<prob[reference]];cal=revealed[y[revealed]==0];w=1/prob
                            f=floors(cal,test,w);need=m*f/ALPHA;hi=strata[test]==1
                            diag=dict(kish_ess=float(w[cal].sum()**2/max((w[cal]**2).sum(),1e-300)),
                                      need_anom_low=float(np.median(need[anom_test&~hi])) if (anom_test&~hi).any() else np.nan,
                                      need_anom_high=float(np.median(need[anom_test&hi])) if (anom_test&hi).any() else np.nan,
                                      frac_anom_floor_ok=float((f[anom_test]<=ALPHA).mean()),
                                      frac_anom_high=float(hi[anom_test].mean()))
                            for mi,(model,rank) in enumerate(ranks.items()):
                                out=methods(rank,cal,test,strata,prob)
                                xi=np.random.default_rng(np.random.SeedSequence([2026092501,dsid,seed,split,di,rep,si,mi])).random()
                                U=np.random.default_rng(np.random.SeedSequence([2026092503,dsid,seed,split,di,rep,si,mi])).random(m)
                                out['wcs_deterministic'],_,_=wcs_select(rank,cal,test,w,1.)
                                out['wcs_homogeneous'],_,_=wcs_select(rank,cal,test,w,xi)
                                out['wcs_rand_deterministic'],_=wcs_rand(rank,cal,test,w,U,1.)
                                out['wcs_rand_homogeneous'],pr=wcs_rand(rank,cal,test,w,U,xi)
                                out['weighted_bh_rand']=bh(pr)
                                # Necessary resolution condition for deterministic weighted p-values.
                                for name,scale in (('weighted_bh',1.),('wcs_deterministic',1.),('wcs_homogeneous',xi)):
                                    rr=out[name]
                                    if rr.any():
                                        checked+=1
                                        if rr.sum()<scale*need[rr].max()-1e-9:violations+=1
                                npos=max(int(anom_test.sum()),1)
                                for method,rr in out.items():
                                    ids=test[rr];tp=int(y[ids].sum());k=len(ids)
                                    rows.append(dict(dataset=ds,model=model,seed=seed,split=split,design=design,scenario=scenario,rep=rep,
                                        method=method,fdp=(k-tp)/max(k,1),power=tp/npos,discoveries=k,nonempty=int(k>0),n_reference=len(cal),**diag))
            print('DONE',ds,seed,round(time.time()-started,1),flush=True)
    assert violations==0,violations
    df=pd.DataFrame(rows)
    old=pd.read_csv(D/'wcs_acquisition_trials.csv.gz');old=old[old.dataset.isin(datasets)]
    keys=['dataset','model','seed','split','design','scenario','rep','method']
    v=old.merge(df,on=keys,validate='one_to_one',suffixes=('_old','_new'));assert len(v)==len(old)
    for m in ('fdp','power','discoveries'):assert np.allclose(v[m+'_old'],v[m+'_new'],rtol=1e-12,atol=1e-12),m
    df.to_csv(D/'wcs_randomized_trials.csv.gz',index=False,compression='gzip')
    groups=['dataset','model','design','scenario','method']
    metrics=['fdp','power','discoveries','nonempty','n_reference','kish_ess','need_anom_low','need_anom_high','frac_anom_floor_ok','frac_anom_high']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/'wcs_randomized_seeds.csv',index=False)
    out=seeds.groupby(groups)[metrics].agg(['mean','std']);out.columns=['_'.join(c) for c in out.columns]
    out.reset_index().to_csv(D/'wcs_randomized_summary.csv',index=False)
    report=dict(rows=len(df),previous_outcomes_reproduced=len(v),resolution_checks=checked,resolution_violations=violations,
                seconds=time.time()-started,protocol_sha256=hashlib.sha256((D/'WCS_RANDOMIZED_PROTOCOL.md').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs)
    (D/'wcs_randomized_manifest.json').write_text(json.dumps(report,indent=2));print(report['rows'],'rows',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--datasets',nargs='*',default=['amazon','tfinance','weibo_gadbench'])
    args=ap.parse_args()
    if args.validate_only:validate()
    else:run(args.datasets)
