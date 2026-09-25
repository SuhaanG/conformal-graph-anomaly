"""WCS under the frozen exposure-biased acquisition design (WCS_ACQUISITION_PROTOCOL.md)."""
from pathlib import Path
import sys,itertools,json,hashlib,time,math,argparse
import numpy as np
import pandas as pd
from scipy import sparse
from stratified_calibration import D,G,MODELS,DATASETS,make_strata
from selection_dose import exposure,score_ranks
from allocation_acquisition import pvalues,bh,methods
sys.path.insert(0,str(D.parent/'aistats_followup'))
import wcs_experiment

ALPHA=.1
SCENARIOS=[('neutral',(.5,.5)),('moderate',(.5,.1)),('severe',(.5,.025))]
DESIGNS=('zero_exposure','exposure_80')

def wcs_sizes(c,wc,t,wt,alpha=ALPHA):
    """Weighted p-values and Jin--Candes BH sizes |R_j| for distinct scores, O(m) per weight value.

    Auxiliary p-values p_l^(j) keep the descending score order of the other tests,
    with the designated test j placed first at value zero, so each BH count reduces
    to a prefix maximum over tests scored above j and a suffix maximum below it."""
    m=len(t);W=float(wc.sum())
    if len(c):
        order=np.argsort(c);tail=np.r_[np.cumsum(wc[order][::-1])[::-1],0.]
        base=tail[np.searchsorted(c[order],t,side='right')]
    else:base=np.zeros(m)
    p=(base+wt)/(W+wt)
    o=np.argsort(-t,kind='stable');b=base[o];pos=np.empty(m,int);pos[o]=np.arange(1,m+1)
    i=np.arange(1,m+1);sizes=np.ones(m,int)
    for u in np.unique(wt):
        A=np.where(b<=alpha*(i+1)*(W+u)/m,i+1,0)
        B=np.where(b+u<=alpha*i*(W+u)/m,i,0)
        pre=np.r_[0,np.maximum.accumulate(A)]            # pre[r-1] = max_{i<r} A_i
        suf=np.r_[np.maximum.accumulate(B[::-1])[::-1],0] # suf[r] = max_{i>r} B_i
        js=np.flatnonzero(wt==u);r=pos[js]
        sizes[js]=np.maximum(1,np.maximum(pre[r-1],suf[r]))
    return p,sizes

def wcs_select(rank,cal,test,w,xi=1.,alpha=ALPHA):
    p,sizes=wcs_sizes(rank[cal].astype(float),w[cal],rank[test].astype(float),w[test],alpha)
    first=p<=alpha*sizes/len(test)
    return wcs_experiment.prune(first,sizes,xi),first,sizes

def homogeneous_exact_fdp(rank,cal,test,w,y,alpha):
    """Average FDP of homogeneous pruning over xi ~ U(0,1), exactly, via breakpoints."""
    _,first,sizes=wcs_select(rank,cal,test,w,1.,alpha)
    if not first.any():return 0.
    bps=np.unique(np.r_[0.,1.,[k/s for s in np.unique(sizes[first]) for k in range(1,len(test)+1) if k/s<1]])
    total=0.
    for lo,hi in zip(bps[:-1],bps[1:]):
        rr=wcs_experiment.prune(first,sizes,(lo+hi)/2)
        total+=(hi-lo)*(1-y[test[rr]]).sum()/max(rr.sum(),1)
    return total

def validate():
    rng=np.random.default_rng(2026092502)
    # 1. Fast implementation equals the validated m-by-m implementation.
    agree=0
    for rep in range(400):
        n=int(rng.integers(0,30));m=int(rng.integers(1,40));vals=rng.permutation(n+m).astype(float)
        c,t=vals[:n],vals[n:]
        if rep%2:wc,wt=rng.choice([2.,10.,40.],n),rng.choice([2.,10.,40.],m)
        else:wc,wt=np.exp(rng.normal(size=n)),np.exp(rng.normal(size=m))
        alpha=float(rng.choice([.1,.2,.5]))
        p,s=wcs_sizes(c,wc,t,wt,alpha)
        if n:
            ref=wcs_experiment.wcs(c,t,wc,wt,alpha=alpha)
            assert np.allclose(p,ref[2]) and np.array_equal(s,ref[3]),rep
        else:
            aux=np.array([[0. if l==j else wt[j]*(t[j]>t[l])/wt[j] for l in range(m)] for j in range(m)])
            assert np.array_equal(s,np.array([bh(a,alpha).sum() for a in aux]))
        agree+=1
    # 2. Exact finite-population FDR under uniform test assignment + Bernoulli(rho_h) acquisition.
    cases=0;assignments=0;worst={'wcs_deterministic':-1.,'wcs_homogeneous':-1.};wbh=[]
    for n in (4,5,6):
        for nt in (1,2,3):
            if nt>=n:continue
            for na in (0,1,2):
                for probs in ((.8,.2),(.5,.1),(.9,.3)):
                    for alpha in (.3,.5):
                        for ordering in range(3):
                            N=n+na;strata=np.arange(N)%2;prob=np.array(probs)[strata];w=1/prob
                            y=np.r_[np.zeros(n),np.ones(na)].astype(int)
                            rank=np.random.default_rng([n,nt,na,ordering]).permutation(N)
                            if ordering==2:rank[n:]=np.arange(N-na,N);rank[:n]=np.random.default_rng(ordering).permutation(n)
                            sums={'wcs_deterministic':0.,'wcs_homogeneous':0.,'weighted_bh':0.};total=0.
                            for tt in itertools.combinations(range(n),nt):
                                rem=np.setdiff1d(np.arange(n),tt);test=np.r_[tt,np.arange(n,N)].astype(int)
                                for flags in itertools.product((False,True),repeat=len(rem)):
                                    flags=np.array(flags,bool);cal=rem[flags]
                                    mass=np.prod(np.where(flags,prob[rem],1-prob[rem]))/math.comb(n,nt)
                                    total+=mass;assignments+=1
                                    rr,_,_=wcs_select(rank,cal,test,w,1.,alpha)
                                    sums['wcs_deterministic']+=mass*(1-y[test[rr]]).sum()/max(rr.sum(),1)
                                    sums['wcs_homogeneous']+=mass*homogeneous_exact_fdp(rank,cal,test,w,y,alpha)
                                    rr=bh(pvalues(rank,cal,test,w),alpha)
                                    sums['weighted_bh']+=mass*(1-y[test[rr]]).sum()/max(rr.sum(),1)
                            assert abs(total-1)<1e-12
                            bound=alpha*nt/(nt+na)
                            for k in worst:
                                assert sums[k]<=bound+1e-12,(k,n,nt,na,probs,alpha,ordering,sums,bound)
                                worst[k]=max(worst[k],sums[k]-bound)
                            wbh.append(sums['weighted_bh']-bound);cases+=1
    report=dict(fast_vs_reference_cases=agree,exact_design_cases=cases,exact_assignments=assignments,
                max_fdr_minus_bound=worst,weighted_bh_max_fdr_minus_bound=max(wbh),
                weighted_bh_cases_above_bound=int(sum(x>1e-12 for x in wbh)))
    (D/'wcs_validation.json').write_text(json.dumps(report,indent=2));print('VALIDATION',report,flush=True)
    return report

def run(datasets):
    validate();rows=[];inputs={};started=time.time()
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
                assert np.array_equal(c['train'],train) and np.array_equal(c['deploy'],deploy) and np.array_equal(c['labels'],y)
                ranks[model]=score_ranks(c['scores'],np.random.SeedSequence([20260923,dsid,seed,mid,7]))
            normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
            marks=np.random.default_rng(np.random.SeedSequence([2026092303,dsid,seed])).random(len(y))
            e_train=exposure(a,y,train)
            fixed={d:make_strata(e_train,deploy,marks,d) for d in DESIGNS}
            for split in range(5):
                rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                reference=np.setdiff1d(deploy,test)
                for di,(design,strata) in enumerate(fixed.items()):
                    for rep in range(10):
                        u=np.random.default_rng(np.random.SeedSequence([2026092407,dsid,seed,split,rep])).random(len(reference))
                        for si,(scenario,pp) in enumerate(SCENARIOS):
                            prob=np.array(pp)[strata];revealed=reference[u<prob[reference]];cal=revealed[y[revealed]==0]
                            w=1/prob
                            for mi,(model,rank) in enumerate(ranks.items()):
                                out=methods(rank,cal,test,strata,prob)
                                xi=np.random.default_rng(np.random.SeedSequence([2026092501,dsid,seed,split,di,rep,si,mi])).random()
                                out['wcs_deterministic'],_,_=wcs_select(rank,cal,test,w,1.)
                                out['wcs_homogeneous'],_,_=wcs_select(rank,cal,test,w,xi)
                                npos=max(int(y[test].sum()),1)
                                for method,rr in out.items():
                                    ids=test[rr];tp=int(y[ids].sum());k=len(ids)
                                    rows.append(dict(dataset=ds,model=model,seed=seed,split=split,design=design,scenario=scenario,rep=rep,
                                        method=method,fdp=(k-tp)/max(k,1),power=tp/npos,discoveries=k,nonempty=int(k>0),n_reference=len(cal)))
            print('DONE',ds,seed,round(time.time()-started,1),flush=True)
    df=pd.DataFrame(rows)
    old=pd.read_csv(D/'allocation_acquisition_trials.csv.gz')
    old=old[(old.phase=='acquisition')&old.dataset.isin(datasets)&old.method.isin(df.method.unique())]
    keys=['dataset','model','seed','split','design','scenario','rep','method']
    v=old.merge(df,on=keys,validate='one_to_one',suffixes=('_old','_new'))
    assert len(v)==len(old)
    for m in ('fdp','power','discoveries'):assert np.allclose(v[m+'_old'],v[m+'_new'],rtol=1e-12,atol=1e-12),m
    df.to_csv(D/'wcs_acquisition_trials.csv.gz',index=False,compression='gzip')
    groups=['dataset','model','design','scenario','method'];metrics=['fdp','power','discoveries','nonempty','n_reference']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/'wcs_acquisition_seeds.csv',index=False)
    out=seeds.groupby(groups)[metrics].agg(['mean','std']);out.columns=['_'.join(c) for c in out.columns]
    out.reset_index().to_csv(D/'wcs_acquisition_summary.csv',index=False)
    report=dict(rows=len(df),previous_outcomes_reproduced=len(v),seconds=time.time()-started,
                protocol_sha256=hashlib.sha256((D/'WCS_ACQUISITION_PROTOCOL.md').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs)
    (D/'wcs_acquisition_manifest.json').write_text(json.dumps(report,indent=2));print(report['rows'],'rows',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--datasets',nargs='*',default=['amazon','tfinance','weibo_gadbench'])
    args=ap.parse_args()
    if args.validate_only:validate()
    else:run(args.datasets)
