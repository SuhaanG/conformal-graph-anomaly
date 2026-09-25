"""Frozen allocation ablation and biased acquisition on existing score caches."""
from pathlib import Path
import itertools,json,hashlib,time,math,argparse
import numpy as np
import pandas as pd
from scipy import sparse
from stratified_calibration import D,G,MODELS,DATASETS,make_strata,reject
from selection_dose import exposure,score_ranks

def pvalues(rank,cal,test,weights=None):
    if not len(cal):return np.ones(len(test))
    order=cal[np.argsort(rank[cal])]
    ix=np.searchsorted(rank[order],rank[test],side='left')
    if weights is None:return (1+len(cal)-ix)/(len(cal)+1)
    prefix=np.r_[0.,np.cumsum(weights[order])]
    return (weights[test]+prefix[-1]-prefix[ix])/(weights[test]+prefix[-1])

def bh(p,alpha=.1):
    if not len(p) or alpha<=0:return np.zeros(len(p),bool)
    order=np.argsort(p,kind='stable');cross=np.flatnonzero(p[order]<=alpha*np.arange(1,len(p)+1)/len(p))
    out=np.zeros(len(p),bool)
    if len(cross):out[order[:cross[-1]+1]]=True
    return out

def methods(rank,cal,test,strata,prob=None):
    p=np.ones(len(test));equal=np.zeros(len(test),bool);size=equal.copy()
    for h in (0,1):
        mask=strata[test]==h;pp=pvalues(rank,cal[strata[cal]==h],test[mask]);p[mask]=pp
        equal[mask]=bh(pp,.05);size[mask]=bh(pp,.1*mask.sum()/max(len(test),1))
    out={'pooled_reference':bh(pvalues(rank,cal,test)),
         'equal_alpha':equal,'size_alpha':size,'pooled_stratum':bh(p)}
    if prob is not None:
        weighted=pvalues(rank,cal,test,1/prob)
        out['weighted_bh']=bh(weighted)
        out['weighted_by']=bh(weighted,.1/np.sum(1/np.arange(1,len(test)+1)))
    return out

def validate():
    # All normal test sets and all revelation subsets; expectation weighted by
    # the exact uniform-test/Bernoulli-acquisition law, not a Monte Carlo estimate.
    cases=0;assignments=0;worst={};marginal_checks=0
    for n in (4,6):
        for nt in (1,2):
            for na in (0,2):
                for scenario in ((.5,.5),(.8,.2)):
                    for ordering in (0,1):
                        N=n+na;strata=np.arange(N)%2
                        rank=np.arange(N) if ordering==0 else np.random.default_rng(100+n+na).permutation(N)
                        prob=np.array(scenario)[strata];y=np.r_[np.zeros(n),np.ones(na)]
                        sums={};total=0.;p_cdf=np.zeros(20);thresholds=np.arange(1,21)/20
                        for tt in itertools.combinations(range(n),nt):
                            remaining=np.setdiff1d(np.arange(n),tt);test=np.r_[tt,np.arange(n,N)].astype(int)
                            for flags in itertools.product((False,True),repeat=len(remaining)):
                                flags=np.array(flags);cal=remaining[flags]
                                mass=np.prod(np.where(flags,prob[remaining],1-prob[remaining]))/math.comb(n,nt)
                                total+=mass;assignments+=1
                                for name,rr in methods(rank,cal,test,strata,prob).items():
                                    f=(1-y[test[rr]]).sum()/max(rr.sum(),1);sums[name]=sums.get(name,0)+mass*f
                                wp=pvalues(rank,cal,test,1/prob)[:nt]
                                p_cdf+=mass*(wp[:,None]<=thresholds).mean(axis=0)
                        assert abs(total-1)<1e-12
                        assert np.all(p_cdf<=thresholds+1e-12),(n,nt,na,scenario,p_cdf)
                        marginal_checks+=len(thresholds)
                        for name in ('equal_alpha','size_alpha','pooled_stratum','weighted_by'):
                            bound=.1*nt/(nt+na)
                            assert sums[name]<=bound+1e-12,(name,sums,bound)
                            worst[name]=max(worst.get(name,-1),sums[name]-bound)
                        cases+=1
    rng=np.random.default_rng(88)
    for _ in range(200):
        rank=rng.permutation(40);cal=rng.choice(20,10,False);test=np.arange(20,40);w=rng.uniform(.2,5,40)
        direct=(w[test]+((rank[cal,None]>=rank[test])*w[cal,None]).sum(axis=0))/(w[test]+w[cal].sum())
        assert np.allclose(pvalues(rank,cal,test,w),direct,rtol=1e-13,atol=1e-13)
        assert np.allclose(pvalues(rank,cal,test,np.ones(40)),pvalues(rank,cal,test))
        assert set(test[bh(pvalues(rank,cal,test))])==set(reject(rank,cal,test,.1))
    report=dict(exact_design_cases=cases,exact_assignments=assignments,marginal_threshold_checks=marginal_checks,
                max_fdr_minus_bound=worst,weighted_direct_checks=200)
    (D/'allocation_validation.json').write_text(json.dumps(report,indent=2));print('VALIDATION',report,flush=True)

def run(datasets):
    validate();rows=[];inputs={};started=time.time()
    def record(phase,ds,model,seed,split,rho,design,scenario,rep,method,rr,test,y,cal,revealed,train):
        ids=test[rr];tp=int(y[ids].sum());k=len(ids)
        rows.append(dict(phase=phase,dataset=ds,model=model,seed=seed,split=split,label_fraction=rho,
            design=design,scenario=scenario,rep=rep,method=method,fdp=(k-tp)/max(k,1),
            power=tp/max(int(y[test].sum()),1),discoveries=k,true_discoveries=tp,nonempty=int(k>0),
            reference_labels=len(revealed),training_labels=len(train),n_reference=len(cal)))
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
            assert not np.intersect1d(train,deploy).size
            normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
            marks=np.random.default_rng(np.random.SeedSequence([2026092303,dsid,seed])).random(len(y))
            e_train=exposure(a,y,train)
            altered=y.copy();altered[deploy]=1-altered[deploy]
            assert np.array_equal(e_train,exposure(a,altered,train))
            fixed={d:make_strata(e_train,deploy,marks,d) for d in ('zero_exposure','exposure_80')}
            for split in range(5):
                rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                reference=np.setdiff1d(deploy,test)
                u=np.random.default_rng(np.random.SeedSequence([20260928,dsid,seed,split])).random(len(reference))
                for rho in (.25,.5,1.):
                    revealed=reference[u<rho];cal=revealed[y[revealed]==0];e=exposure(a,y,np.r_[train,revealed])
                    for design in fixed:
                        strata=make_strata(e,deploy,marks,design)
                        for model,rank in ranks.items():
                            out=methods(rank,cal,test,strata)
                            out['filtered_low']=bh(pvalues(rank,cal[strata[cal]==0],test))
                            for method,rr in out.items():record('allocation',ds,model,seed,split,rho,design,'uniform',0,method,rr,test,y,cal,revealed,train)
                for design,strata in fixed.items():
                    for rep in range(10):
                        u=np.random.default_rng(np.random.SeedSequence([2026092407,dsid,seed,split,rep])).random(len(reference))
                        for scenario,pp in [('neutral',(.5,.5)),('moderate',(.5,.1)),('severe',(.5,.025))]:
                            prob=np.array(pp)[strata];revealed=reference[u<prob[reference]];cal=revealed[y[revealed]==0]
                            q=prob[deploy].mean();uniform_revealed=reference[u<q];uniform_cal=uniform_revealed[y[uniform_revealed]==0]
                            assert not np.intersect1d(cal,test).size
                            for model,rank in ranks.items():
                                out=methods(rank,cal,test,strata,prob)
                                for method,rr in out.items():record('acquisition',ds,model,seed,split,0.,design,scenario,rep,method,rr,test,y,cal,revealed,train)
                                rr=bh(pvalues(rank,uniform_cal,test));record('acquisition',ds,model,seed,split,0.,design,scenario,rep,'uniform_counterfactual',rr,test,y,uniform_cal,uniform_revealed,train)
            print('DONE',ds,seed,round(time.time()-started,1),flush=True)
    df=pd.DataFrame(rows)
    # Exact outcome reproduction for old arms, all previous graph/seed/split cells.
    old=pd.concat([pd.read_csv(p) for p in D.glob('stratified_trials_*.csv')],ignore_index=True)
    old=old[old.dataset.isin(datasets)].copy();old.method=old.method.replace({'full_random':'pooled_reference','stratified':'equal_alpha'})
    keys=['dataset','model','seed','split','label_fraction','design','method'];metrics=['fdp','power','discoveries']
    fresh=df[(df.phase=='allocation')&df.method.isin(old.method.unique())]
    v=old.merge(fresh,on=keys,validate='one_to_one',suffixes=('_old','_new'))
    assert len(v)==len(old)
    for m in metrics:assert np.allclose(v[m+'_old'],v[m+'_new'],rtol=1e-13,atol=1e-13),m
    df.to_csv(D/'allocation_acquisition_trials.csv.gz',index=False,compression='gzip')
    groups=['phase','dataset','model','label_fraction','design','scenario','method'];metrics+=['true_discoveries','nonempty','reference_labels','training_labels','n_reference']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/'allocation_acquisition_seeds.csv',index=False)
    out=seeds.groupby(groups)[metrics].agg(['mean','std']);out.columns=['_'.join(c) for c in out.columns]
    out.reset_index().to_csv(D/'allocation_acquisition_summary.csv',index=False)
    report=dict(rows=len(df),previous_outcomes_reproduced=len(v),seconds=time.time()-started,
                protocol_sha256=hashlib.sha256((D/'ALLOCATION_ACQUISITION_PROTOCOL.md').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs)
    (D/'allocation_acquisition_manifest.json').write_text(json.dumps(report,indent=2));print(report['rows'],'rows',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if args.validate_only:validate()
    else:run(['amazon','tfinance','weibo_gadbench'])
