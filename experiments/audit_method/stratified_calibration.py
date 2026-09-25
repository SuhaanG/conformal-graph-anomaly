"""Role-randomized stratum-specific conformal BH with split alpha budgets."""
from pathlib import Path
import argparse, hashlib, itertools, json, time
import numpy as np
import pandas as pd
from scipy import sparse
from selection_dose import exposure, score_ranks

D=Path(__file__).resolve().parent
G=D.parent/'aistats_followup/h200_results/paper/aistats_followup'
MODELS=('hgb_attributes','hgb_graph_features')
DATASETS={'amazon':0,'tolokers':1,'weibo':2,'tfinance':3,'weibo_gadbench':2}


def reject(rank,cal,test,alpha):
    test=np.asarray(test,dtype=int)
    if len(test)==0 or len(cal)==0 or alpha<=0:
        return np.empty(0,dtype=int)
    p=(1+len(cal)-np.searchsorted(np.sort(rank[cal]),rank[test],side='left'))/(len(cal)+1)
    order=np.argsort(p,kind='stable');cross=np.flatnonzero(p[order]<=alpha*np.arange(1,len(test)+1)/len(test))
    return test[order[:cross[-1]+1]] if len(cross) else np.empty(0,dtype=int)


def stratified_reject(rank,cal,test,stratum,alpha=.1):
    # Two declared strata keep their .05 budgets, including when one is empty.
    return np.concatenate([reject(rank,cal[stratum[cal]==h],test[stratum[test]==h],alpha/2) for h in (0,1)])


def make_strata(e,deploy,marks,design):
    out=np.zeros(len(e),dtype=np.int8)
    if design=='zero_exposure':
        out[deploy]=(e[deploy]>0).astype(np.int8)
    elif design=='exposure_80':
        order=deploy[np.lexsort((marks[deploy],e[deploy]))]
        out[order[int(np.ceil(.8*len(deploy))):]]=1
    else:
        raise ValueError(design)
    return out


def validate():
    rng=np.random.default_rng(2026092304);checks=0;nonempty=0;largest=-1.
    # Fix node-level ranks, all anomaly-side scores and role counts; enumerate
    # every uniform normal role assignment in two independent small strata.
    # Independence here only simplifies enumeration, not the union inequality.
    for N in (4,7,11):
        for nt in (1,2):
            allocations=list(itertools.combinations(range(N),nt))
            for anomalies in (0,1,5):
                for alpha in (.1,.2,.5,.9):
                    local=[]
                    for t in allocations:
                        rank=np.r_[np.arange(N),np.arange(N,N+anomalies)]
                        cal=np.setdiff1d(np.arange(N),t);test=np.r_[t,np.arange(N,N+anomalies)].astype(int)
                        r=reject(rank,cal,test,alpha/2)
                        v=int((r<N).sum());local.append((v,len(r)))
                    single=np.mean([v/max(r,1) for v,r in local])
                    expected_bound=(alpha/2)*nt/(nt+anomalies)
                    assert single<=expected_bound+1e-12,(N,nt,anomalies,alpha,single)
                    total=0.
                    for v1,r1 in local:
                        for v2,r2 in local:
                            f=(v1+v2)/max(r1+r2,1)
                            assert f<=v1/max(r1,1)+v2/max(r2,1)+1e-12
                            total+=f;nonempty+=int(r1+r2>0);checks+=1
                    mean=total/len(local)**2
                    assert mean<=alpha+1e-12
                    largest=max(largest,mean-alpha)
    for _ in range(200):
        raw=rng.integers(0,4,40);rank=score_ranks(raw,int(rng.integers(1000000)))
        cal=rng.choice(20,15,False);test=np.arange(20,40)
        pv=(1+(rank[cal,None]>=rank[test]).sum(axis=0))/(len(cal)+1)
        order=np.argsort(pv);cross=np.flatnonzero(pv[order]<=.4*np.arange(1,21)/20)
        expected=test[order[:cross[-1]+1]] if len(cross) else np.empty(0,int)
        assert set(reject(rank,cal,test,.4))==set(expected)
    assert len(reject(np.arange(3),np.array([],int),np.arange(3),.1))==0
    report=dict(exact_joint_assignments=checks,nonempty_joint_assignments=nonempty,
                max_union_fdr_minus_alpha=largest,independent_rank_checks=200,
                empty_reference_abstains=True)
    (D/'stratified_validation.json').write_text(json.dumps(report,indent=2))
    print('VALIDATION',report,flush=True)


def main(datasets):
    validate();rows=[];checks=0;started=time.time();inputs={}
    for ds in datasets:
        dsid=DATASETS[ds]
        path=(G if ds in ('amazon','tolokers') else D)/f'{ds}_graph.npz'
        g=np.load(path);y=g['labels'].astype(int)
        inputs[str(path.relative_to(D.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
        a=sparse.csr_matrix((np.ones(len(g['indices'])),g['indices'],g['indptr']),shape=(len(y),len(y)))
        for seed in range(10):
            ranks={};base=None
            for mid,model in enumerate(MODELS):
                path=D/f'{ds}_{model}_scores_{seed}.npz';c=np.load(path)
                inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
                if base is None:train=c['train'];deploy=c['deploy'];base=True
                assert np.array_equal(train,c['train']) and np.array_equal(deploy,c['deploy'])
                assert np.array_equal(c['labels'],y)
                ranks[model]=score_ranks(c['scores'],np.random.SeedSequence([20260923,dsid,seed,mid,7]))
            normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
            marks=np.random.default_rng(np.random.SeedSequence([2026092303,dsid,seed])).random(len(y))
            for split in range(5):
                rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                reference=np.setdiff1d(deploy,test)
                rng=np.random.default_rng(np.random.SeedSequence([20260928,dsid,seed,split]));u=rng.random(len(reference))
                for rho in (.25,.5,1.):
                    revealed=reference[u<rho];known=np.r_[train,revealed];cal=revealed[y[revealed]==0]
                    e=exposure(a,y,known)
                    # Equivalent construction using only known anomaly indices.
                    anomaly_known=known[y[known]==1]
                    assert np.array_equal(e,exposure(a,y,anomaly_known))
                    changed=y.copy();unknown=np.setdiff1d(np.arange(len(y)),known);changed[unknown]=1-changed[unknown]
                    assert np.array_equal(e,exposure(a,changed,known))
                    # Any change in revealed normal roles cannot change exposure.
                    fake_normal_known=normals[::2]
                    e2=exposure(a,y,np.r_[anomaly_known,fake_normal_known])
                    assert np.array_equal(e,e2);checks+=1
                    assert not np.intersect1d(cal,test).size and not np.intersect1d(cal,train).size
                    for design in ('zero_exposure','exposure_80'):
                        strata=make_strata(e,deploy,marks,design)
                        assert np.array_equal(strata,make_strata(e2,deploy,marks,design))
                        for model,rank in ranks.items():
                            selected={
                                'full_random':reject(rank,cal,test,.1),
                                'filtered_low':reject(rank,cal[strata[cal]==0],test,.1),
                                'stratified':stratified_reject(rank,cal,test,strata)}
                            for method,rr in selected.items():
                                k=len(rr);tp=int(y[rr].sum())
                                rows.append(dict(dataset=ds,model=model,seed=seed,split=split,label_fraction=rho,
                                    design=design,method=method,fdp=(k-tp)/max(k,1),power=tp/max(int(y[test].sum()),1),
                                    discoveries=k,true_discoveries=tp,nonempty=int(k>0),
                                    reference_labels=len(revealed),training_labels=len(train),n_reference=len(cal),
                                    cal_low=int((strata[cal]==0).sum()),cal_high=int((strata[cal]==1).sum()),
                                    test_low=int((strata[test]==0).sum()),test_high=int((strata[test]==1).sum())))
            print('STRATIFIED',ds,seed,'seconds',round(time.time()-started,1),flush=True)
    df=pd.DataFrame(rows);tag='_'.join(datasets)
    df.to_csv(D/f'stratified_trials_{tag}.csv',index=False)
    groups=['dataset','model','label_fraction','design','method'];metrics=['fdp','power','discoveries','true_discoveries','nonempty','reference_labels','training_labels','n_reference','cal_low','cal_high','test_low','test_high']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/f'stratified_seeds_{tag}.csv',index=False)
    out=seeds.groupby(groups)[metrics].agg(['mean','std']);out.columns=['_'.join(c) for c in out.columns]
    out.reset_index().to_csv(D/f'stratified_summary_{tag}.csv',index=False)
    report=dict(rows=len(df),normal_role_invariance_checks=checks,seconds=time.time()-started,
                protocol_sha256=hashlib.sha256((D/'STRATIFIED_PROTOCOL.md').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=inputs)
    (D/f'stratified_manifest_{tag}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['amazon','tolokers','weibo']);args=parser.parse_args()
    main(args.datasets)
