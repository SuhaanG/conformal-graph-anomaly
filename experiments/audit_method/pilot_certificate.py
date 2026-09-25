"""Pilot chooses one population and audit size; fresh labels certify it."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from certify import upper_null_count
from run_certification import G,D,BUDGETS,REPS,Q,DELTA

def choose_plan(ends,pilot_rank,observed_labels,budget,q=Q,delta=DELTA):
    """Accesses only the pilot labels, not the unqueried label vector."""
    ends=np.asarray(ends,dtype=int)
    included=pilot_rank[:,None]<ends[None,:]
    h=included.sum(axis=0)
    x=((1-observed_labels)[:,None]*included).sum(axis=0)
    sizes=ends-h
    ns=np.array(sorted(set([25,50,100,200,400,budget-len(pilot_rank)])))
    ns=ns[(ns>0)&(ns<=budget-len(pilot_rank))]
    if not len(ns):return None
    valid=ns[None,:]<sizes[:,None]
    N=np.broadcast_to(sizes[:,None],valid.shape)
    n=np.broadcast_to(ns[None,:],valid.shape)
    predicted=np.ceil(((x+1)/(h+2))[:,None]*n).astype(int)
    bound=np.ones(valid.shape)
    u=upper_null_count(N[valid],n[valid],predicted[valid],delta)
    bound[valid]=(u-predicted[valid])/(N[valid]-n[valid])
    utility=np.where(valid&(bound<=q),N-n,-1)
    if utility.max()<0:return None
    k,j=np.unravel_index(np.argmax(utility),utility.shape)
    return int(k),int(ns[j])

def pilot_trials(y,order,ends,budget,rng,reps=REPS,q=Q,delta=DELTA):
    # Rank-space labels keep candidate construction independent of labels.
    yr=np.asarray(y,dtype=int)[order]; cap=int(ends[-1]); rows=[]
    for rep in range(reps):
        pilot=rng.choice(cap,int(.2*budget),replace=False)
        plan=choose_plan(ends,pilot,yr[pilot],budget,q,delta)
        k=-1; n=0; N=0; upper=0; x=0; bound=0.; remaining=0
        cert=np.empty(0,dtype=int)
        if plan is not None:
            k,n=plan
            eligible=np.setdiff1d(np.arange(ends[k]),pilot)
            N=len(eligible)
            cert=rng.choice(eligible,n,replace=False)
            x=int((1-yr[cert]).sum())
            upper=int(upper_null_count(N,n,x,delta))
            bound=(upper-x)/(N-n)
            if bound<=q:remaining=N-n
        # Unqueried labels used only after candidate and certificate are fixed.
        false=0; V=0
        if plan is not None:
            V=int((1-yr[eligible]).sum())
            if remaining:false=V-x
        true=remaining-false; fdp=false/max(remaining,1)
        audited=np.r_[pilot,cert]
        assert len(np.unique(audited))==len(audited)<=budget
        rows.append(dict(rep=rep,candidate_index=k,planned_n=n,candidate_n=N,
            discoveries=remaining,true_discoveries=true,false_discoveries=false,
            fdp=fdp,fdp_bound=bound if remaining else 0.,audit_labels=len(audited),
            reviewed_anomalies=int(yr[audited].sum()),abstain=remaining==0,
            certificate_failure=bool(remaining and fdp>q+1e-12),
            bound_failure=bool(plan is not None and upper<V),
            power=true/max(int(yr.sum()),1)))
    return pd.DataFrame(rows)

def main():
    start=time.time();allrows=[];hashes={}
    models=('dominant_pygod','gae','isolation_forest','dominant_degree_normalized',
            'hgb_attributes','hgb_graph_features')
    for dsid,ds in enumerate(('amazon','tolokers')):
        path=G/f'{ds}_graph.npz';g=np.load(path)
        hashes[str(path.relative_to(D.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
        verified=np.arange(3305,len(g['labels'])) if ds=='amazon' else np.arange(len(g['labels']))
        for mid,model in enumerate(models):
            for seed in range(10):
                if model.startswith('hgb_'):
                    path=D/f'{ds}_{model}_scores_{seed}.npz';cache=np.load(path)
                    ids=cache['deploy'];s=cache['scores'][ids]
                    assert not np.intersect1d(ids,cache['train']).size
                else:
                    base='dominant_pygod' if model=='dominant_degree_normalized' else model
                    path=G/f'{ds}_{base}_scores_{seed}.npz';ids=verified
                    s=np.load(path)['scores'][ids]
                    if model=='dominant_degree_normalized':s=s/np.log1p(g['degree'][ids]+1e-8)
                hashes[str(path.relative_to(D.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
                y=g['labels'][ids].astype(int);order=np.lexsort((ids,-s))
                ends=np.array(sorted(set([25,50,100,200,400,800,1600,int(.2*len(ids))])))
                for b in BUDGETS:
                    rng=np.random.default_rng(np.random.SeedSequence([20260929,dsid,mid,seed,b]))
                    z=pilot_trials(y,order,ends,b,rng)
                    for key,value in dict(dataset=ds,model=model,seed=seed,budget=b,
                                          method='pilot_then_certify').items():z[key]=value
                    allrows.append(z)
            print('PILOT',ds,model,'seconds',round(time.time()-start),flush=True)
    df=pd.concat(allrows,ignore_index=True);df.to_csv(D/'pilot_trials.csv',index=False)
    keys=['dataset','model','budget'];metrics=['discoveries','true_discoveries','fdp','power',
        'audit_labels','abstain','certificate_failure','bound_failure']
    seedmeans=df.groupby(keys+['seed'])[metrics].mean()
    summary=seedmeans.groupby(keys).mean().join(seedmeans.groupby(keys)[
        ['true_discoveries','audit_labels']].std().add_suffix('_seed_sd')).reset_index()
    summary.to_csv(D/'pilot_summary.csv',index=False)
    assert len(df)==72000 and df.groupby(keys+['seed']).size().eq(REPS).all()
    assert (df.audit_labels<=df.budget).all()
    assert df.loc[~df.abstain,'fdp_bound'].le(Q).all()
    report=dict(rows=len(df),master_seed=20260929,q=Q,delta=DELTA,seconds=time.time()-start,
        protocol_sha256=hashlib.sha256((D/'PILOT_PROTOCOL.md').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes=hashes,
        output_sha256=hashlib.sha256((D/'pilot_trials.csv').read_bytes()).hexdigest(),
        certificate_failures=int(df.certificate_failure.sum()),
        bound_failures=int(df.bound_failure.sum()))
    (D/'pilot_manifest.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string(index=False));print('COMPLETE',report['rows'],flush=True)
if __name__=='__main__':main()
