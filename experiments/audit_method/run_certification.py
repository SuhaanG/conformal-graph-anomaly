"""Prespecified fixed-batch certification feasibility on all cached scores."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from certify import uniform_certificate,stratified_certificate,equal_allocation
D=Path(__file__).resolve().parent
G=D.parent/'aistats_followup/h200_results/paper/aistats_followup'
REPS=200;BUDGETS=(100,250,500);Q=.1;DELTA=.05

def trials(y,degree,order,ends,method,budget,rng):
    """Evaluate using only queried labels for selection; hidden labels score it."""
    N=len(y);cap=ends[-1];K=len(ends)
    audit=np.zeros((REPS,N),dtype=bool)
    if method in ('uniform_candidates','uniform_graph'):
        eligible=order[:cap] if method=='uniform_candidates' else np.arange(N)
        for r in range(REPS):audit[r,rng.choice(eligible,budget,False)]=True
        rank_audit=audit[:,order[:cap]]
        audited=np.cumsum(rank_audit,axis=1)[:,ends-1]
        observed=np.cumsum(rank_audit*(1-y[order[:cap]]),axis=1)[:,ends-1]
        result,upper=uniform_certificate(ends,audited,observed,DELTA,Q)
    else:
        cells=[];band=[];start=0
        for k,end in enumerate(ends):
            nodes=order[start:end];start=end
            if method=='score_degree_strata':
                nodes=nodes[np.lexsort((nodes,degree[nodes]))]
                groups=np.array_split(nodes,2)
            else:groups=[nodes]
            for nodes in groups:
                if len(nodes):cells.append(nodes);band.append(k)
        sizes=np.array([len(c) for c in cells]);allocation=equal_allocation(sizes,budget)
        nullcounts=np.empty((REPS,len(cells)),dtype=int)
        for j,(cell,b) in enumerate(zip(cells,allocation)):
            for r in range(REPS):
                chosen=rng.choice(cell,b,False);audit[r,chosen]=True
                nullcounts[r,j]=int((1-y[chosen]).sum())
        result,upper,audited,observed=stratified_certificate(sizes,allocation,nullcounts,band,ends,DELTA,Q)
    index,remaining,bound=result
    assert np.all(audit.sum(axis=1)==budget)
    # Hidden labels enter only the scoring below, after the certificate is fixed.
    full_null=np.cumsum(1-y[order[:cap]])[ends-1]
    unreviewed_null=full_null[None,:]-observed
    unreviewed_n=ends[None,:]-audited
    false=np.where(index>=0,unreviewed_null[np.arange(REPS),np.maximum(index,0)],0)
    true=remaining-false
    fdp=false/np.maximum(remaining,1)
    total_anoms=int(y.sum());reviewed_true=(audit*y).sum(axis=1)
    return pd.DataFrame(dict(rep=np.arange(REPS),candidate_index=index,discoveries=remaining,
        true_discoveries=true,false_discoveries=false,fdp=fdp,fdp_bound=bound,
        power=true/max(total_anoms,1),unreviewed_recall=true/np.maximum(total_anoms-reviewed_true,1),
        audit_labels=audit.sum(axis=1),reviewed_anomalies=reviewed_true,abstain=remaining==0,
        certificate_failure=(remaining>0)&(fdp>Q+1e-12),
        simultaneous_bound_failure=np.any(full_null[None,:]>upper,axis=1)))

def main():
    start=time.time();allrows=[];oracle=[]
    for dsid,ds in enumerate(('amazon','tolokers')):
        g=np.load(G/f'{ds}_graph.npz');verified=np.arange(3305,len(g['labels'])) if ds=='amazon' else np.arange(len(g['labels']))
        y=g['labels'][verified].astype(int);degree=g['degree'][verified];N=len(y)
        ends=np.array(sorted(set([25,50,100,200,400,800,1600,int(.2*N)])))
        for mid,model in enumerate(('dominant_pygod','gae','isolation_forest','dominant_degree_normalized')):
            for seed in range(10):
                base='dominant_pygod' if model=='dominant_degree_normalized' else model
                s=np.load(G/f'{ds}_{base}_scores_{seed}.npz')['scores'][verified]
                if model=='dominant_degree_normalized':s=s/np.log1p(degree+1e-8)
                order=np.lexsort((verified,-s))
                tp=np.cumsum(y[order])[ends-1]
                for n,t in zip(ends,tp):oracle.append(dict(dataset=ds,model=model,seed=seed,n=int(n),fdp=(n-t)/n,power=t/y.sum(),true_discoveries=int(t)))
                for b in BUDGETS:
                    for methodid,method in enumerate(('uniform_candidates','score_strata','score_degree_strata','uniform_graph')):
                        rng=np.random.default_rng(np.random.SeedSequence([20260923,dsid,mid,seed,b,methodid]))
                        z=trials(y,degree,order,ends,method,b,rng)
                        for key,value in dict(dataset=ds,model=model,seed=seed,method=method,budget=b,n_graph=N).items():z[key]=value
                        allrows.append(z)
                print('CERT',ds,model,seed,'elapsed',round(time.time()-start),flush=True)
            pd.concat(allrows,ignore_index=True).to_csv(D/'certification_trials_partial.csv',index=False)
    df=pd.concat(allrows,ignore_index=True);df.to_csv(D/'certification_trials.csv',index=False)
    pd.DataFrame(oracle).to_csv(D/'oracle_prefix_diagnostic.csv',index=False)
    keys=['dataset','model','method','budget']
    metrics=['discoveries','true_discoveries','fdp','power','unreviewed_recall','audit_labels','reviewed_anomalies','abstain','certificate_failure','simultaneous_bound_failure']
    seedmeans=df.groupby(keys+['seed'])[metrics].mean()
    out=seedmeans.groupby(keys).mean().join(seedmeans.groupby(keys)[['true_discoveries','power']].std().add_suffix('_seed_sd')).reset_index()
    out.to_csv(D/'certification_summary.csv',index=False)
    report=dict(rows=len(df),repetitions=REPS,training_seeds=10,master_seed=20260923,q=Q,delta=DELTA,
                sha256=hashlib.sha256((D/'certification_trials.csv').read_bytes()).hexdigest(),seconds=time.time()-start,
                guarantee='P_audit(FDP of selected unreviewed set <= .1) >= .95; fixed graph, scores, candidate family, correct random audit labels',
                max_empirical_failure=float(df.groupby(keys+['seed']).certificate_failure.mean().max()))
    (D/'certification_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
