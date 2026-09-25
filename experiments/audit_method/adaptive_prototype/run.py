"""Exploratory adaptive audit. Allocation sees only requested label counts."""
from pathlib import Path
from functools import lru_cache
import hashlib, itertools, json, math, sys, time
import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln

D = Path(__file__).resolve().parent
P = D.parent
sys.path.insert(0, str(P))
from certify import upper_null_count, certify_candidates, equal_allocation
G = P.parent/'aistats_followup/h200_results/paper/aistats_followup'
Q, DELTA, REPS = .1, .05, 50


def log_ratio(N, n, x, v):
    """Beta(1,1) predictive ordered-path mass / fixed-v ordered-path mass."""
    v = np.asarray(v)
    likelihood = (gammaln(v+1)-gammaln(v-x+1)
                  +gammaln(N-v+1)-gammaln(N-v-n+x+1)
                  -gammaln(N+1)+gammaln(N-n+1))
    return betaln(x+1, n-x+1)-likelihood


@lru_cache(maxsize=300000)
def upper_cs(N, n, x, cells):
    """Upper endpoint of current LR set; historical intersection outside."""
    if not 0 <= x <= n <= N:
        raise ValueError('invalid finite-population counts')
    if n == 0:
        return N
    if n == N:
        return x
    # Likelihood is log-concave in v. Start at its integer maximizer.
    candidates = np.unique(np.clip([math.floor(N*x/n), math.ceil(N*x/n)], x, N-n+x))
    lo = int(candidates[np.argmin(log_ratio(N,n,x,candidates))])
    hi = N-n+x
    cut = math.log(cells/DELTA)
    if log_ratio(N,n,x,lo) >= cut:
        # Empty confidence set: conservative fallback rather than false certainty.
        return hi
    while lo < hi:
        mid = (lo+hi+1)//2
        if log_ratio(N,n,x,mid) < cut:
            lo = mid
        else:
            hi = mid-1
    return lo


def bounds(sizes, n, x):
    H = len(sizes)
    return np.array([upper_cs(int(N),int(h),int(z),H) for N,h,z in zip(sizes,n,x)])


def choose_cell(sizes, n, x, upper, membership, budget_left):
    """Pure planning function. No labels or outcomes of unqueried nodes."""
    remain = sizes-n
    rate = (x+.5)/(n+10)
    metrics = []
    projected_ok = []
    for k, use in enumerate(membership):
        remaining = int(remain[use].sum())
        if not remaining:
            metrics.append(-np.inf); projected_ok.append(-1); continue
        allocation = np.zeros(len(sizes), dtype=int)
        allocation[use] = np.minimum(remain[use], np.floor(
            min(budget_left,remaining)*remain[use]/remaining).astype(int))
        xp = x+np.rint(allocation*rate).astype(int)
        up = np.minimum(upper, bounds(sizes,n+allocation,xp))
        r = remaining-int(allocation[use].sum())
        ub = int((up[use]-xp[use]).clip(min=0).sum())
        projected_ok.append(r if r > 0 and ub <= Q*r else -1)
        deficit = max(0., float((upper[use]-x[use]).sum())-Q*remaining)
        metrics.append(float((remain[use]*(1-rate[use])).sum())/(1+deficit))
    k = int(np.argmax(projected_ok if max(projected_ok) > 0 else metrics))
    eligible = np.flatnonzero(membership[k] & (remain > 0))
    if not len(eligible):
        eligible = np.flatnonzero(remain > 0)
    gains = []
    for j in eligible:
        b = min(25,budget_left,int(remain[j]))
        xp = int(x[j]+round(b*rate[j]))
        up = min(int(upper[j]),upper_cs(int(sizes[j]),int(n[j]+b),xp,len(sizes)))
        gains.append(((upper[j]-x[j])-(up-xp))/b)
    return int(eligible[np.argmax(gains)] if max(gains) > 0 else eligible[np.argmax(remain[eligible])])


def construct_cells(order, ends, degree, method):
    cells=[]; band=[]; start=0
    for k,end in enumerate(ends):
        nodes=order[start:end]; start=end
        if method in ('adaptive_degree','fixed_degree_cs'):
            nodes=nodes[np.lexsort((nodes,degree[nodes]))]
            groups=np.array_split(nodes,2)
        else:
            groups=[nodes]
        for c in groups:
            if len(c):
                cells.append(c); band.append(k)
    return cells, np.arange(len(ends))[:,None] >= np.asarray(band)[None,:]


def adaptive(y, cells, membership, ends, budget, rng, fixed=False):
    sizes=np.array([len(c) for c in cells]); n=np.zeros(len(cells),int); x=n.copy()
    upper=sizes.copy(); queried=[]
    # A fixed random permutation in each cell gives uniform local next draws,
    # regardless of how observations elsewhere determine which cell is queried.
    streams=[rng.permutation(c) for c in cells]
    total=0; bound_failure=False
    allocation=equal_allocation(sizes,min(budget,int(sizes.sum()))) if fixed else None
    while total < min(budget,int(sizes.sum())):
        j=(int(np.flatnonzero(n<allocation)[0]) if fixed else
           choose_cell(sizes,n,x,upper,membership,budget-total))
        take=min(25,budget-total,int(sizes[j]-n[j]))
        if fixed:take=min(take,int(allocation[j]-n[j]))
        nodes=streams[j][n[j]:n[j]+take]
        # Only these labels become visible to the algorithm.
        observed=int((1-y[nodes]).sum())
        n[j]+=take; x[j]+=observed; total+=take; queried.extend(nodes)
        upper[j]=min(upper[j],upper_cs(int(sizes[j]),int(n[j]),int(x[j]),len(sizes)))
        if upper[j]<x[j]:
            # Incompatible historical confidence sets occur only off coverage;
            # do not turn this inconsistency into an anti-conservative bound.
            bound_failure=True
            upper[j]=int(sizes[j]-n[j]+x[j])
        # Ground truth enters this diagnostic only, never selection/planning.
        bound_failure |= bool(upper[j] < (1-y[cells[j]]).sum())
    h=membership@n; z=membership@x; u=membership@upper
    k,r,b=certify_candidates(ends,h[None,:],z[None,:],u[None,:],Q)
    return int(k[0]),int(r[0]),float(b[0]),np.asarray(queried),bound_failure


def uniform(y, order, ends, budget, rng):
    nodes=rng.choice(order[:ends[-1]],min(budget,int(ends[-1])),False)
    audit=np.isin(order[:ends[-1]],nodes)
    h=np.cumsum(audit)[ends-1]; x=np.cumsum(audit*(1-y[order[:ends[-1]]]))[ends-1]
    u=upper_null_count(ends,h,x,DELTA/len(ends))
    k,r,b=certify_candidates(ends,h[None,:],x[None,:],u[None,:],Q)
    failure=bool(np.any(np.cumsum(1-y[order])[ends-1]>u))
    return int(k[0]),int(r[0]),float(b[0]),nodes,failure


def validate():
    checks=0; paths=0; worst=0.
    for N in range(2,11):
        for n in range(N+1):
            for x in range(n+1):
                v=np.arange(x,N-n+x+1)
                ok=v[log_ratio(N,n,x,v)<math.log(1/DELTA)] if n else v
                expected=int(ok.max()) if len(ok) else N-n+x
                assert upper_cs(N,n,x,1)==expected,(N,n,x)
                checks+=1
        for V in range(N+1):
            failures=0; count=0
            for ones in itertools.combinations(range(N),V):
                z=np.zeros(N,int); z[list(ones)]=1; x=0; u=N; failed=False
                for n,a in enumerate(z,1):
                    x+=int(a);u=min(u,upper_cs(N,n,x,1));failed|=u<V
                failures+=failed;count+=1
            rate=failures/count
            assert rate <= DELTA+1e-12,(N,V,rate)
            worst=max(worst,rate);paths+=count
    out=dict(inversion_cases=checks,enumerated_ordered_binary_paths=paths,
             maximum_anytime_noncoverage=worst,delta=DELTA)
    (D/'validation.json').write_text(json.dumps(out,indent=2))
    print('VALIDATED',out,flush=True)


def main():
    validate(); started=time.time(); rows=[]; inputs={}
    for di,ds in enumerate(('amazon','tolokers','weibo')):
        graph=np.load((P if ds=='weibo' else G)/f'{ds}_graph.npz')
        for mi,model in enumerate(('hgb_attributes','hgb_graph_features')):
            for seed in range(10):
                path=P/f'{ds}_{model}_scores_{seed}.npz'; c=np.load(path)
                inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
                ids=c['deploy'];assert not np.intersect1d(ids,c['train']).size
                assert np.array_equal(c['labels'],graph['labels'])
                y=c['labels'][ids].astype(int);s=c['scores'][ids];degree=graph['degree'][ids]
                order=np.lexsort((ids,-s));ends=np.array(sorted(set([25,50,100,200,400,800,1600,int(.2*len(ids))])))
                for methodid,method in enumerate(('uniform_fixed','adaptive_score','adaptive_degree','fixed_score_cs','fixed_degree_cs')):
                    if method!='uniform_fixed':cells,membership=construct_cells(order,ends,degree,method)
                    for budget in (100,250,500):
                        rng=np.random.default_rng(np.random.SeedSequence([2026092301,di,mi,seed,methodid,budget]))
                        for rep in range(REPS):
                            if method=='uniform_fixed':out=uniform(y,order,ends,budget,rng)
                            else:out=adaptive(y,cells,membership,ends,budget,rng,method.startswith('fixed_'))
                            k,r,b,nodes,failed=out
                            assert len(nodes)==min(budget,int(ends[-1]))==len(np.unique(nodes))
                            chosen=np.setdiff1d(order[:ends[k]],nodes) if k>=0 else np.array([],int)
                            assert len(chosen)==r and not np.intersect1d(chosen,nodes).size
                            true=int(y[chosen].sum());fdp=(r-true)/max(r,1)
                            assert r==0 or b<=Q
                            rows.append(dict(dataset=ds,model=model,seed=seed,method=method,budget=budget,rep=rep,
                                discoveries=r,true_discoveries=true,fdp=fdp,fdp_bound=b,nonempty=int(r>0),
                                audit_labels=len(nodes),power=true/max(int(y.sum()),1),
                                certificate_failure=int(r>0 and fdp>Q+1e-12),bound_failure=int(failed)))
                print(ds,model,seed,'elapsed',round(time.time()-started),flush=True)
            pd.DataFrame(rows).to_csv(D/'trials_partial.csv.gz',index=False)
    df=pd.DataFrame(rows);df.to_csv(D/'trials.csv.gz',index=False)
    keys=['dataset','model','method','budget'];metrics=['discoveries','true_discoveries','fdp','power','nonempty','audit_labels','certificate_failure','bound_failure']
    seeds=df.groupby(keys+['seed'])[metrics].mean()
    summary=seeds.groupby(keys).mean().join(seeds.groupby(keys).true_discoveries.std().rename('true_discoveries_seed_sd')).reset_index()
    summary.to_csv(D/'summary.csv',index=False);seeds.to_csv(D/'seed_summary.csv')
    report=dict(rows=len(df),repetitions=REPS,seconds=time.time()-started,input_hashes=inputs,
        protocol_sha256=hashlib.sha256((D/'PROTOCOL.md').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256((D/'trials.csv.gz').read_bytes()).hexdigest(),
        certificate_failures=int(df.certificate_failure.sum()),bound_failures=int(df.bound_failure.sum()))
    (D/'manifest.json').write_text(json.dumps(report,indent=2));print('COMPLETE',len(df),flush=True)


if __name__=='__main__':
    main()
