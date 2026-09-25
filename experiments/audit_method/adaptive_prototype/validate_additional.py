"""Independent enumeration and nonvacuous full-pipeline stress checks."""
import itertools, json, math
import numpy as np
from run import D, DELTA, Q, adaptive, choose_cell, construct_cells, upper_cs


def exact_two_cell():
    worst=0.; count=0
    sizes=np.array([4,4]);membership=np.array([[True,False],[True,True]])
    for K0 in range(5):
        for K1 in range(5):
            failure=0;total=0
            for a in itertools.combinations(range(4),K0):
                for b in itertools.combinations(range(4),K1):
                    streams=np.zeros((2,4),int)
                    streams[0,list(a)]=1;streams[1,list(b)]=1
                    n=np.zeros(2,int);x=n.copy();u=sizes.copy();bad=False
                    for t in range(8):
                        j=choose_cell(sizes,n,x,u,membership,8-t)
                        x[j]+=streams[j,n[j]];n[j]+=1
                        u[j]=min(u[j],upper_cs(4,int(n[j]),int(x[j]),2))
                        bad|=u[0]<K0 or u[1]<K1
                        if u[j]<x[j]:u[j]=4-n[j]+x[j]
                    total+=1;failure+=bad
            assert total==math.comb(4,K0)*math.comb(4,K1)
            assert failure/total<=DELTA+1e-12
            worst=max(worst,failure/total);count+=total
    return dict(paths=count,max_anytime_noncoverage=worst)


def stress():
    rng=np.random.default_rng(2026092302);rows=[]
    N=2000;ends=np.array([25,50,100,200,400,800,1600])
    order=np.arange(N);degree=rng.integers(1,50,N)
    for prefix_size,precision in [(400,p) for p in (0.,.5,.9,.98,1.)]+[(1600,1.)]:
        y=np.zeros(N,int);y[:round(prefix_size*precision)]=1
        # Only the stress generator knows the population; shuffled labels within
        # first 400 test nontrivial tails without tuning the audit algorithm.
        rng.shuffle(y[:prefix_size])
        for method in ('adaptive_score','adaptive_degree'):
            cells,membership=construct_cells(order,ends,degree,method)
            failures=0;nonempty=0
            for rep in range(50):
                k,r,b,queried,failed=adaptive(y,cells,membership,ends,500,rng)
                selected=np.setdiff1d(order[:ends[k]],queried) if k>=0 else np.array([],int)
                assert len(selected)==r and len(np.unique(queried))==500
                fdp=float((1-y[selected]).sum())/max(r,1)
                nonempty+=r>0;failures+=r>0 and fdp>Q+1e-12
            rows.append(dict(prefix_size=prefix_size,prefix_precision=precision,method=method,
                             repetitions=50,nonempty=nonempty,certificate_failures=failures))
    return rows


if __name__=='__main__':
    rows=stress()
    out=dict(exact_adaptive_two_cell=exact_two_cell(),full_pipeline_stress=rows,
             nonvacuous_output_observed=any(r['nonempty'] for r in rows),
             note='The initial 400-node high-precision tests all abstained. A 1600-node all-anomaly positive control was added afterwards to distinguish algorithm inefficiency from an always-abstain implementation.')
    (D/'additional_validation.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
