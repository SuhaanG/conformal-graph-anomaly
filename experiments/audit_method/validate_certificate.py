"""Exhaustive probability checks, not just implementation-identity tests."""
from pathlib import Path
from itertools import combinations,product
import json
import numpy as np
from scipy.stats import hypergeom
from certify import upper_null_count,uniform_certificate,stratified_certificate,equal_allocation
D=Path(__file__).resolve().parent

def main():
    cases=0;max_ratio=0.
    for N in range(1,26):
        for n in range(N+1):
            x=np.arange(n+1)
            for alpha in (.01,.05,.2):
                u=upper_null_count(N,n,x,alpha)
                assert np.all(np.diff(u)>=0)
                for V in range(N+1):
                    prob=float(hypergeom.pmf(x,N,V,n)[u<V].sum())
                    assert prob<=alpha+1e-12,(N,n,V,alpha,prob)
                    max_ratio=max(max_ratio,prob/alpha);cases+=1
    families=0;returned=0;worst_failure=0.
    N=8;ends=np.array([2,4,6,8])
    for budget in (2,4,6):
        subsets=list(combinations(range(N),budget))
        masks=np.zeros((len(subsets),N),dtype=int)
        for r,subset in enumerate(subsets):masks[r,list(subset)]=1
        h=np.cumsum(masks,axis=1)[:,ends-1]
        for labels in product((0,1),repeat=N):
            y=np.array(labels);x=np.cumsum(masks*(1-y),axis=1)[:,ends-1]
            full_null=np.cumsum(1-y)[ends-1]
            for q,delta in ((.1,.05),(.5,.2)):
                (idx,R,bound),upper=uniform_certificate(ends,h,x,delta,q)
                v=np.where(idx>=0,(full_null-x)[np.arange(len(idx)),np.maximum(idx,0)],0)
                fdp=v/np.maximum(R,1)
                prob=float(np.mean(fdp>q+1e-12))
                assert prob<=delta+1e-12,(labels,budget,q,prob)
                assert np.all(bound[idx>=0]<=q)
                assert np.all(R<=N-budget)
                returned+=int((R>0).sum());families+=1;worst_failure=max(worst_failure,prob)
    # All label configurations and all equally likely stratified samples.
    ends=np.array([4,8]);sizes=np.array([4,4]);allocation=np.array([2,2])
    subsets=[tuple(a)+tuple(b) for a in combinations(range(4),2) for b in combinations(range(4,8),2)]
    masks=np.zeros((len(subsets),N),dtype=int)
    for r,subset in enumerate(subsets):masks[r,list(subset)]=1
    for labels in product((0,1),repeat=N):
        y=np.array(labels);x=np.stack([(masks[:,:4]*(1-y[:4])).sum(axis=1),(masks[:,4:]*(1-y[4:])).sum(axis=1)],axis=1)
        for q,delta in ((.1,.05),(.5,.2)):
            (idx,R,bound),upper,h,obs=stratified_certificate(sizes,allocation,x,[0,1],ends,delta,q)
            full_null=np.cumsum(1-y)[ends-1]
            v=np.where(idx>=0,(full_null-obs)[np.arange(len(idx)),np.maximum(idx,0)],0)
            prob=float(np.mean(v/np.maximum(R,1)>q+1e-12))
            assert prob<=delta+1e-12
            returned+=int((R>0).sum());families+=1;worst_failure=max(worst_failure,prob)
    for sizes in ([1,2,10],[0,5,9],[25,25,50,100,200,400]):
        for b in range(sum(sizes)+1):
            n=equal_allocation(sizes,b);assert n.sum()==b and np.all(n<=sizes)
    assert returned>0,'Selection validation must include actual discoveries'
    assert int(upper_null_count(10,0,0,.05))==10
    assert int(upper_null_count(10,10,3,.05))==3
    report=dict(exact_coverage_cases=cases,max_failure_to_alpha_ratio=max_ratio,
                exhaustive_selection_configurations=families,nonempty_selections=returned,
                largest_selection_failure_probability=worst_failure,
                checks=['exact hypergeometric coverage','simultaneous adaptive prefix choice',
                        'uniform and stratified audit enumeration','full and empty audits',
                        'all normal and all anomalous populations','exact budget allocation',
                        'audited nodes excluded from automatic discoveries'],status='passed')
    (D/'certificate_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
