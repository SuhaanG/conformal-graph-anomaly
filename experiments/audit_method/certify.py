"""Finite-batch FDP certification using classical exact hypergeometric bounds.

This module accepts observed audit counts, not hidden deployment labels.
Candidates/strata and sample sizes must be fixed before certification labels.
"""
import numpy as np
from scipy.stats import hypergeom

def upper_null_count(N,n,x,alpha):
    """Largest V with P[Hypergeom(N,V,n)<=x] >= alpha (conservative)."""
    N,n,x=np.broadcast_arrays(np.asarray(N,dtype=int),np.asarray(n,dtype=int),np.asarray(x,dtype=int))
    if not (0<alpha<1):raise ValueError('alpha must be in (0,1)')
    if np.any((N<0)|(n<0)|(n>N)|(x<0)|(x>n)):raise ValueError('infeasible counts')
    shape=N.shape
    N=N.reshape(-1);n=n.reshape(-1);x=x.reshape(-1)
    lo=x.copy();hi=(N-n+x).copy()
    while np.any(lo<hi):
        active=lo<hi;mid=(lo[active]+hi[active]+1)//2
        tail=hypergeom.cdf(x[active],N[active],mid,n[active])
        if not np.isfinite(tail).all():raise ArithmeticError('invalid hypergeometric tail')
        keep=tail>=alpha
        l=lo[active];h=hi[active];l[keep]=mid[keep];h[~keep]=mid[~keep]-1
        lo[active]=l;hi[active]=h
    return lo.reshape(shape)

def equal_allocation(sizes,budget):
    """Capped round-robin allocation. Uses stratum sizes only; exact budget."""
    sizes=np.asarray(sizes,dtype=int);out=np.zeros(len(sizes),dtype=int)
    if budget<0 or budget>sizes.sum():raise ValueError('invalid budget')
    for _ in range(int(budget)):
        candidates=np.flatnonzero(out<sizes)
        j=candidates[np.argmin(out[candidates])];out[j]+=1
    return out

def certify_candidates(sizes,audited,observed_null,upper_total_null,q=.1):
    """Return selected prefix index, remaining size and simultaneous FDP bound.

    Inputs shape (repetitions,candidates); sizes may broadcast. Bound validity
    depends on simultaneous upper_total_null coverage, established externally.
    """
    sizes,audited,observed_null,upper_total_null=np.broadcast_arrays(sizes,audited,observed_null,upper_total_null)
    remaining=sizes-audited
    if np.any((remaining<0)|(upper_total_null<observed_null)):
        raise ValueError('infeasible certificate inputs')
    upper_remaining=np.minimum(remaining,upper_total_null-observed_null)
    bound=upper_remaining/np.maximum(remaining,1)
    accepted=(remaining>0)&(bound<=q)
    utility=np.where(accepted,remaining,-1)
    index=np.argmax(utility,axis=1)
    ok=accepted[np.arange(len(index)),index]
    return np.where(ok,index,-1),np.where(ok,remaining[np.arange(len(index)),index],0),np.where(ok,bound[np.arange(len(index)),index],0.)

def uniform_certificate(sizes,audited,observed_null,delta=.05,q=.1):
    upper=upper_null_count(sizes,audited,observed_null,delta/len(sizes))
    return certify_candidates(sizes,audited,observed_null,upper,q),upper

def stratified_certificate(cell_sizes,cell_samples,cell_observed_null,cell_band,candidate_sizes,delta=.05,q=.1):
    cell_sizes=np.asarray(cell_sizes);cell_band=np.asarray(cell_band)
    upper_cells=upper_null_count(cell_sizes,cell_samples,cell_observed_null,delta/len(cell_sizes))
    upper=np.stack([upper_cells[:,cell_band<=k].sum(axis=1) for k in range(len(candidate_sizes))],axis=1)
    observed=np.stack([cell_observed_null[:,cell_band<=k].sum(axis=1) for k in range(len(candidate_sizes))],axis=1)
    audited=np.stack([np.asarray(cell_samples)[cell_band<=k].sum()+np.zeros(len(observed),dtype=int) for k in range(len(candidate_sizes))],axis=1)
    return certify_candidates(candidate_sizes,audited,observed,upper,q),upper,audited,observed
