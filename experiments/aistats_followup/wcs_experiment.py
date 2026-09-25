"""Hypothesis-conditional WCS: Jin--Candes Algorithm 2, equations (5)--(9).

Scores here increase with anomalousness, opposite to V in that paper. The
simulation has continuous scores. Nonrandomized rank p-values are used for
both weighted BH and WCS; homogeneous pruning is separately randomized.
"""
from pathlib import Path
import sys,csv,json,hashlib,time,argparse
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'aistats_revision'))
from validate_revision import pvals,bh,write_csv

def prune(first,sizes,xi=1.):
    ids=np.flatnonzero(first);out=np.zeros(len(first),bool)
    if not len(ids):return out
    vals=sizes[ids]*xi; order=np.argsort(vals)
    good=np.flatnonzero(vals[order]<=np.arange(1,len(ids)+1))
    if len(good):out[ids[order[:good[-1]+1]]]=True
    return out

def wcs(c,t,w,wt,alpha=.1,xi=.5):
    c,t,w,wt=map(np.asarray,(c,t,w,wt));m=len(t)
    assert np.all(w>0) and np.all(wt>0)
    # Equation (5); continuous simulations ensure strict and non-strict agree.
    order=np.argsort(c); tail=np.r_[np.cumsum(w[order][::-1])[::-1],0.]
    base=tail[np.searchsorted(c[order],t,side='right')]
    p=(base+wt)/(w.sum()+wt)
    # Row j contains auxiliary p_l^(j). Weight is w_j, never w_l.
    aux=(base[None,:]+wt[:,None]*(t[:,None]>t[None,:]))/(w.sum()+wt[:,None])
    np.fill_diagonal(aux,0.)
    cross=np.sort(aux,axis=1)<=alpha*np.arange(1,m+1)[None,:]/m
    sizes=np.max(np.where(cross,np.arange(1,m+1)[None,:],0),axis=1)
    first=p<=alpha*sizes/m
    return prune(first,sizes),prune(first,sizes,xi),p,sizes

def literal(c,t,w,wt,alpha=.1,xi=.5):
    m=len(t);p=np.array([(w[c>s].sum()+wt[j])/(w.sum()+wt[j]) for j,s in enumerate(t)])
    sizes=[]
    for j in range(m):
        aux=np.array([(w[c>t[l]].sum()+wt[j]*(t[j]>t[l]))/(w.sum()+wt[j]) if l!=j else 0 for l in range(m)])
        sizes.append(int(bh(aux,alpha).sum()))
    sizes=np.array(sizes);first=p<=alpha*sizes/m
    return prune(first,sizes),prune(first,sizes,xi),p,sizes

def checks():
    rng=np.random.default_rng(19209)
    for rep in range(80):
        c=rng.normal(size=20);t=rng.normal(size=25);w=np.exp(rng.normal(size=20));wt=np.exp(rng.normal(size=25));xi=rng.random()
        fast=wcs(c,t,w,wt,xi=xi);slow=literal(c,t,w,wt,xi=xi)
        for a,b in zip(fast,slow):assert np.allclose(a,b)
        assert np.allclose(fast[2],pvals(c,t,w,wt))
        scaled=wcs(c,t,w*31,wt*31,xi=xi)
        for a,b in zip(fast,scaled):assert np.allclose(a,b)
        perm=rng.permutation(len(t));z=wcs(c,t[perm],w,wt[perm],xi=xi)
        for a,b in zip(fast,z):assert np.allclose(a[perm],b)
    # The public source uses w_l where the paper's (6) uses w_j, and uses
    # randomized ranks. We verify its constant-weight special case with all
    # random uniforms fixed to one, where those differences disappear.
    import importlib.util
    path=OUT/'vendor/confselect_reference.py';spec=importlib.util.spec_from_file_location('author_reference',path)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    old=np.random.uniform
    try:
        np.random.uniform=lambda low=0.,high=1.,size=None: np.ones(size) if size is not None else 1.
        for rep in range(6):
            c=rng.normal(size=20);t=rng.normal(2,size=16);w=np.ones(20);wt=np.ones(16)
            _,selected,rp=ref.weighted_CS(-c,w,-t,wt,rand='dtm')
            f=wcs(c,t,w,wt);assert set(selected)==set(np.flatnonzero(f[0]));assert np.allclose(rp,f[2])
    finally:np.random.uniform=old
    report=dict(literal_equation_cases=80,weight_scale_and_permutation_checks='passed',reference_constant_weight_cases=6,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reference_discrepancy='Public weighted_CS auxiliary numerator uses test_weights[k]; equation (6) uses test_weights[j]. Public code also randomizes rank p-values; Algorithm 2 refers to nonrandomized equation (5). This implementation follows the paper equations, not the discrepant variable-weight reference code.',
                algorithm_source='https://arxiv.org/html/2307.09291v2#S6',source_url='https://github.com/ying531/conformal-selection/blob/main/confselect-python/ConfSelect/confselect.py')
    (OUT/'wcs_validation.json').write_text(json.dumps(report,indent=2));print('WCS CHECKS PASSED',flush=True)

def main(reps):
    checks(); rows=[]; start=time.time()
    for rep in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([20260916,rep]))
        u=rng.random(200);z=rng.normal(size=200);v=rng.random(400);e=rng.normal(size=400);xi=rng.random()
        for theta in (0.,2.):
            t=theta*(v<.5)+e;t[360:]+=3.
            for q in (1.,.25,.05):
                selected=(u<q/(1+q));c=theta*selected+z;w=np.where(selected,1/q,1.);wt=np.where(v<.5,1/q,1.)
                d,h,p,sizes=wcs(c,t,w,wt,xi=xi)
                results={'selected_bh':bh(pvals(c,t)),'weighted_bh':bh(p),'wcs_deterministic':d,'wcs_homogeneous':h,
                         'weighted_by':bh(p,.1/np.sum(1/np.arange(1,401)))}
                for method,r in results.items():
                    k=int(r.sum());rows.append(dict(rep=rep,theta=theta,q=q,method=method,fdp=float(r[:360].sum()/max(k,1)),
                        power=float(r[360:].mean()),discoveries=k,weighted_rank_null_tail_005=float(np.mean(p[:360]<=.05))))
        if (rep+1)%250==0: print('WCS REP',rep+1,'seconds',round(time.time()-start,1),flush=True)
    write_csv(OUT/'wcs_trials.csv',rows)
    groups={}
    for row in rows:groups.setdefault((row['theta'],row['q'],row['method']),[]).append(row)
    summary=[]
    for (theta,q,method),g in groups.items():
        a=dict(theta=theta,q=q,method=method,repetitions=len(g))
        for k in ('fdp','power','discoveries'):
            x=np.array([r[k] for r in g]);a[k+'_mean']=float(x.mean());a[k+'_mcse']=float(x.std(ddof=1)/np.sqrt(len(x)))
        summary.append(a)
    write_csv(OUT/'wcs_summary.csv',summary)
    (OUT/'wcs_manifest.json').write_text(json.dumps(dict(master_seed=20260916,repetitions=reps,evaluations=len(rows),seconds=time.time()-start,
        n_calibration=200,n_test=400,n_null=360,rank_randomization=False,nominal_alpha=.1),indent=2))
    print('WCS COMPLETE',len(rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=2000);a=ap.parse_args();main(a.reps)
