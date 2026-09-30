"""Validity investigation for randomized weighted BH under the acquisition design (WCS_COUNTEREXAMPLE_PROTOCOL.md)."""
from pathlib import Path
import itertools,json,hashlib,time,math
import numpy as np
import pandas as pd
from wcs_acquisition import D
from wcs_randomized import wcs_rand
from allocation_acquisition import bh

RHO=np.array([.01,.02,.05,.1,.2,.3,.5,.7,.9,1.])
ALPHAS=np.array([.05,.1,.2,.3,.5])
_BANDS={}

def band_table(m,nt):
    """All band vectors L in {1..m+1}^m, BH rejections from bands, and FDP with nulls in positions < nt."""
    if (m,nt) not in _BANDS:
        L=np.array(list(itertools.product(range(1,m+2),repeat=m)))
        ok=np.stack([(L<=r).sum(1)>=r for r in range(1,m+1)],1)
        R=np.where(ok.any(1),m-np.argmax(ok[:,::-1],1),0)
        rej=L<=R[:,None]
        _BANDS[(m,nt)]=(L,rej[:,:nt].sum(1)/np.maximum(rej.sum(1),1))
    return _BANDS[(m,nt)]

def roles(n,nt,rho):
    """Every (test normals, calibration set) with its design probability; nulls come first in test."""
    out=[]
    for tt in itertools.combinations(range(n),nt):
        rem=np.setdiff1d(np.arange(n),tt)
        for flags in itertools.product((False,True),repeat=len(rem)):
            flags=np.array(flags,bool)
            mass=np.prod(np.where(flags,rho[rem],1-rho[rem]))/math.comb(n,nt)
            if mass>0:out.append((mass,np.array(tt),rem[flags]))
    return out

def exact_fdr(d):
    """Exact FDR of BH on randomized (and deterministic) weighted p-values, vectorized over roles and bands."""
    n,nt,na,rho,s,alpha=d['n'],d['nt'],d['na'],d['rho'],d['s'],d['alpha']
    N=n+na;m=nt+na;w=1/rho;L,fdp=band_table(m,nt)
    R=roles(n,nt,rho);mass=np.array([r[0] for r in R])
    calw=np.zeros((len(R),N))
    for i,(_,_,cal) in enumerate(R):calw[i,cal]=w[cal]
    test=np.array([np.r_[tt,np.arange(n,N)] for _,tt,_ in R])
    above=(s[:,None]>s[None,:]).astype(float)                 # above[i,j]=1 if s_i>s_j
    b=np.take_along_axis(calw@above,test,1);W=calw.sum(1)[:,None];wt=w[test]
    t=alpha*np.arange(0,m+1)/m
    F=np.clip((t[None,None,:]*(W+wt)[:,:,None]-b[:,:,None])/wt[:,:,None],0,1);F[:,:,0]=0
    P=np.concatenate([np.diff(F,axis=2),1-F[:,:,-1:]],2)       # P[state,j,k-1]=P(L_j=k)
    probs=P[:,np.arange(m)[None,:],L-1].prod(-1)               # (states, band vectors)
    rand=float(mass@(probs@fdp))
    p=(b+wt)/(W+wt);Ld=np.array([[int(np.argmax(np.r_[pj<=t[1:]+1e-12,True]))+1 for pj in row] for row in p])
    idx=((Ld-1)*((m+1)**np.arange(m-1,-1,-1))).sum(1)          # position of each band vector in L
    return rand,float(mass@fdp[idx]),alpha*nt/m

def exact_fdr_slow(d):
    """Independent re-implementation: explicit loops, BH via allocation_acquisition.bh on representative p-values."""
    n,nt,na,rho,s,alpha=d['n'],d['nt'],d['na'],d['rho'],d['s'],d['alpha']
    N=n+na;m=nt+na;w=1/rho;total=0.;fdr=0.
    thr=[alpha*k/m for k in range(m+1)]
    for tt in itertools.combinations(range(n),nt):
        rem=[i for i in range(n) if i not in tt]
        for flags in itertools.product((0,1),repeat=len(rem)):
            mass=1./math.comb(n,nt);cal=[]
            for i,f in zip(rem,flags):
                mass*=rho[i] if f else 1-rho[i]
                if f:cal.append(i)
            total+=mass
            if mass==0:continue
            test=list(tt)+list(range(n,N));W=sum(w[i] for i in cal);ivals=[]
            for j in test:
                bj=sum(w[i] for i in cal if s[i]>s[j]);lo=bj/(W+w[j]);hi=(bj+w[j])/(W+w[j])
                pr=[];rep=[]
                for k in range(1,m+2):
                    a=thr[k-1] if k>1 else -1.;c=thr[k] if k<=m else 2.
                    x,y=max(a,lo),min(c,hi);pr.append(max(0.,y-x)/(hi-lo));rep.append((x+y)/2)
                ivals.append((pr,rep))
            for combo in itertools.product(range(m+1),repeat=m):
                q=1.
                for j,k in enumerate(combo):q*=ivals[j][0][k]
                if q==0:continue
                pv=np.array([ivals[j][1][k] for j,k in enumerate(combo)])
                rr=bh(pv,alpha);fdr+=mass*q*rr[:nt].sum()/max(rr.sum(),1)
    assert abs(total-1)<1e-9
    return fdr

def simulate(d,draws,seed,wcs=True):
    """Monte Carlo of the full design: test normals, acquisition, U (and xi for WCS)."""
    n,nt,na,rho,s,alpha=d['n'],d['nt'],d['na'],d['rho'],d['s'],d['alpha']
    N=n+na;w=1/rho;rng=np.random.default_rng(seed);fb=np.empty(draws);fw=np.empty(draws)
    anom=np.arange(n,N)
    for r in range(draws):
        tt=rng.choice(n,nt,replace=False);rem=np.setdiff1d(np.arange(n),tt)
        cal=rem[rng.random(len(rem))<rho[rem]];test=np.r_[tt,anom];U=rng.random(len(test))
        W=w[cal].sum();b=np.array([w[cal][s[cal]>s[j]].sum() for j in test])
        p=(b+U*w[test])/(W+w[test]);rr=bh(p,alpha);fb[r]=rr[:nt].sum()/max(rr.sum(),1)
        if wcs:
            rw,_=wcs_rand(s,cal,test,w,U,rng.random(),alpha);fw[r]=rw[:nt].sum()/max(rw.sum(),1)
    se=lambda f:f.std(ddof=1)/np.sqrt(draws)
    return fb.mean(),se(fb),(fw.mean() if wcs else np.nan),(se(fw) if wcs else np.nan)

def random_design(rng):
    n=int(rng.integers(2,7));nt=int(rng.integers(1,min(n-1,4)+1));na=int(rng.integers(0,min(3,4-nt)+1))
    N=n+na
    return dict(n=n,nt=nt,na=na,rho=RHO[rng.integers(0,len(RHO),N)],s=rng.permutation(N).astype(float),alpha=float(rng.choice(ALPHAS)))

def mutate(d,rng):
    e={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()};u=rng.integers(3)
    if u==0:e['rho'][rng.integers(len(e['rho']))]=RHO[rng.integers(len(RHO))]
    elif u==1:i,j=rng.choice(len(e['s']),2,replace=False);e['s'][[i,j]]=e['s'][[j,i]]
    else:e['alpha']=float(rng.choice(ALPHAS))
    return e

def row(d,stage,did):
    r,dt,bnd=exact_fdr(d)
    return dict(stage=stage,design=did,n=d['n'],nt=d['nt'],na=d['na'],alpha=d['alpha'],
                rho=' '.join(f'{x:g}' for x in d['rho']),scores=' '.join(str(int(x)) for x in d['s']),
                fdr_rand_wbh=r,fdr_det_wbh=dt,bound=bnd,excess=r-bnd,excess_alpha=r-d['alpha'])

def parse(rw):
    return dict(n=int(rw.n),nt=int(rw.nt),na=int(rw.na),alpha=float(rw.alpha),
                rho=np.array([float(x) for x in rw.rho.split()]),s=np.array([float(x) for x in rw.scores.split()]))

def checks():
    """Code checks from the protocol: slow re-implementation, Monte Carlo agreement, null uniformity."""
    rng=np.random.default_rng(2026093000);worst_mc=0.;worst_slow=0.
    for i in range(20):
        d=random_design(rng);r,_,_=exact_fdr(d);sl=exact_fdr_slow(d)
        worst_slow=max(worst_slow,abs(r-sl));assert abs(r-sl)<1e-9,(i,r,sl)
        mb,se,_,_=simulate(d,20000,[2026093000,i],wcs=False)
        z=abs(mb-r)/max(se,1e-12);worst_mc=max(worst_mc,z);assert z<4,(i,r,mb,se)
        # design-averaged null p-value is uniform
        n,nt,na,rho,s=d['n'],d['nt'],d['na'],d['rho'],d['s'];w=1/rho;ts=np.linspace(.05,.95,19);cdf=np.zeros(19)
        for mass,tt,cal in roles(n,nt,rho):
            W=w[cal].sum()
            for j in tt:
                bj=w[cal][s[cal]>s[j]].sum();cdf+=mass/nt*np.clip((ts*(W+w[j])-bj)/w[j],0,1)
        assert np.allclose(cdf,ts,atol=1e-12),i
    return dict(designs=20,max_abs_fast_minus_slow=worst_slow,max_mc_z=worst_mc)

def main():
    started=time.time();chk=checks();print('CHECKS',chk,flush=True)
    rng=np.random.default_rng(2026093001);rows=[];designs={}
    for i in range(20000):
        d=random_design(rng);designs[('s1',i)]=d;rows.append(row(d,'random',i))
    print('stage 1',round(time.time()-started),flush=True)
    s1=pd.DataFrame(rows);top=s1.sort_values('excess',ascending=False).drop_duplicates(['n','nt','na','alpha','rho','scores']).head(20)
    rng=np.random.default_rng(2026093002)
    for k,did in enumerate(top.design):
        cur=designs[('s1',did)];cur_ex=s1.excess[did]
        for step in range(300):
            e=mutate(cur,rng);r=row(e,'climb',f'{did}.{step}');rows.append(r)
            if r['excess']>cur_ex:cur,cur_ex=e,r['excess']
    print('stage 2',round(time.time()-started),flush=True)
    led=pd.DataFrame(rows);led.to_csv(D/'wcs_counterexample_ledger.csv.gz',index=False,compression='gzip')
    best=led.sort_values('excess',ascending=False).drop_duplicates(['n','nt','na','alpha','rho','scores']).head(5)
    conf=[]
    for k,rw in enumerate(best.itertuples()):
        d=parse(rw);slow=exact_fdr_slow(d)
        mb,sb,mw,sw=simulate(d,400000,[2026093003,k])
        conf.append(dict(rank=k+1,n=d['n'],nt=d['nt'],na=d['na'],alpha=d['alpha'],rho=rw.rho,scores=rw.scores,bound=rw.bound,
            exact_rand_wbh=rw.fdr_rand_wbh,exact_slow=slow,exact_det_wbh=rw.fdr_det_wbh,mc_rand_wbh=mb,mc_rand_wbh_se=sb,
            mc_wcs=mw,mc_wcs_se=sw,z_wbh=(mb-rw.bound)/sb,z_wcs=(mw-rw.bound)/sw,
            confirmed=bool(rw.excess>1e-9 and abs(slow-rw.fdr_rand_wbh)<1e-9 and (mb-rw.bound)>5*sb),
            exceeds_alpha=bool(rw.fdr_rand_wbh>d['alpha']+1e-9)))
        print('CONFIRM',conf[-1],flush=True)
    conf=pd.DataFrame(conf);conf.to_csv(D/'wcs_counterexample_confirm.csv',index=False)
    scale=[]
    if conf.confirmed.any():
        c=conf[conf.confirmed].iloc[0];d=parse(c)
        for k in (1,2,5,10,25):
            N=d['n']+d['na'];idx=np.r_[np.repeat(np.arange(d['n']),k),np.repeat(np.arange(d['n'],N),k)]
            e=dict(n=k*d['n'],nt=k*d['nt'],na=k*d['na'],alpha=d['alpha'],rho=d['rho'][idx],s=d['s'][idx]*k+np.tile(np.arange(k),N))
            mb,sb,mw,sw=simulate(e,20000,[2026093004,k]);bnd=d['alpha']*d['nt']/(d['nt']+d['na'])
            scale.append(dict(k=k,n=e['n'],m=e['nt']+e['na'],bound=bnd,rand_wbh=mb,rand_wbh_se=sb,wcs=mw,wcs_se=sw,
                              z_wbh=(mb-bnd)/sb,z_wcs=(mw-bnd)/sw,excess_claimed=bool(mb-bnd>3*sb)))
            print('SCALE',scale[-1],flush=True)
    pd.DataFrame(scale).to_csv(D/'wcs_counterexample_scale.csv',index=False)
    ex=led.excess
    report=dict(checks=chk,designs_evaluated=len(led),random_designs=int((led.stage=='random').sum()),
        climb_designs=int((led.stage=='climb').sum()),designs_above_bound=int((ex>1e-9).sum()),
        designs_above_alpha=int((led.excess_alpha>1e-9).sum()),max_excess=float(ex.max()),
        excess_quantiles={q:float(ex.quantile(q)) for q in (.5,.9,.99,1.)},
        det_wbh_above_bound=int((led.fdr_det_wbh-led.bound>1e-9).sum()),confirmed=int(conf.confirmed.sum()),
        seconds=time.time()-started,
        protocol_sha256=hashlib.sha256((D/'WCS_COUNTEREXAMPLE_PROTOCOL.md').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (D/'wcs_counterexample_manifest.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

if __name__=='__main__':main()
