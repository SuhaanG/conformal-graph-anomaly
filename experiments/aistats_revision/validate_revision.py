"""Independent score-level selection experiment and audit of frozen graph summaries.
Run: python paper/aistats_revision/validate_revision.py
No graph rows are overwritten. All trials, including zero discoveries, are retained.
"""
from pathlib import Path
import csv, json, hashlib, platform, time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def pvals(c,t,w=None,wt=None):
    c=np.asarray(c); t=np.asarray(t)
    if w is None:
        return (1+len(c)-np.searchsorted(np.sort(c),t,side='left'))/(len(c)+1)
    w=np.asarray(w,dtype=float); wt=np.asarray(wt,dtype=float)
    if np.any(w<=0) or np.any(wt<=0) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(wt)):
        raise ValueError('Weights must be positive and finite')
    order=np.argsort(c); tail=np.r_[np.cumsum(w[order][::-1])[::-1],0.]
    return (wt+tail[np.searchsorted(c[order],t,side='left')])/(wt+w.sum())

def bh(p,alpha=.1):
    order=np.argsort(p); cross=np.flatnonzero(p[order]<=alpha*np.arange(1,len(p)+1)/len(p))
    out=np.zeros(len(p),dtype=bool)
    if len(cross): out[order[:cross[-1]+1]]=True
    return out

def metrics(p,n0):
    r=bh(p); k=int(r.sum())
    return dict(fdp=float(r[:n0].sum()/max(k,1)),power=float(r[n0:].mean()),discoveries=k,
                null_tail_005=float(np.mean(p[:n0]<=.05)),null_tail_010=float(np.mean(p[:n0]<=.10)))

def checks():
    c=np.array([1.,1.,2.,4.]); t=np.array([0.,1.,1.5,2.,5.]); w=np.array([1.,2.,3.,4.]); wt=np.arange(1.,6.)
    assert np.array_equal(pvals(c,t),pvals(c,t,np.ones(4),np.ones(5)))
    brute=np.array([(wt[j]+w[c>=s].sum())/(wt[j]+w.sum()) for j,s in enumerate(t)])
    assert np.allclose(pvals(c,t,w,wt),brute)
    assert np.allclose(pvals(c,t,w,wt),pvals(c,t,17*w,17*wt))
    assert np.all(pvals(c+2,t)>=pvals(c,t))
    # An attainable common-calibration change loses the true discovery.
    before=np.r_[np.zeros(96),np.full(3,2.5)]
    after=before.copy(); after[:16]=2.5
    test=np.array([3.,2.])  # first null, second anomalous
    assert np.all(after>=before)
    assert np.allclose(pvals(before,test),[.01,.04])
    assert np.allclose(pvals(after,test),[.01,.20])
    assert bh(pvals(before,test)).tolist()==[True,True]
    assert bh(pvals(after,test)).tolist()==[True,False]
    rng=np.random.default_rng(16)
    for _ in range(100):
        p=pvals(rng.normal(size=31),rng.normal(size=60)); r=bh(p)
        if r.any(): assert r.sum()>=int(np.ceil(len(p)/(.1*32)))
    return ['uniform-weight identity','brute-force weighted ranks including ties','common-scale invariance','calibration rank monotonicity','FDP counterexample','BH rank floor']

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def simulate(reps=2000):
    rows=[]; n=200; n0=360; m=400
    for rep in range(reps):
        # Independent trials; shared random numbers across arms within a trial.
        rng=np.random.default_rng(np.random.SeedSequence([20260913,rep]))
        u=rng.random(n); z=rng.normal(size=n); v=rng.random(m); e=rng.normal(size=m)
        wc0=(u<.5).astype(int); wtest=(v<.5).astype(int)
        caids=np.arange(n); tids=np.arange(n,n+m)
        assert not np.intersect1d(caids,tids).size
        # Independent anomalous calibration residuals and selection uniforms.
        za=rng.normal(size=n); wa=(rng.random(n)<.5).astype(int); uc=rng.random(n)
        for theta in (0.,2.):
            target=theta*wtest+e; target[n0:]+=3.
            base=theta*wc0+z
            def add(method,q,p,epsilon=0.):
                rows.append(dict(rep=rep,theta=theta,q=q,method=method,epsilon=epsilon,**metrics(p,n0)))
            add('random',1.,pvals(base,target))
            for q in (.25,.05,0.):
                wc=(u<q/(1+q)).astype(int); selected=theta*wc+z
                add('selected',q,pvals(selected,target))
                if q>0:
                    weights=np.where(wc==1,1/q,1.); target_weights=np.where(wtest==1,1/q,1.)
                    add('weighted_correct',q,pvals(selected,target,weights,target_weights))
                    add('weighted_legacy',q,pvals(selected,target,weights,np.ones(m)))
            for epsilon in (.05,.10):
                contaminated=np.where(uc<epsilon,theta*wa+za+3.,base)
                add('independent_contamination',1.,pvals(contaminated,target),epsilon)
        if (rep+1)%500==0: print('SIM',rep+1,flush=True)
    write_csv(OUT/'synthetic_trials.csv',rows)
    groups={}
    for r in rows: groups.setdefault((r['theta'],r['q'],r['method'],r['epsilon']),[]).append(r)
    summary=[]
    for (theta,q,method,eps),g in groups.items():
        a=dict(theta=theta,q=q,method=method,epsilon=eps,repetitions=len(g))
        for key in ('fdp','power','discoveries','null_tail_005','null_tail_010'):
            x=np.array([r[key] for r in g]); a[key+'_mean']=float(x.mean()); a[key+'_mcse']=float(x.std(ddof=1)/np.sqrt(len(x)))
        summary.append(a)
    write_csv(OUT/'synthetic_summary.csv',summary)
    print(json.dumps(summary,indent=2),flush=True)

def graph_audit():
    summary=[]; paired=[]; manifest={}
    for path in sorted((ROOT/'results/published').glob('calibration_strategy_*.csv')):
        rows=list(csv.DictReader(path.open(encoding='utf8')))
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        groups={}
        for r in rows: groups.setdefault(r['strategy'],[]).append(r)
        for method,g in groups.items():
            if method.startswith('true_contam'): continue
            assert len(g)==5 and len({r['seed'] for r in g})==5
            a=dict(dataset=g[0]['dataset'],detector=g[0]['detector'],strategy=method,seeds=5)
            for key in ('n_calib','m_test','n_null','gamma_t_lo','t_lo','realized_fdr','power','n_discoveries','score_gap_cohens_d'):
                x=np.array([float(r[key]) for r in g]); name='fdp' if key=='realized_fdr' else key
                if key=='score_gap_cohens_d': x=-x; name='delta_test_minus_calib'
                a[name+'_mean']=float(x.mean()); a[name+'_sd']=float(x.std(ddof=1))
            summary.append(a)
        c={r['seed']:r for r in groups['clean']}; r={r['seed']:r for r in groups['random']}
        for s in c:
            for key in ('n_calib','m_test','n_null','t_lo'): assert c[s][key]==r[s][key]
            assert c[s]['calib_frac_anomalous']=='0.0' and r[s]['calib_frac_anomalous']=='0.0'
        delta=np.array([float(c[s]['realized_fdr'])-float(r[s]['realized_fdr']) for s in c])
        paired.append(dict(dataset=c[s]['dataset'],detector=c[s]['detector'],mean_fdp_difference=float(delta.mean()),sd=float(delta.std(ddof=1)),positive_seeds=int((delta>0).sum())))
    write_csv(OUT/'graph_summary.csv',summary); write_csv(OUT/'paired_summary.csv',paired)
    return manifest

def analytic_rank_check():
    """Independent numerical integration of the exact conditional rank law."""
    from scipy.integrate import quad
    from scipy.stats import norm, binom
    simulation=list(csv.DictReader((OUT/'synthetic_summary.csv').open()))
    rows=[]
    for theta in (0.,2.):
        for q in (1.,.25,.05,0.):
            method='random' if q==1 else 'selected'
            r=next(r for r in simulation if float(r['theta'])==theta and float(r['q'])==q and r['method']==method)
            for t,key in ((.05,'null_tail_005'),(.1,'null_tail_010')):
                rank=int(np.floor(t*201)); pc=q/(1+q)
                def integrand(s):
                    fc=(1-pc)*norm.cdf(s)+pc*norm.cdf(s-theta)
                    density=.5*norm.pdf(s)+.5*norm.pdf(s-theta)
                    return binom.cdf(rank-1,200,1-fc)*density
                exact,error=quad(integrand,-12,14,epsabs=1e-10)
                empirical=float(r[key+'_mean']); se=float(r[key+'_mcse'])
                assert error<1e-8
                assert abs(empirical-exact)<5*se, (theta,q,t,empirical,exact,se)
                rows.append(dict(theta=theta,q=q,t=t,exact_null_tail=exact,empirical_null_tail=empirical,mcse=se,z_error=(empirical-exact)/se))
    write_csv(OUT/'analytic_rank_validation.csv',rows)
    print('Analytic binomial-mixture check passed:',len(rows),'threshold comparisons',flush=True)

if __name__=='__main__':
    t=time.time(); passed=checks(); manifest=graph_audit(); simulate(); analytic_rank_check()
    passed.append('independent binomial-mixture quadrature comparison')
    manifest.update(python=platform.python_version(),numpy=np.__version__,checks=passed,seconds=time.time()-t,seed=20260913)
    (OUT/'validation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('VALIDATION PASSED',passed,flush=True)
