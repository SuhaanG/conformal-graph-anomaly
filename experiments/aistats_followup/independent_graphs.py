"""Independent degree-corrected random graphs; analytic scoring controls."""
from pathlib import Path
import sys,json,time,csv,argparse
import numpy as np
from scipy import sparse
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'aistats_revision'))
from validate_revision import pvals,bh,write_csv

def simulate_one(rep,sigma,cross,all_null=False):
    rng=np.random.default_rng(np.random.SeedSequence([20260920,rep,int(sigma*10),int(cross),int(all_null)]))
    N=1200;y=np.zeros(N,int)
    if not all_null:y[rng.choice(N,120,False)]=1
    propensity=np.exp(sigma*rng.normal(size=N)-sigma*sigma/2);propensity/=propensity.mean()
    ii,jj=np.triu_indices(N,1);different=y[ii]!=y[jj];aff=np.where(different,cross,1.)
    prob=12/(N-1)*propensity[ii]*propensity[jj]*aff/aff.mean();clipped=float(np.mean(prob>.9))
    edge=rng.random(len(ii))<np.minimum(prob,.9);i=ii[edge];j=jj[edge]
    a=sparse.coo_matrix((np.ones(2*len(i)),(np.r_[i,j],np.r_[j,i])),shape=(N,N)).tocsr();degree=np.asarray(a.sum(1)).ravel()
    normals=np.flatnonzero(y==0);anoms=np.flatnonzero(y==1)
    tt=np.r_[rng.choice(normals,len(normals)//4,False),rng.choice(anoms,len(anoms)//4,False)]
    ref=np.setdiff1d(np.arange(N),tt);pool=ref[y[ref]==0]
    oracle=np.asarray(a@y).ravel()==0;filtered=pool[oracle[pool]]
    reveal=ref[rng.random(len(ref))<.25];obs=np.zeros(N);obs[reveal]=y[reveal]
    partialpool=reveal[y[reveal]==0];partialfiltered=partialpool[np.asarray(a@obs).ravel()[partialpool]==0]
    assert not np.intersect1d(reveal,tt).size
    n=min(200,len(filtered));np_=min(200,len(partialfiltered));assert n>0 and np_>0
    cf=rng.choice(filtered,n,False);cr=rng.choice(pool,n,False);half=cf[:n//2];cm=np.r_[half,rng.choice(np.setdiff1d(pool,half),n-len(half),False)]
    cp=rng.choice(partialfiltered,np_,False);cpr=rng.choice(partialpool,np_,False)
    z=rng.normal(size=N);neighbor=(a@z)/np.sqrt(np.maximum(degree,1))
    ld=np.log1p(degree);ld=(ld-ld.mean())/max(ld.std(),1e-10)
    results=[]
    conditions=[(2.,.5)] if all_null else [(theta,dep) for theta in (0.,2.) for dep in (0.,.5)]
    for theta,dep in conditions:
        noise=(np.sqrt(1-dep)*z+np.sqrt(dep)*neighbor)/np.sqrt((1-dep)+dep*(degree>0))
        scores=noise+4*y+theta*ld
        for method,cal in [('oracle_filtered',cf),('matched_random',cr),('mixture_half',cm),('partial_filtered',cp),('partial_random',cpr)]:
            assert not np.intersect1d(cal,tt).size and np.all(y[cal]==0)
            p=pvals(scores[cal],scores[tt]);r=bh(p);nr=int(r.sum());yt=y[tt]
            results.append(dict(rep=rep,sigma=sigma,cross_multiplier=cross,theta=theta,neighbor_noise_fraction=dep,all_null=int(all_null),method=method,
                n_calib=len(cal),n_test=len(tt),n_null=int((yt==0).sum()),edges=len(i),clipped_probability_fraction=clipped,
                fdp=float(np.sum(r&(yt==0))/max(nr,1)),power=float(np.sum(r&(yt==1))/max(int(yt.sum()),1)),discoveries=nr,
                null_tail001=float(np.mean(p[yt==0]<=.01))))
    return results

def main(reps):
    start=time.time();rows=[]
    for rep in range(reps):
        for sigma in (0.,1.2):
            for cross in (1.,4.):rows.extend(simulate_one(rep,sigma,cross))
        rows.extend(simulate_one(rep,1.2,1.,True))
        if (rep+1)%50==0:print('GRAPHS REP',rep+1,'seconds',round(time.time()-start,1),flush=True)
    write_csv(OUT/'independent_graph_trials.csv',rows)
    groups={};keys=('sigma','cross_multiplier','theta','neighbor_noise_fraction','all_null','method')
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    summary=[]
    for key,g in groups.items():
        z=dict(zip(keys,key));z['independent_repetitions']=len(g)
        for name in ('fdp','power','discoveries','null_tail001','n_calib','edges','clipped_probability_fraction'):
            v=np.array([r[name] for r in g]);z[name+'_mean']=float(v.mean());z[name+'_mcse']=float(v.std(ddof=1)/np.sqrt(len(v)))
        summary.append(z)
    write_csv(OUT/'independent_graph_summary.csv',summary)
    (OUT/'independent_graph_manifest.json').write_text(json.dumps(dict(master_seed=20260920,repetitions_per_cell=reps,
        unique_graphs=5*reps,method_evaluations=len(rows),seconds=time.time()-start,nodes=1200,anomaly_fraction=.1,
        all_null_control=True,nominal_alpha=.1,score_type='specified analytic scorers; no trained GNN claim'),indent=2))
    print('GRAPHS COMPLETE',len(rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=500);args=ap.parse_args();main(args.reps)
