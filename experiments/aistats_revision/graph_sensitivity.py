"""Fresh graph sensitivity: training fixed within seed, trimming and calibration varied.
Run with the isolated environment. Outputs and score caches stay in this folder.
"""
import sys, time, json, csv, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from detectors import score_nodes
from real_data_experiment import load_pygod_graph
from validate_revision import pvals,bh,write_csv
import pygod,torch_geometric

def main():
    torch.set_num_threads(4)
    graph,x,y=load_pygod_graph('weibo')
    print('DATA',len(y),graph.number_of_edges(),x.shape,int(y.sum()),flush=True)
    normals=np.flatnonzero(y==0); anoms=np.flatnonzero(y==1)
    exp=np.array([sum(y[j]==1 for j in graph.neighbors(int(i)))/max(1,graph.degree(int(i))) for i in normals])
    rows=[]; timings=[]
    for seed in range(5):
        start=time.time(); path=OUT/f'weibo_gae_scores_{seed}.npz'
        if path.exists(): scores=np.load(path)['scores']
        else:
            scores=score_nodes('gae',graph,x,labels=y,seed=seed,n_epochs=100,device='cpu')
            assert scores.shape==y.shape and np.isfinite(scores).all()
            np.savez_compressed(path,scores=scores,labels=y)
        timings.append(time.time()-start)
        for trim in (0.,.01,.05):
            cutoff=np.percentile(scores[normals],100*(1-trim)); eligible=normals[scores[normals]<=cutoff]
            mask=np.isin(normals,eligible)
            unexposed=normals[mask & (exp==0)]; exposed=normals[mask & (exp>0)]
            rng=np.random.default_rng(seed)
            test_normal=rng.choice(eligible,size=min(2000,len(eligible)//2),replace=False)
            test=np.r_[test_normal,anoms]; n0=len(test_normal)
            pools={'clean':np.setdiff1d(unexposed,test_normal),'random':np.setdiff1d(eligible,test_normal),'exposed_only':np.setdiff1d(exposed,test_normal)}
            n=min(map(len,pools.values())); assert n>=50
            for method in ('clean','random','exposed_only','random_full'):
                pool=pools['random'] if method=='random_full' else pools[method]
                nh=min(4000,len(pool)) if method=='random_full' else n
                cal=rng.choice(pool,size=nh,replace=False)
                assert not np.intersect1d(cal,test).size
                assert np.all(y[cal]==0)
                p=pvals(scores[cal],scores[test]); r=bh(p); k=int(r.sum())
                row=dict(dataset='weibo',detector='gae',seed=seed,trim=trim,strategy=method,n_calib=nh,m=len(test),fdp=float(r[:n0].sum()/max(k,1)),power=float(r[n0:].mean()),discoveries=k)
                rows.append(row)
        write_csv(OUT/'graph_sensitivity_trials.csv',rows)
        print('SEED',seed,'seconds',round(timings[-1],1),flush=True)
    summary=[]
    for trim in (0.,.01,.05):
        for method in ('clean','random','exposed_only','random_full'):
            g=[r for r in rows if r['trim']==trim and r['strategy']==method]; a=dict(trim=trim,strategy=method,seeds=len(g))
            for key in ('n_calib','fdp','power','discoveries'):
                z=np.array([r[key] for r in g]); a[key+'_mean']=float(z.mean());a[key+'_sd']=float(z.std(ddof=1))
            summary.append(a)
    write_csv(OUT/'graph_sensitivity_summary.csv',summary)
    (OUT/'graph_sensitivity_manifest.json').write_text(json.dumps(dict(dataset='weibo',detector='gae',seeds=list(range(5)),epochs=100,threads=4,torch=torch.__version__,pygod=pygod.__version__,pyg=torch_geometric.__version__,features_sha256=hashlib.sha256(x.tobytes()).hexdigest(),labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),seconds=timings),indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
