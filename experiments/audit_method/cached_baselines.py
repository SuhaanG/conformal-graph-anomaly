"""Full-reference and fixed degree-normalization comparisons; no new training."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
D=Path(__file__).resolve().parent
G=D.parent/'aistats_followup/h200_results/paper/aistats_followup'
sys.path.insert(0,str(D.parent/'aistats_revision'))
from validate_revision import pvals,bh

def main():
 rows=[]
 for dsid,ds in enumerate(('amazon','tolokers')):
  g=np.load(G/f'{ds}_graph.npz');y=g['labels'];N=len(y);degree=g['degree']
  a=sparse.csr_matrix((np.ones(len(g['indices'])),g['indices'],g['indptr']),shape=(N,N))
  verified=np.arange(3305,N) if ds=='amazon' else np.arange(N)
  normals=verified[y[verified]==0];anoms=verified[y[verified]==1]
  filt=np.asarray(a@y).ravel()==0
  for model in ('dominant_pygod','gae','isolation_forest'):
   for seed in range(10):
    raw=np.load(G/f'{ds}_{model}_scores_{seed}.npz')['scores']
    variants=[('raw',raw)]
    if model=='dominant_pygod': variants.append(('degree_normalized',raw/np.log1p(degree+1e-8)))
    for variant,s in variants:
     assert np.isfinite(s).all()
     for split in range(5):
      rng=np.random.default_rng(np.random.SeedSequence([20260917,dsid,split]))
      test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
      pool=np.setdiff1d(normals,test);filtered=pool[filt[pool]];n=min(1000,len(filtered))
      plans=[('full_reference',pool)]
      if variant=='degree_normalized':plans.append(('filtered',filtered))
      for method,cal in plans:
       r=bh(pvals(s[cal],s[test]));nr=int(r.sum());tp=int(y[test[r]].sum())
       rows.append(dict(dataset=ds,model=model,variant=variant,seed=seed,split=split,method=method,rep=-1,
                        n_calib=len(cal),n_test=len(test),n_reference_labels=len(verified)-len(test),
                        auroc=roc_auc_score(y[verified],s[verified]),fdp=(nr-tp)/max(nr,1),power=tp/int(y[test].sum()),discoveries=nr,true_discoveries=tp))
      if variant=='degree_normalized':
       for rep in range(200):
        rng=np.random.default_rng(np.random.SeedSequence([20260922,dsid,seed,split,rep]))
        cal=rng.choice(pool,n,False);r=bh(pvals(s[cal],s[test]));nr=int(r.sum());tp=int(y[test[r]].sum())
        rows.append(dict(dataset=ds,model=model,variant=variant,seed=seed,split=split,method='matched_random',rep=rep,
                         n_calib=n,n_test=len(test),n_reference_labels=len(verified)-len(test),auroc=roc_auc_score(y[verified],s[verified]),
                         fdp=(nr-tp)/max(nr,1),power=tp/int(y[test].sum()),discoveries=nr,true_discoveries=tp))
   print('BASELINE',ds,model,flush=True)
 df=pd.DataFrame(rows);df.to_csv(D/'cached_baseline_trials.csv',index=False)
 keys=['dataset','model','variant','method'];metrics=['fdp','power','discoveries','true_discoveries','auroc','n_calib','n_reference_labels']
 perseed=df.groupby(keys+['seed','split'])[metrics].mean().groupby(keys+['seed']).mean()
 summary=perseed.groupby(keys).mean().join(perseed.groupby(keys)[['fdp','power']].std().add_suffix('_seed_sd')).reset_index()
 summary.to_csv(D/'cached_baseline_summary.csv',index=False)
 (D/'cached_baseline_manifest.json').write_text(json.dumps(dict(rows=len(df),training_seeds=10,test_splits=5,
  sha256=hashlib.sha256((D/'cached_baseline_trials.csv').read_bytes()).hexdigest(),normalized_formula='score/log1p(degree+1e-8)',no_trimming=True),indent=2))
 print(summary.to_string(index=False))
if __name__=='__main__':main()
