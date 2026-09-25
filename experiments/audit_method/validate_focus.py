"""Independent ledger aggregation and exhaustive finite-design sanity checks."""
from pathlib import Path
from itertools import combinations
import json,sys
import numpy as np
import pandas as pd
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'aistats_revision'))
from validate_revision import pvals

def exact_design():
    # Enumerate every normal-node role assignment, including nonempty rejection.
    total=0;nonempty=0;max_excess=-1.;settings=0
    universe=np.arange(8)
    for alt_tuple in combinations(universe,2):
        alt=np.asarray(alt_tuple);norm=np.setdiff1d(universe,alt)
        for n,m0 in ((2,1),(3,2),(4,2)):
            for alpha in (.1,.25,.5):
                values=[]
                for ct in combinations(norm,n):
                    cal=np.asarray(ct);remaining=np.setdiff1d(norm,cal)
                    for tt in combinations(remaining,m0):
                        test=np.r_[tt,alt];p=pvals(cal,test);order=np.argsort(p)
                        cross=np.flatnonzero(p[order]<=alpha*np.arange(1,len(p)+1)/len(p))
                        k=int(cross[-1]+1) if len(cross) else 0
                        values.append(float(np.sum(order[:k]<m0)/max(k,1)))
                        nonempty+=int(k>0);total+=1
                fdr=float(np.mean(values));bound=alpha*m0/(m0+2)
                assert fdr<=bound+1e-12,(alt_tuple,n,m0,alpha,fdr,bound)
                max_excess=max(max_excess,fdr-bound);settings+=1
    return dict(settings=settings,assignments=total,nonempty=nonempty,max_bound_excess=max_excess)

def main():
    x=pd.read_csv(D/'selection_dose_trials.csv.gz');assert len(x)==810000
    keys=['dataset','model','seed','split','label_fraction','removed','method']
    assert x.groupby(keys).size().eq(50).all()
    assert x[['fdp','power','nonempty']].ge(0).all().all() and x[['fdp','power','nonempty']].le(1).all().all()
    assert x.discoveries.ge(0).all() and x.n_calib.gt(0).all()
    assert x.groupby([k for k in keys if k!='method']+['rep']).n_calib.nunique().eq(1).all()
    z=x[x.removed==0];assert z.groupby([k for k in keys if k!='method']+['rep']).fdp.nunique().eq(1).all()
    metrics=['fdp','power','discoveries','nonempty','n_calib','pool_size']
    rebuilt=x.groupby(keys)[metrics].mean().reset_index()
    saved=pd.read_csv(D/'selection_dose_splits.csv')
    for key in metrics:
        assert np.allclose(rebuilt.set_index(keys)[key].sort_index(),saved.set_index(keys)[key].sort_index(),rtol=1e-12,atol=1e-12),key
    seeds=rebuilt.groupby([k for k in keys if k!='split'])[metrics].mean().reset_index()
    groups=['dataset','model','label_fraction','removed','method']
    agg=seeds.groupby(groups)[metrics].agg(['mean','std']);agg.columns=['_'.join(c) for c in agg.columns]
    target=pd.read_csv(D/'selection_dose_summary.csv').set_index(groups)
    assert np.allclose(agg.sort_index(),target[agg.columns].sort_index(),rtol=1e-11,atol=1e-12)
    report=dict(trial_rows=len(x),split_cells=len(rebuilt),all_outcomes_included=True,summary_recomputed=True,finite_design=exact_design())
    (D/'focus_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
