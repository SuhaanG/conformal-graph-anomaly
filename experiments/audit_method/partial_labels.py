"""Frozen supervised scores, non-test label revelation, matched calibration."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy import sparse
D=Path(__file__).resolve().parent
G=D.parent/'aistats_followup/h200_results/paper/aistats_followup'
sys.path.insert(0,str(D.parent/'aistats_revision'))
from validate_revision import pvals,bh

def observed_filter(a,labels,known):
    observed=np.zeros(len(labels));observed[known]=labels[known]
    return np.asarray(a@observed).ravel()==0

def main():
    rows=[];checks=0
    for dsid,ds in enumerate(('amazon','tolokers')):
        g=np.load(G/f'{ds}_graph.npz');y=g['labels'].astype(int)
        a=sparse.csr_matrix((np.ones(len(g['indices'])),g['indices'],g['indptr']),shape=(len(y),len(y)))
        for seed in range(10):
            c=np.load(D/f'{ds}_hgb_attributes_scores_{seed}.npz');train=c['train'];deploy=c['deploy']
            normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
            scores={model:np.load(D/f'{ds}_{model}_scores_{seed}.npz')['scores'] for model in ('hgb_attributes','hgb_graph_features')}
            for split in range(5):
                rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                reference=np.setdiff1d(deploy,test)
                rng=np.random.default_rng(np.random.SeedSequence([20260928,dsid,seed,split]))
                u=rng.random(len(reference))
                for fraction in (.25,.5):
                    revealed=reference[u<fraction];known=np.r_[train,revealed]
                    assert not np.intersect1d(known,test).size
                    mask=observed_filter(a,y,known)
                    changed=y.copy();unknown=np.setdiff1d(np.arange(len(y)),known);changed[unknown]=1-changed[unknown]
                    assert np.array_equal(mask,observed_filter(a,changed,known));checks+=1
                    pool=revealed[y[revealed]==0];filtered=pool[mask[pool]];n=min(1000,len(filtered))
                    assert n>0
                    for method in ('filtered','matched_random','full_reference'):
                        for rep in range(1 if method=='full_reference' else 200):
                            cal=pool if method=='full_reference' else rng.choice(filtered if method=='filtered' else pool,n,False)
                            for model,s in scores.items():
                                r=bh(pvals(s[cal],s[test]));nr=int(r.sum());ntp=int(y[test[r]].sum())
                                rows.append(dict(dataset=ds,model=model,seed=seed,split=split,label_fraction=fraction,
                                    method=method,rep=rep,n_calib=len(cal),training_labels=len(train),reference_labels=len(revealed),
                                    fdp=(nr-ntp)/max(nr,1),power=ntp/int(y[test].sum()),discoveries=nr))
            print('PARTIAL',ds,seed,flush=True)
    out=D/'partial_label_trials.csv';pd.DataFrame(rows).to_csv(out,index=False)
    (D/'partial_label_manifest.json').write_text(json.dumps(dict(rows=len(rows),hidden_label_invariance_checks=checks,
        master_seed=20260928,sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
