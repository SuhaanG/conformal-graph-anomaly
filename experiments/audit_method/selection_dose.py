"""Frozen observed-exposure sensitivity on cached supervised scores."""
from pathlib import Path
import csv, gzip, hashlib, json, sys, time
import numpy as np
import pandas as pd
from scipy import sparse

D = Path(__file__).resolve().parent
G = D.parent / 'aistats_followup/h200_results/paper/aistats_followup'
FRACTIONS = (0., .05, .10, .20, .40, .60, .80, .90, .97)
REVEAL = (.25, .50, 1.)
REPS = 50
MASTER = 20260923

def exposure(a, y, known):
    observed = np.zeros(len(y)); observed[known] = y[known]
    return np.asarray(a @ observed).ravel() / np.maximum(1, np.asarray(a.sum(axis=1)).ravel())

def score_ranks(s, seed):
    # Lexicographic ordering preserves every strict score comparison.
    u = np.random.default_rng(seed).random(len(s))
    order = np.lexsort((u, s))
    rank = np.empty(len(s), dtype=int); rank[order] = np.arange(len(s))
    return rank

def evaluate(rank, cal, test_order, cumulative_anomalies, thresholds):
    p = (1 + len(cal) - np.searchsorted(np.sort(rank[cal]), rank[test_order], side='left')) / (len(cal) + 1)
    cross = np.flatnonzero(p <= thresholds)
    k = int(cross[-1] + 1) if len(cross) else 0
    tp = int(cumulative_anomalies[k - 1]) if k else 0
    return (k-tp)/max(k,1), tp/max(int(cumulative_anomalies[-1]),1), k, int(k>0)

def validate():
    sys.path.insert(0, str(D.parent/'aistats_revision'))
    from validate_revision import pvals, bh
    rng = np.random.default_rng(84117)
    for i in range(100):
        s = rng.integers(0, 5, 80); y = rng.integers(0, 2, 80)
        rank = score_ranks(s, i); cal = rng.choice(40, 30, False); test = np.arange(40,80)
        order = test[np.argsort(-rank[test])]; cs = np.cumsum(y[order])
        out = evaluate(rank, cal, order, cs, .1*np.arange(1,41)/40)
        reject = bh(pvals(rank[cal], rank[test])); k = int(reject.sum()); tp = int(y[test[reject]].sum())
        assert out == ((k-tp)/max(k,1),tp/max(int(y[test].sum()),1),k,int(k>0))

def main():
    validate(); started = time.time(); summaries=[]; checks=0; nrows=0
    fields=['dataset','model','seed','split','label_fraction','removed','method','rep','n_calib','pool_size','fdp','power','discoveries','nonempty']
    dest=D/'selection_dose_trials.csv.gz'
    with gzip.open(dest,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for dsid, ds in enumerate(('amazon','tolokers','weibo')):
            g=np.load((D if ds=='weibo' else G)/f'{ds}_graph.npz');y=g['labels'].astype(int)
            a=sparse.csr_matrix((np.ones(len(g['indices'])),g['indices'],g['indptr']),shape=(len(y),len(y)))
            for seed in range(10):
                models={}; base=None
                for mid, model in enumerate(('hgb_attributes','hgb_graph_features')):
                    c=np.load(D/f'{ds}_{model}_scores_{seed}.npz')
                    if base is None: train=c['train'];deploy=c['deploy'];base=True
                    assert np.array_equal(c['train'],train) and np.array_equal(c['deploy'],deploy)
                    assert np.array_equal(c['labels'],y)
                    models[model]=score_ranks(c['scores'],np.random.SeedSequence([MASTER,dsid,seed,mid,7]))
                normals=deploy[y[deploy]==0];anoms=deploy[y[deploy]==1]
                for split in range(5):
                    rng=np.random.default_rng(np.random.SeedSequence([20260926,dsid,seed,split]))
                    test=np.r_[rng.choice(normals,int(round(.25*len(normals))),False),rng.choice(anoms,int(round(.25*len(anoms))),False)]
                    reference=np.setdiff1d(deploy,test)
                    rng=np.random.default_rng(np.random.SeedSequence([20260928,dsid,seed,split]));u=rng.random(len(reference))
                    prepared={model:(test[np.argsort(-rank[test])],) for model,rank in models.items()}
                    prepared={model:(order[0],np.cumsum(y[order[0]]),.1*np.arange(1,len(test)+1)/len(test)) for model,order in prepared.items()}
                    for rho in REVEAL:
                        revealed=reference[u<rho];known=np.r_[train,revealed];pool=revealed[y[revealed]==0]
                        e=exposure(a,y,known)
                        changed=y.copy();unknown=np.setdiff1d(np.arange(len(y)),known);changed[unknown]=1-changed[unknown]
                        assert np.array_equal(e,exposure(a,changed,known)); checks+=1
                        assert not np.intersect1d(known,test).size and not np.intersect1d(pool,train).size
                        acc={(m,d,k):[] for m in models for d in FRACTIONS for k in ('exposure','random')}
                        for rep in range(REPS):
                            rng=np.random.default_rng(np.random.SeedSequence([MASTER,dsid,seed,split,int(rho*100),rep]))
                            filtered_order=pool[np.lexsort((rng.random(len(pool)),e[pool]))]
                            random_order=rng.permutation(pool)
                            for drop in FRACTIONS:
                                n=int(np.ceil((1-drop)*len(pool)));assert n>0
                                for model,rank in models.items():
                                    zero=[]
                                    for method,order in (('exposure',filtered_order),('random',random_order)):
                                        cal=order[:n]
                                        val=evaluate(rank,cal,*prepared[model]);zero.append(val)
                                        acc[model,drop,method].append(val)
                                        writer.writerow(dict(zip(fields,[ds,model,seed,split,rho,drop,method,rep,n,len(pool),*val])))
                                        nrows+=1
                                    if drop==0:assert zero[0]==zero[1]
                        for (model,drop,method),values in acc.items():
                            vals=np.asarray(values)
                            row=dict(dataset=ds,model=model,seed=seed,split=split,label_fraction=rho,removed=drop,method=method,n_calib=int(np.ceil((1-drop)*len(pool))),pool_size=len(pool))
                            for j,key in enumerate(('fdp','power','discoveries','nonempty')):
                                row[key]=vals[:,j].mean();row[key+'_mcse']=vals[:,j].std(ddof=1)/np.sqrt(REPS)
                            summaries.append(row)
                pd.DataFrame(summaries).to_csv(D/'selection_dose_splits.csv',index=False)
                print('DOSE',ds,seed,'rows',nrows,'seconds',round(time.time()-started,1),flush=True)
    df=pd.DataFrame(summaries);groups=['dataset','model','label_fraction','removed','method'];metrics=['fdp','power','discoveries','nonempty','n_calib','pool_size']
    seeds=df.groupby(groups+['seed'])[metrics].mean().reset_index();seeds.to_csv(D/'selection_dose_seeds.csv',index=False)
    agg=seeds.groupby(groups)[metrics].agg(['mean','std']);agg.columns=['_'.join(c) for c in agg.columns];agg.reset_index().to_csv(D/'selection_dose_summary.csv',index=False)
    manifest=dict(protocol_sha256=hashlib.sha256((D/'DOSE_PROTOCOL.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=nrows,unknown_label_checks=checks,bh_reference_checks=100,repetitions=REPS,master=MASTER,seconds=time.time()-started,score_ties='independent lexicographic uniforms fixed before splits',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [dest,D/'selection_dose_splits.csv',D/'selection_dose_seeds.csv',D/'selection_dose_summary.csv']})
    (D/'selection_dose_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
