"""Fixed-settings third-graph supervised control; no tuning on its outcomes."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json,sys,time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

def main():
    from real_data_experiment import load_pygod_graph
    graph,x,y=load_pygod_graph('weibo')
    import networkx as nx
    a=sparse.csr_matrix(nx.to_scipy_sparse_array(graph,nodelist=range(len(y)),format='csr',dtype=float))
    degree=np.asarray(a.sum(axis=1)).ravel();avg=(a@x)/np.maximum(1,degree[:,None]);xg=np.c_[x,avg,np.log1p(degree)].astype(np.float32)
    assert len(y)==8405 and y.sum()==347
    np.savez_compressed(D/'weibo_graph.npz',features=x,labels=y,indices=a.indices,indptr=a.indptr,degree=degree)
    diagnostics=[];started=time.time()
    for seed in range(10):
        rng=np.random.default_rng(np.random.SeedSequence([20260924,2,seed]))
        train=np.sort(rng.choice(len(y),int(round(.1*len(y))),False));deploy=np.setdiff1d(np.arange(len(y)),train)
        for model,xx in (('hgb_attributes',x),('hgb_graph_features',xg)):
            p=D/f'weibo_{model}_scores_{seed}.npz'
            if p.exists():
                c=np.load(p);assert np.array_equal(c['train'],train) and np.array_equal(c['labels'],y);s=c['scores']
            else:
                clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=.1,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=seed)
                with threadpool_limits(limits=1):clf.fit(xx[train],y[train]);s=clf.predict_proba(xx)[:,1]
                np.savez_compressed(p,scores=s,train=train,deploy=deploy,labels=y)
            diagnostics.append(dict(dataset='weibo',model=model,seed=seed,training_labels=len(train),training_anomalies=int(y[train].sum()),n_deploy=len(deploy),auroc=roc_auc_score(y[deploy],s[deploy]),average_precision=average_precision_score(y[deploy],s[deploy])))
        print('WEIBO',seed,'seconds',round(time.time()-started,1),flush=True)
    pd.DataFrame(diagnostics).to_csv(D/'weibo_control_diagnostics.csv',index=False)
    files=[D/'weibo_graph.npz',D/'weibo_control_diagnostics.csv',*sorted(D.glob('weibo_hgb_*_scores_*.npz'))]
    (D/'weibo_control_manifest.json').write_text(json.dumps(dict(nodes=len(y),anomalies=int(y.sum()),undirected_edges=graph.number_of_edges(),seconds=time.time()-started,protocol_sha256=hashlib.sha256((D/'WEIBO_CONTROL_PROTOCOL.md').read_bytes()).hexdigest(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))

if __name__=='__main__':main()
