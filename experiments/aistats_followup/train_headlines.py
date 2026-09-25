"""Fresh fixed-hyperparameter scoring; model caches are independent of evaluations."""
from pathlib import Path
import sys, json, time, hashlib, argparse
import numpy as np
from scipy import sparse
from scipy.io import loadmat
import networkx as nx
import torch
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from detectors import score_nodes

def load_graph(name):
    if name=='amazon':
        raw=OUT/'Amazon.mat'; d=loadmat(raw)
        x=d['features'].toarray().astype(np.float32); y=d['label'].reshape(-1).astype(int)
        a=sum((d[k] for k in ('net_upu','net_usu','net_uvu'))).tocsr()
        a.data[:]=1; a=a.maximum(a.T).tocsr()
    elif name=='tolokers':
        raw=ROOT/'tmp_tolokers/tolokers/raw/tolokers.npz';d=np.load(raw)
        x=d['node_features'].astype(np.float32);y=d['node_labels'].astype(int)
        e=d['edges']; a=sparse.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(y),len(y))).tocsr()
        a=a.maximum(a.T).tocsr();a.data[:]=1
    else: raise ValueError(name)
    assert set(np.unique(y))=={0,1}
    sd=x.std(0,keepdims=True);sd[sd==0]=1; x=(x-x.mean(0,keepdims=True))/sd
    g=nx.from_scipy_sparse_array(a,create_using=nx.Graph)
    np.savez_compressed(OUT/f'{name}_graph.npz',features=x,labels=y,indptr=a.indptr,indices=a.indices,
                        degree=np.array([g.degree(i) for i in range(len(y))]))
    manifest=dict(dataset=name,nodes=len(y),undirected_edges=g.number_of_edges(),anomalies=int(y.sum()),
                  raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                  label_sha256=hashlib.sha256(y.tobytes()).hexdigest(),torch=torch.__version__,device='cpu',epochs=100,
                  hidden_units=64,layers=4,learning_rate=.01,dropout=0,threads=4,
                  notes='Fresh canonical sparse-edge load of same public graph. No claim of bitwise GPU reproduction.')
    (OUT/f'{name}_data_manifest.json').write_text(json.dumps(manifest,indent=2))
    return g,x,y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('dataset');ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=10);args=ap.parse_args()
    torch.set_num_threads(4);g,x,y=load_graph(args.dataset)
    print('DATA',args.dataset,len(y),g.number_of_edges(),int(y.sum()),flush=True)
    from sklearn.ensemble import IsolationForest
    for seed in range(args.start,args.start+args.count):
        for detector in ('dominant_pygod','isolation_forest'):
            dest=OUT/f'{args.dataset}_{detector}_scores_{seed}.npz'
            if dest.exists():continue
            t=time.time();print('START',args.dataset,detector,seed,flush=True)
            if detector=='dominant_pygod':s=score_nodes(detector,g,x,seed=seed,n_epochs=100,device='cpu')
            else:s=-IsolationForest(n_estimators=200,max_samples=256,random_state=seed,n_jobs=1).fit(x).score_samples(x)
            assert s.shape==y.shape and np.isfinite(s).all()
            np.savez_compressed(dest,scores=s,labels=y)
            log=dict(dataset=args.dataset,detector=detector,seed=seed,seconds=time.time()-t)
            (OUT/f'{args.dataset}_{detector}_timing_{seed}.json').write_text(json.dumps(log))
            print('DONE',json.dumps(log),flush=True)
if __name__=='__main__':main()
