"""Portable GPU replication. Run only in the isolated supplied bundle."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time, zipfile
import numpy as np
import torch
from train_headlines import load_graph, score_nodes, OUT, ROOT
from evaluate_headlines import main as evaluate, checks

DETECTORS = ('dominant_pygod', 'gae', 'isolation_forest')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def bundle_results():
    dest = ROOT / 'h200_results.zip'
    tmp = dest.with_suffix('.tmp')
    files = []
    for pattern in ('*_scores_*.npz', '*_graph.npz', '*_manifest.json', '*_timing_*.json',
                    'evaluation_*.csv', 'gpu_*.json', 'gpu_*.txt', '*.log'):
        files.extend(OUT.glob(pattern))
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(ROOT))
    tmp.replace(dest)
    print('RESULT ARCHIVE', dest, flush=True)

def smoke(device):
    import networkx as nx
    checks()
    g = nx.path_graph(32)
    x = np.random.default_rng(17).normal(size=(32, 6)).astype(np.float32)
    for name in DETECTORS[:2]:
        s = score_nodes(name, g, x, seed=0, n_epochs=2, device=device)
        assert s.shape == (32,) and np.isfinite(s).all(), name
        print('SMOKE PASSED', name, device, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-only', action='store_true')
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = ap.parse_args()
    if not args.smoke_only and not (ROOT / 'H200_BUNDLE.txt').exists():
        raise RuntimeError('Extract the isolated H200 bundle first; do not mix CPU and GPU caches.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check nvidia-smi and install a CUDA PyTorch wheel.')
    if not args.smoke_only and args.device != 'cuda':
        raise RuntimeError('Full replication requires CUDA; CPU is allowed only for the smoke test.')
    if not args.smoke_only:
        for name, expected in json.loads((ROOT/'input_checksums.json').read_text()).items():
            if sha(ROOT/name) != expected:
                raise RuntimeError(f'Bundle file checksum mismatch: {name}')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    smoke(args.device)
    if args.smoke_only:
        return
    versions = {p: importlib.metadata.version(p) for p in
                ['torch','torch-geometric','pygod','numpy','scipy','scikit-learn','networkx']}
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in
                     [Path(__file__), OUT/'train_headlines.py', OUT/'evaluate_headlines.py',
                      ROOT/'src/detectors.py', ROOT/'src/detector.py',
                      ROOT/'paper/aistats_revision/validate_revision.py']}
    config = dict(versions=versions, sources=source_hashes, python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                  seeds=list(range(10)), detectors=DETECTORS, epochs=100, tf32=False,
                  notes='Independent GPU replication. Float32 scatter reductions may be nondeterministic; no bitwise CPU/GPU equivalence claimed.')
    manifest = OUT/'gpu_environment_manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if json.dumps(previous,sort_keys=True) != json.dumps(config,sort_keys=True):
            raise RuntimeError('Environment/source differs from existing run. Use a fresh extraction instead of mixing caches.')
    atomic_json(manifest, config)
    (OUT/'gpu_pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    from sklearn.ensemble import IsolationForest
    try:
        for dataset in ('amazon','tolokers'):
            g,x,y = load_graph(dataset)
            p = OUT/f'{dataset}_data_manifest.json'
            m = json.loads(p.read_text()); m['device'] = 'cuda'
            m['notes'] = 'Canonical loader, independent GPU replication; see gpu_environment_manifest.json.'
            atomic_json(p,m)
            for seed in range(10):
                for name in DETECTORS:
                    dest = OUT/f'{dataset}_{name}_scores_{seed}.npz'
                    if not dest.exists():
                        print('START',dataset,name,seed,flush=True); t=time.time()
                        if name == 'isolation_forest':
                            s = -IsolationForest(n_estimators=200,max_samples=256,random_state=seed,n_jobs=1).fit(x).score_samples(x)
                        else:
                            s = score_nodes(name,g,x,seed=seed,n_epochs=100,device='cuda')
                        assert s.shape == y.shape and np.isfinite(s).all()
                        tmp = dest.with_suffix('.tmp')
                        with tmp.open('wb') as f:
                            np.savez_compressed(f,scores=s,labels=y)
                        atomic_json(OUT/f'{dataset}_{name}_timing_{seed}.json',
                                    dict(dataset=dataset,detector=name,seed=seed,seconds=time.time()-t,
                                         device='cpu' if name=='isolation_forest' else 'cuda',score_sha256=sha(tmp)))
                        tmp.replace(dest)
                        torch.cuda.empty_cache()
                    else:
                        c=np.load(dest)
                        assert np.array_equal(c['labels'],y) and np.isfinite(c['scores']).all()
                    # A separate atomic marker prevents trusting a partially written CSV.
                    marker=OUT/f'gpu_job_{dataset}_{name}_{seed}.json'
                    artifacts=[dest, OUT/f'evaluation_{dataset}_{name}_{seed}_summary.csv',
                               OUT/f'evaluation_{dataset}_{name}_{seed}_trials.csv']
                    complete=marker.exists() and all(p.exists() for p in artifacts)
                    if complete:
                        saved=json.loads(marker.read_text())
                        complete=all(saved.get(p.name)==sha(p) for p in artifacts)
                    if not complete:
                        evaluate(dataset,name,seed,200)
                        atomic_json(marker,{p.name:sha(p) for p in artifacts})
                    print('COMPLETE',dataset,name,seed,flush=True)
                bundle_results()
            evaluate(dataset,'degree',0,200)
        expected = [OUT/f'evaluation_{d}_{m}_{s}_summary.csv' for d in ('amazon','tolokers')
                    for m in DETECTORS for s in range(10)]
        assert all(p.exists() for p in expected)
        atomic_json(OUT/'gpu_completion.json',dict(status='complete',model_seed_evaluations=len(expected),
                    checksums={p.name:sha(p) for p in expected}))
    except BaseException as e:
        atomic_json(OUT/'gpu_last_failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
    finally:
        bundle_results()

if __name__ == '__main__':
    main()
