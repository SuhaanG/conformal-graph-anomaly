"""Acquire only Weibo/T-Finance members of the official GADBench ZIP.

No models, training, or evaluation. Run with --download for network acquisition.
HTTP ranges avoid downloading unrelated datasets. ZIP CRC and SHA256 are recorded.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import struct
import zlib
import zipfile
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/gadbench'
URL = 'https://drive.usercontent.google.com/download?id=1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1&export=download&confirm=t'
VIEW = 'https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing'


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    r = session.get(URL, headers={'Range': 'bytes=-65536'}, timeout=30)
    r.raise_for_status()
    assert r.status_code == 206 and len(r.content) == 65536
    tail_start = int(r.headers['Content-Range'].split()[1].split('-')[0])
    total = int(r.headers['Content-Range'].split('/')[1])
    (RAW / 'archive_tail.bin').write_bytes(r.content)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    manifest = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    repository='https://github.com/squareRoot3/GADBench',
                    archive_url=VIEW, download_url=URL, archive_bytes=total,
                    last_modified=r.headers.get('Last-Modified'),
                    acquisition='HTTP range extraction of two members only; whole archive not downloaded or hashed',
                    archive_tail_sha256=sha256(RAW / 'archive_tail.bin'), members={})
    for name in ('weibo', 'tfinance'):
        member = z.getinfo('datasets/' + name)
        offset = member.header_offset + tail_start
        assert member.compress_type == zipfile.ZIP_DEFLATED
        h = session.get(URL, headers={'Range': f'bytes={offset}-{offset+1023}'}, timeout=30)
        h.raise_for_status()
        assert h.status_code == 206 and h.content[:4] == b'PK\x03\x04'
        header = struct.unpack('<4s5H3I2H', h.content[:30])
        assert h.content[30:30+header[-2]].decode() == member.filename
        start = offset + 30 + header[-2] + header[-1]
        end = start + member.compress_size - 1
        output = RAW / name
        print(f'{name}: fetching {member.compress_size:,} compressed bytes -> {member.file_size:,} bytes', flush=True)
        decomp = zlib.decompressobj(-15)
        crc = 0
        compressed_hash = hashlib.sha256()
        received = 0
        with session.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=(20,60), stream=True) as resp:
            resp.raise_for_status()
            assert resp.status_code == 206
            assert resp.headers['Content-Range'] == f'bytes {start}-{end}/{total}'
            with output.with_suffix('.part').open('wb') as f:
                for chunk in resp.iter_content(1024*1024):
                    received += len(chunk)
                    compressed_hash.update(chunk)
                    raw = decomp.decompress(chunk)
                    crc = zlib.crc32(raw, crc)
                    f.write(raw)
                raw = decomp.flush()
                crc = zlib.crc32(raw, crc)
                f.write(raw)
        assert decomp.eof and not decomp.unused_data
        assert received == member.compress_size
        assert crc == member.CRC
        assert output.with_suffix('.part').stat().st_size == member.file_size
        output.with_suffix('.part').replace(output)
        manifest['members'][name] = dict(zip_member=member.filename, bytes=member.file_size,
            compressed_bytes=member.compress_size, compressed_range=[start,end],
            crc32=f'{crc:08x}', sha256=sha256(output), compressed_sha256=compressed_hash.hexdigest())
        (RAW / 'acquisition.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(name, manifest['members'][name]['sha256'], flush=True)


def convert():
    # The isolated official Windows wheel contains a ctypes DLL, no Python .pyd.
    # Only DGL graph I/O is used; optional version-specific torch adapters are not used.
    os.environ.setdefault('DGLBACKEND', 'pytorch')
    os.environ.setdefault('DGL_CONF_PATH', str(RAW / 'dgl_config'))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, str(RAW / 'dgl_runtime'))
    import dgl
    import numpy as np
    import scipy
    from scipy import sparse
    import torch
    torch.set_num_threads(1)
    dest = Path(__file__).resolve().parent
    report = dict(acquisition=json.loads((RAW / 'acquisition.json').read_text()),
        upstream_revision=json.loads((RAW / 'upstream_revision.json').read_text()),
        protocol_sha256=sha256(dest / 'TFINANCE_PROTOCOL.md'),
        script_sha256=sha256(__file__), created_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=sys.version, dgl=dgl.__version__, numpy=np.__version__,
                      scipy=scipy.__version__, torch=torch.__version__),
        runtime_note='Official DGL 1.1.2 cp311 Windows wheel extracted locally; ctypes graph-I/O tested under existing Python 3.14.2. No environment packages installed or replaced.',
        dgl_wheel_sha256=sha256(RAW / 'dgl-1.1.2-cp311-cp311-win_amd64.whl'),
        dgl_wheel_url='https://files.pythonhosted.org/packages/5a/ff/2ff1e38194aa66ca2d7393415ad11eccbf8d9ea5e74dfaaa2c145d34a240/dgl-1.1.2-cp311-cp311-win_amd64.whl',
        source_code={p.name:sha256(p) for p in RAW.glob('gadbench_*') if p.is_file()},
        label_semantics='GADBench utils.Dataset loads the graph unchanged; models/detector.py takes ndata[label] directly and evaluates class-1 probabilities as anomalies. No binary inversion, thresholding, argmax, or recoding applied.',
        preprocessing='Keep distributed features and node order unchanged. Sparse boolean adjacency; union with transpose, binarize, remove loops. Ignore distributed train/val/test masks. No training or evaluation.',
        original_BWGNN_fallback_used=False, datasets={})

    def csr(src, dst, n):
        a = sparse.coo_matrix((np.ones(len(src), dtype=bool), (src,dst)), shape=(n,n)).tocsr()
        a.sum_duplicates()
        return a

    def normalize(a):
        a = a.maximum(a.T).tocsr()
        a.setdiag(False)
        a.eliminate_zeros()
        a.sort_indices()
        return a

    def counts(y):
        u,c = np.unique(y, return_counts=True)
        return {str(int(k)):int(v) for k,v in zip(u,c)}

    for name, filename in [('weibo','weibo_gadbench_graph.npz'), ('tfinance','tfinance_graph.npz')]:
        print('Loading', name, flush=True)
        assert sha256(RAW / name) == report['acquisition']['members'][name]['sha256']
        graphs, graph_labels = dgl.load_graphs(str(RAW / name))
        assert len(graphs) == 1
        g = graphs[0]
        x = g.ndata['feature'].numpy().copy()
        y = g.ndata['label'].numpy().copy()
        n = g.num_nodes()
        assert y.shape == (n,) and set(np.unique(y)) == {0,1}
        assert x.ndim == 2 and x.shape[0] == n and np.isfinite(x).all()
        src_t,dst_t = g.edges(order='eid')
        src,dst = src_t.numpy(),dst_t.numpy()
        raw_loops = int(np.count_nonzero(src == dst))
        a = csr(src,dst,n)
        unique_directed = int(a.nnz)
        raw_edges = int(g.num_edges())
        node_schemes = {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in g.ndata.items()}
        edge_schemes = {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in g.edata.items()}
        del graphs,g,src_t,dst_t,src,dst
        a = normalize(a)
        degree = np.diff(a.indptr).astype(np.float64)
        assert (a != a.T).nnz == 0 and not a.diagonal().any()
        assert int(degree.sum()) == a.nnz and a.nnz % 2 == 0
        assert a.has_canonical_format
        output = dest / filename
        np.savez_compressed(output, features=x, labels=y,
                            indices=a.indices.astype(np.int64),
                            indptr=a.indptr.astype(np.int64), degree=degree)
        with np.load(output, allow_pickle=False) as saved:
            assert set(saved.files) == {'features','labels','indices','indptr','degree'}
            assert np.array_equal(saved['features'],x) and np.array_equal(saved['labels'],y)
            assert np.array_equal(saved['indices'],a.indices) and np.array_equal(saved['indptr'],a.indptr)
            assert np.array_equal(saved['degree'],degree)
        item = dict(nodes=n, feature_shape=list(x.shape), feature_dtype=str(x.dtype),
            label_dtype=str(y.dtype), label_counts=counts(y), labels_unchanged=True,
            features_unchanged=True, node_order_unchanged=True, features_all_finite=True,
            raw_directed_edge_entries=raw_edges, raw_self_loop_entries=raw_loops,
            raw_unique_directed_edges=unique_directed, raw_duplicate_entries=raw_edges-unique_directed,
            processed_directed_csr_entries=int(a.nnz), processed_undirected_edges=int(a.nnz//2),
            processed_self_loops=0, processed_is_symmetric=True,
            isolated_nodes=int(np.count_nonzero(degree==0)),
            node_fields=node_schemes, edge_fields=edge_schemes,
            output=str(output.relative_to(ROOT)), output_bytes=output.stat().st_size,
            output_sha256=sha256(output))
        if name == 'weibo':
            pygod_path = Path.home() / '.pygod/data/weibo.pt'
            prior = json.loads((dest / 'WEIBO_DATA_PROVENANCE.json').read_text())
            assert sha256(pygod_path) == prior['raw_sha256']
            pg = torch.load(pygod_path, map_location='cpu', weights_only=False)
            px,py = pg.x.numpy(),pg.y.numpy()
            pa = normalize(csr(pg.edge_index[0].numpy(),pg.edge_index[1].numpy(),len(py)))
            feat_equal = np.array_equal(x,px)
            adjacency_equal = a.shape == pa.shape and (a != pa).nnz == 0
            # Unique feature rows plus exact row equality establish unambiguous alignment.
            unique_features, row_inverse, row_counts = np.unique(px,axis=0,return_inverse=True,return_counts=True)
            unique_rows = len(unique_features)
            duplicate_nodes = np.flatnonzero(row_counts[row_inverse]>1)
            aligned = feat_equal and adjacency_equal and unique_rows == n
            cache = np.load(dest / 'weibo_graph.npz', allow_pickle=False)
            ca = sparse.csr_matrix((np.ones(len(cache['indices']),bool),cache['indices'],cache['indptr']),shape=(n,n))
            std = px.std(axis=0,keepdims=True)
            std[std==0] = 1
            standardized = (px-px.mean(axis=0,keepdims=True))/std
            comparison = dict(pygod_url=prior['url'],
                pygod_raw_sha256=sha256(pygod_path), pygod_raw_label_counts=counts(py),
                pygod_cached_sha256=sha256(dest/'weibo_graph.npz'),
                pygod_cache_labels_equal_raw=bool(np.array_equal(cache['labels'],py)),
                raw_features_exactly_equal=feat_equal, unique_feature_rows=unique_rows,
                duplicate_feature_node_indices=duplicate_nodes.tolist(),
                duplicate_feature_neighbor_sets={str(int(i)):a[i].indices.tolist() for i in duplicate_nodes},
                normalized_adjacency_equal=adjacency_equal,
                cache_adjacency_equal=bool((a != ca).nnz==0),
                cache_features_equal_standardized_raw=bool(np.array_equal(cache['features'],standardized)),
                gadbench_features_equal_cache=bool(np.array_equal(x,cache['features'])),
                same_node_order_established=aligned,
                alignment_evidence='Exact feature-row identity with unique rows AND exact normalized adjacency.' if aligned else 'Node identity/order not established; do not compare labels pointwise.',
                feature_processing_difference='GADBench output preserves distributed raw float32 features; existing PyGOD cache uses per-column z-scores.',
                README_weibo_anomalies=868,
                caveat='Separate source sensitivity, not independent graph replication. Do not overwrite or pool PyGOD results.')
            if aligned:
                comparison['label_disagreements'] = int(np.count_nonzero(y != py))
                comparison['label_transitions_pygod_to_gadbench'] = {
                    f'{i}->{j}':int(np.count_nonzero((py==i)&(y==j))) for i in (0,1) for j in (0,1)}
            item['pygod_comparison'] = comparison
            cache.close()
        report['datasets'][name] = item
        (dest/'GADBENCH_DATA_PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name, 'verified', counts(y), 'undirected edges', a.nnz//2, flush=True)
        del a,x,y


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true')
    p.add_argument('--convert', action='store_true')
    args = p.parse_args()
    if args.download:
        download()
    if args.convert:
        convert()
