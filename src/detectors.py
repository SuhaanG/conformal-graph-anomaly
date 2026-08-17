"""
detectors.py

A single interface for scoring nodes with several anomaly detectors, so every
experiment in this repo can be re-run under more than one scorer.

WHY THIS EXISTS. Every result in the paper so far was produced by one detector:
our DOMINANT implementation in detector.py, whose encoder collapses to zero
(ReLU on the final embedding; see DETECTOR_DIAGNOSTIC.md). Its graph pathway is
inert, so its ~0.92 AUROC comes entirely from raw feature magnitude. PyGOD's
DOMINANT does NOT have that bug -- torch_geometric's BasicGNN gates the
activation behind `i < num_layers - 1`, leaving the final embedding linear
(verified: frac_zero=0.0000, min=-0.219; negatives are impossible after a ReLU).
Measured side by side on the same graph: ours 0.9230, PyGOD's 0.9707.

Two separable goals:
  1. Re-run the study under a CORRECT detector, making results trustworthy.
  2. Run it under SEVERAL detectors, since the discovery-threshold proposition
     is detector-agnostic in principle and should be shown to be so in practice.

THE CONSTRAINT THAT SHAPES THIS FILE. PyGOD's high-level `Detector.fit()` routes
training through `NeighborLoader` even at full batch, which requires pyg-lib or
torch-sparse. On the dev box torch-sparse will not import (needs torch-scatter,
and wheels must match torch 2.4.0+cu121 exactly). So every PyGOD detector here is
driven through its `pygod.nn.*Base` module with an explicit full-batch loop. No
sampler, no extra dependency, and nothing changes if that is later fixed.

CONTRACT. `score_nodes(...)` returns a 1-D float array of per-node anomaly
scores, higher = more anomalous -- the same convention `train_dominant` uses and
what `conformal_p_values` assumes (it tests the upper tail). Scripts can swap
detectors without touching the conformal machinery.

`detector.py` and `conformal_fdr.py` are NOT modified, and `dominant_ours`
delegates to the frozen `train_dominant`, so the default reproduces every
existing result byte-for-byte.

Signatures below were read off pygod 1.1.0 with scripts/pygod_introspect.py
rather than guessed -- an earlier guessed table failed on three of five. Re-run
that script if the PyGOD version changes.

Usage:
    from detectors import score_nodes, available_detectors
    scores = score_nodes("dominant_pygod", graph, features, seed=0, device="cuda")
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from detector import train_dominant

# Scoring modes, i.e. how a per-node anomaly score is obtained:
#   "recon_xs"  forward -> (x_, s_); loss_func(x, x_, s, s_) returns PER-NODE
#   "recon_x"   forward -> x_ only; loss_func is torch's mse_loss and returns a
#               SCALAR, so per-node error is computed here instead
#   "emb_loss"  forward -> emb; loss_func(emb) returns (loss, per-node score)
PYGOD_SPECS = {
    "dominant_pygod": dict(
        cls="DOMINANTBase", mode="recon_xs",
        kwargs=lambda in_dim, n: dict(in_dim=in_dim, hid_dim=64, num_layers=4, dropout=0.0),
    ),
    "anomalydae": dict(
        cls="AnomalyDAEBase", mode="recon_xs",
        # needs num_nodes as well as in_dim -- its structure decoder is sized by it
        kwargs=lambda in_dim, n: dict(in_dim=in_dim, num_nodes=n, emb_dim=64,
                                      hid_dim=64, dropout=0.0),
    ),
    "gae": dict(
        cls="GAEBase", mode="recon_x",
        # recon_s defaults False, so this reconstructs attributes
        kwargs=lambda in_dim, n: dict(in_dim=in_dim, hid_dim=64, num_layers=4, dropout=0.0),
    ),
    "ocgnn": dict(
        cls="OCGNNBase", mode="emb_loss",
        # one-class objective rather than reconstruction -- a genuinely different
        # detector family, which is the point of including it
        kwargs=lambda in_dim, n: dict(in_dim=in_dim, hid_dim=64, num_layers=2, dropout=0.0),
    ),
}

# Deliberately excluded, with reasons, so nobody re-derives this:
#   CoLABase   contrastive; loss_func is BCE over sampled pairs, needs its own
#              sampling + training loop
#   GAANBase   adversarial (generator/discriminator), needs its own loop
#   DMGDBase   overwrites x with the adjacency matrix; forward failed on our
#              feature shapes
#   GADNRBase  requires a precomputed neighbor_num_list
#   DONEBase / AdONEBase   need x_dim AND s_dim plus an 8-argument loss_func
#              (x, x_, s, s_, h_a, h_s, dna, dns); reachable but a bigger job
#   GUIDEBase  needs dim_a and dim_s, where dim_s is a graphlet/motif-degree
#              dimension requiring its own preprocessing
EXCLUDED = ["cola", "gaan", "dmgd", "gadnr", "done", "adone", "guide"]


def available_detectors():
    return ["dominant_ours"] + sorted(PYGOD_SPECS)


def _to_pyg(graph, features, labels=None):
    from torch_geometric.data import Data
    edges = np.array(list(graph.edges()), dtype=np.int64)
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        # networkx stores each undirected edge once; PyG expects both directions
        both = np.concatenate([edges, edges[:, ::-1]], axis=0)
        edge_index = torch.tensor(both.T, dtype=torch.long)
    data = Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edge_index)
    if labels is not None:
        data.y = torch.tensor(np.asarray(labels), dtype=torch.long)
    return data


def _ensure_dense_adj(data, device):
    """recon_xs detectors compare against the dense adjacency as `data.s`. Most
    set it in process_graph, but not all do, so build it if missing."""
    if getattr(data, "s", None) is None:
        from torch_geometric.utils import to_dense_adj
        data.s = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
    return data


def _per_node(mode, model, data, out):
    """Returns (training_loss, per_node_score). They differ only for recon_x,
    where PyGOD's loss_func is a scalar mse and the per-node error has to be
    computed explicitly."""
    if mode == "recon_xs":
        x_, s_ = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)
        val = model.loss_func(data.x, x_, data.s, s_)
        return val.mean(), val

    if mode == "recon_x":
        x_ = out[0] if isinstance(out, (tuple, list)) else out
        per = ((data.x - x_) ** 2).sum(dim=1)
        return per.mean(), per

    if mode == "emb_loss":
        emb = out[0] if isinstance(out, (tuple, list)) else out
        val = model.loss_func(emb)
        if isinstance(val, (tuple, list)):
            # OCGNN returns (loss, per-node distance-to-centre)
            loss, per = val[0], val[1]
            return (loss.mean() if loss.dim() > 0 else loss), per
        return (val.mean() if val.dim() > 0 else val), val

    raise ValueError(f"unknown scoring mode {mode!r}")


def _train_pygod(name, graph, features, labels, seed, n_epochs, device, lr=0.01):
    import importlib
    spec = PYGOD_SPECS[name]
    try:
        Base = getattr(importlib.import_module("pygod.nn"), spec["cls"])
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"detector {name!r} unavailable: could not load pygod.nn.{spec['cls']} "
            f"({type(e).__name__}: {e}). Run scripts/pygod_introspect.py to see what "
            f"this pygod version exposes."
        ) from e

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = _to_pyg(graph, features, labels)
    if hasattr(Base, "process_graph"):
        try:
            Base.process_graph(data)
        except Exception:
            pass  # some set nothing; _ensure_dense_adj covers recon_xs below
    data = data.to(device)
    if spec["mode"] == "recon_xs":
        _ensure_dense_adj(data, device)

    kwargs = spec["kwargs"](features.shape[1], graph.number_of_nodes())
    try:
        model = Base(**kwargs).to(device)
    except TypeError as e:
        import inspect
        raise RuntimeError(
            f"detector {name!r}: kwargs {sorted(kwargs)} rejected by {spec['cls']}. "
            f"Signature is {inspect.signature(Base.__init__)}. Fix PYGOD_SPECS. ({e})"
        ) from e

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss, _ = _per_node(spec["mode"], model, data, model(data.x, data.edge_index))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, per = _per_node(spec["mode"], model, data, model(data.x, data.edge_index))
    scores = per.detach().cpu().numpy()
    if scores.ndim > 1:
        scores = scores.reshape(scores.shape[0], -1).mean(axis=1)
    return np.asarray(scores, dtype=np.float64).ravel()


def score_nodes(detector, graph, features, labels=None, *, seed=0, n_epochs=100,
                device="cpu", use_sparse_prop=False, score_alpha=0.5):
    """Per-node anomaly scores, higher = more anomalous.

    detector: see available_detectors(). "dominant_ours" delegates to the frozen
        train_dominant and reproduces existing results byte-for-byte.
    use_sparse_prop: only meaningful for "dominant_ours" (large-graph path,
        required for Yelp). PyGOD detectors already use sparse message passing.
    """
    if detector == "dominant_ours":
        scores, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed,
                                   verbose=False, device=device, alpha=score_alpha,
                                   use_sparse_prop=use_sparse_prop)
        return np.asarray(scores, dtype=np.float64)

    if detector in PYGOD_SPECS:
        return _train_pygod(detector, graph, features, labels, seed, n_epochs, device)

    raise ValueError(f"unknown detector {detector!r}. Available: {available_detectors()}")


if __name__ == "__main__":
    # Smoke test: which detectors construct, train, and produce a sane score on
    # THIS install? Run first on any new machine -- PYGOD_SPECS is
    # version-sensitive and a wrong kwarg should surface in seconds, not three
    # hours into an experiment.
    import argparse
    from graph_gen import GraphGenConfig, ContaminatedGraphGenerator

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_nodes", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = GraphGenConfig(n_nodes=args.n_nodes, p_aa=0.3, p_an=0.002, p_nn=0.005,
                         feature_shift=1.0, n_anomaly_clusters=3, random_state=0)
    graph, features, labels = ContaminatedGraphGenerator(cfg).generate()

    def auroc(s, l):
        s = np.asarray(s, dtype=np.float64); l = np.asarray(l)
        pos, neg = s[l == 1], s[l == 0]
        if not len(pos) or not len(neg):
            return float("nan")
        order = np.argsort(s, kind="mergesort")
        r = np.empty(len(s), dtype=float); r[order] = np.arange(1, len(s) + 1)
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        ts = np.zeros(len(cnt)); np.add.at(ts, inv, r); r = (ts / cnt)[inv]
        return float((r[l == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    print(f"device={device}  n_nodes={args.n_nodes}  epochs={args.epochs}\n")
    print(f"{'detector':18} {'status':6} {'AUROC':>8}  note")
    for name in available_detectors():
        try:
            s = score_nodes(name, graph, features, labels, seed=0,
                            n_epochs=args.epochs, device=device)
            a = auroc(s, labels)
            note = ""
            if not np.isfinite(a):
                note = "degenerate scores"
            elif a < 0.5:
                note = "INVERTED -- anti-correlates with labels"
            elif np.std(s) == 0:
                note = "constant scores"
            print(f"{name:18} {'OK':6} {a:8.4f}  {note}")
        except Exception as e:
            print(f"{name:18} {'FAIL':6} {'':>8}  {type(e).__name__}: {str(e)[:80]}")
    print(f"\nexcluded (need their own training loops): {', '.join(EXCLUDED)}")
    print("AUROC below 0.5 means the score is inverted on this generator -- dense "
          "anomaly clusters reconstruct easily, so structure-based scorers can\n"
          "anti-correlate. That is a real finding, not necessarily a bug; exclude "
          "such detectors or flip the sign deliberately and say so in the paper.")
