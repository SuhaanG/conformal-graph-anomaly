"""
detectors.py

A single interface for scoring nodes with several anomaly detectors, so every
experiment in this repo can be re-run under more than one scorer.

WHY THIS EXISTS. Every result in the paper so far was produced by one detector:
our DOMINANT implementation in detector.py, whose encoder collapses to zero
(ReLU on the final embedding; see DETECTOR_DIAGNOSTIC.md). Its graph pathway is
inert, so its ~0.91 AUROC comes entirely from raw feature magnitude. PyGOD's
DOMINANT does NOT have that bug -- torch_geometric's BasicGNN gates the
activation behind `i < num_layers - 1`, leaving the final embedding linear
(verified: frac_zero=0.0000, min=-0.219; negatives are impossible after a ReLU).

So there are two goals here, and they are separable:
  1. Re-run the study under a CORRECT detector, making the results trustworthy.
  2. Run it under SEVERAL detectors, since the discovery-threshold proposition
     is detector-agnostic in principle and should be shown to be so in practice.

THE CONSTRAINT THAT SHAPES THIS FILE. PyGOD's high-level `Detector.fit()` routes
training through `NeighborLoader` even at full batch, which requires pyg-lib or
torch-sparse. On the dev box torch-sparse will not import (it needs
torch-scatter, and the wheels must match torch 2.4.0+cu121 exactly). Rather than
depend on that, every PyGOD detector here is driven through its `pygod.nn.*Base`
module with an explicit full-batch training loop. No sampler, no extra
dependency. If torch-scatter/torch-sparse are later installed, nothing here
needs to change.

CONTRACT. `score_nodes(...)` returns a 1-D float array of per-node anomaly
scores, higher = more anomalous -- the same convention `train_dominant` uses and
the same one `conformal_p_values` assumes (it tests the upper tail). Existing
scripts can therefore swap detectors without touching the conformal machinery.

`detector.py` and `conformal_fdr.py` are NOT modified. `dominant_ours` delegates
to the frozen `train_dominant`, so passing it reproduces every existing result
byte-for-byte.

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

# name -> (pygod.nn class name, constructor kwargs). Only reconstruction-style
# detectors are listed: they share the `model(x, edge_index) -> reconstructions`
# plus `model.loss_func(...) -> per-node loss` shape that _train_pygod relies on.
# Adversarial (GAAN) and contrastive (CoLA) detectors need their own training
# loops and are deliberately excluded rather than half-supported.
PYGOD_SPECS = {
    "dominant_pygod": ("DOMINANTBase", dict(hid_dim=64, num_layers=4, dropout=0.0)),
    "anomalydae":     ("AnomalyDAEBase", dict(emb_dim=64, hid_dim=64, dropout=0.0)),
    "gae":            ("GAEBase", dict(hid_dim=64, num_layers=4, dropout=0.0)),
    "guide":          ("GUIDEBase", dict(dim_a=64, dim_s=4, num_layers=4, dropout=0.0)),
    "done":           ("DONEBase", dict(hid_dim=64, num_layers=4, dropout=0.0)),
}


def available_detectors():
    """Names accepted by score_nodes. PyGOD entries are advertised whether or
    not they turn out to construct on this install -- score_nodes raises with a
    readable message if one is unavailable, which is more useful than silently
    hiding it."""
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


def _call_loss(model, data, out):
    """PyGOD loss_func signatures vary by detector. The reconstruction family
    all take the originals alongside the reconstructions, so dispatch on how
    many tensors forward() returned rather than hardcoding per detector."""
    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            x_, s_ = out
            # `s` is the dense adjacency, set by the model's own process_graph
            return model.loss_func(data.x, x_, data.s, s_)
        if len(out) == 3:
            return model.loss_func(data.x, out[0], data.s, out[1], out[2])
        if len(out) >= 4:
            return model.loss_func(data.x, out[0], data.s, out[1], *out[2:])
    return model.loss_func(data.x, out)


def _train_pygod(name, graph, features, labels, seed, n_epochs, device, lr=0.01):
    import importlib
    cls_name, kwargs = PYGOD_SPECS[name]
    try:
        nn_mod = importlib.import_module("pygod.nn")
        Base = getattr(nn_mod, cls_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"detector {name!r} unavailable: could not load pygod.nn.{cls_name} "
            f"({type(e).__name__}: {e}). Check the installed pygod version."
        ) from e

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = _to_pyg(graph, features, labels)
    # Most reconstruction detectors need the dense adjacency as `data.s`; the
    # class knows how to build its own, so ask it rather than guessing.
    if hasattr(Base, "process_graph"):
        Base.process_graph(data)
    data = data.to(device)

    try:
        model = Base(in_dim=features.shape[1], **kwargs).to(device)
    except TypeError as e:
        import inspect
        raise RuntimeError(
            f"detector {name!r}: constructor kwargs {sorted(kwargs)} rejected by "
            f"{cls_name}. Its signature is {inspect.signature(Base.__init__)}. "
            f"Fix PYGOD_SPECS in src/detectors.py. ({e})"
        ) from e

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = _call_loss(model, data, model(data.x, data.edge_index))
        (loss.mean() if loss.dim() > 0 else loss).backward()
        opt.step()

    # The per-node loss IS the anomaly score for this family -- reconstruction
    # error per node, which is what PyGOD's own detectors report as
    # decision_score_.
    model.eval()
    with torch.no_grad():
        per_node = _call_loss(model, data, model(data.x, data.edge_index))
    scores = per_node.detach().cpu().numpy()
    if scores.ndim > 1:
        scores = scores.reshape(len(scores), -1).mean(axis=1)
    return np.asarray(scores, dtype=np.float64)


def score_nodes(detector, graph, features, labels=None, *, seed=0, n_epochs=100,
                device="cpu", use_sparse_prop=False, score_alpha=0.5):
    """Per-node anomaly scores, higher = more anomalous.

    detector: see available_detectors(). "dominant_ours" delegates to the frozen
        train_dominant and reproduces existing results byte-for-byte; everything
        else routes through PyGOD's nn modules with full-batch training.
    use_sparse_prop: only meaningful for "dominant_ours" (large-graph path,
        required for Yelp). Ignored otherwise, since PyGOD detectors use PyG's
        sparse message passing already.
    """
    if detector == "dominant_ours":
        scores, _ = train_dominant(graph, features, n_epochs=n_epochs, seed=seed,
                                   verbose=False, device=device, alpha=score_alpha,
                                   use_sparse_prop=use_sparse_prop)
        return np.asarray(scores, dtype=np.float64)

    if detector in PYGOD_SPECS:
        return _train_pygod(detector, graph, features, labels, seed, n_epochs, device)

    raise ValueError(
        f"unknown detector {detector!r}. Available: {available_detectors()}"
    )


if __name__ == "__main__":
    # Smoke test: which detectors actually construct and train on this install?
    # Run this first on any new machine -- PYGOD_SPECS is version-sensitive and
    # a wrong kwarg surfaces here rather than three hours into an experiment.
    import argparse
    sys.path.insert(0, os.path.dirname(__file__))
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
    print(f"{'detector':18} {'status':10} {'AUROC':>8}  note")
    for name in available_detectors():
        try:
            s = score_nodes(name, graph, features, labels, seed=0,
                            n_epochs=args.epochs, device=device)
            a = auroc(s, labels)
            note = "inverted -- scores anti-correlate with labels" if a < 0.5 else ""
            print(f"{name:18} {'OK':10} {a:8.4f}  {note}")
        except Exception as e:
            print(f"{name:18} {'FAIL':10} {'':>8}  {type(e).__name__}: {str(e)[:90]}")
    print("\nAUROC below 0.5 means the score is inverted for that detector on this "
          "generator (dense anomaly clusters reconstruct easily) -- a real finding, "
          "not necessarily a bug. Exclude such detectors or flip the sign "
          "deliberately and say so.")
