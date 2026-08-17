"""
pygod_architecture_check.py

Sampler-free fallback for pygod_dominant_collapse_check.py, which fails on
PyGOD >= 1.1 unless pyg-lib or torch-sparse is installed (its detector wrapper
routes training through NeighborLoader even at full batch).

This bypasses the PyGOD detector wrapper entirely: it instantiates PyGOD's
DOMINANTBase module directly, trains it full-batch with a plain loop, and
inspects the embedding. No NeighborLoader, no extra dependencies.

THE QUESTION. Our DOMINANT encoder's final layer is dead -- DETECTOR_DIAGNOSTIC.md
has the full trace. Our encode() is:

    for layer in self.encoder_layers:
        H = F.relu(layer(A_norm, H))     # ReLU on EVERY layer, including the last

ReLU on the final layer forces Z >= 0, so sigmoid(z_i . z_j) >= 0.5 for every
pair, while ~99.7% of node pairs are non-edges pulling it toward 0. The nearest
reachable point is exactly 0.5, attained at Z = 0, so the structure loss drives
the embedding to zero.

The specific thing to check in PyGOD is therefore whether ITS encoder applies an
activation to the final embedding layer. PyGOD's DOMINANTBase delegates to a
torch_geometric backbone (GCN by default), and PyG's BasicGNN.forward is written
to skip norm/activation/dropout after the final conv. If that holds here, PyGOD
is structurally immune to this failure and the collapse is ours alone.

Do not take the previous paragraph on faith -- this script prints the actual
source of both the PyGOD module and the backbone's forward so it can be read
directly, and then measures the trained embedding empirically. Source inspection
and measurement have to agree before either is trusted.

WHAT THE ANSWER MEANS.
  - PyGOD CLEAN     -> the collapse is our implementation bug. It stays in the
    paper as a documented pitfall, but it does not generalize, and the venue
    ceiling is correspondingly lower.
  - PyGOD COLLAPSES -> a widely-used reference implementation silently disables
    its own graph pathway while still reporting respectable AUROC. Much
    stronger finding, much higher ceiling.

Run:
  ~/envs/dgl311/bin/python scripts/pygod_architecture_check.py --n_nodes 3000 --epochs 100
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import inspect
import numpy as np
import torch

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator


def to_pyg(graph, features, labels):
    from torch_geometric.data import Data
    edges = np.array(list(graph.edges()), dtype=np.int64)
    both = np.concatenate([edges, edges[:, ::-1]], axis=0)
    return Data(x=torch.tensor(features, dtype=torch.float32),
                edge_index=torch.tensor(both.T, dtype=torch.long),
                y=torch.tensor(labels, dtype=torch.long))


def show_source(obj, label):
    print(f"\n  ### {label}")
    try:
        for line in inspect.getsource(obj).splitlines():
            print("    " + line)
    except Exception as e:
        print(f"    (unavailable: {e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hid_dim", type=int, default=64)
    args = parser.parse_args()

    cfg = GraphGenConfig(n_nodes=args.n_nodes, p_aa=0.3, p_an=0.002, p_nn=0.005,
                         feature_shift=1.0, n_anomaly_clusters=3, random_state=0)
    graph, features, labels = ContaminatedGraphGenerator(cfg).generate()
    data = to_pyg(graph, features, labels)
    print(f"graph: {graph.number_of_nodes():,} nodes, {int(labels.sum()):,} anomalies, "
          f"{graph.number_of_edges():,} edges")

    try:
        import pygod
        from pygod.nn import DOMINANTBase
    except Exception as e:
        print(f"\nCANNOT IMPORT PyGOD internals ({type(e).__name__}: {e})")
        print("Try: ~/envs/dgl311/bin/pip install pygod --upgrade")
        return
    print(f"pygod version: {getattr(pygod, '__version__', 'unknown')}")

    # ---------------- source inspection ----------------
    print(f"\n{'=' * 78}\nSOURCE -- does PyGOD activate the FINAL encoder layer?\n{'=' * 78}")
    show_source(DOMINANTBase, "pygod.nn.DOMINANTBase")

    # ---------------- build + train full batch ----------------
    print(f"\n{'=' * 78}\nTRAINING (full batch, no NeighborLoader)\n{'=' * 78}")
    model = None
    for kwargs in (
        dict(in_dim=features.shape[1], hid_dim=args.hid_dim, num_layers=4, dropout=0.0, act=torch.nn.functional.relu),
        dict(in_dim=features.shape[1], hid_dim=args.hid_dim, num_layers=4),
        dict(in_dim=features.shape[1], hid_dim=args.hid_dim),
    ):
        try:
            model = DOMINANTBase(**kwargs)
            print(f"  constructed with: {sorted(kwargs)}")
            break
        except Exception as e:
            print(f"  ctor attempt failed ({sorted(kwargs)}): {type(e).__name__}: {e}")
    if model is None:
        print("  Could not construct DOMINANTBase; signature is:")
        print("   ", inspect.signature(DOMINANTBase.__init__))
        return

    backbone = getattr(model, "shared_encoder", None)
    if backbone is not None:
        show_source(type(backbone).forward, f"backbone {type(backbone).__name__}.forward")

    # double_recon_loss needs the DENSE adjacency as its `s` argument. PyGOD
    # supplies it via DOMINANTBase.process_graph, which sets data.s -- use that
    # rather than building it by hand, so the loss matches PyGOD's exactly.
    DOMINANTBase.process_graph(data)

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(args.epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        x_, s_ = out[0], out[1]
        loss = model.loss_func(data.x, x_, data.s, s_)
        loss = loss.mean() if loss.dim() > 0 else loss
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss={loss.item():.6f}")

    # ---------------- measure the embedding ----------------
    print(f"\n{'=' * 78}\nEMBEDDING\n{'=' * 78}")
    model.eval()
    with torch.no_grad():
        Z = model.shared_encoder(data.x, data.edge_index).cpu().numpy()

    frac_zero = float((Z == 0).mean())
    collapsed = bool((Z == 0).all())
    per_dim_std = Z.std(axis=0)
    print(f"  Z shape={Z.shape}  frac_zero={frac_zero:.4f}  std={Z.std():.8f}  "
          f"dead dims={int((per_dim_std == 0).sum())}/{Z.shape[1]}")
    print(f"  min={Z.min():.6f}  max={Z.max():.6f}")
    print(f"  any negative values: {bool((Z < 0).any())}   "
          f"(a NEGATIVE value proves no final ReLU -- the structural difference)")

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(f"  ours (known)  : COLLAPSED  frac_zero=1.0000  std=0.0")
    print(f"  pygod         : {'COLLAPSED' if collapsed else 'ALIVE'}  "
          f"frac_zero={frac_zero:.4f}  std={Z.std():.8f}")
    if collapsed:
        print("""
  PYGOD COLLAPSES TOO. Not our bug -- a property of the reference
  implementation. This becomes the paper's headline: a widely-used detector's
  graph pathway is silently inert under common settings while still reporting
  respectable AUROC, because feature magnitude carries the signal.""")
    else:
        print("""
  PYGOD IS CLEAN. The collapse is specific to our encoder, which applies ReLU
  to EVERY layer including the last; PyGOD's backbone leaves the final
  embedding linear. Check "any negative values" above -- negatives confirm no
  final activation, which is the structural difference.

  For the paper: report this as a documented implementation pitfall with a
  concrete fix, not as a general property of DOMINANT. It does not generalize,
  so it does not carry a top-tier venue on its own.""")


if __name__ == "__main__":
    main()
