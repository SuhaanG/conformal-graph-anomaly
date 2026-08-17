"""
pygod_dominant_collapse_check.py

THE PIVOTAL QUESTION for this project's scope: does the dead-embedding collapse
we found in our DOMINANT implementation also affect PyGOD's DOMINANT -- the
widely-used reference implementation -- or is it specific to our code?

WHAT WE FOUND IN OUR IMPLEMENTATION (see DETECTOR_DIAGNOSTIC.md). The GCN
encoder's final layer is dead: ReLU on the final embedding forces Z >= 0, so
sigmoid(z_i . z_j) >= 0.5 for every pair, but ~99.7% of node pairs are
non-edges pulling that toward 0. The loss is minimized at Z = 0 exactly.
Confirmed 6/6 across sizes and seeds including n=15,000. Consequences:
A_hat = 0.5 uniformly, struct_err = 0.25n for every node (a constant that
cannot affect ranking), and the detector's ~0.91 AUROC comes entirely from raw
feature magnitude -- the graph pathway contributes nothing.

WHY IT MATTERS WHICH ANSWER WE GET.
  - PyGOD is CLEAN  -> this is our bug. It becomes a footnote in the paper,
    and the honest venue ceiling is a workshop or a mid-tier journal.
  - PyGOD COLLAPSES TOO -> a widely-used detector's graph pathway is silently
    inert under common settings while still reporting respectable AUROC,
    because feature magnitude carries the signal. That is a finding with real
    reach: a body of published reconstruction-based GAD results may rest on
    detectors that are not using the graph, and nobody would notice from the
    reported metrics. That is a different, much stronger paper.

HOW THIS CHECKS IT. Deliberately API-agnostic, since PyGOD's internals differ
across versions and we should not assume an attribute name:
  1. Prints the fitted model's full module tree.
  2. Registers forward hooks on EVERY submodule and records the fraction of
     exactly-zero entries and the standard deviation of each output. A dead
     embedding shows frac_zero = 1.0 and std = 0.
  3. Prints the source of the encoder/forward path so the activation on the
     final embedding layer can be read directly.
  4. Runs our own detector on the identical graph as a positive control -- we
     already know it collapses, so if our run does NOT collapse here, the
     harness itself is wrong and the PyGOD result cannot be trusted either.

Interpreting the output: look for a hooked module whose output is
frac_zero=1.0000 with std=0.000000, sitting at the END of the encoder (not a
decoder). That is the collapse. Intermediate ReLUs legitimately produce
frac_zero around 0.4-0.6; that is normal and not the finding.

Run in the dgl311 env (CPU is fine, the graph is small):
  ~/envs/dgl311/bin/python scripts/pygod_dominant_collapse_check.py --n_nodes 3000 --epochs 100
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
from detector import train_dominant


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / counts)[inv]
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def to_pyg(graph, features, labels):
    from torch_geometric.data import Data
    edges = np.array(list(graph.edges()), dtype=np.int64)
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        # undirected -> both directions, as PyG expects
        both = np.concatenate([edges, edges[:, ::-1]], axis=0)
        edge_index = torch.tensor(both.T, dtype=torch.long)
    return Data(x=torch.tensor(features, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(labels, dtype=torch.long))


def our_control(graph, features, labels, epochs):
    """Positive control. We already know our implementation collapses. If this
    does not reproduce, the harness is wrong and nothing else here is valid."""
    print(f"\n{'=' * 78}\nPOSITIVE CONTROL -- our implementation on the same graph\n{'=' * 78}")
    from detector import normalize_adj
    import networkx as nx

    scores, model = train_dominant(graph, features, n_epochs=epochs, seed=0, verbose=False)
    n = graph.number_of_nodes()
    A = nx.to_numpy_array(graph, nodelist=range(n))
    A_norm = normalize_adj(A)
    X = torch.tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Z = model.encode(A_norm, X)
    Z = Z.cpu().numpy()
    collapsed = bool((Z == 0).all())
    print(f"  embedding Z: shape={Z.shape}  frac_zero={(Z == 0).mean():.4f}  std={Z.std():.8f}")
    print(f"  AUROC={auroc(scores, labels):.4f}")
    print(f"  -> COLLAPSED: {collapsed}   (expected True; if False the harness is suspect)")
    return collapsed


def pygod_check(graph, features, labels, epochs, device):
    print(f"\n{'=' * 78}\nPYGOD DOMINANT\n{'=' * 78}")
    try:
        import pygod
        from pygod.detector import DOMINANT as PyGODDominant
    except Exception as e:
        print(f"  CANNOT IMPORT PyGOD ({type(e).__name__}: {e})")
        return None
    print(f"  pygod version: {getattr(pygod, '__version__', 'unknown')}")

    data = to_pyg(graph, features, labels)
    det = PyGODDominant(epoch=epochs, hid_dim=64, num_layers=2, gpu=-1 if device == "cpu" else 0)

    print("\n  --- fitting ---")
    det.fit(data)

    model = getattr(det, "model", None)
    if model is None:
        print("  Could not locate .model on the detector; attributes available:")
        print("   ", [a for a in dir(det) if not a.startswith("_")][:40])
        return None

    print(f"\n  --- module tree ---")
    print("   ", str(model).replace("\n", "\n    "))

    # Hook every submodule and record what its output looks like on a forward pass.
    records = {}

    def make_hook(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
                arr = t.detach().cpu()
                records[name] = (tuple(arr.shape), float((arr == 0).float().mean()), float(arr.std()))
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in model.named_modules() if n]

    model.eval()
    with torch.no_grad():
        try:
            model(data.x, data.edge_index)
        except Exception:
            # signature differs across versions; try the detector's own path
            try:
                det.predict(data)
            except Exception as e2:
                print(f"  Could not run a forward pass: {e2}")
                for h in handles:
                    h.remove()
                return None
    for h in handles:
        h.remove()

    print(f"\n  --- per-module output statistics ---")
    print(f"  {'module':38} {'shape':>18} {'frac_zero':>10} {'std':>12}")
    dead = []
    for name, (shape, fz, sd) in records.items():
        flag = ""
        if fz >= 0.9999 and sd == 0.0:
            flag = "  <-- DEAD"
            dead.append(name)
        print(f"  {name[:38]:38} {str(shape):>18} {fz:10.4f} {sd:12.6f}{flag}")

    scores = np.asarray(det.decision_score_, dtype=np.float64)
    print(f"\n  PyGOD AUROC on this graph: {auroc(scores, labels):.4f}")

    print(f"\n  --- source of the encoder/forward path ---")
    for obj, label in [(type(model), "model class")]:
        try:
            src = inspect.getsource(obj)
            print(f"  ### {label}: {obj.__name__}")
            for line in src.splitlines():
                print("    " + line)
        except Exception as e:
            print(f"  (could not read source for {label}: {e})")

    return dead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = GraphGenConfig(n_nodes=args.n_nodes, p_aa=0.3, p_an=0.002, p_nn=0.005,
                         feature_shift=1.0, n_anomaly_clusters=3, random_state=0)
    graph, features, labels = ContaminatedGraphGenerator(cfg).generate()
    print(f"graph: {graph.number_of_nodes():,} nodes, {int(labels.sum()):,} anomalies, "
          f"{graph.number_of_edges():,} edges")

    ours_collapsed = our_control(graph, features, labels, args.epochs)
    dead = pygod_check(graph, features, labels, args.epochs, args.device)

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(f"  our implementation collapsed : {ours_collapsed}")
    if dead is None:
        print("  PyGOD                        : CHECK DID NOT COMPLETE (see errors above)")
    elif dead:
        print(f"  PyGOD dead modules           : {dead}")
        print("""
  PyGOD COLLAPSES TOO. This is no longer our bug -- it is a property of the
  reference implementation. Verify the dead module sits at the END of the
  encoder (not a decoder), then this becomes the paper's headline: a widely
  used detector's graph pathway is silently inert while still reporting
  respectable AUROC, because feature magnitude carries the signal.""")
    else:
        print("""  PyGOD dead modules           : none

  PyGOD IS CLEAN. The collapse is specific to our implementation. It stays in
  the paper as a documented methodological pitfall, but it does not generalize,
  and the venue ceiling is correspondingly lower. Worth diffing our encoder
  against PyGOD's (printed above) to identify exactly which difference matters
  -- most likely the activation on the final embedding layer.""")


if __name__ == "__main__":
    main()
