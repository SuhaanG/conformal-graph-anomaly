"""
pygod_introspect.py

Prints the real constructor and loss_func signatures for every detector module
PyGOD exposes, plus what forward() actually returns on a small graph.

The PYGOD_SPECS table in src/detectors.py has to name each detector's exact
constructor kwargs, and those differ per detector and per PyGOD version. Guessing
them produced three failures on pygod 1.1.0 (anomalydae, done, guide rejected
their kwargs; gae returned a scalar loss where the generic scorer expects a
per-node vector). Rather than guess again, read the ground truth off the
installed package and fix the specs once.

Run:
  ~/envs/dgl311/bin/python scripts/pygod_introspect.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import inspect
import numpy as np
import torch


def to_pyg(graph, features):
    from torch_geometric.data import Data
    edges = np.array(list(graph.edges()), dtype=np.int64)
    both = np.concatenate([edges, edges[:, ::-1]], axis=0)
    return Data(x=torch.tensor(features, dtype=torch.float32),
                edge_index=torch.tensor(both.T, dtype=torch.long))


def main():
    import pygod
    from pygod import nn as pygod_nn
    print(f"pygod {getattr(pygod, '__version__', '?')}\n")

    names = [n for n in dir(pygod_nn) if n.endswith("Base") and not n.startswith("_")]
    print(f"exported *Base modules: {names}\n")

    from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
    cfg = GraphGenConfig(n_nodes=300, p_aa=0.3, p_an=0.002, p_nn=0.005,
                         feature_shift=1.0, n_anomaly_clusters=3, random_state=0)
    graph, features, _ = ContaminatedGraphGenerator(cfg).generate()
    in_dim = features.shape[1]

    for name in names:
        Base = getattr(pygod_nn, name)
        print("=" * 78)
        print(name)
        print("=" * 78)
        try:
            sig = inspect.signature(Base.__init__)
            print("  __init__:")
            for pname, p in sig.parameters.items():
                if pname == "self":
                    continue
                default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
                print(f"      {pname}{default}")
        except Exception as e:
            print(f"  (could not read __init__: {e})")

        # loss_func may be a bound method, a staticmethod, or an attribute
        # pointing at a module-level function -- check the instance later too.
        lf = getattr(Base, "loss_func", None)
        if lf is not None:
            try:
                print(f"  loss_func: {inspect.signature(lf)}")
            except Exception:
                print(f"  loss_func: {lf!r} (signature unavailable)")

        # Try to actually build and run it with minimal args, to see what
        # forward returns and whether the loss is per-node or scalar.
        try:
            model = Base(in_dim=in_dim)
        except Exception as e:
            print(f"  construct(in_dim=...) FAILED: {type(e).__name__}: {e}")
            print()
            continue

        data = to_pyg(graph, features)
        if hasattr(Base, "process_graph"):
            try:
                Base.process_graph(data)
                print(f"  process_graph: sets {[k for k in ('s',) if hasattr(data, k)]}")
            except Exception as e:
                print(f"  process_graph FAILED: {type(e).__name__}: {e}")

        try:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
            if isinstance(out, (tuple, list)):
                print(f"  forward -> tuple of {len(out)}: "
                      f"{[tuple(o.shape) if torch.is_tensor(o) else type(o).__name__ for o in out]}")
            else:
                print(f"  forward -> {tuple(out.shape)}")
        except Exception as e:
            print(f"  forward FAILED: {type(e).__name__}: {e}")
            print()
            continue

        # instance-level loss_func, and whether it returns per-node or scalar
        ilf = getattr(model, "loss_func", None)
        if ilf is not None:
            try:
                print(f"  instance loss_func: {inspect.signature(ilf)}")
            except Exception:
                pass
            for label, args in [
                ("(x, x_, s, s_)", lambda o: (data.x, o[0], getattr(data, "s", None), o[1])),
                ("(x, x_)",        lambda o: (data.x, o[0])),
            ]:
                try:
                    with torch.no_grad():
                        val = ilf(*args(out))
                    shape = tuple(val.shape) if torch.is_tensor(val) else type(val).__name__
                    kind = "PER-NODE" if torch.is_tensor(val) and val.dim() == 1 else "scalar/other"
                    print(f"  loss_func{label} -> {shape}  [{kind}]")
                    break
                except Exception as e:
                    print(f"  loss_func{label} rejected: {type(e).__name__}: {str(e)[:70]}")
        print()

    print("=" * 78)
    print("""
Use this to fix PYGOD_SPECS in src/detectors.py:
  - copy the exact kwarg NAMES from __init__ (defaults are usually fine; the
    only ones worth setting are hidden-dim and dropout, for comparability)
  - note which loss_func arity each accepts, and whether it returns a PER-NODE
    vector. A scalar loss cannot be used as an anomaly score directly and needs
    per-node reconstruction error computed explicitly instead.
""")


if __name__ == "__main__":
    main()
